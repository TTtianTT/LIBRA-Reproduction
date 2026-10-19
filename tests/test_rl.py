from dataclasses import replace
from itertools import product
import math

import numpy as np
import pytest

pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env

from libra_ml.features import newsvendor_tau
from libra_ml.policies import window_quantile_estimator
from libra_sim import Config, simulate
from libra_rl.dp import evaluate_schedule, solve_dp
from libra_rl.env import LibraEnv
from libra_rl.model import CostModel
from libra_rl.queueing import erlang_c, violation_probability


@pytest.mark.parametrize("interval", [120, 300])
@pytest.mark.parametrize("routing", ["estimate", "capacity"])
def test_fixed_policy_matches_original(interval, routing):
    rng = np.random.default_rng(13)
    x = np.concatenate([rng.poisson(v, n) for v, n in
                        ((30, 400), (190, 400), (0, 400), (70, 513))]).tolist()
    c = replace(Config(), scaling_interval_seconds=interval,
                rho=0.8 if routing == "estimate" else 1.0)
    estimator = None if routing == "estimate" else window_quantile_estimator(newsvendor_tau(c))
    ref = simulate(x, config=c, routing=routing, estimator=estimator)
    env = LibraEnv(x, model=CostModel(c), decision_seconds=interval, episode_seconds=None)
    env.reset(seed=0)
    for decision in ref["decisions"]:
        env.step_target(decision["target_vms"], estimate=decision["estimate"], rho=c.rho, routing=routing)
    got = env.summary()
    for key in ("vm_cost_usd", "faas_cost_usd", "gateway_cost_usd", "total_cost_usd"):
        assert got[key] == pytest.approx(ref["summary"][key], abs=1e-9, rel=0)
    assert got["expected_violations"] == 0


@pytest.mark.parametrize("version,service", [("v1", 1.0), ("v2", 0.5), ("v2", 0.7)])
@pytest.mark.parametrize("action_mode", ["residual", "raw"])
def test_gym_contract(version, service, action_mode):
    env = LibraEnv(np.arange(500) % 80, model=CostModel(version=version, service_seconds=service),
                   action_mode=action_mode, episode_seconds=240)
    check_env(env, skip_render_check=True)


def test_overlapping_startups_cancel_newest_and_charge_all_time():
    env = LibraEnv([100] * 240, decision_seconds=60, episode_seconds=None)
    env.reset()
    _, _, _, _, first = env.step_target(2)
    assert first["vm_requests"] == 0
    assert first["starting_at_end"] == 2
    _, _, _, _, second = env.step_target(3)
    assert second["vm_requests"] == 20 * 32  # first cohort ready at t=100
    assert second["starting_at_end"] == 1
    _, _, _, _, third = env.step_target(1)
    assert third["vm_requests"] == 60 * 16
    assert env.summary()["cancelled_startup_vm_seconds"] == 60  # cancelled t=60 cohort
    assert env.summary()["vm_seconds"] == (2 + 3 + 1) * 60
    assert env.summary()["starting_vm_seconds"] == 2 * 100 + 60


def test_causal_observations_and_rule():
    a = np.arange(900) % 83
    b = a.copy()
    b[301:] = 10000
    envs = [LibraEnv(x, episode_seconds=None) for x in (a, b)]
    obs = [env.reset(seed=0)[0] for env in envs]
    np.testing.assert_array_equal(*obs)
    for _ in range(5):
        out = [env.step(3)[0] for env in envs]
        np.testing.assert_array_equal(*out)
        assert envs[0].rule_target() == envs[1].rule_target()


def test_forecast_slot_and_seeded_augmentation():
    class ConstantForecast:
        def predict(self, features):
            return np.full(len(features), 1.25)
    env = LibraEnv(np.arange(1000) % 21, augment=True, episode_seconds=300,
                   forecast_model=ConstantForecast())
    a, info_a = env.reset(seed=42)
    first = env.step(3)
    b, info_b = env.reset(seed=42)
    second = env.step(3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[0], second[0])
    assert info_a == info_b
    assert a[21:23].tolist() == [1.25, 1.0]
    assert first[4] == second[4]


def test_all_faas_is_billed_and_episode_boundary():
    env = LibraEnv([10] * 121, episode_seconds=None, penalty=0.02)
    env.reset()
    done = False
    while not done:
        _, _, terminated, done, _ = env.step_target(0)
        assert not terminated
    got = env.summary()
    assert got["faas_requests"] == 1210
    assert got["total_cost_usd"] == pytest.approx(1210 * Config().faas_cost_per_request
                                                  + 121 * Config().gateway_hourly_price / 3600)
    with pytest.raises(RuntimeError):
        env.step_target(0)


def test_latency_table_and_full_response_difference():
    table = np.array([violation_probability([0.7, 0.8, 0.9], s) for s in (1.0, 0.7, 0.5)])
    np.testing.assert_allclose(table, [[0.130, 0.305, 0.591], [0.017, 0.077, 0.298],
                                     [0.001, 0.012, 0.119]], atol=0.001)
    assert np.all(np.diff(table, axis=1) > 0)
    assert erlang_c(0) == 0
    assert erlang_c(1) == 1
    for s in (0.5, 0.7, 1.0):
        assert violation_probability(0, s, mode="mmc_response") == pytest.approx(math.exp(-1 / s))
        assert violation_probability(1 - 1 / 16, s, mode="mmc_response") == pytest.approx(
            (1 - erlang_c(15 / 16)) * math.exp(-1 / s)
            + erlang_c(15 / 16) * math.exp(-1 / s) * (1 + 1 / s))


@pytest.mark.parametrize("interval", [60, 120, 300])
@pytest.mark.parametrize("version,service,penalty", [("v1", 1.0, 0.0), ("v2", 0.7, 0.001)])
def test_dp_matches_exhaustive_and_environment(interval, version, service, penalty):
    # Four decisions, a truncated final interval, all 3**4 VM sequences.
    x = np.concatenate([np.full(interval, v, dtype=int) for v in (11, 39, 2, 28)])[:-13]
    model = CostModel(version=version, service_seconds=service)
    rho_grid = (1.0,) if version == "v1" else (0.7, 0.9)
    dp = solve_dp(x, model=model, decision_seconds=interval, max_vms=2,
                  rho_grid=rho_grid, penalty=penalty)
    optimum = math.inf
    # Independent enumeration with per-second startup clocks, rather than DP states.
    for targets in product(range(3), repeat=4):
        for rhos in product(rho_grid, repeat=4):
            out = evaluate_schedule(x, targets, model=model, decision_seconds=interval, rhos=rhos)
            optimum = min(optimum, out["total_cost_usd"] + penalty * out["expected_violations"])
    assert dp.objective_usd == pytest.approx(optimum, abs=1e-12, rel=0)
    env = LibraEnv(x, model=model, decision_seconds=interval, episode_seconds=None, max_vms=2,
                   penalty=penalty)
    env.reset()
    for target, rho in zip(dp.targets, dp.rhos):
        env.step_target(target, rho=rho)
    assert env.summary()["total_cost_usd"] == pytest.approx(dp.total_cost_usd, abs=1e-12, rel=0)
    assert env.summary()["expected_violations"] == pytest.approx(dp.expected_violations, abs=1e-9, rel=0)


def test_dp_rejects_unsupported_domain_and_unbounded_v2():
    with pytest.raises(ValueError, match="ceil"):
        solve_dp([1] * 500, decision_seconds=30)
    with pytest.raises(ValueError, match="explicit"):
        solve_dp([1] * 500, model=CostModel(version="v2", service_seconds=0.5))
    zero = solve_dp([0] * 127, decision_seconds=60)
    assert zero.max_vms == 0
    assert zero.total_cost_usd == pytest.approx(127 * Config().gateway_hourly_price / 3600)
