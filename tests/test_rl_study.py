from dataclasses import replace
import json

import numpy as np
import pytest

from libra_rl.dp import solve_dp
from libra_rl.fast_dp import optimal_bound
from libra_rl.env import LibraEnv
from libra_rl.model import CostModel
from libra_rl.evaluation import evaluate_targets
from libra_rl.training import ObservationTransform, StudyEnv, choose_penalty
from libra_rl.baselines import BaselineEvaluator, grids, select_on_training
from libra_sim import Config, simulate


@pytest.mark.parametrize("k", [60, 120, 300])
@pytest.mark.parametrize("service,penalty", [(1.0, 0), (0.5, 0.0001), (0.7, 0.01)])
def test_fast_bound_matches_trajectory_dp(k, service, penalty):
    x = np.random.default_rng(3).poisson(36, 1007)
    model = CostModel(version="v1" if service == 1 else "v2", service_seconds=service)
    rho = (1.0,) if service == 1 else (0.6, 0.8, 1.0)
    expected = solve_dp(x, model=model, decision_seconds=k, max_vms=5, rho_grid=rho, penalty=penalty)
    got = optimal_bound(x, model=model, decision_seconds=k, max_vms=5, rho_grid=rho, penalty=penalty)
    assert got["objective_usd"] == pytest.approx(expected.objective_usd, abs=1e-10)
    assert got["total_cost_usd"] + penalty * got["expected_violations"] == pytest.approx(expected.objective_usd, abs=1e-10)


@pytest.mark.parametrize("service", [1.0, 0.5, 0.7])
def test_vectorized_evaluator_matches_env_with_cancelled_startups(service):
    x = np.arange(671) % 111
    model = CostModel(version="v1" if service == 1 else "v2", service_seconds=service)
    targets = [2, 3, 1, 5, 0, 7, 2, 2, 3, 0, 9, 1]
    rho = [0.6, 0.8, 1.0] * 4
    got = evaluate_targets(x, targets, model=model, rhos=rho)
    json.dumps(got)  # Persist cancelled-startup counts as native JSON numbers.
    env = LibraEnv(x, model=model, episode_seconds=None)
    env.reset()
    for n, r in zip(targets, rho):
        env.step_target(n, rho=r)
    for key, value in env.summary().items():
        assert got[key] == pytest.approx(value, abs=1e-9)


def test_vectorized_paper_baseline_matches_original():
    x = np.random.default_rng(4).poisson(25, 2003).tolist()
    c = Config()
    expected = simulate(x, config=c)["summary"]
    got = BaselineEvaluator(x, CostModel(c), None, max_vms=292).run("LIBRA", {"rho": 0.8, "phi": 1})
    assert got["total_cost_usd"] == pytest.approx(expected["total_cost_usd"], abs=1e-9)


def test_validation_only_selection_and_train_only_baseline_tuning():
    rows = [dict(split="validation", penalty=p, total_cost_usd=c, violation_rate=v)
            for p, c, v in ((0, 1, 0.02), (0.01, 2, 0.0001), (1, 3, 0))]
    assert choose_penalty(rows, 0.001) == 0.01
    rows[0]["split"] = "test"
    with pytest.raises(ValueError):
        choose_penalty(rows, 0.001)
    with pytest.raises(ValueError):
        select_on_training(rows, 0)
    assert all(len(candidates) == 12 for candidates in grids().values())


def test_finite_horizon_no_bootstrap_and_fixed_observation_transform():
    env = ObservationTransform(StudyEnv([15] * 121, episode_seconds=None), hide_forecast=True)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs[21] == obs[22] == 0
    while True:
        _, _, terminated, truncated, _ = env.step(3)
        assert not truncated
        if terminated:
            break
    ref = LibraEnv([15] * 121, episode_seconds=None)
    ref.reset()
    for _ in range(3):
        ref.step(3)
    assert env.unwrapped.summary() == ref.summary()


def test_sb3_checkpoint_reloads_selected_updated_policy(tmp_path):
    sb3 = pytest.importorskip("stable_baselines3")
    import torch
    from libra_rl.study import train_agent, evaluate_agent
    torch.set_num_threads(1)
    protocol = {"episode_seconds": 600, "epsilon": 0.001, "cost_normalizer_usd": 0.01,
                "network": [8, 8], "checkpoint_fractions": [0.25, 0.5, 0.75, 1.0],
                "ppo": {"n_steps": 128, "batch_size": 64, "n_epochs": 2, "gamma": 0.99}}
    job = {"id": "unit", "decision_seconds": 60, "action_mode": "residual", "augment": False,
           "penalty": 0, "hide_forecast": False, "seed": 0, "samples": 512}
    x = np.arange(1800) % 50
    model = CostModel()
    run = train_agent(x, x[:601], job=job, protocol=protocol, model=model, forecast_model=None,
                       max_vms=8, output=tmp_path, fingerprint="unit-test")
    agent = sb3.PPO.load(tmp_path / run["checkpoint"], device="cpu")
    assert agent.num_timesteps == run["selected_step"]
    assert agent._n_updates == run["selected_step"] // 128 * 2
    replay = evaluate_agent(agent, x[:601], model=model, forecast_model=None,
                             max_vms=8, penalty=0, episode_seconds=600)
    assert replay == run["validation"]
