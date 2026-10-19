import math

import numpy as np
import pytest

pytest.importorskip("scipy")
from libra_sim import Config, simulate                      # noqa: E402
from libra_ml import (feature_matrix, model_estimator,      # noqa: E402
                      newsvendor_tau, oracle_estimator, training_set,
                      window_quantile_estimator)
from libra_ml.policies import pinball                       # noqa: E402

RNG = np.random.default_rng(0)
TRACE = [int(v) for v in 150 + 60 * np.sin(np.arange(4000) / 300) + RNG.poisson(20, 4000)]


def test_tau_matches_paper_prices():
    tau = newsvendor_tau(Config())
    slot = 0.1 / 3600 / 16
    assert tau == pytest.approx(1 - slot / 8.535e-6)
    assert tau == pytest.approx(0.79659, abs=1e-5)


def test_pinball_loss_equals_provisioning_cost_up_to_constant():
    c = Config()
    tau = newsvendor_tau(c)
    slot = c.vm_hourly_price / 3600 / c.vm_capacity
    faas = c.faas_cost_per_request
    d = RNG.integers(0, 400, 1000).astype(float)
    for x in (0.0, 37.5, 160.0, 512.0):
        cost = np.mean(slot * x + faas * np.maximum(0, d - x))
        assert cost == pytest.approx(faas * (pinball(d, x, tau) + (1 - tau) * d.mean()))


@pytest.mark.parametrize("scale_quantile", [None, 0.8])
def test_features_are_causal(scale_quantile):
    for t in (0, 1, 299, 300, 1799, 2500, len(TRACE) - 1):
        full, s_full = feature_matrix(TRACE, [t], scale_quantile)
        past, s_past = feature_matrix(TRACE[:t + 1], [t], scale_quantile)
        np.testing.assert_allclose(full, past)
        np.testing.assert_allclose(s_full, s_past)


def test_ewma_feature_matches_simulator_monitor():
    from libra_sim.engine import Monitor
    from libra_ml.features import FEATURE_NAMES
    m = Monitor(0.2, 0.2)
    for n in TRACE[:1001]:
        avg, dev = m.observe(n)
    X, scale = feature_matrix(TRACE, [1000])
    names = list(FEATURE_NAMES)
    assert X[0, names.index("ewma_0.2")] * scale[0] == pytest.approx(avg)
    assert X[0, names.index("abs_dev_ewma_0.2")] * scale[0] == pytest.approx(dev)


def test_training_targets_stay_inside_given_series():
    X, y = training_set(TRACE[:1000], stride=100)
    times = np.arange(0, 1000 - 400 + 1, 100)
    assert X.shape[0] == len(y) == len(times) * 30
    assert np.isfinite(X).all() and np.isfinite(y).all()


def test_oracle_picks_cheapest_vm_count():
    c = Config()
    est = oracle_estimator([40] * 1000, c)
    # 3 VMs (48 slots) beat 2 VMs + 8 FaaS requests/s at the paper's prices.
    assert est(0, [40]) == 48.0
    assert oracle_estimator([0] * 1000, c)(0, [0]) == 0.0


def test_window_quantile_and_model_estimators():
    tau = newsvendor_tau(Config())
    hist = TRACE[:700]
    assert window_quantile_estimator(tau)(699, hist) == pytest.approx(np.quantile(hist[-300:], tau))

    class Unit:
        def predict(self, X):
            return np.ones(len(X))
    _, scale = feature_matrix(hist, [699])
    assert model_estimator(Unit())(699, hist) == pytest.approx(scale[0])


def test_learned_policy_end_to_end():
    pytest.importorskip("sklearn")
    from dataclasses import replace
    from libra_ml.models import fit_gbdt
    c = Config()
    tau = newsvendor_tau(c)
    X, y = training_set(TRACE[:2000], stride=60)
    model = fit_gbdt(X, y, tau)
    test = TRACE[2000:]
    r = simulate(test, "LIBRA", replace(c, rho=1.0), routing="capacity",
                 estimator=model_estimator(model))
    s = r["summary"]
    assert s["vm_requests"] + s["faas_requests"] == sum(test)
    assert s["sla_violations"] == 0
    assert all(d["target_vms"] == math.ceil(d["estimate"] / 16) for d in r["decisions"] if d["estimate"])
