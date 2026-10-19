from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
from libra_sim import Config
from libra_ml import newsvendor_tau, oracle_estimator
from scripts.prepare_azure_2021 import HORIZON, aggregate
from scripts.azure_ml_experiments import DAY, day_features, offline_levels


def test_azure_uses_seconds_and_reconstructs_arrivals(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("app,func,end_timestamp,duration\na,f,100.5,0.5\na,f,2.7,0.1\na,f,1.0,2.0\na,f,3,-1\n")
    counts, meta = aggregate(path)
    assert len(counts) == HORIZON
    assert counts[100] == counts[2] == 1
    assert counts.sum() == 2
    assert meta["source_rows"] == 4
    assert meta["outside_14_day_arrivals"] == meta["invalid_rows"] == 1


def test_calendar_and_day_lag_features_are_causal():
    x = np.asarray([int(v) for v in (10 + 5 * np.sin(np.arange(3 * DAY) / 86400 * 2 * np.pi))])
    times = np.asarray([0, DAY - 300, DAY + 300, 2 * DAY + 600])
    lag, clock, scale, seasonal = day_features(x, times, .8)
    for i, t in enumerate(times):
        a, b, c, d = day_features(x[:t + 1], np.asarray([t]), .8)
        np.testing.assert_allclose(a[0], lag[i])
        np.testing.assert_allclose(b[0], clock[i])
        assert c[0] == scale[i]
        assert d[0] == seasonal[i]


def test_fast_integer_oracle_matches_exhaustive_vm_search():
    c = Config()
    tau = newsvendor_tau(c)
    rng = np.random.default_rng(23)
    for x in (rng.integers(0, 500, 1000).tolist(), [0] * 899 + [10000]):
        expected = oracle_estimator(x, c)
        actual = offline_levels(x, c, tau)
        assert actual == [expected(t, x[:t + 1]) for t in range(0, len(x), 300)]
