"""Estimators for ``libra_sim.simulate(..., estimator=...)``.

Each returns a provisioning level in req/s; the simulator turns it into
ceil(level / 16) VMs and applies Algorithm 1's CIP rule.

Why a quantile: with fill-ready-capacity routing, provisioning x request slots
for a second with demand d costs  c_slot*x + c_faas*(d - x)^+
  = c_faas * [pinball_tau(d - x) + (1 - tau)*d],   tau = 1 - c_slot/c_faas,
so minimising expected cost is exactly minimising the tau-pinball loss
(quantile regression), up to VM granularity and cold-start billing.
"""
import math

import numpy as np

from .features import feature_matrix


def model_estimator(model, scale_quantile=None):
    """Wrap a fitted regressor that predicts future demand / scale (see feature_matrix)."""
    def estimate(t, history):
        X, scale = feature_matrix(history, [t], scale_quantile)
        return max(0.0, float(model.predict(X)[0]) * float(scale[0]))
    return estimate


def window_quantile_estimator(tau, window=300):
    """Non-ML baseline: empirical tau-quantile of the last ``window`` seconds."""
    def estimate(t, history):
        return float(np.quantile(np.asarray(history[-window:], dtype=float), tau))
    return estimate


def oracle_estimator(counts, config, *, lead=100, horizon=300):
    """Offline reference that READS FUTURE DEMAND; never a deployable policy.

    Picks the VM count that minimises the exact cost of [t+lead, t+lead+horizon)
    under fill routing. It ignores cold-start billing and the overlap with the
    next decision, so it is a reference point, not a proven lower bound.
    """
    x = np.asarray(counts, dtype=float)
    cap = config.vm_capacity
    vm_second = config.vm_hourly_price / 3600
    faas = config.faas_cost_per_request

    def estimate(t, history):
        w = x[t + lead: t + lead + horizon]
        if len(w) == 0:
            w = x[t: t + horizon]
        n_max = math.ceil(w.max() / cap)
        costs = [n * len(w) * vm_second + faas * np.maximum(0.0, w - n * cap).sum()
                 for n in range(n_max + 1)]
        return float(int(np.argmin(costs)) * cap)
    return estimate


def pinball(y, q, tau):
    diff = np.asarray(y, dtype=float) - q
    return np.mean(np.maximum(tau * diff, (tau - 1) * diff))
