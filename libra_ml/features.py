"""Causal features and training targets for the scaling decision.

At a decision time t a feature may use counts[0..t] only (the same history the
simulator hands to an estimator). Every demand-valued feature is divided by
``scale`` = mean demand of the last 300 s, and the target is future demand divided
by the same scale, so one model can serve traces with different absolute rates.
"""
import math

import numpy as np
from scipy.signal import lfilter

SCALE_WINDOW = 300
MEAN_WINDOWS = (30, 60, 120, 600, 1800)
EWMA_ALPHAS = (0.2, 0.02, 0.005)
FEATURE_NAMES = (
    ("current",)
    + tuple(f"mean_{w}s" for w in MEAN_WINDOWS)
    + ("std_60s", "std_300s", "q50_300s", "q80_300s", "q95_300s", "max_300s",
       "q80_prev_300s", "q80_prev_600s", "slope_300s", "slope_1800s")
    + tuple(f"ewma_{a}" for a in EWMA_ALPHAS)
    + ("abs_dev_ewma_0.2", "history_fraction")
)


def newsvendor_tau(config):
    """Cost-optimal quantile of per-second demand to provision for.

    One VM request slot costs vm_price/3600/capacity per second whether used or not;
    a request it does not absorb costs one FaaS invocation. The extra slot pays off
    when P(demand > slot) > slot_cost / faas_cost, i.e. provision the tau-quantile
    with tau = 1 - slot_cost / faas_cost (0.797 with the paper's prices).
    """
    if config.service_seconds != 1:
        raise ValueError("Derivation assumes one-second service, as in the simulator")
    slot = config.vm_hourly_price / 3600 / config.vm_capacity
    tau = 1 - slot / config.faas_cost_per_request
    if not 0 < tau < 1:
        raise ValueError(f"No interior optimum: tau={tau}")
    return tau


def _ewma(x, alpha):
    y = np.empty_like(x)
    y[0] = x[0]
    if len(x) > 1:
        y[1:] = lfilter([alpha], [1, -(1 - alpha)], x[1:], zi=[(1 - alpha) * x[0]])[0]
    return y


def _abs_dev_ewma(x, avg, beta):
    """Paper Eq. (7) with the updated mean, initial deviation 0 (as libra_sim.Monitor)."""
    d = np.zeros_like(x)
    if len(x) > 1:
        d[1:] = lfilter([beta], [1, -(1 - beta)], np.abs(x[1:] - avg[1:]), zi=[0.0])[0]
    return d


def _slope(w):
    if len(w) < 2:
        return 0.0
    i = np.arange(len(w), dtype=float)
    i -= i.mean()
    return float(np.dot(i, w - w.mean()) / np.dot(i, i))


def feature_matrix(counts, times, scale_quantile=None):
    """Features at each decision time; returns (X, scale).

    ``scale`` is the mean of the last 300 s, or with ``scale_quantile`` that
    quantile of the last 300 s. With scale_quantile=tau a model predicts a
    multiplicative correction to the window-quantile rule (1.0 = the rule itself).
    """
    times = np.asarray(times, dtype=int)
    if times.size == 0 or times.min() < 0:
        raise ValueError("times must be nonempty and nonnegative")
    x = np.asarray(counts, dtype=float)[: times.max() + 1]
    if len(x) <= times.max():
        raise ValueError("decision time beyond the available history")
    cs = np.concatenate(([0.0], np.cumsum(x)))
    cs2 = np.concatenate(([0.0], np.cumsum(x * x)))
    ewmas = [_ewma(x, a) for a in EWMA_ALPHAS]
    dev = _abs_dev_ewma(x, ewmas[0], EWMA_ALPHAS[0])

    def window(t, w):
        lo = max(0, t + 1 - w)
        n = t + 1 - lo
        mean = (cs[t + 1] - cs[lo]) / n
        var = max(0.0, (cs2[t + 1] - cs2[lo]) / n - mean * mean)
        return mean, math.sqrt(var)

    rows, scales = [], []
    for t in times:
        last = x[max(0, t + 1 - SCALE_WINDOW): t + 1]
        scale = max(1.0, window(t, SCALE_WINDOW)[0] if scale_quantile is None
                    else float(np.quantile(last, scale_quantile)))
        q50, q80, q95 = np.quantile(last, (0.5, 0.8, 0.95))
        prev1 = x[max(0, t + 1 - 600): max(0, t + 1 - 300)]
        prev2 = x[max(0, t + 1 - 900): max(0, t + 1 - 600)]
        q_prev1 = np.quantile(prev1, 0.8) if len(prev1) else q80
        q_prev2 = np.quantile(prev2, 0.8) if len(prev2) else q_prev1
        long = x[max(0, t + 1 - 1800): t + 1]
        values = ([x[t]] + [window(t, w)[0] for w in MEAN_WINDOWS]
                  + [window(t, 60)[1], window(t, 300)[1], q50, q80, q95, last.max(),
                     q_prev1, q_prev2, _slope(last) * 300, _slope(long) * 300]
                  + [e[t] for e in ewmas] + [dev[t]])
        rows.append([v / scale for v in values] + [min(1.0, (t + 1) / 1800)])
        scales.append(scale)
    return np.asarray(rows), np.asarray(scales)


def training_set(counts, *, stride=30, lead=100, horizon=300, target_step=10,
                 scale_quantile=None):
    """Rows (features at t, demand at s / scale_t) for s in [t+lead, t+lead+horizon).

    New VMs requested at t serve from t+lead (cold start) until the next decision
    takes effect, so the pinball loss over these seconds is the provisioning cost
    of the decision (see policies.py). Only targets inside ``counts`` are used, so
    a caller that passes a training prefix never sees later data.
    """
    x = np.asarray(counts, dtype=float)
    last_t = len(x) - lead - horizon
    if last_t < 0:
        raise ValueError("trace shorter than one decision window")
    times = np.arange(0, last_t + 1, stride)
    X, scale = feature_matrix(x, times, scale_quantile)
    offsets = np.arange(lead, lead + horizon, target_step)
    rows = np.repeat(X, len(offsets), axis=0)
    y = np.concatenate([x[t + offsets] / s for t, s in zip(times, scale)])
    return rows, y
