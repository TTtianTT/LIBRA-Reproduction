"""Train-tuned baseline grids with common v2 routing/latency accounting.

AUTO/SPOCK here are explicitly adapted target controllers. AUTO observes routed
VM utilization; SPOCK observes offered-load pressure including overflow. Their
shared hybrid router has no persistent FIFO, unlike original AUTO/SPOCK.
"""
from itertools import product
import math

import numpy as np
from scipy.signal import lfilter

from libra_ml.features import feature_matrix
from .evaluation import evaluate_targets


def grids():
    rng = np.random.default_rng(20261019)
    thresholds = list(product((0.4, 0.5, 0.6, 0.7, 0.8), (0.1, 0.2, 0.3),
                              (5, 10, 20), (30, 60, 120), (5, 10)))
    selected = rng.choice(len(thresholds), 12, replace=False)
    controllers = [dict(zip(("out", "in", "group", "window", "minimum"), thresholds[i]),
                        rho=(0.6, 0.8, 1.0)[j % 3]) for j, i in enumerate(selected)]
    quantiles = [dict(rho=rho, window=window) for window, rho in
                 product((120, 300, 600), (0.6, 0.7, 0.8, 0.9))]
    # Include the strong v1 fill-capacity baseline within the same 12-candidate budget.
    quantiles[-1] = dict(rho=1.0, window=300)
    return {"LIBRA": [dict(rho=r, phi=p) for r, p in product((0.6, 0.8, 1.0), (0, 1, 2, 4))],
            "Quantile": quantiles,
            "GBDT": [dict(rho=r, multiplier=m) for r, m in
                     product((0.6, 0.8, 0.9, 1.0), (0.8, 1.0, 1.2))],
            "AUTO-adapted": controllers, "SPOCK-adapted": controllers}


def _ewma(x, alpha=0.2):
    y = np.empty(len(x), dtype=float)
    y[0] = x[0]
    if len(x) > 1:
        y[1:] = lfilter([alpha], [1, -(1 - alpha)], x[1:], zi=[(1 - alpha) * x[0]])[0]
    return y


class BaselineEvaluator:
    def __init__(self, counts, model, forecast_model, *, max_vms, decision_seconds=60):
        self.x = np.asarray(counts, dtype=np.int64)
        self.model, self.forecast_model = model, forecast_model
        self.max_vms, self.k = max_vms, decision_seconds
        self.times = np.arange(0, len(self.x), self.k)
        self._windows = {}
        self._predictions = None
        self._paper = None

    def _quantile(self, window):
        if window not in self._windows:
            self._windows[window] = np.array([np.quantile(self.x[max(0, t-window+1):t+1], self.model.tau)
                                             for t in self.times])
        return self._windows[window]

    def run(self, family, params):
        rho = params.get("rho", 1.0)
        estimates, k = None, self.k
        if family == "FaaS":
            targets = np.zeros(len(self.times), dtype=int)
        elif family == "Quantile":
            q = self._quantile(params["window"])
            targets = np.where(q < 4, 0, np.ceil(q / self.model.throughput / rho)).astype(int)
        elif family == "GBDT":
            if self._predictions is None:
                features, scales = feature_matrix(self.x, self.times)
                self._predictions = np.maximum(0, self.forecast_model.predict(features)) * scales
            q = self._predictions * params["multiplier"]
            targets = np.where(q < 4, 0, np.ceil(q / self.model.throughput / rho)).astype(int)
        elif family == "LIBRA":
            k = 300  # Keep paper's original decision clock/routing semantics.
            if self._paper is None:
                avg = _ewma(self.x)
                delta = np.abs(self.x - avg)
                delta[0] = 0
                dev = _ewma(delta)
                self._paper = avg[::k], dev[::k]
            avg, dev = self._paper
            estimates = np.where(avg < 4, 0, np.maximum(0, avg + params["phi"] * dev))
            targets = np.ceil(estimates / self.model.throughput).astype(int)
        elif family in ("AUTO-adapted", "SPOCK-adapted"):
            targets = self._controller(family, params)
        else:
            raise ValueError(f"unknown baseline {family}")
        targets = np.clip(targets, 0, self.max_vms)
        return evaluate_targets(self.x, targets, model=self.model, decision_seconds=k,
                                rhos=rho, estimates=estimates)

    def _controller(self, family, p):
        ready_at, utilization, targets = [], [], []
        target = p["minimum"]
        for t in self.times:
            if utilization:
                u = float(np.mean(utilization[-p["window"]:]))
                if u > p["out"]:
                    target = min(self.max_vms, target + p["group"])
                elif u < p["in"]:
                    target = max(p["minimum"], target - p["group"])
            before = len(ready_at)
            ready_at = (ready_at[:target] if target <= before else
                        ready_at + [t + self.model.config.vm_cold_start_seconds] * (target - before))
            targets.append(target)
            end = min(t + self.k, len(self.x))
            ready = np.sum(np.asarray(ready_at)[None, :] <= np.arange(t, end)[:, None], axis=1)
            capacity = ready * self.model.throughput
            offered = self.x[t:end]
            if family == "AUTO-adapted":
                numerator = np.minimum(offered, np.floor(p["rho"] * capacity))
            else:
                numerator = offered
            u = np.divide(numerator, capacity, out=np.where(numerator > 0, 1.0, 0.0), where=capacity > 0)
            utilization = (utilization + u.tolist())[-max(self.k, p["window"]):]
        return np.array(targets, dtype=int)


def select_on_training(candidates, penalty):
    if not candidates or any(row["split"] != "train" for row in candidates):
        raise ValueError("baseline tuning requires train records")
    return min(candidates, key=lambda row: row["summary"]["total_cost_usd"]
               + penalty * row["summary"]["expected_violations"])
