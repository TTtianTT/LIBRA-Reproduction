"""Gymnasium v1/v2 environment with explicit startup cohorts.

An action is taken after observing the current second's arrivals, matching
libra_sim. New instances are billed immediately and become ready after D seconds.
Scale-in removes the newest instances first. Episodes start with zero VMs.
"""
import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from libra_ml.features import FEATURE_NAMES, feature_matrix
from .model import CostModel, validate_counts

RAW_DELTAS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
RHO_GRID = (0.6, 0.7, 0.8, 0.9, 1.0)


class LibraEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, counts, *, model=None, decision_seconds=60,
                 episode_seconds=21600, action_mode="residual", max_vms=292,
                 forecast_model=None, penalty=0.0, epsilon=0.001,
                 cost_normalizer=1.0, augment=False, rule_rho=None,
                 rule_uses_forecast=False):
        self.counts = validate_counts(counts)
        self.model = model or CostModel()
        if not isinstance(decision_seconds, int) or decision_seconds < 1:
            raise ValueError("decision_seconds must be a positive integer")
        if episode_seconds is not None and (not isinstance(episode_seconds, int) or episode_seconds < 1):
            raise ValueError("episode_seconds must be positive or None")
        if action_mode not in ("residual", "raw"):
            raise ValueError("action_mode must be residual or raw")
        if not isinstance(max_vms, int) or max_vms < 0:
            raise ValueError("max_vms must be a nonnegative integer")
        if not math.isfinite(cost_normalizer) or cost_normalizer <= 0:
            raise ValueError("cost_normalizer must be finite and positive")
        if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1]")
        self.decision_seconds, self.episode_seconds = decision_seconds, episode_seconds
        self.action_mode, self.max_vms = action_mode, max_vms
        self.forecast_model, self.augment = forecast_model, augment
        self.rule_rho = (1.0 if self.model.version == "v1" else 0.8) if rule_rho is None else rule_rho
        if not math.isfinite(self.rule_rho) or not 0 < self.rule_rho <= 1:
            raise ValueError("rule_rho must be in (0, 1]")
        if rule_uses_forecast and forecast_model is None:
            raise ValueError("forecast-based rule requires a fitted model")
        self.rule_uses_forecast = rule_uses_forecast
        self.epsilon, self.cost_normalizer = epsilon, cost_normalizer
        self.set_penalty(penalty)
        choices = 7 if action_mode == "residual" else len(RAW_DELTAS)
        self.action_space = (spaces.Discrete(choices) if self.model.version == "v1"
                             else spaces.MultiDiscrete([choices, len(RHO_GRID)]))
        self.observation_names = tuple(FEATURE_NAMES) + (
            "forecast", "has_forecast", "ready_capacity", "starting_capacity",
            "allocated_capacity", "utilization", "violation_rate", "since_expansion",
            "episode_remaining", "lagrange_penalty",
        ) + tuple(f"startup_remaining_{s}s" for s in range(1, self.cold_start + 1))
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(len(self.observation_names),), dtype=np.float32)
        self._has_reset = False

    @property
    def cold_start(self):
        return self.model.config.vm_cold_start_seconds

    def set_penalty(self, value):
        if not math.isfinite(value) or value < 0:
            raise ValueError("penalty must be finite and nonnegative")
        self.penalty = float(value)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        length = min(len(self.counts), self.episode_seconds or len(self.counts))
        start = int(options.get("start", self.np_random.integers(0, len(self.counts) - length + 1)))
        end = int(options.get("end", min(len(self.counts), start + length)))
        if not 0 <= start < end <= len(self.counts):
            raise ValueError("episode must stay inside the provided data partition")
        history_start = max(0, start - 1800)
        data = self.counts[history_start:end].copy()
        self.augmentation = None
        if self.augment:
            amplitude = float(self.np_random.uniform(0.5, 2.0))
            speed = float(self.np_random.uniform(0.8, 1.25))
            # Time-warp inside this partition; endpoint padding, no held-out data.
            grid = np.arange(len(data), dtype=float)
            mean = np.interp(grid * speed, grid, data) * amplitude
            data = self.np_random.poisson(mean).astype(np.int64)
            self.augmentation = {"amplitude": amplitude, "time_scale": speed,
                                 "noise": "Poisson resampling"}
        self.source_start, self.source_end = start, end
        self._offset = start - history_start
        self._history = data
        self._x = data[self._offset:]
        self._t = 0
        self._ready_at = np.empty(0, dtype=np.int64)
        self._routing_estimate = 0.0
        self._pending_routing = None
        self._last_expansion = None
        self._last_utilization = self._last_violation = 0.0
        self._cancelled_startup = 0.0
        self._totals = {k: 0.0 for k in ("vm_seconds", "starting_vm_seconds", "vm_requests",
                                        "faas_requests", "expected_violations", "vm_cost_usd",
                                        "faas_cost_usd", "gateway_cost_usd")}
        local_times = list(range(0, len(self._x), self.decision_seconds)) + [len(self._x)]
        feature_times = self._offset + np.minimum(local_times, len(self._x) - 1)
        features, scales = feature_matrix(self._history, feature_times)
        predictions = (np.zeros(len(local_times)) if self.forecast_model is None
                       else np.maximum(0, np.asarray(self.forecast_model.predict(features), dtype=float)))
        if predictions.shape != (len(local_times),) or not np.all(np.isfinite(predictions)):
            raise ValueError("forecast model must return one finite normalized quantile per row")
        self._features = {t: (f, float(s), float(p)) for t, f, s, p in
                          zip(local_times, features, scales, predictions)}
        self._has_reset = True
        return self._observation(), {"source_start": start, "source_end": end,
                                     "augmentation": self.augmentation}

    def _observation(self):
        f, scale, pred = self._features[self._t]
        remaining = self._ready_at - self._t
        ready = int(np.sum(remaining <= 0))
        pending = len(remaining) - ready
        capacity_scale = self.model.throughput / scale
        since = (self._t + self.cold_start if self._last_expansion is None
                 else self._t - self._last_expansion)
        resource = [pred, float(self.forecast_model is not None),
                    ready * capacity_scale, pending * capacity_scale,
                    len(remaining) * capacity_scale, self._last_utilization,
                    self._last_violation, since / 3600,
                    (len(self._x) - self._t) / len(self._x), self.penalty]
        # The remaining-time histogram preserves the full resource delay state.
        cohorts = np.bincount(remaining[remaining > 0], minlength=self.cold_start + 1)[1:]
        return np.concatenate((f, resource, cohorts * capacity_scale)).astype(np.float32)

    def rule_estimate(self):
        if self.rule_uses_forecast:
            _, scale, pred = self._features[self._t]
            return scale * pred
        i = min(self._offset + self._t, len(self._history) - 1)
        return float(np.quantile(self._history[max(0, i - 299):i + 1], self.model.tau))

    def rule_target(self):
        estimate = self.rule_estimate()
        # Keep paper CIP=4 as an explicit baseline convention in both versions.
        return (0 if estimate < self.model.config.cip_requests_per_second
                else math.ceil(estimate / self.model.throughput / self.rule_rho))

    def step(self, action):
        if not self._has_reset or self._t >= len(self._x):
            raise RuntimeError("call reset before stepping, including after truncation")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} outside {self.action_space}")
        if self.model.version == "v1":
            index, rho = int(action), 1.0
        else:
            index, rho_index = map(int, action)
            rho = RHO_GRID[rho_index]
        delta = index - 3 if self.action_mode == "residual" else RAW_DELTAS[index]
        base = self.rule_target() if self.action_mode == "residual" else len(self._ready_at)
        target = int(np.clip(base + delta, 0, self.max_vms))
        return self.step_target(target, rho=rho)

    def step_target(self, target, *, rho=1.0, estimate=None, routing="capacity"):
        """Replay a fixed policy/DP action, bypassing the residual action space.

        ``routing='estimate'`` reproduces LIBRA's delayed routing update for
        parity tests (decision interval > startup). RL uses capacity routing.
        """
        if not self._has_reset or self._t >= len(self._x):
            raise RuntimeError("call reset before stepping, including after truncation")
        if not isinstance(target, (int, np.integer)) or not 0 <= target <= self.max_vms:
            raise ValueError("target outside VM limits")
        if not math.isfinite(rho) or not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        if routing not in ("capacity", "estimate"):
            raise ValueError("routing must be capacity or estimate")
        if routing == "estimate" and (estimate is None or not math.isfinite(estimate) or estimate < 0):
            raise ValueError("estimate routing requires a finite nonnegative estimate")
        if routing == "estimate" and self.decision_seconds <= self.cold_start:
            raise ValueError("paper routing replay requires interval > startup; use capacity for RL")
        t = self._t
        if self._pending_routing is not None and self._pending_routing[0] <= t:
            self._routing_estimate = self._pending_routing[1]
            self._pending_routing = None
        before = len(self._ready_at)
        if target < before:
            removed = self._ready_at[target:]
            self._cancelled_startup += float(np.sum(t - (removed[removed > t] - self.cold_start)))
            self._ready_at = self._ready_at[:target]
        elif target > before:
            self._ready_at = np.concatenate((self._ready_at,
                                            np.full(target - before, t + self.cold_start, dtype=np.int64)))
            self._last_expansion = t
        if routing == "estimate":
            if target > before and self.cold_start:
                self._pending_routing = (t + self.cold_start, estimate)
            else:
                self._routing_estimate = estimate
                self._pending_routing = None
        end = min(t + self.decision_seconds, len(self._x))
        times = np.arange(t, end)
        ready = np.sum(self._ready_at[None, :] <= times[:, None], axis=1)
        routing_values = None
        if routing == "estimate":
            routing_values = np.full(end - t, self._routing_estimate)
            if self._pending_routing is not None:
                due, value = self._pending_routing
                routing_values[times >= due] = value
                if due < end:
                    self._routing_estimate = value
                    self._pending_routing = None
        traffic = self.model.traffic(self._x[t:end], ready, rho, routing_values)
        vm_seconds = target * (end - t)
        step = {"vm_seconds": float(vm_seconds),
                "starting_vm_seconds": float(np.sum(target - ready)),
                "vm_requests": float(np.sum(traffic["vm_requests"])),
                "faas_requests": float(np.sum(traffic["faas_requests"])),
                "expected_violations": float(np.sum(traffic["expected_violations"])),
                "vm_cost_usd": vm_seconds * self.model.vm_second_price,
                "faas_cost_usd": float(np.sum(traffic["faas_requests"])) * self.model.faas_request_price,
                "gateway_cost_usd": (end - t) * self.model.config.gateway_hourly_price / 3600}
        requests = step["vm_requests"] + step["faas_requests"]
        violation = step["expected_violations"]
        controllable_cost = step["vm_cost_usd"] + step["faas_cost_usd"]
        reward = -(controllable_cost + self.penalty * (violation - self.epsilon * requests)) / self.cost_normalizer
        for key, value in step.items():
            self._totals[key] += value
        self._last_utilization = float(np.mean(traffic["utilization"]))
        self._last_violation = violation / max(1.0, requests)
        self._t = end
        info = {**step, "time_s": t, "end_s": end, "target_vms": int(target), "rho": rho,
                "ready_at_end": int(np.sum(self._ready_at <= end)),
                "starting_at_end": int(np.sum(self._ready_at > end)),
                "total_cost_usd": controllable_cost + step["gateway_cost_usd"],
                "summary": self.summary()}
        return self._observation(), float(reward), False, end == len(self._x), info

    def summary(self):
        totals = dict(self._totals)
        totals["total_cost_usd"] = sum(totals[k] for k in
                                       ("vm_cost_usd", "faas_cost_usd", "gateway_cost_usd"))
        requests = totals["vm_requests"] + totals["faas_requests"]
        totals["requests"] = requests
        totals["violation_rate"] = totals["expected_violations"] / max(1.0, requests)
        totals["cancelled_startup_vm_seconds"] = self._cancelled_startup
        unfinished = self._ready_at[self._ready_at > self._t]
        totals["unfinished_startup_vm_seconds"] = float(np.sum(self._t - (unfinished - self.cold_start)))
        return totals
