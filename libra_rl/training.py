"""SB3 training adapter: finite episodes, bounded observations, causal caching."""
import math

import gymnasium as gym
import numpy as np

from .env import LibraEnv


class StudyEnv(LibraEnv):
    """Cache a causal rule per episode; preserve RL-0's accounting unchanged.

    Unlike a continuing TimeLimit, this study's finite six-hour task liquidates
    at the end. Mark it terminated so SB3 does not bootstrap beyond that end.
    """
    def reset(self, **kwargs):
        self._rule_cache = None
        observation, info = super().reset(**kwargs)
        self._rule_cache = {}
        for t in self._features:
            if t < len(self._x):
                i = self._offset + t
                self._rule_cache[t] = float(np.quantile(
                    self._history[max(0, i - 299):i + 1], self.model.tau))
        return observation, info

    def rule_estimate(self):
        if self.rule_uses_forecast or self._rule_cache is None:
            return super().rule_estimate()
        return self._rule_cache[self._t]

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        return observation, reward, terminated or truncated, False, info


class ObservationTransform(gym.ObservationWrapper):
    """Fixed transform fitted on no data; avoid future/test normalization stats.

    Signed log compresses potentially huge normalized resource counts at idle
    demand. GBDT ablation zeros its value/mask without removing dimensions.
    """
    def __init__(self, env, *, hide_forecast=False):
        super().__init__(env)
        self.hide_forecast = hide_forecast
        self.observation_space = gym.spaces.Box(-20.0, 20.0,
                                                 shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        x = np.asarray(observation, dtype=np.float32).copy()
        if self.hide_forecast:
            x[21:23] = 0
        return np.clip(np.sign(x) * np.log1p(np.abs(x)), -20, 20).astype(np.float32)


def choose_penalty(rows, epsilon):
    """Select a lambda from validation means only; never accept test records."""
    if any(r.get("split") != "validation" for r in rows):
        raise ValueError("penalty selection accepts validation records only")
    summaries = []
    for penalty in sorted({r["penalty"] for r in rows}):
        subset = [r for r in rows if r["penalty"] == penalty]
        summaries.append((penalty, float(np.mean([r["total_cost_usd"] for r in subset])),
                          float(np.mean([r["violation_rate"] for r in subset]))))
    if not summaries:
        raise ValueError("no validation records")
    feasible = [s for s in summaries if s[2] <= epsilon]
    return min(feasible, key=lambda s: (s[1], s[2]))[0] if feasible else min(summaries, key=lambda s: (s[2], s[1]))[0]


def confidence_interval(values):
    """Student t two-sided 95% interval, five independent training seeds."""
    from scipy.stats import t
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    if len(x) < 2:
        return mean, mean, mean
    half = float(t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return mean, mean - half, mean + half
