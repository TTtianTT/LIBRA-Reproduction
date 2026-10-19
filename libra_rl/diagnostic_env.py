"""A separate diagnostic action space; historical env/checkpoints stay intact."""
import math

import gymnasium as gym
import numpy as np

from .env import RHO_GRID
from .training import StudyEnv


class DiagnosticEnv(StudyEnv):
    """Joint categorical residual actions plus an explicit zero-VM/FaaS action.

    The residual center is a specified causal window rule, chosen on train only.
    Full FaaS clears the fleet immediately; it does not retain free warm VMs.
    """
    def __init__(self, counts, *, rule_window=300, allow_faas=True, **kwargs):
        super().__init__(counts, **kwargs)
        if self.action_mode != "residual":
            raise ValueError("diagnostic action space supports residual control only")
        if not isinstance(rule_window, int) or rule_window < 1:
            raise ValueError("rule_window must be a positive integer")
        self.rule_window = rule_window
        self.allow_faas = bool(allow_faas)
        self.rhos = (1.0,) if self.model.version == "v1" else RHO_GRID
        self.residual_choices = 7 * len(self.rhos)
        self.action_space = gym.spaces.Discrete(self.residual_choices + int(self.allow_faas))

    @property
    def teacher_action(self):
        index = min(range(len(self.rhos)), key=lambda i: abs(self.rhos[i] - self.rule_rho))
        if abs(self.rhos[index] - self.rule_rho) > 1e-12:
            raise ValueError("teacher routing ratio must belong to the action grid")
        return 3 * len(self.rhos) + index

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        if self.rule_window != 300:
            for t in self._rule_cache:
                i = self._offset + t
                self._rule_cache[t] = float(np.quantile(
                    self._history[max(0, i-self.rule_window+1):i+1], self.model.tau))
        return observation, info

    def decode(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid diagnostic action {action}")
        action = int(action)
        if action == self.residual_choices:
            return 0, 1.0
        delta = action // len(self.rhos) - 3
        rho = self.rhos[action % len(self.rhos)]
        return int(np.clip(self.rule_target() + delta, 0, self.max_vms)), rho

    def step(self, action):
        if not self._has_reset or self._t >= len(self._x):
            raise RuntimeError("reset the environment before stepping")
        target, rho = self.decode(action)
        observation, reward, terminated, truncated, info = self.step_target(target, rho=rho)
        info["diagnostic_action"] = int(action)
        info["full_faas_action"] = int(action) == self.residual_choices
        return observation, reward, terminated or truncated, False, info


def initialize_rule_prior(agent, action, probability=0.9):
    """Explicit logit prior, not behavior-cloning training or a learned policy."""
    import torch
    head = agent.policy.action_net
    n = agent.action_space.n
    if not 0 <= action < n or not 0 < probability < 1:
        raise ValueError("invalid rule prior")
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[action] = math.log(probability * (n-1) / (1-probability))


def base_targets(counts, model, *, rho, window=300, decision_seconds=60, max_vms=512):
    x = np.asarray(counts)
    levels = [float(np.quantile(x[max(0, t-window+1):t+1], model.tau))
              for t in range(0, len(x), decision_seconds)]
    return np.array([min(max_vms, 0 if q < model.config.cip_requests_per_second else
                             math.ceil(q / model.throughput / rho)) for q in levels], dtype=int)
