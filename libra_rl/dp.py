"""Finite-horizon, clairvoyant DP for oldest-first VM retention.

For K >= D, state is the previous target (all old VMs are ready).
For D/2 <= K < D, state is (previous target, ready old VMs). The one
pending cohort becomes ready after D-K seconds; new VMs after this interval.
This exactly covers K=60/120/300, D=100. Shorter K is explicitly rejected.

v1 with fill routing has a sufficient cap ceil(peak / throughput). v2 requires
an explicit finite cap and rho grid: its result is exact only in that domain,
and a penalty sweep is needed for cost/SLA tradeoffs.
"""
from dataclasses import dataclass
import math

import numpy as np

from .model import CostModel, validate_counts


@dataclass(frozen=True)
class DPResult:
    total_cost_usd: float
    objective_usd: float
    expected_violations: float
    targets: tuple
    rhos: tuple
    max_vms: int
    decision_seconds: int
    penalty: float
    domain: str


def evaluate_schedule(counts, targets, *, model=None, decision_seconds=60, rhos=None):
    """Replay targets using per-second startup clocks, without DP recurrence."""
    x = validate_counts(counts)
    model = model or CostModel()
    steps = math.ceil(len(x) / decision_seconds)
    targets = tuple(targets)
    rhos = tuple(rhos) if rhos is not None else (1.0,) * steps
    if len(targets) != steps or len(rhos) != steps:
        raise ValueError("one target/rho is required per decision")
    ready_at = np.empty(0, dtype=np.int64)
    vm_seconds = faas = violations = 0.0
    for t, target, rho in zip(range(0, len(x), decision_seconds), targets, rhos):
        if not isinstance(target, (int, np.integer)) or target < 0:
            raise ValueError("targets must be nonnegative integers")
        n = len(ready_at)
        ready_at = (ready_at[:target] if target <= n else np.concatenate(
            (ready_at, np.full(target - n, t + model.config.vm_cold_start_seconds, dtype=np.int64))))
        end = min(t + decision_seconds, len(x))
        ready = np.sum(ready_at[None, :] <= np.arange(t, end)[:, None], axis=1)
        traffic = model.traffic(x[t:end], ready, rho)
        vm_seconds += target * (end - t)
        faas += float(np.sum(traffic["faas_requests"]))
        violations += float(np.sum(traffic["expected_violations"]))
    vm_cost = vm_seconds * model.vm_second_price
    faas_cost = faas * model.faas_request_price
    gateway = len(x) * model.config.gateway_hourly_price / 3600
    return {"total_cost_usd": vm_cost + faas_cost + gateway,
            "vm_cost_usd": vm_cost, "faas_cost_usd": faas_cost,
            "gateway_cost_usd": gateway, "vm_seconds": vm_seconds,
            "faas_requests": faas, "expected_violations": violations,
            "violation_rate": violations / max(1, int(x.sum()))}


def solve_dp(counts, *, model=None, decision_seconds=300, max_vms=None,
             rho_grid=(1.0,), penalty=0.0):
    x = validate_counts(counts)
    model = model or CostModel()
    d, k = model.config.vm_cold_start_seconds, decision_seconds
    if not isinstance(k, int) or k < 1 or 2 * k < d:
        raise ValueError("exact DP currently requires K >= ceil(D/2)")
    rho_grid = tuple(float(r) for r in rho_grid)
    if not rho_grid or any(not math.isfinite(r) or not 0 < r <= 1 for r in rho_grid):
        raise ValueError("rho_grid must contain values in (0, 1]")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("penalty must be finite and nonnegative")
    if max_vms is None:
        if model.version != "v1":
            raise ValueError("v2 requires an explicit finite max_vms")
        max_vms = math.ceil(float(x.max()) / model.throughput / min(rho_grid))
    if not isinstance(max_vms, int) or max_vms < 0:
        raise ValueError("max_vms must be a nonnegative integer")
    levels = np.arange(max_vms + 1, dtype=np.int64)
    short = k < d
    if short:
        pairs = [(n, r) for n in levels for r in range(n + 1)]
        old = np.array([p[0] for p in pairs], dtype=np.int64)
        ready_old = np.array([p[1] for p in pairs], dtype=np.int64)
        first_ready = np.minimum(ready_old[:, None], levels[None, :])
        second_ready = np.minimum(old[:, None], levels[None, :])
        next_state = levels[None, :] * (levels[None, :] + 1) // 2 + second_ready
        first_length = d - k
    else:
        old = levels
        first_ready = np.minimum(old[:, None], levels[None, :])
        second_ready = np.broadcast_to(levels, first_ready.shape)
        next_state = np.broadcast_to(levels, first_ready.shape)
        first_length = d
    value = np.zeros(len(old))
    times = list(range(0, len(x), k))
    action_table, rho_table = [], []
    for t in reversed(times):
        end = min(t + k, len(x))
        middle = min(t + first_length, end)
        future_and_vm = value[next_state] + levels[None, :] * (end - t) * model.vm_second_price
        best_value = np.full(len(old), np.inf)
        best_action = np.zeros(len(old), dtype=np.int32)
        best_rho = np.zeros(len(old), dtype=np.int32)
        for j, rho in enumerate(rho_grid):
            phases = []
            for a, b in ((t, middle), (middle, end)):
                traffic = model.traffic(x[a:b, None], levels[None, :], rho)
                phase = (np.sum(traffic["faas_requests"], axis=0) * model.faas_request_price
                         + penalty * np.sum(traffic["expected_violations"], axis=0))
                phases.append(phase)
            q = future_and_vm + phases[0][first_ready] + phases[1][second_ready]
            action = np.argmin(q, axis=1)
            candidate = q[np.arange(len(old)), action]
            improve = candidate < best_value
            best_value[improve] = candidate[improve]
            best_action[improve] = action[improve]
            best_rho[improve] = j
        value = best_value
        action_table.append(best_action)
        rho_table.append(best_rho)
    state = 0  # empty fleet, no prewarmed or free instances
    targets, rhos = [], []
    for actions, rho_indices in zip(reversed(action_table), reversed(rho_table)):
        target = int(actions[state])
        targets.append(target)
        rhos.append(rho_grid[int(rho_indices[state])])
        state = int(next_state[state, target])
    replay = evaluate_schedule(x, targets, model=model, decision_seconds=k, rhos=rhos)
    objective = replay["total_cost_usd"] + penalty * replay["expected_violations"]
    gateway = len(x) * model.config.gateway_hourly_price / 3600
    if not math.isclose(objective, float(value[0]) + gateway, abs_tol=1e-9, rel_tol=1e-12):
        raise AssertionError("DP recurrence disagrees with per-second schedule replay")
    return DPResult(replay["total_cost_usd"], objective, replay["expected_violations"],
                    tuple(targets), tuple(rhos), max_vms, k, penalty,
                    f"{model.version}; 0..{max_vms} VMs; rho={rho_grid}; oldest-first scale-in")
