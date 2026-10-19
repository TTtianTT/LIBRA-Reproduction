"""Vectorized full-trace policy evaluation with the RL-0 accounting contract."""
import numpy as np

from .model import validate_counts


def evaluate_targets(counts, targets, *, model, decision_seconds=60, rhos=1.0,
                     estimates=None):
    x = validate_counts(counts)
    times = np.arange(0, len(x), decision_seconds)
    targets = np.asarray(targets, dtype=int)
    rho_steps = np.broadcast_to(rhos, targets.shape)
    if len(targets) != len(times) or np.any(targets < 0):
        raise ValueError("one nonnegative target per decision is required")
    if estimates is not None and decision_seconds <= model.config.vm_cold_start_seconds:
        raise ValueError("paper estimate routing requires decision interval > startup")
    events = np.zeros(len(x) + 1, dtype=np.int64)
    allocated = []
    lifetimes = []
    startup_waste = 0
    d = model.config.vm_cold_start_seconds
    route = np.zeros(len(x)) if estimates is not None else None
    last_estimate = 0.0
    for j, (t, target) in enumerate(zip(times, targets)):
        before = len(allocated)
        for created in allocated[target:]:
            lifetimes.append((created, int(t)))
            if t - created < d:
                startup_waste += int(t) - created
        if target < before:
            allocated = allocated[:target]
        else:
            allocated.extend([int(t)] * (target - before))
        if estimates is not None:
            end = min(int(t) + decision_seconds, len(x))
            if target > before:
                due = min(int(t) + d, end)
                route[t:due] = last_estimate
                route[due:end] = estimates[j]
                if int(t) + d < end:
                    last_estimate = estimates[j]
                elif int(t) + d == end:
                    last_estimate = estimates[j]
            else:
                last_estimate = estimates[j]
                route[t:end] = last_estimate
    unfinished = sum(len(x) - created for created in allocated if len(x) - created < d)
    lifetimes.extend((created, len(x)) for created in allocated)
    vm_seconds = startup = 0
    for created, end in lifetimes:
        vm_seconds += end - created
        startup += min(d, end - created)
        ready = created + d
        if ready < end:
            events[ready] += 1
            events[end] -= 1
    ready = np.cumsum(events[:-1])
    rho_timeline = np.repeat(rho_steps, decision_seconds)[:len(x)]
    vm_requests = faas_requests = violations = 0.0
    for rho in np.unique(rho_steps):
        mask = rho_timeline == rho
        traffic = model.traffic(x[mask], ready[mask], float(rho), None if route is None else route[mask])
        vm_requests += float(traffic["vm_requests"].sum())
        faas_requests += float(traffic["faas_requests"].sum())
        violations += float(traffic["expected_violations"].sum())
    vm_cost = vm_seconds * model.vm_second_price
    faas_cost = faas_requests * model.faas_request_price
    gateway = len(x) * model.config.gateway_hourly_price / 3600
    return {"total_cost_usd": vm_cost + faas_cost + gateway, "vm_cost_usd": vm_cost,
            "faas_cost_usd": faas_cost, "gateway_cost_usd": gateway, "requests": int(x.sum()),
            "vm_requests": vm_requests, "faas_requests": faas_requests,
            "expected_violations": violations, "violation_rate": violations / max(1, int(x.sum())),
            "vm_seconds": vm_seconds, "starting_vm_seconds": startup,
            "cancelled_startup_vm_seconds": startup_waste,
            "unfinished_startup_vm_seconds": unfinished, "max_target_vms": int(targets.max())}
