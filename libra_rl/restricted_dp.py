"""Clairvoyant DP in exactly the finite residual action domain.

Each time has at most eight allowed targets, so a short-interval state update
uses O(M² * 8) work. Lambda sweeps give feasible primal plans and dual bounds,
not an exact solution to every deterministic resource-constrained problem.
"""
import math
import numpy as np

from .model import CostModel, validate_counts


def residual_bound(counts, bases, *, model=None, decision_seconds=60, max_vms=512,
                   rho_grid=(1.0,), allow_faas=False, penalty=0.0,
                   cost_weight=1.0, return_plan=False):
    x = validate_counts(counts)
    model = model or CostModel()
    k, d = decision_seconds, model.config.vm_cold_start_seconds
    if not isinstance(k, int) or k < 1 or 2*k < d:
        raise ValueError("DP requires K >= ceil(D/2)")
    if not isinstance(max_vms, int) or max_vms < 0:
        raise ValueError("max_vms must be nonnegative")
    bases = np.asarray(bases)
    if bases.shape != (math.ceil(len(x)/k),) or np.any(bases != np.floor(bases)) or np.any(bases < 0):
        raise ValueError("one nonnegative integer rule target is required per decision")
    if not rho_grid or any(not np.isfinite(r) or not 0 < r <= 1 for r in rho_grid):
        raise ValueError("invalid routing grid")
    if not np.isfinite(penalty) or penalty < 0 or not np.isfinite(cost_weight) or cost_weight < 0:
        raise ValueError("objective weights must be finite and nonnegative")
    if cost_weight == penalty == 0:
        raise ValueError("objective cannot be identically zero")
    m_max = min(max_vms, int(bases.max()) + 3)
    levels = np.arange(m_max+1)
    short = k < d
    n, r = np.tril_indices(m_max+1) if short else (levels, levels)
    costs, violations = np.zeros(len(n)), np.zeros(len(n))
    values = np.zeros(len(n))
    plans = []
    times = list(range(0, len(x), k))
    for index in reversed(range(len(times))):
        t = times[index]
        end = min(t+k, len(x))
        middle = min(t+(d-k if short else d), end)
        actions = np.unique(np.clip(int(bases[index]) + np.arange(-3,4), 0, max_vms))
        if allow_faas:
            actions = np.unique(np.append(actions, 0))
        first_ready = np.minimum(r[:,None], actions[None,:])
        second_ready = (np.minimum(n[:,None], actions[None,:]) if short
                        else np.broadcast_to(actions, first_ready.shape))
        following = (actions[None,:]*(actions[None,:]+1)//2 + second_ready if short
                     else np.broadcast_to(actions, first_ready.shape))
        vm_charge = actions*(end-t)*model.vm_second_price
        best = np.full(len(n), np.inf)
        chosen_cost, chosen_viol = np.zeros(len(n)), np.zeros(len(n))
        best_target, best_rho = np.zeros(len(n), dtype=np.int32), np.zeros(len(n), dtype=np.int32)
        for j, rho in enumerate(rho_grid):
            stage_cost, stage_viol = [], []
            for a,b in ((t,middle),(middle,end)):
                traffic = model.traffic(x[a:b,None], levels[None,:], rho)
                stage_cost.append(traffic["faas_requests"].sum(axis=0)*model.faas_request_price)
                stage_viol.append(traffic["expected_violations"].sum(axis=0))
            c = vm_charge[None,:] + stage_cost[0][first_ready] + stage_cost[1][second_ready] + costs[following]
            v = stage_viol[0][first_ready] + stage_viol[1][second_ready] + violations[following]
            q = cost_weight*c + penalty*v
            choice = np.argmin(q, axis=1)
            candidate = q[np.arange(len(n)),choice]
            improve = candidate < best
            best[improve] = candidate[improve]
            chosen_cost[improve] = c[np.arange(len(n)),choice][improve]
            chosen_viol[improve] = v[np.arange(len(n)),choice][improve]
            best_target[improve] = actions[choice[improve]]
            best_rho[improve] = j
        values, costs, violations = best, chosen_cost, chosen_viol
        if return_plan:
            plans.append((best_target, best_rho))
    gateway = len(x)*model.config.gateway_hourly_price/3600
    result = {"total_cost_usd": float(costs[0]+gateway), "expected_violations": float(violations[0]),
              "violation_rate": float(violations[0]/max(1,int(x.sum()))),
              "objective_usd": float(values[0]+cost_weight*gateway), "penalty": penalty,
              "cost_weight": cost_weight, "decision_seconds": k, "max_vms": m_max,
              "allow_faas": allow_faas, "scope": "time-dependent residual targets, finite rho grid"}
    if return_plan:
        state, targets, rhos = 0, [], []
        for target_table, rho_table in reversed(plans):
            target = int(target_table[state])
            targets.append(target)
            rhos.append(1.0 if target == 0 and allow_faas else float(rho_grid[int(rho_table[state])]))
            ready = min(int(n[state]), target)
            state = target*(target+1)//2+ready if short else target
        result.update(targets=targets, rhos=rhos)
    return result
