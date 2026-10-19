"""Value-only exact DP with O(M²) rather than O(M³) short-interval updates.

For a state (n,r), partition targets m into m<=r, r<m<=n, and m>n.
The first part is a prefix minimum, the second an interval minimum, the third
a column minimum of V(m,n)+VM_price*m. No trajectory table is retained.
The result is checked against the trajectory-producing DP on small instances.
"""
import math
import numpy as np

from .model import CostModel, validate_counts


def optimal_bound(counts, *, model=None, decision_seconds=60, max_vms=None,
                  rho_grid=(1.0,), penalty=0.0):
    x = validate_counts(counts)
    model = model or CostModel()
    d, k = model.config.vm_cold_start_seconds, decision_seconds
    if not isinstance(k, int) or k < 1 or 2 * k < d:
        raise ValueError("K must be >= ceil(D/2)")
    if not rho_grid or any(not np.isfinite(r) or not 0 < r <= 1 for r in rho_grid):
        raise ValueError("invalid rho grid")
    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("invalid penalty")
    if max_vms is None:
        if model.version != "v1":
            raise ValueError("v2 requires an explicit finite max_vms")
        max_vms = math.ceil(float(x.max()) / model.throughput / min(rho_grid))
    if not isinstance(max_vms, int) or max_vms < 0:
        raise ValueError("invalid VM bound")
    levels = np.arange(max_vms + 1)
    short = k < d
    if short:
        n, r = np.tril_indices(max_vms + 1)
        state_index = lambda a, b: a * (a + 1) // 2 + b
        diagonal = state_index(levels, levels)
        greater = levels[:, None] > levels[None, :]
    else:
        n = r = levels
    value = np.zeros(len(n))
    costs = np.zeros(len(n))
    violations = np.zeros(len(n))
    for t in reversed(range(0, len(x), k)):
        end = min(t + k, len(x))
        middle = min(t + (d - k if short else d), end)
        charge = levels * (end - t) * model.vm_second_price
        if short:
            v_square = np.full((max_vms + 1, max_vms + 1), np.inf)
            v_square[n, r] = value
            suffix_matrix = np.where(greater, v_square + charge[:, None], np.inf)
            suffix = np.min(suffix_matrix, axis=0)
            suffix_action = np.argmin(suffix_matrix, axis=0)
        best = np.full(len(n), np.inf)
        chosen_cost = np.zeros(len(n))
        chosen_violations = np.zeros(len(n))
        for rho in rho_grid:
            fc, fv, sc, sv = [], [], [], []
            for a, b, c_out, v_out in ((t, middle, fc, fv), (middle, end, sc, sv)):
                traffic = model.traffic(x[a:b, None], levels[None, :], rho)
                c_out.extend((np.sum(traffic["faas_requests"], axis=0) * model.faas_request_price).tolist())
                v_out.extend(np.sum(traffic["expected_violations"], axis=0).tolist())
            fc, fv, sc, sv = map(np.asarray, (fc, fv, sc, sv))
            f, s = fc + penalty * fv, sc + penalty * sv
            if short:
                a = charge + f + s + value[diagonal]
                prefix = np.minimum.accumulate(a)
                prefix_action = np.maximum.accumulate(np.where(a == prefix, levels, 0))
                b = charge + s + value[diagonal]
                intervals = np.minimum.accumulate(np.where(greater, b[:, None], np.inf), axis=0)
                interval_action = np.maximum.accumulate(
                    np.where(greater & (b[:, None] == intervals), levels[:, None], 0), axis=0)
                q = np.stack((prefix[r], f[r] + intervals[n, r], f[r] + s[n] + suffix[n]))
                option = np.argmin(q, axis=0)
                action = np.choose(option, (prefix_action[r], interval_action[n, r], suffix_action[n]))
                candidate = np.min(q, axis=0)
                ready_first, ready_second = np.minimum(r, action), np.minimum(n, action)
                following = state_index(action, ready_second)
            else:
                first = np.minimum(n[:, None], levels[None, :])
                q = charge[None, :] + f[first] + s[None, :] + value[None, :]
                action = np.argmin(q, axis=1)
                candidate = q[levels, action]
                ready_first, ready_second = np.minimum(n, action), action
                following = action
            improve = candidate < best
            selected_cost = charge[action] + fc[ready_first] + sc[ready_second] + costs[following]
            selected_violations = fv[ready_first] + sv[ready_second] + violations[following]
            best[improve] = candidate[improve]
            chosen_cost[improve] = selected_cost[improve]
            chosen_violations[improve] = selected_violations[improve]
        value, costs, violations = best, chosen_cost, chosen_violations
    gateway = len(x) * model.config.gateway_hourly_price / 3600
    if not np.isclose(value[0], costs[0] + penalty * violations[0], atol=1e-8, rtol=1e-10):
        raise AssertionError("cost/violation propagation disagrees with Bellman value")
    return {"total_cost_usd": float(costs[0] + gateway),
            "expected_violations": float(violations[0]),
            "objective_usd": float(value[0] + gateway), "penalty": penalty,
            "max_vms": max_vms, "decision_seconds": k, "rho_grid": list(rho_grid),
            "scope": "global v1 fill-routing bound" if model.version == "v1" else "finite VM/rho domain only"}
