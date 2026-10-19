"""Anytime exact queue-state search with a valid frontier lower bound.

No beam pruning. A limit stops search with an optimal-cost interval, rather
than relabeling a feasible heuristic solution as the true optimum.
"""
import heapq
import time

import numpy as np

from .physical import QueueFleet,actions,evaluate_plan
from .physical_planning import cost_relaxation
from .physical_dual import phase_dual_tables


def optimal_interval(stream,bases,*,service_seconds,incumbent_targets,incumbent_rhos,
                     epsilon=0.001,max_seconds=10.0,max_expansions=10000,max_live_nodes=50000):
    started = time.perf_counter()
    relaxation = cost_relaxation(stream.counts,bases,service_seconds)
    penalties,dual = phase_dual_tables(stream,bases,service_seconds)
    incumbent = evaluate_plan(stream,incumbent_targets,incumbent_rhos,service_seconds=service_seconds)
    if incumbent["violation_rate"]>epsilon+1e-12:
        raise ValueError("a feasible incumbent is required")
    upper = incumbent["total_cost_usd"]
    targets,rhos = list(incumbent_targets),list(incumbent_rhos)
    fleet = QueueFleet(stream,service_seconds=service_seconds)
    budget = epsilon*int(stream.counts.sum())
    gateway_rate = fleet.model.config.gateway_hourly_price/3600
    def estimate(f):
        n = len(f.ready_at)
        r = int(np.sum(f.ready_at<=f.t))
        stage = len(bases) if f.t==len(stream.counts) else f.t//60
        state = n*(n+1)//2+r
        summary = f.summary()
        remaining_budget = budget-summary["violations"]
        future = max(table[stage][state]-penalty*remaining_budget for penalty,table in zip(penalties,dual))
        return float(summary["total_cost_usd"]+future+(len(stream.counts)-f.t)*gateway_rate)
    # Nodes carry a live flag so dominated heap entries can be discarded lazily.
    root = dict(fleet=fleet,targets=[],rhos=[],live=True,cost=0.0,violations=0)
    queue = [(estimate(fleet),0,root)]
    frontiers = {fleet.state_key():[root]}
    generated,expanded,dominated = 1,0,0
    stopped = "complete"
    while queue:
        while queue and not queue[0][2]["live"]:
            heapq.heappop(queue)
        if not queue:
            break
        if queue[0][0]>=upper-1e-12:
            break
        if time.perf_counter()-started>=max_seconds:
            stopped = "time_limit"; break
        if expanded>=max_expansions:
            stopped = "expansion_limit"; break
        if len(queue)>=max_live_nodes:
            stopped = "frontier_limit"; break
        _,_,node = heapq.heappop(queue)
        f = node["fleet"]
        if f.t==len(stream.counts):
            value = f.summary(terminal=True)
            if value["total_cost_usd"]<upper:
                upper,incumbent = value["total_cost_usd"],value
                targets,rhos = node["targets"],node["rhos"]
            continue
        expanded += 1
        i = f.t//60
        for target,rho in actions(int(bases[i])):
            candidate = f.clone(); candidate.step(target,rho,60)
            summary = candidate.summary()
            if summary["violations"]>budget+1e-12:
                continue
            lower = estimate(candidate)
            if lower>=upper-1e-12:
                continue
            key = candidate.state_key()
            frontier = frontiers.setdefault(key,[])
            c,v = summary["total_cost_usd"],summary["violations"]
            if any(old["cost"]<=c+1e-12 and old["violations"]<=v for old in frontier):
                dominated += 1; continue
            keep = []
            for old in frontier:
                if c<=old["cost"]+1e-12 and v<=old["violations"]:
                    old["live"] = False
                else:
                    keep.append(old)
            child = dict(fleet=candidate,targets=node["targets"]+[target],rhos=node["rhos"]+[rho],
                         live=True,cost=c,violations=v)
            keep.append(child); frontiers[key] = keep
            heapq.heappush(queue,(lower,generated,child)); generated += 1
    while queue and not queue[0][2]["live"]:
        heapq.heappop(queue)
    lower = min(upper,queue[0][0]) if queue else upper
    exact = upper-lower<=1e-10
    return dict(summary=incumbent,targets=targets,rhos=rhos,cost_lower_bound_usd=float(lower),
        feasible_upper_bound_usd=float(upper),exact=exact,stopped="optimality_certified" if exact else stopped,
        expanded_states=expanded,generated_states=generated,dominated_states=dominated,
        frontier_entries=len(queue),seconds=time.perf_counter()-started,
        relaxation_lower_bound_usd=relaxation["lower_bound_usd"],sla_phase_dual_lower_bound_usd=estimate(fleet),
        dual_penalties=penalties.tolist())
