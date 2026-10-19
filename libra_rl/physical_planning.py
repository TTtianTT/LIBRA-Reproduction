"""Finite-budget clairvoyant search and exact optimistic cost relaxation.

The queue-state search is exact only when no state is pruned. Its feasible plan
is an upper bound on offline optimal cost, never a lower bound on online RL.
"""
import math
import numpy as np

from .model import CostModel
from .physical import QueueFleet,actions


def cost_relaxation(counts,bases,service_seconds,*,decision_seconds=60,cold=100,max_vms=512):
    """Exact DP omitting SLA and all draining costs; a genuine cost lower bound."""
    if not 0<decision_seconds<cold<=2*decision_seconds:
        raise ValueError("cost table currently requires K < D <= 2K")
    x = np.asarray(counts)
    if len(bases)!=math.ceil(len(x)/decision_seconds):
        raise ValueError("base length mismatch")
    model = CostModel(version="v2",service_seconds=service_seconds)
    maximum = min(max_vms,int(np.max(bases))+3)
    n,r = np.tril_indices(maximum+1)
    levels = np.arange(maximum+1)
    tables = [None]*(len(bases)+1)
    tables[-1] = np.zeros(len(n))
    for i in reversed(range(len(bases))):
        t = i*decision_seconds
        end = min(t+decision_seconds,len(x))
        middle = min(t+cold-decision_seconds,end)
        targets = np.array(sorted({m for m,_ in actions(int(bases[i]),max_vms)}))
        first,second = np.minimum(r[:,None],targets),np.minimum(n[:,None],targets)
        following = targets*(targets+1)//2+second
        charge = []
        for a,b in ((t,middle),(middle,end)):
            # rho=1 maximizes routed requests and minimizes cost when queues/SLA are relaxed.
            vm = np.minimum(x[a:b,None],np.floor(levels[None,:]*model.throughput))
            charge.append((x[a:b,None]-vm).sum(axis=0)*model.faas_request_price)
        q = targets*(end-t)*model.vm_second_price+charge[0][first]+charge[1][second]+tables[i+1][following]
        tables[i] = np.min(q,axis=1)
    gateway = len(x)*model.config.gateway_hourly_price/3600
    return dict(lower_bound_usd=float(tables[0][0]+gateway),tables=tables,scope="finite residual targets, SLA/draining relaxed")


def fixed_plan(bases,action):
    if action==35:
        return np.zeros(len(bases),dtype=int),np.ones(len(bases))
    delta,rho = action//5-3,(0.6,0.7,0.8,0.9,1.0)[action%5]
    return np.clip(np.asarray(bases)+delta,0,512).astype(int),np.full(len(bases),rho)


def perfect_window(stream,bases,*,service_seconds,epsilon=0.001,decision_seconds=60,cold=100):
    """Perfect D+K arrival lookahead, execute K; local SLA only on executed K.

    It sees the replayed subsecond arrival realization, a stronger oracle than
    knowing only future per-second counts. It never reads past this episode.
    """
    fleet = QueueFleet(stream,service_seconds=service_seconds,cold_start_seconds=cold)
    targets,rhos = [],[]
    for base in bases:
        current = fleet.summary()
        t = fleet.t
        window = min(cold+decision_seconds,len(stream.counts)-t)
        executed_requests = int(stream.counts[t:min(t+decision_seconds,len(stream.counts))].sum())
        feasible = []
        for target,rho in actions(int(base)):
            candidate = fleet.clone()
            step = candidate.step(target,rho,decision_seconds)
            if step["violations"]>epsilon*executed_requests+1e-12:
                continue
            executed = candidate.clone()
            if window>decision_seconds:
                candidate.step(target,rho,window-decision_seconds)
            score = candidate.summary(terminal=candidate.t==len(stream.counts))["total_cost_usd"]-current["total_cost_usd"]
            feasible.append((score,target,rho,executed))
        if not feasible:
            raise AssertionError("all FaaS should satisfy local SLA")
        _,target,rho,fleet = min(feasible,key=lambda p:p[:3])
        targets.append(target); rhos.append(rho)
    return dict(summary=fleet.summary(terminal=True),targets=targets,rhos=rhos,
                oracle="perfect D+K arrivals, local executed-window SLA",lookahead_seconds=cold+decision_seconds)


def clairvoyant_search(stream,bases,*,service_seconds,incumbent_targets,incumbent_rhos,
                      epsilon=0.001,decision_seconds=60,cold=100,beam_width=8,exact=False):
    """Forward DP over full queue states, with declared beam pruning if enabled."""
    if beam_width<2:
        raise ValueError("beam width must be at least two")
    relaxation = cost_relaxation(stream.counts,bases,service_seconds,decision_seconds=decision_seconds,cold=cold)
    budget = epsilon*int(stream.counts.sum())
    initial = QueueFleet(stream,service_seconds=service_seconds,cold_start_seconds=cold)
    states = [(initial,[],[])]
    incumbent = initial.clone()
    incumbent_path = ([],[])
    discarded,expanded,max_states = 0,0,1
    for i,base in enumerate(bases):
        retained = {}
        for fleet,targets,rhos in states:
            for target,rho in actions(int(base)):
                candidate = fleet.clone()
                candidate.step(target,rho,decision_seconds)
                expanded += 1
                summary = candidate.summary()
                if summary["violations"]>budget+1e-12:
                    continue
                key = candidate.state_key()
                item = (candidate,targets+[target],rhos+[rho])
                # For hard constraints, equal physical state can have different cost/violation tradeoffs.
                frontier = retained.setdefault(key,[])
                if any(old[0].summary()["total_cost_usd"]<=summary["total_cost_usd"]+1e-12 and
                       old[0].summary()["violations"]<=summary["violations"] for old in frontier):
                    continue
                frontier[:] = [old for old in frontier if not (summary["total_cost_usd"]<=old[0].summary()["total_cost_usd"]+1e-12
                               and summary["violations"]<=old[0].summary()["violations"])]
                frontier.append(item)
        candidates = [item for frontier in retained.values() for item in frontier]
        max_states = max(max_states,len(candidates))
        if not candidates:
            raise AssertionError("zero VM path must remain feasible")
        def priority(item):
            fleet = item[0]
            n = len(fleet.ready_at)
            ready = int(np.sum(fleet.ready_at<=fleet.t))
            return fleet.summary()["total_cost_usd"]+relaxation["tables"][i+1][n*(n+1)//2+ready]
        if not exact and len(candidates)>beam_width:
            # Preserve both low-violation and low-cost states, plus the fixed incumbent below.
            half = max(1,beam_width//4)
            safe = sorted(candidates,key=lambda p:(p[0].summary()["violations"],priority(p)))[:half]
            ids = {id(p) for p in safe}
            others = sorted((p for p in candidates if id(p) not in ids),key=priority)[:beam_width-len(safe)]
            discarded += len(candidates)-len(safe)-len(others)
            candidates = safe+others
        incumbent.step(int(incumbent_targets[i]),float(incumbent_rhos[i]),decision_seconds)
        incumbent_path[0].append(int(incumbent_targets[i])); incumbent_path[1].append(float(incumbent_rhos[i]))
        if incumbent.summary()["violations"]<=budget+1e-12:
            if not any(p[0].state_key()==incumbent.state_key() and abs(p[0].summary()["total_cost_usd"]-incumbent.summary()["total_cost_usd"])<1e-12
                       and p[0].summary()["violations"]==incumbent.summary()["violations"] for p in candidates):
                candidates.append((incumbent.clone(),list(incumbent_path[0]),list(incumbent_path[1])))
        states = candidates
    best = min(states,key=lambda p:p[0].summary(terminal=True)["total_cost_usd"])
    return dict(summary=best[0].summary(terminal=True),targets=best[1],rhos=best[2],
        exact=discarded==0,pruned_states=discarded,expanded_states=expanded,max_states=max_states,
        cost_lower_bound_usd=relaxation["lower_bound_usd"],beam_width=None if exact else beam_width,
        meaning="feasible offline plan upper bound; exact optimum only if no states were pruned")
