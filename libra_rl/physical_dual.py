"""Admissible SLA-aware bounds by resetting queues at readiness phase borders.

Resetting queues can only accelerate deterministic FCFS departures. The DP
therefore relaxes both cost (free draining) and violations. Nonnegative
Lagrange multipliers give genuine constrained-cost lower bounds.
"""
import numpy as np

from .model import CostModel
from .physical import QueueFleet,actions


def phase_dual_tables(stream,bases,service_seconds,penalties=(0,1e-5,3e-5,1e-4,3e-4,0.001,0.003,0.01)):
    model = CostModel(version="v2",service_seconds=service_seconds)
    maximum = min(512,int(max(bases))+3)
    n,r = np.tril_indices(maximum+1)
    levels = np.arange(maximum+1)
    states = len(n)
    stage_data = []
    for i,base in enumerate(bases):
        t,end = i*60,min((i+1)*60,len(stream.counts))
        middle = min(t+40,end)
        targets = np.array(sorted({m for m,_ in actions(int(base))}))
        first,second = np.minimum(r[:,None],targets),np.minimum(n[:,None],targets)
        following = targets*(targets+1)//2+second
        by_rho = []
        for rho in (0.6,0.7,0.8,0.9,1.0):
            charges,risks = [],[]
            for a,b in ((t,middle),(middle,end)):
                demand = stream.counts[a:b]
                vm = np.minimum(demand[:,None],np.floor(rho*(levels[None,:]*model.throughput)))
                charges.append((demand[:,None]-vm).sum(axis=0)*model.faas_request_price)
                viol = np.zeros(maximum+1)
                if b>a:
                    for level in range(1,maximum+1):
                        empty = QueueFleet(stream,service_seconds=service_seconds,cold_start_seconds=0)
                        empty.t = a
                        empty.ready_at = np.zeros(level)
                        empty.availability = np.zeros((level,16))
                        empty.pointers = np.zeros(level,dtype=np.int64)
                        empty.step(level,rho,b-a)
                        viol[level] = empty.summary()["violations"]
                risks.append(viol)
            c = targets*(end-t)*model.vm_second_price+charges[0][first]+charges[1][second]
            v = risks[0][first]+risks[1][second]
            by_rho.append((c,v))
        stage_data.append((following,by_rho))
    tables = []
    for penalty in penalties:
        value = [None]*(len(bases)+1)
        value[-1] = np.zeros(states)
        for i in reversed(range(len(bases))):
            following,by_rho = stage_data[i]
            value[i] = np.min(np.stack([np.min(c+penalty*v+value[i+1][following],axis=1) for c,v in by_rho]),axis=0)
        tables.append(value)
    return np.asarray(penalties),tables
