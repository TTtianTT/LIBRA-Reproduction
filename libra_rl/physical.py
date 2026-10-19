"""Transient deterministic-service FCFS queues, independent of old tail formulas.

Each VM has 16 slots and retains scheduled completions across seconds. Random
within-second arrivals and routing uniforms are shared across
policies. Scale-in drains accepted work and bills the draining VM. No Gym/Torch.
"""
from dataclasses import dataclass
import copy
import math

import numpy as np

from .model import CostModel, validate_counts


def _serve(arrivals, destinations, availability, pointers, service, sla):
    violations, waited, response_sum, maximum = 0,0,0.0,0.0
    for i in range(len(arrivals)):
        vm = destinations[i]
        slot = pointers[vm]
        start = max(arrivals[i],availability[vm,slot])
        departure = start+service
        availability[vm,slot] = departure
        pointers[vm] = (slot+1)%availability.shape[1]
        response = departure-arrivals[i]
        violations += int(response>sla+1e-10)
        waited += int(start>arrivals[i]+1e-10)
        response_sum += response
        maximum = max(maximum,response)
    return violations,waited,response_sum,maximum


try:
    from numba import njit
    serve = njit(cache=True)(_serve)
    ACCELERATOR = "numba"
except ImportError:
    serve = _serve
    ACCELERATOR = "python"


def staggered_copies(counts, offsets):
    """Past-only noncircular copies within one split; discard the common prefix."""
    x = validate_counts(counts)
    offsets = np.asarray(offsets)
    if offsets.ndim!=1 or not len(offsets) or np.any(offsets<0) or np.any(offsets!=offsets.astype(int)):
        raise ValueError("nonnegative integer offsets required")
    maximum = int(offsets.max())
    if maximum>=len(x):
        raise ValueError("partition too short for specified shifts")
    result = np.zeros(len(x)-maximum,dtype=np.int64)
    for offset in offsets:
        result += x[maximum-int(offset):len(x)-int(offset)]
    return result


@dataclass
class ArrivalStream:
    counts: np.ndarray
    offsets: np.ndarray
    times: np.ndarray
    ranks: np.ndarray
    routing: np.ndarray

    @classmethod
    def generate(cls,counts,seed=0):
        x = validate_counts(counts)
        offsets = np.concatenate(([0],np.cumsum(x)))
        times,routing = np.empty(int(offsets[-1])),np.empty(int(offsets[-1]))
        ranks = np.empty(int(offsets[-1]),dtype=np.int32)
        rng = np.random.default_rng(seed)
        for t,n in enumerate(x):
            a,b = offsets[t:t+2]
            times[a:b] = t+np.sort(rng.random(int(n)))
            # A causal quota admits the first arrivals, without seeing later requests.
            ranks[a:b] = np.arange(int(n))
            routing[a:b] = rng.random(int(n))
        return cls(x,offsets,times,ranks,routing)


class QueueFleet:
    def __init__(self,stream,*,service_seconds=0.5,cold_start_seconds=100,sla_seconds=1.0,max_vms=512):
        if service_seconds<=0 or cold_start_seconds<0 or sla_seconds<=0 or max_vms<1:
            raise ValueError("invalid physical parameters")
        self.stream = stream
        self.model = CostModel(version="v2",service_seconds=service_seconds)
        self.service,self.cold,self.sla,self.max_vms = service_seconds,cold_start_seconds,sla_seconds,max_vms
        self.t = 0
        self.ready_at = np.empty(0)
        self.availability = np.empty((0,16))
        self.pointers = np.empty(0,dtype=np.int64)
        self.totals = dict(vm_seconds=0.0,drain_vm_seconds=0.0,starting_vm_seconds=0.0,
            requests=0,vm_requests=0,faas_requests=0,violations=0,waited_requests=0,
            response_seconds=0.0,max_response_seconds=0.0)

    def clone(self):
        other = copy.copy(self)
        other.ready_at = self.ready_at.copy()
        other.availability = self.availability.copy()
        other.pointers = self.pointers.copy()
        other.totals = self.totals.copy()
        return other

    def state_key(self):
        # Absolute completion times and ring indices are required; (VM,ready) is insufficient.
        columns = (np.arange(16)[None,:]+self.pointers[:,None])%16
        ordered = np.take_along_axis(self.availability,columns,axis=1)
        return (self.t,np.maximum(self.ready_at,self.t).tobytes(),np.maximum(ordered,self.t).tobytes())

    def backlog(self):
        return int(np.ceil(np.maximum(self.availability-self.t,0)/self.service-1e-12).clip(0).sum())

    def step(self,target,rho=1.0,seconds=60):
        if not isinstance(target,(int,np.integer)) or not 0<=target<=self.max_vms:
            raise ValueError("target outside VM domain")
        if not 0<rho<=1 or seconds<1 or self.t>=len(self.stream.counts):
            raise ValueError("invalid step")
        end = min(self.t+seconds,len(self.stream.counts))
        before = self.summary()
        n = len(self.ready_at)
        if target<n:
            drain = np.maximum(self.availability[target:].max(axis=1)-self.t,0).sum()
            self.totals["drain_vm_seconds"] += float(drain)
            self.ready_at = self.ready_at[:target].copy()
            self.availability = self.availability[:target].copy()
            self.pointers = self.pointers[:target].copy()
        elif target>n:
            self.ready_at = np.concatenate((self.ready_at,np.full(target-n,self.t+self.cold)))
            self.availability = np.concatenate((self.availability,np.zeros((target-n,16))))
            self.pointers = np.concatenate((self.pointers,np.zeros(target-n,dtype=np.int64)))
        self.totals["vm_seconds"] += target*(end-self.t)
        self.totals["starting_vm_seconds"] += float(np.clip(self.ready_at-self.t,0,end-self.t).sum())
        # Readiness changes at integer seconds; each phase shares the routing population.
        boundaries = sorted({self.t,end,*[int(v) for v in self.ready_at if self.t<v<end]})
        stream = self.stream
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            ready = np.flatnonzero(self.ready_at<=a)
            demand = stream.counts[a:b]
            cap = int(np.floor(rho*(len(ready)*self.model.throughput)))
            admitted = np.minimum(demand,cap)
            self.totals["requests"] += int(demand.sum())
            self.totals["faas_requests"] += int(demand.sum()-admitted.sum())
            m = int(admitted.sum())
            self.totals["vm_requests"] += m
            if not m:
                continue
            left,right = stream.offsets[[a,b]]
            budgets = np.repeat(admitted,demand)
            mask = stream.ranks[left:right]<budgets
            arrivals = stream.times[left:right][mask]
            destinations = ready[np.minimum((stream.routing[left:right][mask]*len(ready)).astype(np.int64),len(ready)-1)]
            violations,waited,response,maximum = serve(arrivals,destinations,self.availability,self.pointers,self.service,self.sla)
            self.totals["violations"] += int(violations)
            self.totals["waited_requests"] += int(waited)
            self.totals["response_seconds"] += float(response)
            self.totals["max_response_seconds"] = max(self.totals["max_response_seconds"],float(maximum))
        self.t = end
        after = self.summary()
        return dict(cost_usd=after["total_cost_usd"]-before["total_cost_usd"],
            violations=after["violations"]-before["violations"],backlog=self.backlog(),target_vms=int(target),rho=rho)

    def summary(self,*,terminal=False):
        result = self.totals.copy()
        terminal_drain = float(np.maximum(self.availability.max(axis=1)-self.t,0).sum()) if terminal and len(self.ready_at) else 0.0
        result["terminal_drain_vm_seconds"] = terminal_drain
        result["vm_cost_usd"] = (result["vm_seconds"]+result["drain_vm_seconds"]+terminal_drain)*self.model.vm_second_price
        result["faas_cost_usd"] = result["faas_requests"]*self.model.faas_request_price
        result["gateway_cost_usd"] = self.t*self.model.config.gateway_hourly_price/3600
        result["total_cost_usd"] = result["vm_cost_usd"]+result["faas_cost_usd"]+result["gateway_cost_usd"]
        result["violations"] += result["faas_requests"]*int(self.service>self.sla)
        result["violation_rate"] = result["violations"]/max(1,result["requests"])
        return result


def actions(base,max_vms=512):
    """Unique resource/routing choices; zero VM has one canonical action."""
    choices = {(int(np.clip(base+d,0,max_vms)),rho) for d in range(-3,4) for rho in (0.6,0.7,0.8,0.9,1.0)}
    return sorted({(m,rho) for m,rho in choices if m>0}|{(0,1.0)})


def evaluate_plan(stream,targets,rhos,*,service_seconds=0.5,decision_seconds=60,**kwargs):
    if len(targets)!=math.ceil(len(stream.counts)/decision_seconds) or len(rhos)!=len(targets):
        raise ValueError("one target and rho per decision required")
    fleet = QueueFleet(stream,service_seconds=service_seconds,**kwargs)
    for target,rho in zip(targets,rhos):
        fleet.step(int(target),float(rho),decision_seconds)
    return fleet.summary(terminal=True)
