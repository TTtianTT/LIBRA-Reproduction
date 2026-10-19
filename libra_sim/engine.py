"""One-second batch simulator. No AWS SDK, network calls, or wall-clock waiting.

Arrivals at t execute over [t,t+1). FIFO cohorts retain original arrival times,
so requests delayed even once violate the paper's one-second response SLA.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import math

from .config import Config

POLICIES = ("LIBRA", "FAAS", "MAX", "SPOCK", "AUTO")


@dataclass
class VM:
    created: int
    ready: int


@dataclass
class Monitor:
    alpha: float
    beta: float
    avg: float | None = None
    deviation: float = 0

    def observe(self, count):
        if self.avg is None:
            self.avg = float(count)
        else:
            self.avg = (1 - self.alpha) * self.avg + self.alpha * count
            # Eq. (7) uses the updated avg from Eq. (6).
            self.deviation = ((1 - self.beta) * self.deviation
                              + self.beta * abs(count - self.avg))
        return self.avg, self.deviation


def libra_target(avg, deviation, config):
    """Algorithm 1; clamp negative estimates for negative-phi sensitivity runs."""
    if avg < config.cip_requests_per_second:
        return 0, 0.0
    estimate = max(0.0, avg + config.phi * deviation)
    return math.ceil(estimate / config.vm_capacity), estimate


def simulate(counts, policy="LIBRA", config=None):
    c = config or Config()
    policy = policy.upper()
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy: {policy}")
    counts = list(counts)
    if not counts or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts):
        raise ValueError("Trace must contain nonnegative integer request counts")
    horizon = len(counts)
    monitor = Monitor(c.alpha, c.beta)
    vms = []
    queue = deque()
    queued = 0
    vm_seconds = 0
    faas_total = 0
    vm_total = 0
    histogram = Counter()
    rows = []
    decisions = []
    routing_estimate = 0.0
    pending_routing = None
    utilization = []

    def resize(target, t, prewarm=False):
        nonlocal vms, vm_seconds
        before = len(vms)
        # Requests finish at the epoch boundary, so scale-in loses no in-flight work.
        if target < before:
            vm_seconds += sum(max(0, t - v.created) for v in vms[target:])
            vms = vms[:target]
        elif target > before:
            vms.extend(VM(t, t if prewarm else t + c.vm_cold_start_seconds)
                       for _ in range(target - before))
        return before

    if policy == "MAX":
        resize(math.ceil(max(counts) / c.vm_capacity), 0, prewarm=True)
    elif policy in ("AUTO", "SPOCK"):
        resize(c.auto_min_instances, 0)

    for t, demand in enumerate(counts):
        avg, deviation = monitor.observe(demand)
        if pending_routing is not None and t >= pending_routing[0]:
            routing_estimate = pending_routing[1]
            pending_routing = None
        if policy == "LIBRA" and t % c.scaling_interval_seconds == 0:
            target, estimate = libra_target(avg, deviation, c)
            previous = resize(target, t)
            if target > previous:
                pending_routing = (t + c.vm_cold_start_seconds, estimate)
                if c.vm_cold_start_seconds == 0:
                    routing_estimate = estimate
                    pending_routing = None
            else:
                routing_estimate = estimate
                pending_routing = None
            decisions.append({"time_s": t, "target_vms": target, "avg": avg,
                              "deviation": deviation, "estimate": estimate})

        ready = sum(v.ready <= t for v in vms)
        capacity = ready * c.vm_capacity
        starting = len(vms) - ready
        faas = 0
        if policy == "FAAS":
            faas = demand
        elif policy == "LIBRA":
            budget = min(capacity, math.floor(c.rho * routing_estimate))
            faas = max(0, demand - budget)
        elif policy == "SPOCK" and starting:
            # Existing FIFO backlog is kept on VMs. FaaS only absorbs new overflow
            # while an initial/scale-out VM is starting (Section IV-C.3).
            faas = max(0, demand - max(0, capacity - queued))

        vm_arrivals = demand - faas
        faas_total += faas
        histogram[c.service_seconds] += faas
        if vm_arrivals:
            queue.append([t, vm_arrivals])
            queued += vm_arrivals
        served = min(queued, capacity)
        left = served
        while left:
            arrival, n = queue[0]
            take = min(left, n)
            histogram[t - arrival + c.service_seconds] += take
            left -= take
            queue[0][1] -= take
            if queue[0][1] == 0:
                queue.popleft()
        queued -= served
        vm_total += served

        rows.append({"time_s": t, "requests": demand, "ewma": avg,
                     "deviation": deviation, "allocated_vms": len(vms),
                     "ready_vms": ready, "starting_vms": starting,
                     "routing_estimate": routing_estimate, "vm_arrivals": vm_arrivals,
                     "vm_served": served, "faas_requests": faas, "queue_requests": queued})

        if policy in ("AUTO", "SPOCK"):
            # Average resource utilization includes allocated VMs while starting,
            # matching the denominator in the author's simulator.
            utilization.append(served / (len(vms) * c.vm_capacity))
            if len(utilization) >= c.auto_window_seconds:
                u = sum(utilization) / len(utilization)
                old = len(vms)
                target = old
                if u > c.auto_scale_out_threshold:
                    target = min(c.auto_max_instances, old + c.auto_scale_group)
                elif u < c.auto_scale_in_threshold:
                    target = max(c.auto_min_instances, old - c.auto_scale_group)
                if target != old:
                    # Decision uses this epoch's completed utilization; new VMs
                    # are requested at the next boundary.
                    resize(target, t + 1)
                    decisions.append({"time_s": t + 1, "target_vms": target,
                                      "utilization": u})
                    utilization = []
                else:
                    utilization = utilization[-(c.auto_window_seconds - 1):] if c.auto_window_seconds > 1 else []

    # Drain all arrivals, including those queued in the last epoch, without
    # further policy changes or new arrivals. Billing includes the drain.
    t = horizon
    while queued:
        capacity = sum(v.ready <= t for v in vms) * c.vm_capacity
        if not vms:
            raise RuntimeError("Queue cannot drain without provisioned VMs")
        left = min(queued, capacity)
        queued -= left
        vm_total += left
        while left:
            arrival, n = queue[0]
            take = min(left, n)
            histogram[t - arrival + c.service_seconds] += take
            left -= take
            queue[0][1] -= take
            if not queue[0][1]:
                queue.popleft()
        t += 1
    billing_end = t
    vm_seconds += sum(max(0, billing_end - v.created) for v in vms)
    vm_cost = vm_seconds * c.vm_hourly_price / 3600
    faas_cost = faas_total * c.faas_cost_per_request
    gateway_cost = horizon * c.gateway_hourly_price / 3600 if policy == "LIBRA" else 0
    total = sum(counts)
    violations = sum(n for latency, n in histogram.items() if latency > c.sla_seconds)
    if sum(histogram.values()) != total or vm_total + faas_total != total:
        raise RuntimeError("Request accounting mismatch")
    summary = {
        "policy": policy, "duration_seconds": horizon, "total_requests": total,
        "vm_requests": vm_total, "faas_requests": faas_total,
        "sla_violations": violations, "sla_violation_percent": 100 * violations / total if total else 0,
        "mean_response_seconds": sum(k * v for k, v in histogram.items()) / total if total else 0,
        "max_response_seconds": max((k for k, v in histogram.items() if v), default=0),
        "vm_uptime_seconds": vm_seconds, "vm_cost_usd": vm_cost,
        "faas_cost_usd": faas_cost, "gateway_cost_usd": gateway_cost,
        "total_cost_usd": vm_cost + faas_cost + gateway_cost,
        "drain_seconds": billing_end - horizon,
        "response_histogram": {str(k): v for k, v in sorted(histogram.items()) if v},
        "config": asdict(c),
    }
    return {"summary": summary, "timeline": rows, "decisions": decisions}
