"""Independent continuous-time FCFS multi-server simulation, no queue formulas.

Stationary constant-rate calibration only. This does not validate transient
one-second rates, traffic balancing, real FaaS latency, or AWS execution times.
"""
import heapq
import numpy as np


def simulate_queue(*, utilization, service_seconds, distribution, servers=16,
                    sla_seconds=1.0, warmup=50000, measured=500000, seed=0):
    if not 0 < utilization < 1 or service_seconds <= 0 or sla_seconds <= 0:
        raise ValueError("positive times and stable utilization are required")
    if not isinstance(servers,int) or servers < 1 or warmup < 0 or measured < 1:
        raise ValueError("invalid queue size or sample count")
    if distribution not in ("deterministic", "exponential"):
        raise ValueError("unknown service distribution")
    rng = np.random.default_rng(seed)
    total = warmup+measured
    arrivals = np.cumsum(rng.exponential(service_seconds/(servers*utilization), total))
    services = (np.full(total, service_seconds) if distribution == "deterministic"
                else rng.exponential(service_seconds, total))
    available = [0.0]*servers
    violations, waited, response_sum = 0, 0, 0.0
    for i in range(total):
        arrival = float(arrivals[i])
        start = max(arrival, available[0])
        departure = start+float(services[i])
        heapq.heapreplace(available, departure)
        if i >= warmup:
            response = departure-arrival
            # Absorb arithmetic roundoff at deterministic response==SLA.
            violations += int(response > sla_seconds+1e-12)
            waited += int(start > arrival+1e-12)
            response_sum += response
    return {"utilization": utilization, "service_seconds": service_seconds,
            "distribution": distribution, "servers": servers, "seed": seed,
            "warmup_requests": warmup, "measured_requests": measured,
            "violations": violations, "violation_rate": violations/measured,
            "wait_probability": waited/measured, "mean_response_seconds": response_sum/measured}
