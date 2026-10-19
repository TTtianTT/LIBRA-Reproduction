"""Pure, vectorized accounting shared by the environment and bounded DP."""
from dataclasses import dataclass, field
import math

import numpy as np

from libra_sim import Config
from .queueing import violation_probability


def validate_counts(counts):
    x = np.asarray(counts)
    if x.ndim != 1 or not x.size or not np.issubdtype(x.dtype, np.number):
        raise ValueError("counts must be a nonempty one-dimensional numeric array")
    if np.any(~np.isfinite(x)) or np.any(x < 0) or np.any(x != np.floor(x)):
        raise ValueError("counts must be finite nonnegative integers")
    return x.astype(np.int64)


@dataclass(frozen=True)
class CostModel:
    config: Config = field(default_factory=Config)
    version: str = "v1"
    service_seconds: float = 1.0
    latency_mode: str = "wait_budget"

    def __post_init__(self):
        if self.version not in ("v1", "v2"):
            raise ValueError("version must be v1 or v2")
        if not math.isfinite(self.service_seconds) or self.service_seconds <= 0:
            raise ValueError("service_seconds must be finite and positive")
        if self.version == "v1" and self.service_seconds != 1:
            raise ValueError("v1 reproduces the original one-second service")
        if self.latency_mode not in ("wait_budget", "mmc_response"):
            raise ValueError("unknown latency model")

    @property
    def throughput(self):
        return self.config.vm_capacity / self.service_seconds

    @property
    def vm_second_price(self):
        return self.config.vm_hourly_price / 3600

    @property
    def faas_request_price(self):
        c = self.config
        return c.request_memory_mb / 1024 * self.service_seconds * c.faas_gb_second_price + c.faas_request_price

    @property
    def tau(self):
        if self.faas_request_price == 0:
            return 0.0
        return float(np.clip(1 - self.vm_second_price / self.throughput / self.faas_request_price, 0, 1))

    def traffic(self, demand, ready, rho=1.0, estimate=None):
        """Return VM/FaaS requests, utilization and expected SLA violations.

        v2 assumes balanced traffic over ready VMs. Faster service changes both
        VM throughput and FaaS compute charges; invocation fees stay fixed.
        ``estimate`` enables exact replay of the paper's estimate-based routing.
        """
        if not math.isfinite(rho) or not 0 < rho <= 1:
            raise ValueError("rho must be in (0, 1]")
        d, n = np.broadcast_arrays(np.asarray(demand, dtype=float), np.asarray(ready, dtype=float))
        capacity = n * self.throughput
        budget = np.floor(rho * capacity)
        if estimate is not None:
            budget = np.minimum(capacity, np.floor(rho * np.asarray(estimate)))
        vm = np.minimum(d, budget)
        faas = d - vm
        utilization = np.divide(vm, capacity, out=np.zeros_like(vm), where=capacity > 0)
        violations = np.zeros_like(vm)
        if self.version == "v2":
            violations = vm * violation_probability(utilization, self.service_seconds,
                                                    self.config.sla_seconds,
                                                    self.config.vm_capacity, self.latency_mode)
            # FaaS execution is deterministic and has no queue/cold-start penalty.
            if self.service_seconds > self.config.sla_seconds:
                violations = violations + faas
        return {"vm_requests": vm, "faas_requests": faas,
                "utilization": utilization, "expected_violations": violations}
