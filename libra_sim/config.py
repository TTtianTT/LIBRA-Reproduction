from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    epoch_seconds: int = 1
    request_memory_mb: int = 512
    service_seconds: int = 1
    sla_seconds: float = 1
    vm_memory_mb: int = 8192
    vm_hourly_price: float = 0.1
    vm_cold_start_seconds: int = 100
    faas_gb_second_price: float = 0.00001667
    faas_request_price: float = 0.0000002
    gateway_hourly_price: float = 0.0116
    alpha: float = 0.2
    beta: float = 0.2
    scaling_interval_seconds: int = 300
    rho: float = 0.8
    phi: float = 1
    cip_requests_per_second: float = 4
    auto_min_instances: int = 10
    auto_max_instances: int = 292
    auto_scale_group: int = 10
    auto_scale_out_threshold: float = 0.7
    auto_scale_in_threshold: float = 0.3
    auto_window_seconds: int = 120

    def __post_init__(self):
        if any(not isinstance(v, (int, float)) or isinstance(v, bool)
               or not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("All configuration values must be finite numbers")
        for key in ("epoch_seconds", "service_seconds", "request_memory_mb", "vm_memory_mb",
                    "scaling_interval_seconds", "vm_cold_start_seconds", "auto_min_instances",
                    "auto_max_instances", "auto_scale_group", "auto_window_seconds"):
            if not isinstance(getattr(self, key), int):
                raise ValueError(f"{key} must be an integer")
        if self.epoch_seconds != 1 or self.service_seconds != 1:
            raise ValueError("This reproduction implements the paper's one-second epochs/service only")
        if self.request_memory_mb <= 0 or self.vm_memory_mb < self.request_memory_mb:
            raise ValueError("A VM must have space for at least one request")
        if self.vm_cold_start_seconds < 0 or self.scaling_interval_seconds <= 0:
            raise ValueError("Invalid startup/scaling interval")
        if self.scaling_interval_seconds <= self.vm_cold_start_seconds:
            raise ValueError("LIBRA scaling interval must exceed VM startup time (Section III-D)")
        if not (0 < self.alpha <= 1 and 0 < self.beta <= 1 and 0 < self.rho <= 1):
            raise ValueError("alpha, beta and rho must be in (0, 1]")
        if not (0 <= self.auto_scale_in_threshold < self.auto_scale_out_threshold <= 1):
            raise ValueError("Invalid autoscaling thresholds")
        if not (1 <= self.auto_min_instances <= self.auto_max_instances):
            raise ValueError("Invalid autoscaling instance bounds")
        if self.auto_scale_group < 1 or self.auto_window_seconds < 1:
            raise ValueError("Autoscaling group/window must be positive")
        if any(getattr(self, k) < 0 for k in ("vm_hourly_price", "faas_gb_second_price",
               "faas_request_price", "gateway_hourly_price", "cip_requests_per_second")):
            raise ValueError("Prices and CIP must be nonnegative")
        if self.sla_seconds <= 0:
            raise ValueError("SLA must be positive")

    @property
    def vm_capacity(self):
        return self.vm_memory_mb // self.request_memory_mb

    @property
    def faas_cost_per_request(self):
        return (self.request_memory_mb / 1024 * self.service_seconds
                * self.faas_gb_second_price + self.faas_request_price)

    @property
    def price_derived_cip(self):
        if self.faas_cost_per_request == 0:
            return None
        return self.vm_hourly_price / 3600 / self.faas_cost_per_request

    @classmethod
    def read(cls, path):
        values = json.loads(Path(path).read_text())
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values)
