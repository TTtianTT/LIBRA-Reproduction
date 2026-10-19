"""Balanced per-VM Erlang-C latency approximations, evaluated per second.

``wait_budget`` matches the proposed table: exponential M/M/c waiting tail
with a *fixed* execution time. It is a hybrid approximation, not the complete
M/M/c response-time distribution. ``mmc_response`` includes exponential service.
Neither mode simulates transient queue buildup across seconds.
"""
import numpy as np


def erlang_c(utilization, servers=16):
    if not isinstance(servers, int) or servers < 1:
        raise ValueError("servers must be a positive integer")
    u = np.asarray(utilization, dtype=float)
    if np.any(~np.isfinite(u)) or np.any(u < 0):
        raise ValueError("utilization must be finite and nonnegative")
    offered = servers * np.minimum(u, 1.0)
    b = np.ones_like(u)
    for k in range(1, servers + 1):
        b = offered * b / (k + offered * b)
    c = b / (1 - np.minimum(u, 1.0) + np.minimum(u, 1.0) * b)
    return np.where(u >= 1, 1.0, c)


def violation_probability(utilization, service_seconds, sla_seconds=1.0,
                          servers=16, mode="wait_budget"):
    if service_seconds <= 0 or sla_seconds <= 0:
        raise ValueError("service and SLA must be positive")
    u = np.asarray(utilization, dtype=float)
    c = erlang_c(u, servers)
    mu = 1 / service_seconds
    k = servers * mu * np.maximum(0.0, 1 - u)
    if mode == "wait_budget":
        slack = sla_seconds - service_seconds
        p = np.ones_like(u) if slack < 0 else c * np.exp(-k * slack)
    elif mode == "mmc_response":
        # Conditional on waiting, W~Exp(k), S~Exp(mu), independently.
        delta = k - mu
        numerator = k * np.exp(-mu * sla_seconds) - mu * np.exp(-k * sla_seconds)
        conditional = np.divide(numerator, delta, out=np.zeros_like(u),
                                where=np.abs(delta) > 1e-9 * mu)
        conditional = np.where(np.abs(delta) <= 1e-9 * mu,
                               np.exp(-mu * sla_seconds) * (1 + mu * sla_seconds),
                               conditional)
        p = (1 - c) * np.exp(-mu * sla_seconds) + c * conditional
    else:
        raise ValueError(f"Unknown latency mode: {mode}")
    # No stationary distribution exists at saturation; conservatively use 1.
    return np.clip(np.where(u >= 1, 1.0, p), 0.0, 1.0)
