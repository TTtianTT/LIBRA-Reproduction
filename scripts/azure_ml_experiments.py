#!/usr/bin/env python3
"""Chronological Azure 2021 experiment: train days 1-7, test days 8-14.

Uses all functions aggregated without test-dependent selection. Runs both the
original counts and a declared x64 workload-replication scenario. Calendar/day
lag features are an extension tested against the same lag-only GBDT and strong
non-ML window/previous-day quantile rules. No test-based hyperparameter tuning.
"""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_sim import Config, simulate
from libra_ml import feature_matrix, newsvendor_tau
from libra_ml.models import GBDT_PARAMS, fit_gbdt
from libra_ml.policies import pinball

DAY = 86400
TRAIN_END = 7 * DAY
TOTAL = 14 * DAY
EWMA_GRID = [(a, p) for a in (.2, .05, .02, .005) for p in (0, .5, 1, 1.5, 2, 3)]


def day_features(counts, times, tau):
    """Every demand-valued feature uses <=t; day phase is known at decision t."""
    X, scales = feature_matrix(counts, times)
    x = np.asarray(counts, dtype=float)
    calendar, seasonal = [], []
    for t, scale in zip(times, scales):
        if t + 100 >= DAY:
            # Yesterday at the NEXT service window's phase; its end is <=t.
            lo, hi = max(0, t + 100 - DAY), t + 400 - DAY
            previous = x[lo:hi]
            has_day = bool(len(previous))
        else:
            previous = x[max(0, t + 1 - 300):t + 1]
            has_day = False
        q = float(np.quantile(previous, tau))
        seasonal.append(q)
        day_angle = 2 * np.pi * (t % DAY) / DAY
        week_angle = 2 * np.pi * (t % (7 * DAY)) / (7 * DAY)
        calendar.append([math.sin(day_angle), math.cos(day_angle), math.sin(week_angle), math.cos(week_angle),
                         q / scale, float(np.mean(previous)) / scale, float(has_day)])
    return X, np.column_stack([X, np.asarray(calendar)]), scales, np.asarray(seasonal)


def training_arrays(counts, tau):
    # Caller passes ONLY the training prefix; targets cannot extend into test.
    x = np.asarray(counts, dtype=float)
    times = np.arange(0, len(x) - 400 + 1, 300)
    lag, clock, scales, _ = day_features(x, times, tau)
    offsets = np.arange(100, 400, 10)
    y = np.concatenate([x[t + offsets] / scale for t, scale in zip(times, scales)])
    return np.repeat(lag, len(offsets), axis=0), np.repeat(clock, len(offsets), axis=0), y


def cached_estimator(levels):
    def estimate(t, history):
        return max(0.0, float(levels[t // 300]))
    return estimate


def offline_levels(test, config, tau):
    """Convex single-window integer-capacity oracle; ignores inter-window effects."""
    levels = []
    x = np.asarray(test, dtype=float)
    for t in range(0, len(x), 300):
        window = x[t + 100:t + 400]
        if not len(window):
            window = x[t:t + 300]
        # An empirical inverse-CDF quantile is a continuous convex minimizer.
        q = float(np.quantile(window, tau, method="inverted_cdf"))
        candidates = sorted({0, math.floor(q / config.vm_capacity), math.ceil(q / config.vm_capacity)})
        def cost(n):
            return n * len(window) * config.vm_hourly_price / 3600 + config.faas_cost_per_request * np.maximum(0, window - n * config.vm_capacity).sum()
        levels.append(min(candidates, key=cost) * config.vm_capacity)
    return levels


def run_scenario(raw, factor, config, out):
    started = time.perf_counter()
    train = [int(x) * factor for x in raw[:TRAIN_END]]
    test = [int(x) * factor for x in raw[TRAIN_END:]]
    full = np.asarray(raw, dtype=np.float64) * factor
    tau = newsvendor_tau(config)
    X_lag, X_clock, y = training_arrays(train, tau)
    lag_model = fit_gbdt(X_lag, y, tau)
    clock_model = fit_gbdt(X_clock, y, tau)
    times = TRAIN_END + np.arange(0, len(test), 300)
    lag, clock, scale, seasonal = day_features(full, times, tau)
    # Check vectorized feature computation against strict prefixes at the start,
    # middle and end of the held-out week. This tests temporal causality directly.
    for i in (0, len(times) // 2, len(times) - 1):
        pl, pc, ps, pq = day_features(full[:times[i] + 1], np.asarray([times[i]]), tau)
        np.testing.assert_allclose(lag[i], pl[0])
        np.testing.assert_allclose(clock[i], pc[0])
        np.testing.assert_allclose(scale[i], ps[0])
        np.testing.assert_allclose(seasonal[i], pq[0])
    predictions = {
        "fill_gbdt_lag": np.maximum(0, lag_model.predict(lag) * scale),
        "fill_gbdt_calendar": np.maximum(0, clock_model.predict(clock) * scale),
        "fill_previous_day_quantile": seasonal,
        "fill_window_quantile": np.asarray([np.quantile(full[t + 1 - 300:t + 1], tau) for t in times]),
        "oracle": offline_levels(test, config, tau)}
    fill = replace(config, rho=1.0)
    scores = []
    print(f"Azure x{factor}: {len(y)} training rows; tuning EWMA on training prefix only", flush=True)
    for i, (alpha, phi) in enumerate(EWMA_GRID):
        s = simulate(train, "LIBRA", replace(fill, alpha=alpha, beta=alpha, phi=phi), routing="capacity")["summary"]
        scores.append((s["total_cost_usd"], alpha, phi))
        if (i + 1) % 6 == 0:
            print(f"Azure x{factor}: EWMA grid {i+1}/{len(EWMA_GRID)}", flush=True)
    train_cost, alpha, phi = min(scores)
    runs = {"paper": (config, "estimate", None), "fill_paper": (fill, "capacity", None),
            "fill_tuned": (replace(fill, alpha=alpha, beta=alpha, phi=phi), "capacity", None)}
    runs.update({name: (fill, "capacity", cached_estimator(levels)) for name, levels in predictions.items()})
    rows, daily = [], []
    for name, (cfg, routing, estimator) in runs.items():
        result = simulate(test, "LIBRA", cfg, routing=routing, estimator=estimator)
        summary = result["summary"]
        row = {"load_multiplier": factor, "policy": name, "total_requests": summary["total_requests"],
               "total_cost_usd": summary["total_cost_usd"], "vm_cost_usd": summary["vm_cost_usd"],
               "faas_cost_usd": summary["faas_cost_usd"], "gateway_cost_usd": summary["gateway_cost_usd"],
               "sla_violation_percent": summary["sla_violation_percent"],
               "faas_request_percent": 100 * summary["faas_requests"] / summary["total_requests"]}
        losses = [pinball(test[d["time_s"] + 100:d["time_s"] + 400], d["estimate"], tau)
                  for d in result["decisions"] if d["time_s"] + 100 < len(test)]
        row["normalized_pinball"] = float(np.mean(losses) / np.mean(test))
        rows.append(row)
        for day in range(7):
            timeline = result["timeline"][day * DAY:(day + 1) * DAY]
            # Sum allocated VM seconds over this day: exact for this no-queue,
            # one-second-service simulation, including startup allocation time.
            vm_seconds = sum(r["allocated_vms"] for r in timeline)
            faas = sum(r["faas_requests"] for r in timeline)
            cost = vm_seconds * config.vm_hourly_price / 3600 + faas * config.faas_cost_per_request + DAY * config.gateway_hourly_price / 3600
            daily.append({"load_multiplier": factor, "test_day": day + 8, "policy": name, "total_cost_usd": cost})
        assert summary["total_requests"] == sum(test) and summary["sla_violations"] == 0
        assert math.isclose(sum(r["total_cost_usd"] for r in daily if r["policy"] == name), summary["total_cost_usd"], abs_tol=1e-8)
        print(f"Azure x{factor} {name}: ${summary['total_cost_usd']:.6f}, SLA={summary['sla_violation_percent']}%", flush=True)
        del result
        gc.collect()
    meta = {"load_multiplier": factor, "train_seconds": TRAIN_END, "test_seconds": len(test),
            "train_mean_rps": float(np.mean(train)), "test_mean_rps": float(np.mean(test)),
            "training_rows": len(y), "train_feature_rows_stride_seconds": 300, "training_target_sample_step_seconds": 10,
            "tuned_ewma": {"alpha_beta": alpha, "phi": phi, "training_cost_usd": train_cost},
            "elapsed_seconds": time.perf_counter() - started}
    return rows, daily, meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/ml_azure2021")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = Config.read(ROOT / "configs/paper.json")
    manifest = json.loads((ROOT / "data/azure2021/manifest.json").read_text())
    raw = np.load(ROOT / "data/azure2021/aggregate_counts.npz")["counts"]
    assert len(raw) == TOTAL and hashlib.sha256(raw.astype('<i8').tobytes()).hexdigest() == manifest["aggregate_sha256"]
    rows, daily, scenarios = [], [], []
    with threadpool_limits(limits=4):
        for factor in (1, 64):
            r, d, m = run_scenario(raw, factor, config, args.output)
            rows.extend(r)
            daily.extend(d)
            scenarios.append(m)
            payload = {"metadata": {"created_utc": datetime.now(timezone.utc).isoformat(),
                "platform": platform.platform(), "python": platform.python_version(), "config": asdict(config),
                "tau": newsvendor_tau(config), "gbdt_params": GBDT_PARAMS, "ewma_grid": EWMA_GRID,
                "dataset": manifest, "protocol": "first 7 days train/tune, next 7 days holdout; no refitting on test",
                "calendar_caveat": "Relative daily/weekly phase; source timestamps were modified",
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted([*ROOT.glob('libra_ml/*.py'), *ROOT.glob('libra_sim/*.py'), Path(__file__)])}},
                "scenarios": scenarios, "rows": rows, "daily": daily}
            (args.output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
            for filename, values in (("results.csv", rows), ("daily_costs.csv", daily)):
                with (args.output / filename).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(values[0]), lineterminator="\n")
                    writer.writeheader()
                    writer.writerows(values)
    print(f"Azure experiment complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
