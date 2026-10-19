#!/usr/bin/env python3
"""Run independent study seeds in local worker processes, then freeze/evaluate.

Calls the same train_agent/selection code as rl_study.py; does not change sample
budgets or policy algorithms. Completed runs are checked and reused unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
from pathlib import Path
import time

import torch
from threadpoolctl import threadpool_limits

import rl_study as driver
from libra_rl.model import CostModel
from libra_rl.training import choose_penalty

STATE = None
LIMITER = None


def initialize(output):
    global STATE, LIMITER
    torch.set_num_threads(1)
    LIMITER = threadpool_limits(limits=1)
    STATE = (Path(output), driver.prepare(Path(output)))


def fit(j):
    output, (protocol, manifest, train, validation, cap, forecasts) = STATE
    return driver.train_agent(train, validation, job=j, protocol=protocol,
                               model=CostModel(version=j["version"], service_seconds=j["service_seconds"]),
                               forecast_model=forecasts[j["service_seconds"]], max_vms=cap,
                               output=output, fingerprint=manifest["fingerprint"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=driver.ROOT / "outputs/rl_study")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        protocol, manifest, train, validation, cap, forecasts = driver.prepare(args.output)
        main_jobs = [driver.job("v1", 1.0, 0.0, s, protocol) for s in protocol["seeds"]]
        main_jobs += [driver.job("v2", service, penalty, seed, protocol) for service in (0.5, 0.7)
                      for penalty in protocol["penalties_usd_per_violation"] for seed in protocol["seeds"]]
        results = {}
        def collect(pool, jobs):
            futures = {pool.submit(fit, j): j for j in jobs}
            for future in as_completed(futures):
                run = future.result()
                j = run["job"]
                results[j["id"]] = run
                print(f"{len(results):02d}/85 {j['id']} fit={run['fit_seconds']:.1f}s "
                      f"val_cost={run['validation']['total_cost_usd']:.4f} "
                      f"val_violation={run['validation']['violation_rate']:.6f}", flush=True)
                driver.write(args.output / "parallel_progress.json", {"completed": len(results),
                               "planned": 85, "last_job": j["id"], "workers": args.workers})
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                 initializer=initialize, initargs=(str(args.output),)) as pool:
            collect(pool, main_jobs)
            ordered = [results[j["id"]] for j in main_jobs]
            chosen = {}
            for service in (0.5, 0.7):
                validation_rows = [{"split": "validation", "penalty": r["job"]["penalty"], **r["validation"]}
                                   for r in ordered if r["job"]["service_seconds"] == service]
                chosen[str(service)] = choose_penalty(validation_rows, protocol["epsilon"])
            ablations = [driver.job("v2", service, chosen[str(service)], seed, protocol, variant)
                         for service in (0.5, 0.7)
                         for variant in ("no_augmentation", "no_forecast", "raw_actions", "interval120", "interval300")
                         for seed in protocol["seeds"]]
            collect(pool, ablations)
        driver.write(args.output / "parallel_driver.json", {"source": "scripts/rl_study_parallel.py",
                      "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "workers": args.workers,
                      "training_wall_seconds": time.perf_counter()-started,
                      "semantics": "same train_agent, per-seed budgets and validation selection; process scheduling only"})
        driver.train_stage(args.output, protocol, manifest, train, validation, cap, forecasts)
        driver.baseline_stage(args.output, protocol, manifest, train, cap, forecasts)
        driver.final_stage(args.output, protocol, manifest, cap, forecasts)


if __name__ == "__main__":
    main()
