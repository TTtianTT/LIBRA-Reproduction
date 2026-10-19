#!/usr/bin/env python3
"""Run the frozen local PPO study; checkpoints are chosen before final tests.

Stages are restartable with source/data hashes. Model ZIPs stay local; structured
run records, learning curves, tuning grids and final summaries are reviewable.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from libra_ml.features import training_set
from libra_ml.models import fit_gbdt, GBDT_PARAMS
from libra_rl.baselines import BaselineEvaluator, grids, select_on_training
from libra_rl.data import load_azure_partition
from libra_rl.env import RHO_GRID
from libra_rl.fast_dp import optimal_bound
from libra_rl.model import CostModel
from libra_rl.study import aggregate, chunks, evaluate_agent, train_agent
from libra_rl.training import choose_penalty
from libra_sim.traces import load_trace


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def tag(service):
    return f"s{round(service * 10):02d}"


def source_hashes():
    files = [Path(__file__), ROOT / "configs/rl_study.json", ROOT / "pyproject.toml",
             ROOT / "requirements-rl.txt"]
    for package in ("libra_rl", "libra_ml", "libra_sim"):
        files += sorted((ROOT / package).glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def prepare(output, pilot=False):
    protocol = json.loads((ROOT / "configs/rl_study.json").read_text())
    hashes = source_hashes()
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("source changed since this study started; use a new output directory")
    else:
        import stable_baselines3, gymnasium, sklearn, scipy
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint,
                    "source_sha256": hashes, "protocol": protocol, "pilot": pilot,
                    "platform": platform.platform(), "python": platform.python_version(),
                    "versions": {"stable-baselines3": stable_baselines3.__version__, "torch": torch.__version__,
                                 "gymnasium": gymnasium.__version__, "numpy": np.__version__,
                                 "scikit-learn": sklearn.__version__, "scipy": scipy.__version__},
                    "azure_aggregate_sha256": hashlib.sha256((ROOT / "data/azure2021/aggregate_counts.npz").read_bytes()).hexdigest(),
                    "validation_unit": "eight nonoverlapping 6h episodes, fresh VMs and local history per episode",
                    "test_unit": "nonoverlapping <=6h episodes, fresh VMs and local history per episode",
                    "test_exposure": protocol["test_exposure_caveat"]}
        write(path, manifest)
    train = load_azure_partition("train", root=ROOT / "data/azure2021", multiplier=protocol["azure_replication"])
    validation = load_azure_partition("validation", root=ROOT / "data/azure2021", multiplier=protocol["azure_replication"])
    cap = 2 ** math.ceil(math.log2(max(8, math.ceil(float(train.max()) * 2 / 16) + 8)))
    forecasts = {}
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_target = None
    for version, service in (("v1", 1.0), ("v2", 0.5), ("v2", 0.7)):
        path = model_dir / f"gbdt_{tag(service)}.joblib"
        model = CostModel(version=version, service_seconds=service)
        if path.exists():
            forecasts[service] = joblib.load(path)
        else:
            print(f"Fit causal GBDT, service={service}, train days 1-9 only", flush=True)
            if feature_target is None:
                feature_target = training_set(train, stride=300, target_step=10)
            start = time.perf_counter()
            features, target = feature_target
            forecasts[service] = fit_gbdt(features, target, model.tau, seed=0)
            joblib.dump(forecasts[service], path)
            write(output / f"forecast_{tag(service)}.json", {"train_rows": len(target), "features": features.shape[1],
                  "tau": model.tau, "fit_seconds": time.perf_counter() - start, "parameters": GBDT_PARAMS,
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "training_split": "Azure days 1-9, 16x replication"})
    return protocol, manifest, train, validation, cap, forecasts


def job(version, service, penalty, seed, protocol, variant="main"):
    settings = {"action_mode": "raw" if variant == "raw_actions" else "residual",
                "decision_seconds": {"interval120": 120, "interval300": 300}.get(variant, 60),
                "augment": variant != "no_augmentation", "hide_forecast": variant == "no_forecast"}
    return {"id": f"{version}_{tag(service)}_{variant}_l{penalty:g}_seed{seed}", "version": version,
            "service_seconds": service, "penalty": penalty, "seed": seed, "variant": variant,
            "samples": protocol["v1_samples"] if version == "v1" else protocol["v2_samples"], **settings}


def train_stage(output, protocol, manifest, train, validation, cap, forecasts, pilot=False):
    main_jobs = [job("v1", 1.0, 0.0, s, protocol) for s in protocol["seeds"]]
    main_jobs += [job("v2", service, penalty, seed, protocol) for service in (0.5, 0.7)
                  for penalty in protocol["penalties_usd_per_violation"] for seed in protocol["seeds"]]
    if pilot:
        main_jobs = [main_jobs[0], next(j for j in main_jobs if j["service_seconds"] == 0.7)]
        main_jobs = [{**j, "samples": 2048} for j in main_jobs]
        validation = validation[:21600]
    runs = []

    def execute(j):
        model = CostModel(version=j["version"], service_seconds=j["service_seconds"])
        result = train_agent(train, validation, job=j, protocol=protocol, model=model,
                             forecast_model=forecasts[j["service_seconds"]], max_vms=cap,
                             output=output, fingerprint=manifest["fingerprint"])
        runs.append(result)
        write(output / "progress.json", {"completed": len(runs), "last_job": j["id"],
                                         "stage": "pilot" if pilot else "training"})
        print(f"{len(runs):02d}: {j['id']} samples={result['samples']} fit={result['fit_seconds']:.1f}s "
              f"val_cost={result['validation']['total_cost_usd']:.4f} "
              f"val_violation={result['validation']['violation_rate']:.6f}", flush=True)
    for j in main_jobs:
        execute(j)
    if not pilot:
        chosen = {}
        for service in (0.5, 0.7):
            candidates = [{"split": "validation", "penalty": r["job"]["penalty"], **r["validation"]}
                          for r in runs if r["job"]["service_seconds"] == service]
            chosen[str(service)] = choose_penalty(candidates, protocol["epsilon"])
        write(output / "ablation_selection.json", {"split": "validation", "penalties": chosen,
                                                     "fingerprint": manifest["fingerprint"]})
        for service in (0.5, 0.7):
            for variant in ("no_augmentation", "no_forecast", "raw_actions", "interval120", "interval300"):
                for seed in protocol["seeds"]:
                    execute(job("v2", service, chosen[str(service)], seed, protocol, variant))
    write(output / "runs_index.json", {"fingerprint": manifest["fingerprint"],
                                        "runs": [r["job"]["id"] for r in runs], "count": len(runs), "max_vms": cap})
    return runs


def baseline_stage(output, protocol, manifest, train, cap, forecasts):
    path = output / "selected_baselines.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("stale baseline selection")
        return result
    selected, all_rows = [], []
    for version, service in (("v1", 1.0), ("v2", 0.5), ("v2", 0.7)):
        model = CostModel(version=version, service_seconds=service)
        evaluators = [BaselineEvaluator(x, model, forecasts[service], max_vms=cap) for x in chunks(train)]
        for family, candidates in grids().items():
            family_rows = []
            for parameters in candidates:
                summary = aggregate([e.run(family, parameters) for e in evaluators])
                family_rows.append({"split": "train", "version": version, "service_seconds": service,
                                    "family": family, "parameters": parameters, "summary": summary})
            all_rows.extend(family_rows)
            penalties = (0.0,) if version == "v1" else protocol["penalties_usd_per_violation"]
            for penalty in penalties:
                winner = select_on_training(family_rows, penalty)
                selected.append({**winner, "penalty": penalty})
            print(f"Baseline tuning {version}, s={service}, {family}: {len(candidates)} train candidates", flush=True)
            write(output / "baseline_training_grid.json", {"fingerprint": manifest["fingerprint"], "rows": all_rows})
        for family, parameters, label in (("LIBRA", {"rho": 0.8, "phi": 1}, "LIBRA-paper"),
                                           ("Quantile", {"rho": 1.0 if version == "v1" else 0.8, "window": 300}, "Quantile-reference"),
                                           ("FaaS", {}, "FaaS")):
            selected.append({"split": "fixed", "version": version, "service_seconds": service,
                             "family": family, "label": label, "parameters": parameters, "penalty": None})
    result = {"fingerprint": manifest["fingerprint"], "selection_split": "train",
              "candidate_budget": 12, "selection": selected,
              "budget_caveat": "PPO has 3 lambdas x 4 validation checkpoints per seed; static baselines have 12 train candidates per family. These caps do not equalize training FLOPs or validation queries."}
    write(path, result)
    return result


def final_stage(output, protocol, manifest, cap, forecasts):
    from stable_baselines3 import PPO
    final_path = output / "final_results.json"
    if final_path.exists():
        existing = json.loads(final_path.read_text())
        if existing["fingerprint"] != manifest["fingerprint"]:
            raise ValueError("stale final results")
        print("Final evaluation already exists; preserving its single frozen test batch.", flush=True)
        return
    index = json.loads((output / "runs_index.json").read_text())
    if index["count"] != 85 or index["fingerprint"] != manifest["fingerprint"]:
        raise ValueError("all 85 training configurations must finish before test evaluation")
    runs = [json.loads((output / "runs" / f"{name}.json").read_text()) for name in index["runs"]]
    baselines = json.loads((output / "selected_baselines.json").read_text())
    for run in runs:
        assert run["fingerprint"] == manifest["fingerprint"]
        assert hashlib.sha256((output / run["checkpoint"]).read_bytes()).hexdigest() == run["checkpoint_sha256"]
    freeze = {"fingerprint": manifest["fingerprint"], "frozen_utc": datetime.now(timezone.utc).isoformat(),
              "checkpoint_sha256": {r["job"]["id"]: r["checkpoint_sha256"] for r in runs},
              "ablation_selection": json.loads((output / "ablation_selection.json").read_text()),
              "baseline_selection": baselines}
    freeze_path = output / "test_freeze.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text())
        assert previous["checkpoint_sha256"] == freeze["checkpoint_sha256"]
        assert previous["fingerprint"] == freeze["fingerprint"]
        freeze = previous
    else:
        write(freeze_path, freeze)  # Created BEFORE opening/evaluating test-day counts.
    datasets = {"Azure16x": load_azure_partition("test", root=ROOT / "data/azure2021", multiplier=16),
                "Azure1x": load_azure_partition("test", root=ROOT / "data/azure2021", multiplier=1)}
    provenance = {}
    for name, filename, seconds in (("WITS", "wits_logs.txt", 43200), ("Berkeley", "berkeley_logs.txt", 43200),
                                     ("Sin", "sin_logs.txt", 10000)):
        datasets[name], provenance[name] = load_trace(ROOT / "data/traces" / filename, seconds=seconds)
    rows = []
    row_path = output / "evaluation_rows.json"
    if row_path.exists():
        partial = json.loads(row_path.read_text())
        assert partial["fingerprint"] == manifest["fingerprint"]
        rows = partial["rows"]
    seen = {r["key"] for r in rows}

    def save_row(key, row):
        rows.append({"key": key, **row})
        seen.add(key)
        write(row_path, {"fingerprint": manifest["fingerprint"], "rows": rows})

    for run in runs:
        j = run["job"]
        agent = None
        modes = ("wait_budget", "mmc_response") if j["version"] == "v2" and j["variant"] == "main" else ("wait_budget",)
        for mode in modes:
            model = CostModel(version=j["version"], service_seconds=j["service_seconds"], latency_mode=mode)
            for dataset, x in datasets.items():
                if mode == "mmc_response" and dataset not in ("Azure16x", "WITS", "Berkeley"):
                    continue
                key = f"PPO:{j['id']}:{mode}:{dataset}"
                if key in seen:
                    continue
                if agent is None:
                    agent = PPO.load(output / run["checkpoint"], device="cpu")
                start = time.perf_counter()
                summary = evaluate_agent(agent, x, model=model, forecast_model=forecasts[j["service_seconds"]],
                                         max_vms=cap, penalty=j["penalty"], decision_seconds=j["decision_seconds"],
                                         action_mode=j["action_mode"], hide_forecast=j["hide_forecast"])
                save_row(key, {"kind": "PPO", "dataset": dataset, "latency_mode": mode, "split": "test",
                               "job": j, "summary": summary, "evaluation_seconds": time.perf_counter() - start})
        print(f"Test batch: {j['id']}", flush=True)
    cache = {}
    for selected in baselines["selection"]:
        service, version = selected["service_seconds"], selected["version"]
        modes = ("wait_budget", "mmc_response") if version == "v2" else ("wait_budget",)
        for mode in modes:
            model = CostModel(version=version, service_seconds=service, latency_mode=mode)
            for dataset, x in datasets.items():
                if mode == "mmc_response" and dataset not in ("Azure16x", "WITS", "Berkeley"):
                    continue
                label = selected.get("label", selected["family"] + "-tuned")
                key = f"baseline:{version}:{service}:{label}:{selected['penalty']}:{mode}:{dataset}"
                if key in seen:
                    continue
                cache_key = (service, mode, dataset)
                if cache_key not in cache:
                    cache[cache_key] = [BaselineEvaluator(block, model, forecasts[service], max_vms=cap)
                                        for block in chunks(x)]
                summary = aggregate([e.run(selected["family"], selected["parameters"]) for e in cache[cache_key]])
                save_row(key, {"kind": "baseline", "dataset": dataset, "latency_mode": mode, "split": "test",
                               "version": version, "service_seconds": service, "label": label,
                               "penalty": selected["penalty"], "parameters": selected["parameters"], "summary": summary})
    bounds_path = output / "dp_bounds.json"
    bounds = json.loads(bounds_path.read_text())["rows"] if bounds_path.exists() else []
    done_bounds = {(r["dataset"], r["version"], r["service_seconds"], r["penalty"]) for r in bounds}
    for dataset, x in datasets.items():
        blocks = chunks(x)
        for version, service in (("v1", 1.0), ("v2", 0.5), ("v2", 0.7)):
            if version == "v2" and dataset not in ("Azure16x", "WITS", "Berkeley"):
                continue
            penalties = (0.0,) if version == "v1" else protocol["penalties_usd_per_violation"]
            model = CostModel(version=version, service_seconds=service)
            matching = [r["summary"]["max_target_vms"] for r in rows if r["dataset"] == dataset and
                        (r.get("service_seconds", r.get("job", {}).get("service_seconds")) == service) and
                        r["latency_mode"] == "wait_budget"]
            observed_cap = max(matching, default=0)
            domain_cap = max(observed_cap, math.ceil(float(np.max(x)) / model.throughput / 0.6) + 8)
            for penalty in penalties:
                if (dataset, version, service, penalty) in done_bounds:
                    continue
                print(f"DP final batch: {dataset} {version} s={service} lambda={penalty} cap={domain_cap}", flush=True)
                start = time.perf_counter()
                solved = [optimal_bound(block, model=model, decision_seconds=60, penalty=penalty,
                                        max_vms=None if version == "v1" else domain_cap,
                                        rho_grid=(1.0,) if version == "v1" else RHO_GRID) for block in blocks]
                bounds.append({"dataset": dataset, "version": version, "service_seconds": service,
                               "penalty": penalty, "decision_seconds": 60, "episodes": len(blocks),
                               "total_cost_usd": sum(r["total_cost_usd"] for r in solved),
                               "expected_violations": sum(r["expected_violations"] for r in solved),
                               "objective_usd": sum(r["objective_usd"] for r in solved),
                               "max_vms": max(r["max_vms"] for r in solved),
                               "scope": solved[0]["scope"], "observed_policy_max_vms": observed_cap,
                               "seconds": time.perf_counter() - start})
                write(bounds_path, {"fingerprint": manifest["fingerprint"], "rows": bounds})
    write(final_path, {"fingerprint": manifest["fingerprint"], "freeze_utc": freeze["frozen_utc"],
                       "completed_utc": datetime.now(timezone.utc).isoformat(), "rows": rows,
                       "dp_bounds": bounds, "trace_provenance": provenance,
                       "azure_test_sha256": hashlib.sha256(np.asarray(datasets["Azure1x"], dtype="<i8").tobytes()).hexdigest(),
                       "interpretation": "one frozen final batch; no test-based checkpoint or hyperparameter selection"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("pilot", "train", "baselines", "final", "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / ("outputs/rl_pilot" if args.stage == "pilot" else "outputs/rl_study")
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        protocol, manifest, train, validation, cap, forecasts = prepare(output, args.stage == "pilot")
        if args.stage in ("pilot", "train", "all"):
            train_stage(output, protocol, manifest, train, validation, cap, forecasts, args.stage == "pilot")
        if args.stage in ("baselines", "all"):
            baseline_stage(output, protocol, manifest, train, cap, forecasts)
        if args.stage in ("final", "all"):
            final_stage(output, protocol, manifest, cap, forecasts)


if __name__ == "__main__":
    main()
