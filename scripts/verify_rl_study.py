#!/usr/bin/env python3
"""Audit PPO run completeness, frozen selection, accounting and seed statistics."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from libra_rl.training import confidence_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rl_study")
    parser.add_argument("--require-models", action="store_true")
    args = parser.parse_args()
    output = args.output
    def read(name):
        return json.loads((output / name).read_text())
    manifest, index, freeze, final, summary = [read(name) for name in
        ("manifest.json", "runs_index.json", "test_freeze.json", "final_results.json", "summary.json")]
    fingerprint = manifest["fingerprint"]
    for path, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, f"source changed: {path}"
    if (output / "parallel_driver.json").exists():
        parallel = read("parallel_driver.json")
        assert hashlib.sha256((ROOT / parallel["source"]).read_bytes()).hexdigest() == parallel["sha256"]
    assert hashlib.sha256((ROOT / "data/azure2021/aggregate_counts.npz").read_bytes()).hexdigest() == manifest["azure_aggregate_sha256"]
    for obj in (index, freeze, final, summary, read("selected_baselines.json"), read("ablation_selection.json")):
        assert obj["fingerprint"] == fingerprint
    assert datetime.fromisoformat(freeze["frozen_utc"]) <= datetime.fromisoformat(final["completed_utc"])
    assert final["freeze_utc"] == freeze["frozen_utc"]
    assert len(index["runs"]) == len(set(index["runs"])) == index["count"] == 85
    runs = [read(f"runs/{name}.json") for name in index["runs"]]
    local_weights = 0
    for run in runs:
        j = run["job"]
        assert run["fingerprint"] == fingerprint
        assert run["checkpoint_selection_split"] == "validation"
        assert run["samples"] == j["samples"]
        assert run["fit_seconds"] > 0
        curve = run["learning_curve"]
        assert len(curve) == 4
        assert [p["samples"] for p in curve] == [j["samples"] * f for f in (0.25, 0.5, 0.75, 1.0)]
        assert all(p["split"] == "validation" for p in curve)
        best = min(curve, key=lambda p: p["selection_objective_usd"])
        assert run["selected_step"] == best["samples"]
        assert run["validation"] == best["summary"]
        assert freeze["checkpoint_sha256"][j["id"]] == run["checkpoint_sha256"]
        path = output / run["checkpoint"]
        if args.require_models or path.exists():
            assert path.exists(), f"missing local model: {path}"
            assert hashlib.sha256(path.read_bytes()).hexdigest() == run["checkpoint_sha256"]
            local_weights += 1
    assert sum(r["samples"] for r in runs) == summary["training_samples"] == 1351680
    grid = read("baseline_training_grid.json")["rows"]
    assert len(grid) == 180
    assert all(row["split"] == "train" for row in grid)
    assert freeze["baseline_selection"] == read("selected_baselines.json")
    assert freeze["ablation_selection"] == read("ablation_selection.json")
    rows = final["rows"]
    assert len({r["key"] for r in rows}) == len(rows)
    for row in rows:
        assert row["split"] == "test"
        s = row["summary"]
        assert np.isclose(s["total_cost_usd"], s["vm_cost_usd"]+s["faas_cost_usd"]+s["gateway_cost_usd"], atol=1e-9, rtol=0)
        assert np.isclose(s["requests"], s["vm_requests"]+s["faas_requests"], atol=1e-9, rtol=0)
        assert np.isclose(s["violation_rate"], s["expected_violations"]/max(1,s["requests"]), atol=1e-12)
        assert 0 <= s["expected_violations"] <= s["requests"] + 1e-9
        assert s["cancelled_startup_vm_seconds"] <= s["starting_vm_seconds"]
        if row["kind"] == "baseline" and row["label"] == "FaaS":
            assert s["vm_seconds"] == s["expected_violations"] == 0
    for group in summary["grouped_seed_results"]:
        matching = [row for row in rows if row["kind"] == "PPO" and
                    row["dataset"] == group["dataset"] and row["latency_mode"] == group["latency_mode"] and
                    all(row["job"][k] == group[k] for k in
                        ("version", "service_seconds", "variant", "penalty", "decision_seconds"))]
        assert sorted(row["job"]["seed"] for row in matching) == [0, 1, 2, 3, 4]
        for metric in ("total_cost_usd", "violation_rate"):
            mean, lo, hi = confidence_interval([r["summary"][metric] for r in matching])
            assert np.isclose(mean, group[metric], atol=1e-12)
            np.testing.assert_allclose([lo, hi], group[metric+"_ci95"], atol=1e-12)
    for bound in final["dp_bounds"]:
        # A finite rho grid need not bound LIBRA's estimate-based routing.
        policies = [row for row in rows if row["kind"] == "PPO" and row["dataset"] == bound["dataset"]
                    and row["latency_mode"] == "wait_budget" and
                    row["job"]["version"] == bound["version"] and
                    row["job"]["service_seconds"] == bound["service_seconds"] and
                    row["job"]["penalty"] == bound["penalty"]]
        for row in policies:
            s = row["summary"]
            assert bound["objective_usd"] <= s["total_cost_usd"] + bound["penalty"]*s["expected_violations"] + 1e-7
    for name in ("REPORT.md", "summary.csv", "learning_curves.csv", "pareto.png", "learning_curves.png", "ablations.png"):
        assert (output / name).stat().st_size > 0
    print(f"Verified 85 trained PPO runs, 1,351,680 samples, 180 baseline candidates, {len(rows)} frozen test rows, "
          f"{len(summary['grouped_seed_results'])} five-seed intervals, {len(final['dp_bounds'])} DP bounds; "
          f"{local_weights} local checkpoints checked.")


if __name__ == "__main__":
    main()
