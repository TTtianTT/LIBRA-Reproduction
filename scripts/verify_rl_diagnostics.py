#!/usr/bin/env python3
"""Verify diagnostic provenance, action bounds, budgets, and non-test scope."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from libra_rl.training import confidence_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/rl_diagnostics")
    parser.add_argument("--require-models",action="store_true")
    args = parser.parse_args()
    output = args.output
    read = lambda name:json.loads((output/name).read_text())
    sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    manifest,index,summary,gate = [read(n) for n in ("manifest.json","runs_index.json","summary.json","toy_gate.json")]
    protocol,fingerprint = manifest["protocol"],manifest["fingerprint"]
    for path,expected in manifest["source_sha256"].items():
        assert sha(ROOT/path)==expected,f"source changed: {path}"
    assert sha(ROOT/"data/azure2021/aggregate_counts.npz")==manifest["azure_aggregate_sha256"]
    assert manifest["test_evaluated"] is index["test_evaluated"] is summary["test_evaluated"] is False
    assert summary["partial"] is False
    assert index["fingerprint"]==summary["fingerprint"]==gate["fingerprint"]==fingerprint
    for name in ("action_audit.json","domain_bounds.json","queue_calibration.json",
                 "raw_failure_summary.json","penalties.json"):
        assert read(name)["fingerprint"]==fingerprint,f"stale artifact: {name}"
    assert len(set(index["runs"]))==index["count"]==30
    main_runs = [read(f"runs/{name}.json") for name in index["runs"]]
    toy_runs = [read(f"runs/toy_random_seed{s}.json") for s in protocol["toy"]["seeds"]]
    bounds = read("domain_bounds.json")["rows"]
    assert len(bounds)==60
    assert all(r["split"]=="validation" for r in bounds)
    for r in bounds:
        if r["domain"]=="original_plus_faas":
            old = next(o for o in bounds if o["case"]==r["case"] and o["domain"]=="original"
                       and o["penalty"]==r["penalty"] and o["cost_weight"]==r["cost_weight"])
            assert r["objective"]<=old["objective"]+1e-8
        if r["cost_weight"]==0 and r["domain"]!="original":
            assert r["expected_violations"]==0
    for r in summary["domain_frontiers"]:
        if r["feasible_upper_usd"] is not None:
            assert r["dual_lower_usd"]<=r["feasible_upper_usd"]+1e-8
    models_checked = 0
    assert len(summary["budget_groups"])==12
    assert read("penalties.json")["split"]=="validation"
    for case in ("v1","v2_s05","v2_s07"):
        metadata = read(f"forecast_{case}.json")
        assert metadata["training_split"]=="train"
        path = output/"models"/f"gbdt_{case}.joblib"
        if args.require_models or path.exists():
            assert path.exists() and sha(path)==metadata["sha256"]
    for run in main_runs+toy_runs:
        job = run["job"]
        assert run["fingerprint"]==fingerprint
        assert run["selection_split"]=="validation" and run["test_evaluated"] is False
        assert run["samples"]==job["samples"]
        if job["case"]!="toy":
            assert job["samples"]==max(protocol["budgets"]) and job["augment"] is True
            assert job["initialization"] in protocol["initializations"]
            assert run["penalty"]==read("penalties.json")["penalties"][job["case"]]
        assert run["fit_seconds"]>0
        curve = run["curve"]
        expected = [n for n in protocol["checkpoints"] if n<=job["samples"]]
        assert [p["samples"] for p in curve]==expected
        if job["initialization"]=="rule_prior":
            assert abs(run["initial_validation"]["total_cost_usd"]-run["teacher_validation"]["total_cost_usd"])<=1e-9
        for point in curve:
            s = point["summary"]
            assert point["split"]=="validation"
            assert np.isclose(s["total_cost_usd"],s["vm_cost_usd"]+s["faas_cost_usd"]+s["gateway_cost_usd"],atol=1e-9,rtol=0)
            assert np.isclose(s["requests"],s["vm_requests"]+s["faas_requests"],atol=1e-9,rtol=0)
            assert sum(s["action_counts"])==s["decisions"]
            assert np.isclose(point["objective_usd"],s["total_cost_usd"]+run["penalty"]*s["expected_violations"],atol=1e-9,rtol=0)
            if job["case"]!="toy":
                bound = next(b for b in bounds if b["case"]==job["case"] and b["domain"]=="teacher_plus_faas"
                             and b["penalty"]==run["penalty"] and b["cost_weight"]==1)
                assert point["objective_usd"]>=bound["objective"]-1e-8,"policy beats same-domain clairvoyant DP"
            path = output/point["checkpoint"]
            if args.require_models or path.exists():
                assert path.exists() and sha(path)==point["checkpoint_sha256"]
                models_checked += 1
        for s in run["selections"]:
            budget = s["budget"]
            candidates = [p for p in curve if p["samples"] in {budget//4,budget//2,3*budget//4,budget}]
            assert len(candidates)==4
            best = min(candidates,key=lambda p:p["objective_usd"])
            assert s["selected_samples"]==best["samples"] and s["summary"]==best["summary"]
            assert s["checkpoint_sha256"]==best["checkpoint_sha256"]
            assert s["validation_feasible_fallback"]["summary"]["violation_rate"]<=protocol["epsilon"]
    assert sum(r["samples"] for r in main_runs)==index["samples"]==summary["actual_main_samples"]==7864320
    assert sum(r["samples"] for r in toy_runs)==gate["samples"]==summary["actual_toy_samples"]==327680
    assert gate["passed"] and gate["mean_regret_fraction"]<=gate["threshold"]
    for group in summary["budget_groups"]:
        matching = [s["summary"] for r in main_runs for s in r["selections"]
                    if r["job"]["case"]==group["case"] and r["job"]["initialization"]==group["initialization"] and s["budget"]==group["budget"]]
        assert len(matching)==group["seeds"]==5
        assert sorted(r["job"]["seed"] for r in main_runs if r["job"]["case"]==group["case"]
                      and r["job"]["initialization"]==group["initialization"])==protocol["seeds"]
        for metric in ("total_cost_usd","violation_rate","behavioral_teacher_fraction","explicit_faas_action_fraction"):
            np.testing.assert_allclose(confidence_interval([s[metric] for s in matching]),
                [group[metric],group[metric+"_lo"],group[metric+"_hi"]],atol=1e-12)
        sample = next(r for r in main_runs if r["job"]["case"]==group["case"])
        teacher = sample["teacher_validation"]
        j_teacher = teacher["total_cost_usd"]+sample["penalty"]*teacher["expected_violations"]
        bound = next(b for b in bounds if b["case"]==group["case"] and b["domain"]=="teacher_plus_faas"
                     and b["penalty"]==sample["penalty"] and b["cost_weight"]==1)
        gaps = [(j_teacher-s["total_cost_usd"]-sample["penalty"]*s["expected_violations"])/
                (j_teacher-bound["objective"]) for s in matching]
        np.testing.assert_allclose(confidence_interval(gaps),[group["objective_gap_closed_fraction"],
            group["objective_gap_closed_lo"],group["objective_gap_closed_hi"]],atol=1e-12)
    queue = read("queue_calibration.json")["rows"]
    assert len(queue)==100
    assert len({(r["service_seconds"],r["distribution"],r["utilization"],r["seed"]) for r in queue})==100
    assert all(r["measured_requests"]==500000 and r["warmup_requests"]==50000 for r in queue)
    assert all(0<=r["violation_rate"]<=1 for r in queue)
    assert all(abs(r["violation_rate"]-r["violations"]/r["measured_requests"])<1e-15 for r in queue)
    raw = read("raw_failure_summary.json")["rows"]
    assert {r["seed"] for r in raw}=={0,1}
    for r in raw:
        source = ROOT/"outputs/rl_study/runs"/f"v2_s07_raw_actions_l0.01_seed{r['seed']}.json"
        old = json.loads(source.read_text())
        assert sha(source)==r["source_record_sha256"]
        assert abs(old["validation"]["total_cost_usd"]-r["summary"]["total_cost_usd"])<=1e-9
        assert r["split"]=="validation" and r["steps"]==2880
        with (output/f"raw_trace_seed{r['seed']}.csv").open() as handle:
            trace = list(csv.DictReader(handle))
        assert len(trace)==r["steps"]
        assert abs(sum(float(p["total_cost_usd"]) for p in trace)-r["summary"]["total_cost_usd"])<1e-8
        rmse = np.sqrt(np.mean([(float(p["value"])-float(p["discounted_return"]))**2 for p in trace]))
        assert abs(float(rmse)-r["value_rmse"])<1e-8
    from PIL import Image
    for stem in ("learning_curves","queue_calibration","diagnostic_pareto"):
        with Image.open(output/(stem+".png")) as picture:
            assert picture.width>500 and picture.height>300
            picture.verify()
        ET.parse(output/(stem+".svg"))
    assert (output/"REPORT.md").stat().st_size>1000
    for name,expected in (("budget_summary",12),("learning_curves",210),("paired_differences",12),
                          ("queue_summary",20),("domain_frontiers",9)):
        with (output/(name+".csv")).open() as handle:
            assert len(list(csv.DictReader(handle)))==expected,f"incomplete {name}"
    print(f"Verified 30 main + 5 toy runs, 8,192,000 actual samples, 60 DP bounds, 100 queue runs, {models_checked} local checkpoints; no test re-evaluation")


if __name__=="__main__":
    main()
