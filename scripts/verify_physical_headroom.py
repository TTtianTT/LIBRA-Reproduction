#!/usr/bin/env python3
"""Rebuild arrival streams and replay every corrected-model published plan."""
import json
from datetime import datetime
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import physical_headroom as driver
import forecast_rule_controls as forecast
from libra_rl.physical import ArrivalStream,evaluate_plan
from libra_rl.physical_lookup import MD16Lookup
from libra_rl.physical_planning import cost_relaxation,actions
from summarize_physical_headroom import ci


def main():
    output = ROOT/"outputs/physical_headroom"
    read = lambda name:json.loads((output/name).read_text())
    protocol,manifest,data = driver.prepare(output)
    fp = manifest["fingerprint"]
    original,extended,summary = read("evaluation.json"),read("forecast_rule_evaluation.json"),read("summary.json")
    exact = read("exact_intervals.json")
    assert exact["physical_fingerprint"]==fp and len(exact["records"])==20
    for path,expected in exact["source_sha256"].items():
        assert driver.sha(ROOT/path)==expected
    assert original["fingerprint"]==summary["fingerprint"]==extended["physical_fingerprint"]==fp
    assert manifest["test_evaluated"] is original["test_evaluated"] is extended["test_evaluated"] is summary["test_evaluated"] is False
    assert manifest["ppo_trained"] is original["ppo_trained"] is extended["ppo_trained"] is summary["ppo_trained"] is False
    assert len(original["runs"])==20 and len(original["rows"])==62 and len(extended["rows"])==60
    assert len(summary["groups"])==10 and len(summary["per_seed"])==50
    assert protocol["history_seconds"]==1800
    grid,selection = read("fixed_training_grid.json"),read("fixed_selection.json")
    forecast_grid = read("forecast_rule_grid.json")
    assert extended["source_sha256"]==forecast_grid["source_sha256"]==driver.sha(ROOT/"scripts/forecast_rule_controls.py")
    assert extended["training_selection_sha256"]==driver.sha(output/"forecast_rule_grid.json")
    assert grid["selection_split"]==selection["selection_split"]==forecast_grid["selection_split"]=="train"
    assert datetime.fromisoformat(selection["frozen_utc"])<=datetime.fromisoformat(original["completed_utc"])
    assert len(grid["rows"])==72 and len(forecast_grid["rows"])==144
    for service in protocol["services_seconds"]:
        candidates = [r for r in grid["rows"] if r["service_seconds"]==service and r["summary"]["violation_rate"]<=protocol["epsilon"]]
        assert min(candidates,key=lambda r:(r["summary"]["total_cost_usd"],r["action"]))==selection["selected"][str(service)]
        assert selection["selected"][str(service)]==grid["selected"][str(service)]
        for variant in ("causal_economic","perfect_quantile"):
            candidates = [r for r in forecast_grid["rows"] if r["service_seconds"]==service and r["variant"]==variant
                          and r["summary"]["violation_rate"]<=protocol["epsilon"]]
            assert min(candidates,key=lambda r:(r["summary"]["total_cost_usd"],r["action"]))==forecast_grid["selected"][f"{service}_{variant}"]
    replayed = 0
    metrics = ("total_cost_usd","vm_cost_usd","faas_cost_usd","gateway_cost_usd","requests","vm_requests","faas_requests",
               "violations","violation_rate","vm_seconds","drain_vm_seconds","terminal_drain_vm_seconds","response_seconds")
    for name in original["runs"]:
        run = read("runs/"+name+".json")
        service,seed,window,start = run["service_seconds"],run["arrival_seed"],run["window"],run["start"]
        assert run["fingerprint"]==fp and run["split"]=="validation" and run["history_seconds"]==1800
        stream = ArrivalStream.generate(data["validation"][start:start+3600],seed=run["stream_seed"])
        bases = driver.bases_for(data["validation"],start,3600,service)
        np.testing.assert_array_equal(bases,run["bases"])
        lower = cost_relaxation(stream.counts,bases,service)["lower_bound_usd"]
        records = [(policy,p) for policy,p in run["results"].items()]
        new = [r for r in extended["rows"] if (r["service_seconds"],r["arrival_seed"],r["window"])==(service,seed,window)]
        for r in new:
            if r["policy"] in ("causal_economic","perfect_quantile"):
                targets,rhos = forecast.rule_plan(data["validation"],start,3600,service,r["action"],bases,
                                                  perfect=r["policy"]=="perfect_quantile")
                np.testing.assert_array_equal(targets,r["targets"])
                np.testing.assert_array_equal(rhos,r["rhos"])
            records.append((r["policy"],r))
        certified = next(r for r in exact["records"] if (r["service_seconds"],r["arrival_seed"],r["window"])==(service,seed,window))
        assert certified["exact"] and abs(certified["feasible_upper_bound_usd"]-certified["cost_lower_bound_usd"])<=1e-10
        assert certified["relaxation_lower_bound_usd"]<=certified["sla_phase_dual_lower_bound_usd"]+1e-9
        assert certified["sla_phase_dual_lower_bound_usd"]<=certified["cost_lower_bound_usd"]+1e-9
        records.append(("certified_clairvoyant",certified))
        for policy,p in records:
            for base,target,rho in zip(bases,p["targets"],p["rhos"]):
                canonical = (0,1.0) if target==0 else (target,rho)
                assert canonical in actions(int(base)),"policy escaped the shared action domain"
            observed = evaluate_plan(stream,p["targets"],p["rhos"],service_seconds=service)
            for metric in metrics:
                assert np.isclose(observed[metric],p["summary"][metric],atol=1e-8,rtol=0),f"replay mismatch {name} {policy} {metric}"
            assert observed["requests"]==observed["vm_requests"]+observed["faas_requests"]
            assert observed["total_cost_usd"]>=lower-1e-9
            if policy in ("clairvoyant_dp","wider_dp","refined_clairvoyant","perfect_window","certified_clairvoyant"):
                assert observed["violation_rate"]<=protocol["epsilon"]+1e-12
            if "cost_lower_bound_usd" in p and policy!="certified_clairvoyant":
                assert abs(p["cost_lower_bound_usd"]-lower)<1e-9
            replayed += 1
        refined = next(r for r in new if r["policy"]=="refined_clairvoyant")
        assert certified["feasible_upper_bound_usd"]<=refined["summary"]["total_cost_usd"]+1e-9
        assert refined["summary"]["total_cost_usd"]<=run["results"]["clairvoyant_dp"]["summary"]["total_cost_usd"]+1e-9
        if "wider_dp" in run["results"]:
            assert run["results"]["wider_dp"]["summary"]["total_cost_usd"]<=run["results"]["clairvoyant_dp"]["summary"]["total_cost_usd"]+1e-9
    all_rows = original["rows"]+extended["rows"]+[dict(policy="certified_clairvoyant",**r) for r in exact["records"]]
    for group in summary["groups"]:
        aggregates = []
        for seed in protocol["validation_arrival_seeds"]:
            matching = [r["summary"] for r in all_rows if (r["service_seconds"],r["arrival_seed"],r["policy"])==
                        (group["service_seconds"],seed,group["policy"])]
            assert len(matching)==2
            aggregates.append(driver.aggregate(matching))
        for metric in ("total_cost_usd","violation_rate","drain_vm_seconds","terminal_drain_vm_seconds"):
            np.testing.assert_allclose(ci([r[metric] for r in aggregates]),
                [group[metric],group[metric+"_lo"],group[metric+"_hi"]],atol=1e-12)
    for certificate in summary["certificates"]:
        service = certificate["service_seconds"]
        candidates = [("fixed",grid["selected"][str(service)]["summary"]["total_cost_usd"]),
                      ("causal_economic",forecast_grid["selected"][f"{service}_causal_economic"]["summary"]["total_cost_usd"])]
        family = min(candidates,key=lambda p:p[1])[0]
        assert certificate["strong_causal_family_selected_on_train"]==family
        group = next(g for g in summary["groups"] if g["service_seconds"]==service and g["policy"]==family)
        lower = np.mean([sum(r["cost_lower_bound_usd"] for r in exact["records"]
                            if r["service_seconds"]==service and r["arrival_seed"]==seed)
                         for seed in protocol["validation_arrival_seeds"]])
        assert certificate["certified_windows"]==10
        np.testing.assert_allclose(
            [certificate["cost_lower_bound_usd"],certificate["strong_causal_cost_usd"],
             certificate["strong_causal_violation_rate"],certificate["improvement_upper_fraction_vs_strong_causal"]],
            [lower,group["total_cost_usd"],group["violation_rate"],
             (group["total_cost_usd"]-lower)/group["total_cost_usd"]],atol=1e-12,rtol=0)
    lookup = read("md16_lookup.json")
    assert lookup["fingerprint"]==fp and len(lookup["rows"])==80
    assert lookup["source_queue_calibration_sha256"]==driver.sha(ROOT/"outputs/rl_diagnostics/queue_calibration.json")
    table = MD16Lookup.load(output/"md16_lookup.json")
    for service in protocol["services_seconds"]:
        points = lookup["tables"][str(service)]
        assert np.all(np.diff([p["monotone_estimate"] for p in points])>=0)
        try:
            table.probability(1.0,service)
        except ValueError:
            pass
        else:
            raise AssertionError("lookup extrapolated saturation")
    audit = json.loads((ROOT/"outputs/constant_action_audit/summary.json").read_text())
    assert audit["source_sha256"]==driver.sha(ROOT/"scripts/audit_constant_actions.py")
    assert sum(bool(m["matching_metric_actions"]) for r in audit["comparisons"] for m in r["high_budget_matches"])==15
    assert [r["dominant_at_least_95_percent"] for r in audit["comparisons"]]==[35,29,29]
    reward = json.loads((ROOT/"outputs/constant_action_audit/reward_signal.json").read_text())
    assert reward["source_sha256"]==driver.sha(ROOT/"scripts/audit_reward_signal.py") and len(reward["rows"])==6
    raw = read("raw_action_behavior.json")
    assert raw["source_sha256"]==driver.sha(ROOT/"outputs/rl_diagnostics/raw_trace_seed1.csv")
    assert raw["delta_counts"]=={"-8":39,"4":2788,"8":53} and raw["decisions"]==2880
    from PIL import Image
    for name in ("controls","md16_lookup"):
        with Image.open(output/(name+".png")) as img:
            img.verify()
        ET.parse(output/(name+".svg"))
    assert (output/"REPORT.md").stat().st_size>1000
    assert (output/"COURSE_REPORT.md").stat().st_size>1000
    print(f"Verified corrected source/data hashes, 72+144 train-grid candidates, {replayed} transient queue plan replays, "
          "20 optimality certificates, fixed-action 15/15 equivalence, 80 lookup runs; no PPO training or test use")


if __name__=="__main__":
    main()
