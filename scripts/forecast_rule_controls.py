#!/usr/bin/env python3
"""Development extension: rho-aware causal and perfect-forecast quantile rules.

All 36 candidates are selected on train only. This extension was motivated by
the weak fixed-target 160s planner; it is not another blind validation test.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import physical_headroom as driver
from libra_rl.model import CostModel
from libra_rl.physical import ArrivalStream,evaluate_plan
from libra_rl.physical_planning import clairvoyant_search


def rule_plan(x,start,length,service,action,bases,*,perfect):
    if action==35:
        return np.zeros(len(bases),dtype=int),np.ones(len(bases))
    delta,rho = action//5-3,(0.6,0.7,0.8,0.9,1.0)[action%5]
    model = CostModel(version="v2",service_seconds=service)
    # The marginal usable capacity is rho*throughput, not full throughput.
    tau = float(np.clip(1-model.vm_second_price/(rho*model.throughput*model.faas_request_price),0,1))
    result = []
    for i,base in enumerate(bases):
        t = start+60*i
        a,b = ((t+100,min(t+400,start+length)) if perfect else (max(0,t-300),t))
        q = float(np.quantile(x[a:b],tau)) if a<b else 0.0
        target = 0 if q<4 else int(np.ceil(q/(rho*model.throughput)))
        result.append(int(np.clip(target+delta,max(0,int(base)-3),min(512,int(base)+3))))
    return np.array(result),np.full(len(bases),rho)


def main():
    output = driver.ROOT/"outputs/physical_headroom"
    protocol,manifest,data = driver.prepare(output)
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    grid_path = output/"forecast_rule_grid.json"
    if grid_path.exists():
        grid = json.loads(grid_path.read_text())
        if grid["source_sha256"]!=fingerprint:
            raise ValueError("forecast control source changed")
    else:
        train = data["train"]
        length = protocol["episode_seconds"]
        starts = list(range(protocol["history_seconds"],len(train)-length+1,length))
        totals = {(s,variant,a):[] for s in protocol["services_seconds"]
                  for variant in ("causal_economic","perfect_quantile") for a in range(36)}
        began = time.perf_counter()
        for j,start in enumerate(starts):
            stream = ArrivalStream.generate(train[start:start+length],seed=protocol["training_arrival_seed"]+j)
            for service in protocol["services_seconds"]:
                bases = driver.bases_for(train,start,length,service)
                cached = {}
                for variant in ("causal_economic","perfect_quantile"):
                    for action in range(36):
                        targets,rhos = rule_plan(train,start,length,service,action,bases,perfect=variant=="perfect_quantile")
                        key = targets.tobytes(),np.where(targets>0,rhos,1.0).tobytes()
                        if key not in cached:
                            cached[key] = evaluate_plan(stream,targets,rhos,service_seconds=service)
                        totals[(service,variant,action)].append(cached[key])
            if j%40==0 or j+1==len(starts):
                print(f"Forecast control grid {j+1}/{len(starts)} {time.perf_counter()-began:.1f}s",flush=True)
        rows,selected = [],{}
        for service in protocol["services_seconds"]:
            for variant in ("causal_economic","perfect_quantile"):
                candidates = [dict(service_seconds=service,variant=variant,action=a,split="train",
                                   summary=driver.aggregate(totals[(service,variant,a)])) for a in range(36)]
                rows += candidates
                feasible = [r for r in candidates if r["summary"]["violation_rate"]<=protocol["epsilon"]]
                selected[f"{service}_{variant}"] = min(feasible,key=lambda r:(r["summary"]["total_cost_usd"],r["action"]))
        grid = dict(source_sha256=fingerprint,physical_fingerprint=manifest["fingerprint"],rows=rows,selected=selected,
            selection_split="train",development_extension=True,training_seconds=time.perf_counter()-began,
            frozen_utc=driver.datetime.now(driver.timezone.utc).isoformat(),
            definition="rho-aware critical quantile; future [t+100,t+400) versus causal past 300s; targets clipped to shared past-base residual domain")
        driver.write(grid_path,grid)
    rows = []
    for old_path in sorted((output/"runs").glob("*.json")):
        old = json.loads(old_path.read_text())
        start,service,seed,window = old["start"],old["service_seconds"],old["arrival_seed"],old["window"]
        length = protocol["episode_seconds"]
        validation = data["validation"]
        stream = ArrivalStream.generate(validation[start:start+length],seed=old["stream_seed"])
        bases = np.array(old["bases"])
        candidates = [old["results"]["clairvoyant_dp"]]
        for variant in ("causal_economic","perfect_quantile"):
            action = grid["selected"][f"{service}_{variant}"]["action"]
            targets,rhos = rule_plan(validation,start,length,service,action,bases,perfect=variant=="perfect_quantile")
            summary = evaluate_plan(stream,targets,rhos,service_seconds=service)
            rows.append(dict(service_seconds=service,arrival_seed=seed,window=window,policy=variant,
                summary=summary,action=action,targets=targets.tolist(),rhos=rhos.tolist(),selection_split="train"))
            if summary["violation_rate"]<=protocol["epsilon"]:
                candidates.append(dict(summary=summary,targets=targets,rhos=rhos))
        incumbent = min(candidates,key=lambda r:r["summary"]["total_cost_usd"])
        refined = clairvoyant_search(stream,bases,service_seconds=service,epsilon=protocol["epsilon"],
            incumbent_targets=incumbent["targets"],incumbent_rhos=incumbent["rhos"],beam_width=protocol["beam_width"])
        rows.append(dict(service_seconds=service,arrival_seed=seed,window=window,policy="refined_clairvoyant",
            summary=refined["summary"],targets=refined["targets"],rhos=refined["rhos"],
            exact=refined["exact"],cost_lower_bound_usd=refined["cost_lower_bound_usd"],pruned_states=refined["pruned_states"]))
        print(f"Extended controls s={service} seed={seed} window={window}",flush=True)
    driver.write(output/"forecast_rule_evaluation.json",dict(source_sha256=fingerprint,physical_fingerprint=manifest["fingerprint"],
        training_selection_sha256=driver.sha(grid_path),rows=rows,test_evaluated=False,ppo_trained=False))


if __name__=="__main__":
    main()
