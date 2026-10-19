#!/usr/bin/env python3
"""Corrected queue/load model and a gated headroom study; no PPO training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from libra_rl.data import load_azure_partition
from libra_rl.model import CostModel
from libra_rl.physical import ArrivalStream,QueueFleet,evaluate_plan,staggered_copies,ACCELERATOR
from libra_rl.physical_planning import fixed_plan,perfect_window,clairvoyant_search,cost_relaxation
from libra_rl.queue_validation import simulate_queue


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bases_for(x,start,length,service,k=60):
    model = CostModel(version="v2",service_seconds=service)
    quantiles = np.array([np.quantile(x[max(0,t-300):t],model.tau) for t in range(start,start+length,k)])
    return np.where(quantiles<4,0,np.ceil(quantiles/model.throughput)).astype(int)


def prepare(output):
    protocol = json.loads((ROOT/"configs/physical_headroom.json").read_text())
    if (protocol["decision_seconds"],protocol["cold_start_seconds"],protocol["sla_seconds"])!=(60,100,1.0):
        raise ValueError("this frozen study implements K=60, D=100, SLA=1")
    files = [Path(__file__),ROOT/"configs/physical_headroom.json",
             *[ROOT/"libra_rl"/name for name in ("physical.py","physical_lookup.py","physical_planning.py","model.py","data.py","queue_validation.py")],
             *sorted((ROOT/"libra_sim").glob("*.py"))]
    sources = {str(p.relative_to(ROOT)):sha(p) for p in files}
    fingerprint = hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()
    data_hash = sha(ROOT/"data/azure2021/aggregate_counts.npz")
    path = output/"manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"]!=fingerprint or manifest["azure_aggregate_sha256"]!=data_hash:
            raise ValueError("physical study source/data changed; use another output directory")
    else:
        manifest = dict(fingerprint=fingerprint,source_sha256=sources,protocol=protocol,
            created_utc=datetime.now(timezone.utc).isoformat(),azure_aggregate_sha256=data_hash,
            platform=platform.platform(),python=platform.python_version(),numpy=np.__version__,
            accelerator=ACCELERATOR,test_evaluated=False,ppo_trained=False)
        write(path,manifest)
    data,statistics = {},[]
    for split in ("train","validation"):
        raw = load_azure_partition(split)
        combined = staggered_copies(raw,protocol["offsets_seconds"])
        data[split] = combined
        aligned = raw[max(protocol["offsets_seconds"]):]*16
        statistics.append(dict(split=split,seconds=len(combined),staggered_requests=int(combined.sum()),
            synchronous_requests=int(aligned.sum()),staggered_mean=float(combined.mean()),
            synchronous_mean=float(aligned.mean()),staggered_cv=float(combined.std()/combined.mean()),
            synchronous_cv=float(aligned.std()/aligned.mean()),staggered_peak=int(combined.max()),
            synchronous_peak=int(aligned.max()),
            meaning="time-shifted replicas are correlated copies, not sixteen statistically independent users"))
    write(output/"load_statistics.json",dict(fingerprint=fingerprint,rows=statistics))
    return protocol,manifest,data


def aggregate(rows):
    keys = ("total_cost_usd","vm_cost_usd","faas_cost_usd","gateway_cost_usd","requests","vm_requests",
            "faas_requests","violations","vm_seconds","drain_vm_seconds","terminal_drain_vm_seconds",
            "starting_vm_seconds","waited_requests","response_seconds")
    result = {key:sum(r[key] for r in rows) for key in keys}
    result["violation_rate"] = result["violations"]/max(1,result["requests"])
    result["max_response_seconds"] = max(r["max_response_seconds"] for r in rows)
    return result


def lookup_task(job):
    return simulate_queue(**job)


def lookup_stage(output,protocol,manifest,workers):
    path = output/"md16_lookup.json"
    if path.exists():
        return
    source = ROOT/"outputs/rl_diagnostics/queue_calibration.json"
    rows = [{k:v for k,v in r.items() if k not in ("analytic_violation_rate","erlang_c_wait_probability")}
            for r in json.loads(source.read_text())["rows"] if r["distribution"]=="deterministic"]
    jobs = [dict(utilization=u,service_seconds=s,distribution="deterministic",servers=16,
                 warmup=protocol["lookup_extra_warmup_requests"],measured=protocol["lookup_extra_measured_requests"],seed=seed)
            for s in protocol["services_seconds"] for u in protocol["lookup_extra_utilizations"]
            for seed in protocol["validation_arrival_seeds"]]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool:
        rows += list(pool.map(lookup_task,jobs))
    tables = {}
    for service in protocol["services_seconds"]:
        points = [dict(utilization=0.0,raw_mean=0.0,monotone_estimate=0.0,ci95=[0.0,0.0])]
        previous = 0.0
        for u in sorted({r["utilization"] for r in rows if r["service_seconds"]==service}):
            rates = [r["violation_rate"] for r in rows if r["service_seconds"]==service and r["utilization"]==u]
            mean = float(np.mean(rates)); half = 2.7764451051977987*float(np.std(rates,ddof=1))/np.sqrt(5)
            previous = max(previous,mean)
            points.append(dict(utilization=u,raw_mean=mean,monotone_estimate=previous,ci95=[mean-half,mean+half]))
        tables[str(service)] = points
    write(path,dict(fingerprint=manifest["fingerprint"],source_queue_calibration_sha256=sha(source),
        tables=tables,rows=rows,scope="steady M/D/16 reference only; u>=1 has no stationary table, use transient queues"))
    print("M/D/16 lookup built, including near-saturation calibration",flush=True)


def training_stage(output,protocol,manifest,data):
    path = output/"fixed_training_grid.json"
    if path.exists():
        return json.loads(path.read_text())
    x = data["train"]
    length = protocol["episode_seconds"]
    starts = list(range(protocol["history_seconds"],len(x)-length+1,length))
    totals = {(s,a):[] for s in protocol["services_seconds"] for a in range(36)}
    started = time.perf_counter()
    for j,start in enumerate(starts):
        stream = ArrivalStream.generate(x[start:start+length],seed=protocol["training_arrival_seed"]+j)
        for service in protocol["services_seconds"]:
            bases = bases_for(x,start,length,service,protocol["decision_seconds"])
            cached = {}
            for action in range(36):
                targets,rhos = fixed_plan(bases,action)
                key = targets.tobytes(),np.where(targets>0,rhos,1.0).tobytes()
                if key not in cached:
                    cached[key] = evaluate_plan(stream,targets,rhos,service_seconds=service)
                result = cached[key]
                totals[(service,action)].append(result)
        if j%20==0 or j+1==len(starts):
            print(f"Fixed grid train windows {j+1}/{len(starts)} elapsed={time.perf_counter()-started:.1f}s",flush=True)
            write(output/"training_progress.json",dict(completed=j+1,total=len(starts),elapsed_seconds=time.perf_counter()-started))
    rows,selected = [],{}
    for service in protocol["services_seconds"]:
        grid = [dict(service_seconds=service,action=action,split="train",summary=aggregate(totals[(service,action)]))
                for action in range(36)]
        rows += grid
        feasible = [r for r in grid if r["summary"]["violation_rate"]<=protocol["epsilon"]]
        selected[str(service)] = min(feasible,key=lambda r:(r["summary"]["total_cost_usd"],r["action"]))
    result = dict(fingerprint=manifest["fingerprint"],rows=rows,selected=selected,selection_split="train",
                  windows=starts,episode_seconds=length,fit_seconds=time.perf_counter()-started)
    write(path,result)
    # This file is persisted before validation arrival streams are generated.
    write(output/"fixed_selection.json",dict(fingerprint=manifest["fingerprint"],selected=selected,
        frozen_utc=datetime.now(timezone.utc).isoformat(),selection_split="train"))
    return result


def evaluation_stage(output,protocol,manifest,data,training):
    x = data["validation"]
    rows,plans,sensitivity = [],[],[]
    for seed in protocol["validation_arrival_seeds"]:
        for window,start in enumerate(protocol["validation_starts_after_staggering_seconds"]):
            length = protocol["episode_seconds"]
            if start<protocol["history_seconds"] or start+length>len(x):
                raise ValueError("validation window does not fit its split")
            stream = ArrivalStream.generate(x[start:start+length],seed=seed+100*window)
            for service in protocol["services_seconds"]:
                name = f"s{int(service*10):02d}_seed{seed}_window{window}"
                path = output/"runs"/(name+".json")
                if path.exists():
                    result = json.loads(path.read_text())
                    if result["fingerprint"]!=manifest["fingerprint"]:
                        raise ValueError("stale physical run")
                else:
                    began = time.perf_counter()
                    bases = bases_for(x,start,length,service,protocol["decision_seconds"])
                    action = training["selected"][str(service)]["action"]
                    targets,rhos = fixed_plan(bases,action)
                    fixed = dict(summary=evaluate_plan(stream,targets,rhos,service_seconds=service),
                                 targets=targets.tolist(),rhos=rhos.tolist(),action=action)
                    rolling = perfect_window(stream,bases,service_seconds=service,epsilon=protocol["epsilon"])
                    incumbent = rolling if rolling["summary"]["total_cost_usd"]<fixed["summary"]["total_cost_usd"] or fixed["summary"]["violation_rate"]>protocol["epsilon"] else fixed
                    dp = clairvoyant_search(stream,bases,service_seconds=service,epsilon=protocol["epsilon"],
                        incumbent_targets=incumbent["targets"],incumbent_rhos=incumbent["rhos"],beam_width=protocol["beam_width"])
                    results = dict(fixed=fixed,perfect_window=rolling,clairvoyant_dp=dp)
                    if seed==0 and window==0:
                        wide = clairvoyant_search(stream,bases,service_seconds=service,epsilon=protocol["epsilon"],
                            incumbent_targets=dp["targets"],incumbent_rhos=dp["rhos"],beam_width=protocol["beam_sensitivity_width"])
                        results["wider_dp"] = wide
                    result = dict(fingerprint=manifest["fingerprint"],service_seconds=service,arrival_seed=seed,
                        window=window,start=start,split="validation",history_seconds=protocol["history_seconds"],
                        results=results,bases=bases.tolist(),seconds=time.perf_counter()-began,
                        stream_seed=seed+100*window,test_evaluated=False)
                    write(path,result)
                    print(f"{name}: fixed=${fixed['summary']['total_cost_usd']:.4f}, "
                          f"perfect=${rolling['summary']['total_cost_usd']:.4f}, DP=${dp['summary']['total_cost_usd']:.4f}, "
                          f"lower=${dp['cost_lower_bound_usd']:.4f}; exact={dp['exact']} elapsed={result['seconds']:.1f}s",flush=True)
                plans.append(name)
                for policy,p in result["results"].items():
                    rows.append(dict(service_seconds=service,arrival_seed=seed,window=window,policy=policy,
                                     summary=p["summary"],exact=p.get("exact"),cost_lower_bound_usd=p.get("cost_lower_bound_usd")))
    write(output/"evaluation.json",dict(fingerprint=manifest["fingerprint"],rows=rows,runs=plans,
        completed_utc=datetime.now(timezone.utc).isoformat(),test_evaluated=False,ppo_trained=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/physical_headroom")
    parser.add_argument("--stage",choices=("prepare","lookup","train","evaluate","all"),default="all")
    parser.add_argument("--workers",type=int,default=4)
    args = parser.parse_args()
    protocol,manifest,data = prepare(args.output)
    if args.stage in ("lookup","all"):
        lookup_stage(args.output,protocol,manifest,args.workers)
    if args.stage in ("train","all"):
        training = training_stage(args.output,protocol,manifest,data)
    if args.stage in ("evaluate","all"):
        training = json.loads((args.output/"fixed_training_grid.json").read_text())
        evaluation_stage(args.output,protocol,manifest,data,training)


if __name__=="__main__":
    main()
