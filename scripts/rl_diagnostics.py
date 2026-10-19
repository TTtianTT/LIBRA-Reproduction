#!/usr/bin/env python3
"""Train/validation-only action, latency, and nested-budget learning diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor  # load OpenMP before limiting pools
from stable_baselines3 import PPO
from threadpoolctl import threadpool_limits, threadpool_info
from libra_ml.features import training_set
from libra_ml.models import fit_gbdt
from libra_rl.data import load_azure_partition
from libra_rl.diagnostic_env import base_targets
from libra_rl.diagnostics import train_nested, write_json
from libra_rl.env import RHO_GRID
from libra_rl.evaluation import evaluate_targets
from libra_rl.model import CostModel
from libra_rl.queue_validation import simulate_queue
from libra_rl.queueing import erlang_c, violation_probability
from libra_rl.restricted_dp import residual_bound
from libra_rl.study import aggregate, chunks
from libra_rl.training import StudyEnv, ObservationTransform

STATE = None
LIMITER = None
CASES = (("v1",1.0),("v2_s05",0.5),("v2_s07",0.7))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(output):
    protocol = json.loads((ROOT/"configs/rl_diagnostics.json").read_text())
    files = [Path(__file__),ROOT/"configs/rl_diagnostics.json",ROOT/"requirements-rl.txt",
             ROOT/"outputs/rl_study/baseline_training_grid.json"]
    for package in ("libra_rl","libra_ml","libra_sim"):
        files += sorted((ROOT/package).glob("*.py"))
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in files}
    fingerprint = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    archive_hash = sha(ROOT/"data/azure2021/aggregate_counts.npz")
    path = output/"manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["fingerprint"]!=fingerprint or manifest["azure_aggregate_sha256"]!=archive_hash:
            raise ValueError("diagnostic source/data changed; use a new output directory")
    else:
        import stable_baselines3, gymnasium, scipy, sklearn
        manifest = dict(fingerprint=fingerprint,source_sha256=hashes,protocol=protocol,
            azure_aggregate_sha256=archive_hash,created_utc=datetime.now(timezone.utc).isoformat(),
            platform=platform.platform(),python=platform.python_version(),
            versions={"numpy":np.__version__,"torch":torch.__version__,"scipy":scipy.__version__,
                      "sklearn":sklearn.__version__,"gymnasium":gymnasium.__version__,
                      "stable_baselines3":stable_baselines3.__version__},test_evaluated=False)
        write_json(path,manifest)
    train = load_azure_partition("train",multiplier=protocol["azure_replication"])
    validation = load_azure_partition("validation",multiplier=protocol["azure_replication"])
    cap = 2**math.ceil(math.log2(max(8,math.ceil(float(train.max())*2/16)+8)))
    grid = json.loads((ROOT/"outputs/rl_study/baseline_training_grid.json").read_text())["rows"]
    teachers, forecasts, features = {}, {}, None
    for case,service in CASES:
        feasible = [r for r in grid if r["split"]=="train" and r["family"]=="Quantile"
                    and r["service_seconds"]==service and r["summary"]["violation_rate"]<=protocol["epsilon"]]
        best = min(feasible,key=lambda r:r["summary"]["total_cost_usd"])
        teachers[case] = {**best["parameters"],"selection_split":"train","training_summary":best["summary"]}
        p = output/"models"/f"gbdt_{case}.joblib"
        meta = output/f"forecast_{case}.json"
        if p.exists():
            if sha(p)!=json.loads(meta.read_text())["sha256"]:
                raise ValueError("modified GBDT")
            forecasts[service] = joblib.load(p)
        else:
            print(f"Fit train-only GBDT {case}",flush=True)
            if features is None:
                features = training_set(train,stride=300,target_step=10)
            model = CostModel(version="v1" if service==1 else "v2",service_seconds=service)
            forecasts[service] = fit_gbdt(*features,model.tau,seed=0)
            p.parent.mkdir(parents=True,exist_ok=True)
            joblib.dump(forecasts[service],p)
            write_json(meta,dict(sha256=sha(p),training_split="train",rows=len(features[1]),tau=model.tau))
    teacher_record = dict(teachers=teachers,max_vms=cap,selection_split="train")
    teacher_path = output/"teachers.json"
    if teacher_path.exists():
        if json.loads(teacher_path.read_text()) != teacher_record:
            raise ValueError("modified teacher selection")
    else:
        write_json(teacher_path,teacher_record)
    return protocol,manifest,train,validation,cap,forecasts,teachers


def audit_stage(output, state):
    protocol,manifest,train,validation,cap,forecasts,teachers = state
    rows = []
    for case,service in CASES:
        model = CostModel(version="v1" if service==1 else "v2",service_seconds=service)
        teacher = teachers[case]
        for split,x in (("train",train),("validation",validation)):
            original, desired, summaries = [], [], []
            for block in chunks(x):
                original.extend(base_targets(block,model,rho=1.0 if service==1 else 0.8,max_vms=cap))
                target = base_targets(block,model,rho=teacher["rho"],window=teacher["window"],max_vms=cap)
                desired.extend(target)
                summaries.append(evaluate_targets(block,target,model=model,rhos=teacher["rho"]))
            a,b = np.asarray(original),np.asarray(desired)
            summary = aggregate(summaries)
            if split=="train" and abs(summary["total_cost_usd"]-teacher["training_summary"]["total_cost_usd"])>1e-9:
                raise AssertionError("teacher billing differs from frozen training grid")
            rows.append(dict(case=case,split=split,decisions=len(a),
                teacher_unreachable_fraction=float(np.mean(abs(a-b)>3)),
                full_faas_unreachable_fraction=float(np.mean(a>3)),
                teacher_delta_min=int((b-a).min()),teacher_delta_max=int((b-a).max()),
                teacher_summary=summary))
    write_json(output/"action_audit.json",dict(fingerprint=manifest["fingerprint"],rows=rows))


def bounds_stage(output,state):
    protocol,manifest,train,validation,cap,forecasts,teachers = state
    path = output/"domain_bounds.json"
    rows = json.loads(path.read_text())["rows"] if path.exists() else []
    complete = {(r["case"],r["domain"],r["penalty"],r["cost_weight"]) for r in rows}
    for case,service in CASES:
        model = CostModel(version="v1" if service==1 else "v2",service_seconds=service)
        teacher = teachers[case]
        blocks = chunks(validation)
        for domain in ("original","original_plus_faas","teacher_plus_faas"):
            rho = teacher["rho"] if domain=="teacher_plus_faas" else (1.0 if service==1 else 0.8)
            window = teacher["window"] if domain=="teacher_plus_faas" else 300
            bases = [base_targets(b,model,rho=rho,window=window,max_vms=cap) for b in blocks]
            objectives = [(p,1.0) for p in ([0.0] if service==1 else protocol["lambda_grid_for_dp"])]
            objectives.append((1.0,0.0))
            for penalty,weight in objectives:
                if (case,domain,penalty,weight) in complete:
                    continue
                started = time.perf_counter()
                parts = [residual_bound(b,base,model=model,max_vms=cap,
                    rho_grid=(1.0,) if service==1 else RHO_GRID,allow_faas=domain!="original",
                    penalty=penalty,cost_weight=weight) for b,base in zip(blocks,bases)]
                row = dict(case=case,domain=domain,penalty=penalty,cost_weight=weight,split="validation",
                    total_cost_usd=sum(p["total_cost_usd"] for p in parts),
                    expected_violations=sum(p["expected_violations"] for p in parts),
                    objective=sum(p["objective_usd"] for p in parts),requests=int(validation.sum()),
                    seconds=time.perf_counter()-started,episodes=len(blocks))
                row["violation_rate"] = row["expected_violations"]/max(1,row["requests"])
                row["dual_lower_bound_usd"] = (row["objective"]-penalty*protocol["epsilon"]*row["requests"]
                                                if weight==1 else None)
                rows.append(row)
                write_json(path,dict(fingerprint=manifest["fingerprint"],rows=rows))
                print(f"DP {case} {domain} lambda={penalty:g} weight={weight:g} cost={row['total_cost_usd']:.4f} "
                      f"violation={row['violation_rate']:.6g}",flush=True)
    chosen = {}
    for case,service in CASES:
        points = [r for r in rows if r["case"]==case and r["domain"]=="teacher_plus_faas" and r["cost_weight"]==1]
        feasible = [r for r in points if r["violation_rate"]<=protocol["epsilon"]]
        best = min(feasible,key=lambda r:r["total_cost_usd"]) if feasible else min(points,key=lambda r:r["violation_rate"])
        chosen[case] = best["penalty"]
    write_json(output/"penalties.json",dict(fingerprint=manifest["fingerprint"],penalties=chosen,
        split="validation",criterion=protocol["lambda_selection"],clairvoyant_selection=True))


def queue_job(j):
    return simulate_queue(**j)


def queue_stage(output,state,workers):
    protocol,manifest,*_ = state
    path = output/"queue_calibration.json"
    if path.exists():
        return
    q = protocol["queue"]
    jobs = [dict(utilization=u,service_seconds=s,distribution=d,servers=q["servers"],
                 warmup=q["warmup_requests"],measured=q["measured_requests"],seed=seed)
            for s in q["services"] for u in q["utilizations"] for d in q["service_distributions"] for seed in q["seeds"]]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn")) as pool:
        rows = list(pool.map(queue_job,jobs))
    for r in rows:
        r["analytic_violation_rate"] = float(violation_probability(r["utilization"],r["service_seconds"],
            servers=r["servers"],mode="mmc_response" if r["distribution"]=="exponential" else "wait_budget"))
        r["erlang_c_wait_probability"] = float(erlang_c(r["utilization"],r["servers"]))
    write_json(path,dict(fingerprint=manifest["fingerprint"],rows=rows,
        scope="stationary FCFS calibration, not transient/AWS validation",test_evaluated=False))
    print(f"Queue calibration: {len(rows)} independent runs",flush=True)


def trace_stage(output,state):
    from libra_rl.diagnostics import write_json
    import csv
    protocol,manifest,_,validation,cap,_,_ = state
    if (output/"raw_failure_summary.json").exists():
        return
    source = ROOT/"outputs/rl_study"
    forecast = joblib.load(source/"models/gbdt_s07.joblib")
    if sha(source/"models/gbdt_s07.joblib")!=json.loads((source/"forecast_s07.json").read_text())["sha256"]:
        raise ValueError("historical forecast changed")
    results = []
    for seed in (0,1):
        record_path = source/"runs"/f"v2_s07_raw_actions_l0.01_seed{seed}.json"
        record = json.loads(record_path.read_text())
        path = source/record["checkpoint"]
        if sha(path)!=record["checkpoint_sha256"]:
            raise ValueError("historical checkpoint changed")
        agent = PPO.load(path,device="cpu")
        rows, summaries = [], []
        for episode,block in enumerate(chunks(validation)):
            env = ObservationTransform(StudyEnv(block,model=CostModel(version="v2",service_seconds=0.7),
                action_mode="raw",max_vms=cap,forecast_model=forecast,penalty=0.01,
                episode_seconds=None,cost_normalizer=0.01))
            obs,_ = env.reset(seed=0,options={"start":0})
            start_index = len(rows)
            while True:
                tensor,_ = agent.policy.obs_to_tensor(obs)
                with torch.no_grad():
                    entropy = float(agent.policy.get_distribution(tensor).entropy().item())
                    value = float(agent.policy.predict_values(tensor).item())
                action,_ = agent.predict(obs,deterministic=True)
                obs,reward,done,truncated,info = env.step(action)
                row = {k:info[k] for k in ("time_s","target_vms","rho","ready_at_end","starting_at_end",
                    "total_cost_usd","vm_cost_usd","faas_cost_usd","expected_violations","starting_vm_seconds")}
                row.update(episode=episode,seed=seed,action_delta=int((-8,-4,-2,-1,0,1,2,4,8)[int(action[0])]),
                    entropy=entropy,value=value,reward=reward,
                    requests=float(block[info["time_s"]:info["end_s"]].sum()))
                rows.append(row)
                if done or truncated:
                    break
            ret = 0.0
            for row in reversed(rows[start_index:]):
                ret = row["reward"]+protocol["ppo"]["gamma"]*ret
                row["discounted_return"] = ret
            summaries.append({**env.unwrapped.summary(),"max_target_vms":max(r["target_vms"] for r in rows[start_index:])})
        summary = aggregate(summaries)
        if abs(summary["total_cost_usd"]-record["validation"]["total_cost_usd"])>1e-9:
            raise AssertionError("historical raw-policy validation replay changed")
        with (output/f"raw_trace_seed{seed}.csv").open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        truth,pred = np.array([r["discounted_return"] for r in rows]),np.array([r["value"] for r in rows])
        results.append(dict(seed=seed,summary=summary,checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),
            positive_delta_fraction=float(np.mean([r["action_delta"]>0 for r in rows])),
            at_vm_cap_fraction=float(np.mean([r["target_vms"]==cap for r in rows])),
            mean_entropy=float(np.mean([r["entropy"] for r in rows])),
            value_rmse=float(np.sqrt(np.mean((truth-pred)**2))),
            value_explained_variance=float(1-np.var(truth-pred)/np.var(truth)) if np.var(truth)>0 else None,
            source_record_sha256=sha(record_path),split="validation",steps=len(rows)))
    write_json(output/"raw_failure_summary.json",dict(fingerprint=manifest["fingerprint"],rows=results,test_evaluated=False))
    print("Historical raw policies replay exactly; failure traces saved",flush=True)


def initialize(output):
    global STATE,LIMITER
    torch.set_num_threads(1)
    LIMITER = threadpool_limits(limits=1)
    STATE = (Path(output),prepare(Path(output)))
    LIMITER = threadpool_limits(limits=1)
    if any(pool["num_threads"]!=1 for pool in threadpool_info()):
        raise RuntimeError("a numerical runtime escaped the single-thread worker limit")


def fit(j):
    output,state = STATE
    protocol,manifest,train,validation,cap,forecasts,teachers = state
    if j["case"]=="toy":
        train = np.full(36000,protocol["toy"]["demand_rps"],dtype=np.int64)
        validation = np.full(3600,protocol["toy"]["demand_rps"],dtype=np.int64)
        teacher,forecast,cap,penalty = dict(rho=1.0,window=300),None,8,0.0
    else:
        teacher,forecast = teachers[j["case"]],forecasts[j["service_seconds"]]
        penalty = json.loads((output/"penalties.json").read_text())["penalties"][j["case"]]
    return train_nested(train,validation,model=CostModel(version=j["version"],service_seconds=j["service_seconds"]),
        teacher=teacher,forecast=forecast,max_vms=cap,penalty=penalty,job=j,protocol=protocol,
        output=output,fingerprint=manifest["fingerprint"])


def train_stage(output,state,workers,toy=False):
    protocol,manifest,*_ = state
    if toy:
        jobs = [dict(id=f"toy_random_seed{seed}",case="toy",version="v1",service_seconds=1.0,
            initialization="random",seed=seed,augment=False,episode_seconds=protocol["toy"]["episode_seconds"],
            samples=protocol["toy"]["samples"]) for seed in protocol["toy"]["seeds"]]
    else:
        gate = json.loads((output/"toy_gate.json").read_text())
        if not gate["passed"]:
            raise ValueError("toy learning gate failed; diagnose before main training")
        jobs = [dict(id=f"{case}_{init}_seed{seed}",case=case,version="v1" if s==1 else "v2",service_seconds=s,
            initialization=init,seed=seed,augment=True,episode_seconds=protocol["episode_seconds"],
            samples=max(protocol["budgets"])) for case,s in CASES for init in protocol["initializations"] for seed in protocol["seeds"]]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"),initializer=initialize,
                             initargs=(str(output),)) as pool:
        futures = [pool.submit(fit,j) for j in jobs]
        try:
            for future in as_completed(futures):
                run = future.result()
                results.append(run)
                s = run["selections"][-1]["summary"]
                print(f"{'Toy' if toy else 'Main'} {len(results)}/{len(jobs)} {run['job']['id']} "
                      f"fit={run['fit_seconds']:.1f}s cost={s['total_cost_usd']:.6f} violation={s['violation_rate']:.6g}",flush=True)
                write_json(output/("toy_progress.json" if toy else "progress.json"),dict(completed=len(results),planned=len(jobs),
                    last_job=run["job"]["id"],wall_seconds=time.perf_counter()-started))
        except BaseException:
            # Python 3.12 has no public terminate_workers(); stop only our children.
            for process in pool._processes.values():
                process.terminate()
            raise
    if toy:
        regrets = [r["selections"][0]["summary"]["total_cost_usd"]/r["teacher_validation"]["total_cost_usd"]-1 for r in results]
        mean = float(np.mean(regrets))
        write_json(output/"toy_gate.json",dict(fingerprint=manifest["fingerprint"],regrets=regrets,
            mean_regret_fraction=mean,passed=mean<=protocol["toy"]["gate_max_mean_regret_fraction"],
            threshold=protocol["toy"]["gate_max_mean_regret_fraction"],split="validation",samples=sum(r["samples"] for r in results)))
    else:
        write_json(output/"runs_index.json",dict(fingerprint=manifest["fingerprint"],runs=[j["id"] for j in jobs],
            count=len(jobs),samples=sum(r["samples"] for r in results),workers=workers,
            wall_seconds=time.perf_counter()-started,test_evaluated=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/rl_diagnostics")
    parser.add_argument("--stage",choices=("prepare","audit","bounds","queue","trace","toy","train","all"),default="all")
    parser.add_argument("--workers",type=int,default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        state = prepare(args.output)
        for stage,call in (("audit",lambda:audit_stage(args.output,state)),
                           ("bounds",lambda:bounds_stage(args.output,state)),
                           ("queue",lambda:queue_stage(args.output,state,args.workers)),
                           ("trace",lambda:trace_stage(args.output,state)),
                           ("toy",lambda:train_stage(args.output,state,args.workers,True)),
                           ("train",lambda:train_stage(args.output,state,args.workers))):
            if args.stage in (stage,"all"):
                call()


if __name__=="__main__":
    main()
