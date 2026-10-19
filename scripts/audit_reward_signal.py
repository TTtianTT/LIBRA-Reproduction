#!/usr/bin/env python3
"""One-window counterfactual costs along the *historical* teacher trajectory."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from libra_rl.data import load_azure_partition
from libra_rl.diagnostic_env import base_targets
from libra_rl.diagnostics import write_json
from libra_rl.env import RHO_GRID
from libra_rl.model import CostModel
from libra_rl.study import chunks


def main():
    source = ROOT/"outputs/rl_diagnostics"
    output = ROOT/"outputs/constant_action_audit"
    teachers = json.loads((source/"teachers.json").read_text())["teachers"]
    rows = []
    for split,case,service in ((split,case,service) for split in ("train","validation")
                             for case,service in (("v1",1.0),("v2_s05",0.5),("v2_s07",0.7))):
        counts = load_azure_partition(split,multiplier=16)
        model = CostModel(version="v1" if service==1 else "v2",service_seconds=service)
        teacher = teachers[case]
        teacher_costs,teacher_rewards,gains,cost_gains = [],[],[],[]
        penalty = 0.0 if service==1 else 0.0001
        for block in chunks(counts):
            bases = base_targets(block,model,rho=teacher["rho"],window=teacher["window"])
            ready_at = np.empty(0,dtype=int)
            for i,base in enumerate(bases):
                t,end = i*60,min((i+1)*60,len(block))
                times = np.arange(t,end)
                def counterfactual(target,rho):
                    fleet = (ready_at[:target] if target<len(ready_at) else
                             np.r_[ready_at,np.full(target-len(ready_at),t+100)])
                    ready = (fleet[:,None]<=times[None,:]).sum(axis=0)
                    traffic = model.traffic(block[t:end],ready,rho)
                    return float(target*(end-t)*model.vm_second_price+
                                 traffic["faas_requests"].sum()*model.faas_request_price),float(traffic["expected_violations"].sum()),fleet
                cost,violations,following = counterfactual(int(base),teacher["rho"])
                choices = [(0,1.0)]+[(int(max(0,base+d)),rho) for d in range(-3,4)
                                     for rho in ((1.0,) if service==1 else RHO_GRID)]
                candidates = [counterfactual(target,rho) for target,rho in choices]
                best = min(c+penalty*v for c,v,_ in candidates)
                teacher_costs.append(cost/0.01)
                teacher_rewards.append(-(cost+penalty*(violations-0.001*block[t:end].sum()))/0.01)
                gains.append((cost+penalty*violations-best)/0.01)
                cost_gains.append((cost-min(c for c,_,_ in candidates))/0.01)
                ready_at = following
        runs = [json.loads(p.read_text()) for p in (source/"runs").glob(case+"_random_seed*.json")]
        diagnostics = [p["optimizer_diagnostics"] for r in runs for p in r["curve"]]
        losses = [p["train/value_loss"] for p in diagnostics if "train/value_loss" in p]
        ev = [p["train/explained_variance"] for p in diagnostics if "train/explained_variance" in p]
        row = dict(case=case,split=split,penalty=penalty,steps=len(gains),cost_normalizer_usd=0.01,
            teacher_cost_std=float(np.std(teacher_costs)),perfect_immediate_gain_mean=float(np.mean(gains)),
            perfect_immediate_gain_median=float(np.median(gains)),
            teacher_actual_reward_std=float(np.std(teacher_rewards)),
            cost_only_gain_mean=float(np.mean(cost_gains)),cost_only_gain_median=float(np.median(cost_gains)),
            gain_to_cost_std_ratio=float(np.mean(gains)/np.std(teacher_costs)),
            critic_value_loss_range=[min(losses),max(losses)],critic_explained_variance_range=[min(ev),max(ev)])
        rows.append(row)
        print(row,flush=True)
    write_json(output/"reward_signal.json",dict(rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        meaning="immediate counterfactual on teacher trajectory, cost-only and actual reward both reported; no continuation value, not a policy upper bound or an advantage-variance proof"))


if __name__=="__main__":
    main()
