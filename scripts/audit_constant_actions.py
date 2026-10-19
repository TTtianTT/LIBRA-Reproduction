#!/usr/bin/env python3
"""Train-only fixed-action selection and recorded PPO action-collapse audit."""
import argparse
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
from libra_rl.evaluation import evaluate_targets
from libra_rl.model import CostModel
from libra_rl.study import aggregate, chunks


def constant_grid(counts,model,teacher,max_vms):
    blocks = chunks(counts)
    bases = [base_targets(b,model,rho=teacher["rho"],window=teacher["window"],max_vms=max_vms)
             for b in blocks]
    rhos = (1.0,) if model.version=="v1" else RHO_GRID
    result = []
    for action in range(7*len(rhos)+1):
        faas = action==7*len(rhos)
        delta,rho = (None,1.0) if faas else (action//len(rhos)-3,rhos[action%len(rhos)])
        summary = aggregate([evaluate_targets(b,np.zeros_like(base) if faas else np.clip(base+delta,0,max_vms),
            model=model,rhos=rho) for b,base in zip(blocks,bases)])
        result.append(dict(action=action,delta=delta,rho=rho,all_faas=faas,summary=summary))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=ROOT/"outputs/rl_diagnostics")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/constant_action_audit")
    args = parser.parse_args()
    source,output = args.source,args.output
    manifest = json.loads((source/"manifest.json").read_text())
    for path,expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,f"source changed: {path}"
    teacher_record = json.loads((source/"teachers.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted((source/"runs").glob("v*.json"))]
    train = load_azure_partition("train",multiplier=16)
    validation = load_azure_partition("validation",multiplier=16)
    rows,comparisons,checkpoint_rows = [],[],[]
    for case,service in (("v1",1.0),("v2_s05",0.5),("v2_s07",0.7)):
        model = CostModel(version="v1" if service==1 else "v2",service_seconds=service)
        teacher = teacher_record["teachers"][case]
        grid_train = constant_grid(train,model,teacher,teacher_record["max_vms"])
        candidates = [r for r in grid_train if r["summary"]["violation_rate"]<=0.001]
        best = min(candidates,key=lambda r:(r["summary"]["total_cost_usd"],r["action"]))
        # The action is frozen before validation candidates are evaluated.
        write_json(output/f"selected_{case}.json",dict(split="train",selected=best,
                   historical_source_fingerprint=manifest["fingerprint"]))
        grid_val = constant_grid(validation,model,teacher,teacher_record["max_vms"])
        val = grid_val[best["action"]]
        best_val = min([r for r in grid_val if r["summary"]["violation_rate"]<=0.001],
                       key=lambda r:(r["summary"]["total_cost_usd"],r["action"]))
        rows += [dict(case=case,split=split,**r) for split,grid in (("train",grid_train),("validation",grid_val)) for r in grid]
        selected_runs = [r for r in runs if r["job"]["case"]==case and r["job"]["initialization"]=="random"]
        matches = []
        for run in selected_runs:
            for point in run["curve"]:
                counts = np.array(point["summary"]["action_counts"])
                checkpoint_rows.append(dict(case=case,seed=run["job"]["seed"],samples=point["samples"],
                    dominant_fraction=float(counts.max()/counts.sum()),actions_used=int(np.count_nonzero(counts))))
            selection = next(s for s in run["selections"] if s["budget"]==262144)
            same_cost = [r["action"] for r in grid_val if abs(r["summary"]["total_cost_usd"]-
                         selection["summary"]["total_cost_usd"])<1e-9]
            same_metrics = [r["action"] for r in grid_val if all(abs(r["summary"][k]-selection["summary"][k])<1e-9
                           for k in ("total_cost_usd","expected_violations","vm_seconds","faas_requests","starting_vm_seconds"))]
            matches.append(dict(seed=run["job"]["seed"],matching_cost_actions=same_cost,matching_metric_actions=same_metrics))
        selected_checkpoint_rows = [r for r in checkpoint_rows if r["case"]==case]
        comparisons.append(dict(case=case,selected_action=best["action"],delta=best["delta"],rho=best["rho"],
            selection_split="train",validation=val["summary"],validation_grid_best_action_for_audit_only=best_val["action"],
            high_budget_matches=matches,checkpoints=len(selected_checkpoint_rows),
            dominant_at_least_95_percent=sum(r["dominant_fraction"]>=0.95 for r in selected_checkpoint_rows),
            median_actions_used=float(np.median([r["actions_used"] for r in selected_checkpoint_rows])),
            teacher_validation=selected_runs[0]["teacher_validation"],
            ppo_high_budget_mean_cost=float(np.mean([next(s for s in r["selections"] if s["budget"]==262144)["summary"]["total_cost_usd"] for r in selected_runs]))))
        print(f"{case}: best train action={best['action']} delta={best['delta']} rho={best['rho']}; "
              f"validation ${val['summary']['total_cost_usd']:.6f}, violations={100*val['summary']['violation_rate']:.6f}%",flush=True)
    write_json(output/"grid.json",dict(source_fingerprint=manifest["fingerprint"],rows=rows))
    write_json(output/"summary.json",dict(source_fingerprint=manifest["fingerprint"],comparisons=comparisons,
        checkpoint_rows=checkpoint_rows,test_evaluated=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    lines = ["# 固定残差对照：PPO 没有展示状态相关的额外收益","",
        "固定动作在训练集上按 SLA≤0.1% 选择，再评估验证集。固定的是规则上的偏移和路由，VM 数仍随因果规则变化。","",
        "| 场景 | ≥95% 单一动作 checkpoint | 动作数中位数 | 最佳训练固定动作 | 验证费用 $ | 验证违约率 % |",
        "|---|---:|---:|---|---:|---:|"]
    for r in comparisons:
        lines.append(f"| {r['case']} | {r['dominant_at_least_95_percent']}/{r['checkpoints']} | {r['median_actions_used']:g} | δ={r['delta']}, ρ={r['rho']} | {r['validation']['total_cost_usd']:.6f} | {100*r['validation']['violation_rate']:.6f} |")
    matches = sum(bool(m["matching_cost_actions"]) for r in comparisons for m in r["high_budget_matches"])
    exact = sum(bool(m["matching_metric_actions"]) for r in comparisons for m in r["high_budget_matches"])
    lines += ["",f"高预算随机初始化选中策略：{matches}/15 与某固定动作费用相同，{exact}/15 的费用、违约数、VM 秒、FaaS 请求和启动 VM 秒均相同。",
        "这说明本批结果可以由固定偏移解释，不能据此认定 PPO 学到了时变控制。仅费用相等本身不是策略等价的数学证明。","",
        "旧 v2 的混合排队近似无效，因此此表只审计旧算法行为，不作物理 SLA 结论。原 85 次 RL-study 只有 8k–16k 步，应归为有限预算、训练不足的探索记录。",
        "完整网格、训练选择、checkpoint 动作直方图统计和逐种子匹配均保留；验证网格最优动作只用于核查，不参与选择。"]
    output.mkdir(parents=True,exist_ok=True)
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
