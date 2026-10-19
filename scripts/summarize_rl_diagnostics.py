#!/usr/bin/env python3
"""Summarize diagnostic measurements without opening the historical test split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from libra_rl.diagnostics import write_json
from libra_rl.training import confidence_interval
from libra_sim.plotting import plt


def csv_write(path,rows):
    if not rows:
        return
    with path.open("w",newline="") as handle:
        writer = csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def summarize(output,partial=False):
    read = lambda name:json.loads((output/name).read_text())
    manifest = read("manifest.json")
    for path,expected in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"source changed: {path}")
    if not partial and not (output/"runs_index.json").exists():
        raise ValueError("main training is incomplete; use --partial for a clearly labeled interim report")
    runs = [json.loads(p.read_text()) for p in sorted((output/"runs").glob("*.json"))]
    main = [r for r in runs if r["job"]["case"]!="toy"]
    toys = [r for r in runs if r["job"]["case"]=="toy"]
    queue = read("queue_calibration.json")["rows"] if (output/"queue_calibration.json").exists() else []
    bounds = read("domain_bounds.json")["rows"] if (output/"domain_bounds.json").exists() else []
    audit = read("action_audit.json")["rows"] if (output/"action_audit.json").exists() else []
    raw = read("raw_failure_summary.json")["rows"] if (output/"raw_failure_summary.json").exists() else []
    groups,points,paired,calibration,frontiers = [],[],[],[],[]
    metrics = ("total_cost_usd","violation_rate","behavioral_teacher_fraction","explicit_faas_action_fraction")
    keys = sorted({(r["job"]["case"],r["job"]["initialization"],s["budget"])
                   for r in main for s in r["selections"]})
    for case,init,budget in keys:
        selected = [(r,s) for r in main for s in r["selections"]
                    if (r["job"]["case"],r["job"]["initialization"],s["budget"])==(case,init,budget)]
        row = dict(case=case,initialization=init,budget=budget,seeds=len(selected))
        for metric in metrics:
            mean,lo,hi = confidence_interval([s["summary"][metric] for _,s in selected])
            row.update({metric:mean,metric+"_lo":lo,metric+"_hi":hi})
        row.update(teacher_cost_usd=selected[0][0]["teacher_validation"]["total_cost_usd"],
                   teacher_violation_rate=selected[0][0]["teacher_validation"]["violation_rate"],
                   penalty=selected[0][0]["penalty"])
        bound = next((b for b in bounds if b["case"]==case and b["domain"]=="teacher_plus_faas"
                      and b["penalty"]==row["penalty"] and b["cost_weight"]==1),None)
        if bound:
            teacher = selected[0][0]["teacher_validation"]
            j_teacher = teacher["total_cost_usd"]+row["penalty"]*teacher["expected_violations"]
            denominator = j_teacher-bound["objective"]
            if denominator>1e-12:
                closed = [(j_teacher-s["objective_usd"])/denominator for _,s in selected]
                mean,lo,hi = confidence_interval(closed)
                row.update(objective_gap_closed_fraction=mean,objective_gap_closed_lo=lo,
                           objective_gap_closed_hi=hi,dp_objective_usd=bound["objective"])
        groups.append(row)
    for r in main:
        for p in r["curve"]:
            points.append(dict(case=r["job"]["case"],initialization=r["job"]["initialization"],
                seed=r["job"]["seed"],samples=p["samples"],cost_usd=p["summary"]["total_cost_usd"],
                violation_rate=p["summary"]["violation_rate"],objective_usd=p["objective_usd"],
                teacher_fraction=p["summary"]["behavioral_teacher_fraction"],
                faas_fraction=p["summary"]["explicit_faas_action_fraction"],fit_seconds=p["fit_seconds"],
                value_loss=p["optimizer_diagnostics"].get("train/value_loss"),
                entropy_loss=p["optimizer_diagnostics"].get("train/entropy_loss")))
    def selected_value(r,budget,metric):
        return next(s["summary"][metric] for s in r["selections"] if s["budget"]==budget)
    for case in sorted({r["job"]["case"] for r in main}):
        for comparison in ("budget_random","budget_rule_prior","initialization_65536","initialization_262144"):
            differences = []
            for seed in manifest["protocol"]["seeds"]:
                same = {r["job"]["initialization"]:r for r in main if r["job"]["case"]==case and r["job"]["seed"]==seed}
                if comparison.startswith("budget_"):
                    init = comparison.removeprefix("budget_")
                    if init not in same:
                        continue
                    a,b = (same[init],65536),(same[init],262144)
                else:
                    if not {"random","rule_prior"}<=same.keys():
                        continue
                    budget = int(comparison.split("_")[-1])
                    a,b = (same["random"],budget),(same["rule_prior"],budget)
                differences.append(selected_value(b[0],b[1],"total_cost_usd")-selected_value(a[0],a[1],"total_cost_usd"))
            if differences:
                mean,lo,hi = confidence_interval(differences)
                paired.append(dict(case=case,comparison=comparison,seeds=len(differences),
                    cost_difference_usd=mean,lo=lo,hi=hi,meaning="later minus earlier, or rule_prior minus random"))
    for service,distribution,u in sorted({(r["service_seconds"],r["distribution"],r["utilization"]) for r in queue}):
        selected = [r for r in queue if (r["service_seconds"],r["distribution"],r["utilization"])==(service,distribution,u)]
        mean,lo,hi = confidence_interval([r["violation_rate"] for r in selected])
        formula = selected[0]["analytic_violation_rate"]
        calibration.append(dict(service_seconds=service,distribution=distribution,utilization=u,seeds=len(selected),
            simulated_rate=mean,lo=lo,hi=hi,analytic_rate=formula,absolute_error=formula-mean,
            ratio_formula_to_mc=formula/mean if mean>0 else None))
    for case,domain in sorted({(r["case"],r["domain"]) for r in bounds}):
        selected = [r for r in bounds if (r["case"],r["domain"])==(case,domain)]
        opt = [r for r in selected if r["cost_weight"]==1]
        feasible = [r for r in opt if r["violation_rate"]<=manifest["protocol"]["epsilon"]]
        minviol = [r for r in selected if r["cost_weight"]==0]
        frontiers.append(dict(case=case,domain=domain,
            dual_lower_usd=max(r["dual_lower_bound_usd"] for r in opt),
            feasible_upper_usd=min(r["total_cost_usd"] for r in feasible) if feasible else None,
            min_violation_rate=minviol[0]["violation_rate"] if minviol else None,
            infeasible_proven=bool(minviol and minviol[0]["violation_rate"]>manifest["protocol"]["epsilon"])))
    summary = dict(fingerprint=manifest["fingerprint"],partial=partial,main_runs=len(main),toy_runs=len(toys),
        actual_main_samples=sum(r["samples"] for r in main),actual_toy_samples=sum(r["samples"] for r in toys),
        training_seconds=sum(r["fit_seconds"] for r in runs),evaluation_seconds=sum(r["evaluation_seconds"] for r in runs),
        budget_groups=groups,paired_cost_differences=paired,queue_groups=calibration,domain_frontiers=frontiers,
        test_evaluated=False)
    write_json(output/"summary.json",summary)
    for name,rows in (("budget_summary",groups),("learning_curves",points),("paired_differences",paired),
                       ("queue_summary",calibration),("domain_frontiers",frontiers)):
        csv_write(output/(name+".csv"),rows)
    if points:
        fig,axes = plt.subplots(2,3,figsize=(13,7),sharex=True)
        for i,case in enumerate(("v1","v2_s05","v2_s07")):
            for init in ("random","rule_prior"):
                for axis,metric in zip(axes[:,i],("cost_usd","violation_rate")):
                    sample_grid = sorted({p["samples"] for p in points if p["case"]==case and p["initialization"]==init})
                    triples = [confidence_interval([p[metric] for p in points if p["case"]==case and
                               p["initialization"]==init and p["samples"]==s]) for s in sample_grid]
                    if triples:
                        means,lo,hi = np.array(triples).T
                        factor = 100 if metric=="violation_rate" else 1
                        axis.plot(sample_grid,means*factor,label=init)
                        axis.fill_between(sample_grid,np.maximum(0,lo)*factor,hi*factor,alpha=0.15)
                        axis.set_xscale("log",base=2)
            selected = next((r for r in main if r["job"]["case"]==case),None)
            if selected:
                axes[0,i].axhline(selected["teacher_validation"]["total_cost_usd"],color="black",ls="--",label="teacher")
                axes[1,i].axhline(0.1,color="red",ls=":",label="SLA target")
            axes[0,i].set_title(case+" (validation only)")
            axes[1,i].set_xlabel("training samples")
            if axes[0,i].lines:
                axes[0,i].legend(fontsize=8)
        axes[0,0].set_ylabel("cost (USD / validation 48h)")
        axes[1,0].set_ylabel("expected violations (%)")
        fig.tight_layout(); fig.savefig(output/"learning_curves.png",dpi=160); fig.savefig(output/"learning_curves.svg"); plt.close(fig)
    if calibration:
        fig,axes = plt.subplots(1,2,figsize=(11,4))
        for ax,service in zip(axes,(0.5,0.7)):
            for distribution in ("deterministic","exponential"):
                selected = [r for r in calibration if r["service_seconds"]==service and r["distribution"]==distribution]
                x = [r["utilization"] for r in selected]
                y = np.array([100*r["simulated_rate"] for r in selected])
                lo = np.array([100*max(0,r["lo"]) for r in selected])
                hi = np.array([100*r["hi"] for r in selected])
                ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="o-",capsize=3,label=distribution+" MC (95% CI)")
                formula_label = "fixed service: hybrid approx." if distribution=="deterministic" else "exponential: M/M/16 formula"
                ax.plot(x,[100*r["analytic_rate"] for r in selected],"--",label=formula_label)
            ax.set(title=f"16 servers, mean service {service}s",xlabel="utilization",ylabel="P(response > 1s) (%)")
            ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(output/"queue_calibration.png",dpi=160); fig.savefig(output/"queue_calibration.svg"); plt.close(fig)
    if groups:
        fig,axes = plt.subplots(1,2,figsize=(12,4.5))
        legends = {}
        for ax,case in zip(axes,("v2_s05","v2_s07")):
            for init,color in (("random","tab:blue"),("rule_prior","tab:orange")):
                for budget,marker in ((65536,"o"),(262144,"s")):
                    r = next((r for r in groups if (r["case"],r["initialization"],r["budget"])==(case,init,budget)),None)
                    if not r:
                        continue
                    x,y = 100*r["violation_rate"],r["total_cost_usd"]
                    ax.errorbar(x,y,xerr=[[x-100*max(0,r["violation_rate_lo"])],[100*r["violation_rate_hi"]-x]],
                        yerr=[[y-r["total_cost_usd_lo"]],[r["total_cost_usd_hi"]-y]],
                        fmt=marker,color=color,capsize=3,label=f"{init}, {budget:,}")
            selected_run = next((r for r in main if r["job"]["case"]==case),None)
            if selected_run:
                t = selected_run["teacher_validation"]
                ax.scatter(100*t["violation_rate"],t["total_cost_usd"],marker="*",s=130,c="black",label="fixed teacher")
                f = selected_run["faas_validation"]
                ax.scatter(0,f["total_cost_usd"],marker="x",s=65,c="gray",label="all FaaS",clip_on=False)
            selected_bounds = sorted([r for r in bounds if r["case"]==case and r["domain"]=="teacher_plus_faas"
                                      and r["cost_weight"]==1],key=lambda r:r["violation_rate"])
            ax.plot([100*r["violation_rate"] for r in selected_bounds],[r["total_cost_usd"] for r in selected_bounds],
                    ":",color="purple",alpha=0.7,label="clairvoyant DP lambda sweep")
            ax.axvline(0.1,color="red",ls="--",lw=1,label="SLA target")
            ax.set_xscale("symlog",linthresh=0.001)
            ax.set_xlim(left=0)
            ax.set(title=case+" (validation only)",xlabel="expected violations (%)",ylabel="cost (USD / 48h)")
            handles,labels = ax.get_legend_handles_labels()
            legends.update(zip(labels,handles))
        fig.legend(legends.values(),legends.keys(),loc="lower center",ncol=4,fontsize=7)
        fig.tight_layout(rect=(0,0.14,1,1)); fig.savefig(output/"diagnostic_pareto.png",dpi=160); fig.savefig(output/"diagnostic_pareto.svg"); plt.close(fig)
    lines = ["# RL 诊断实验：动作范围、排队校准与训练预算", "",
        "**仅训练/验证集；不重新评估历史测试集。** "+("此报告尚未完成主训练。" if partial else "所有预定训练已完成。"),"",
        "Azure 16× 调用量复制；第 1–9 天训练，第 10–11 天验证。验证按 8 个独立 6 小时回合计费，每回合清空 VM 和历史。",
        "旧测试结果已参与提出本次诊断，后续泛化结论必须另找独立数据；这里的区间只反映随机种子差异。","",
        "## 动作范围和已知未来的上限","",
        "同域 DP 使用相同的 ±3 残差目标、60 秒决策、100 秒冷启动和路由网格。分别加入全 FaaS 动作、再更换规则中心，避免将两项变化混为一谈。",
        "全 FaaS 会立即清空 VM，并对全部请求正常计费。更换中心前后的动作集合不互相包含。","",
        "| 场景 | 分区 | 原动作不能表达教师的比例 | 原动作不能全 FaaS 的比例 | 教师成本 ($) | 教师违约率 (%) |",
        "|---|---|---:|---:|---:|---:|"]
    for r in audit:
        lines.append(f"| {r['case']} | {r['split']} | {100*r['teacher_unreachable_fraction']:.2f}% | {100*r['full_faas_unreachable_fraction']:.2f}% | {r['teacher_summary']['total_cost_usd']:.4f} | {100*r['teacher_summary']['violation_rate']:.5f} |")
    lines += ["","| 场景 | 动作域 | 约束成本下界 ($) | 已找到可行成本 ($) | 最低可达违约率 (%) |","|---|---|---:|---:|---:|"]
    for r in frontiers:
        upper = "未找到" if r["feasible_upper_usd"] is None else f"{r['feasible_upper_usd']:.6f}"
        minimum = "待算" if r["min_violation_rate"] is None else f"{100*r['min_violation_rate']:.6f}"
        lines.append(f"| {r['case']} | {r['domain']} | {r['dual_lower_usd']:.6f} | {upper} | {minimum} |")
    lines += ["","对每个 λ，DP 精确最小化 C+λV。约束成本下界为 maxλ[Jλ−λεN]；满足约束的 DP 计划给出可行上界。",
        "有限 λ 扫描不等于精确求解所有 ε 约束问题；下界只针对表中有限动作域，且 DP 使用未来信息，不能视为在线 RL 可达收益。","",
        "## 原始动作失控重放","",
        "以下全部来自旧模型在验证集的逐步重放；成本必须与原记录一致到 1e-9。失败种子完整保留，没有改 VM 上限或剔除异常值。","",
        "| 种子 | 验证成本 ($) | 正扩容动作比例 | 位于 512 VM 上限比例 | 平均动作熵 | value RMSE |","|---|---:|---:|---:|---:|---:|"]
    for r in raw:
        lines.append(f"| {r['seed']} | {r['summary']['total_cost_usd']:.6f} | {100*r['positive_delta_fraction']:.2f}% | {100*r['at_vm_cap_fraction']:.2f}% | {r['mean_entropy']:.4f} | {r['value_rmse']:.3f} |")
    lines += ["","`raw_trace_seed*.csv` 包含动作、VM、费用、奖励、熵、critic value 和实际折扣回报。",
        "critic 对照使用训练时的奖励尺度和有限回合 γ=0.99；这些观测说明行为及预测误差，不能单独证明优化器失效的因果机制。",
        "原始动作空间最大熵为 log(9×5)≈3.807。失控种子的平均熵仍接近最大值：贪心持续扩容不意味着模型对此具有很高置信度，微小 logit 差异也可能反复选择同一扩容动作。","",
        "## 独立排队校准","",
        "连续时间 FCFS 离散事件模拟，每项配置 5 个独立种子；各预热 50,000、测量 500,000 个请求。区间基于种子均值，不把相关请求当作独立伯努利样本。",
        "固定服务时间的 M/D/16 对照历史 wait_budget 混合近似；指数服务时间的 M/M/16 对照完整响应时间公式。两者假设不同，不根据 RL 是否获利来挑选。","",
        "| 服务 (s) | 分布 | 利用率 | 仿真违约 (%) | 95% 区间 (%) | 公式违约 (%) |","|---:|---|---:|---:|---|---:|"]
    for r in calibration:
        lines.append(f"| {r['service_seconds']} | {r['distribution']} | {r['utilization']} | {100*r['simulated_rate']:.5f} | [{100*r['lo']:.5f}, {100*r['hi']:.5f}] | {100*r['analytic_rate']:.5f} |")
    lines += ["","![独立排队校准](queue_calibration.png)","",
        "零事件表示有限样本未观测到违约；此时种子 t 区间为零，并不意味着真实违约概率严格为零。稳态校准不能验证秒级非平稳队列、跨 VM 负载均衡或 AWS/FaaS 延迟。",
        "本次 PPO 为隔离学习问题继续使用历史 wait_budget；其 ε 合格只表示在该近似环境合格，不构成校准后真实云 SLA 结论。","",
        "## 简单负载学习门槛与两档预算","",
        "先在恒定 32 req/s 的 v1 环境训练 5 个随机种子，65,536 步内选 4 个更新后 checkpoint；平均成本相对规则不超过 1% 才运行主实验。"]
    if (output/"toy_gate.json").exists():
        gate = read("toy_gate.json")
        lines += [f"实际平均 regret：{100*gate['mean_regret_fraction']:.4f}%；门槛{'通过' if gate['passed'] else '未通过'}。"]
    lines += ["","主实验比较随机初始化与显式规则 logit 先验（初始教师动作概率 0.9，贪心执行等于规则）。这是预置知识，不能算学习所得。",
        "每个种子只训练到 262,144 步；低预算从 16,384/32,768/49,152/65,536 选，高预算从 65,536/131,072/196,608/262,144 选，均为 4 个验证 checkpoint。",
        "高预算不额外保留低预算的全部早期 checkpoint，以保持四次选择机会相同；因此预算差异也受候选 checkpoint 时间位置影响，不能把高预算成本上升直接判作训练过拟合。",
        "λ 从验证集上的 teacher_plus_faas DP 可行点选择，不使用测试；这种理想化 oracle 辅助选择也必须披露。两初始化使用相同动作域、λ、预算、GBDT 和 5 个种子。","",
        "| 场景 | 初始化 | 预算 | 验证成本 $，均值 [95% CI] | 违约率 %，均值 [95% CI] | DP 目标差距关闭比例 | 教师动作行为比例 |","|---|---|---:|---|---|---:|---:|"]
    for r in groups:
        closed = f"{100*r['objective_gap_closed_fraction']:.2f}%" if "objective_gap_closed_fraction" in r else "待算"
        lines.append(f"| {r['case']} | {r['initialization']} | {r['budget']} | {r['total_cost_usd']:.4f} [{r['total_cost_usd_lo']:.4f}, {r['total_cost_usd_hi']:.4f}] | {100*r['violation_rate']:.5f} [{100*r['violation_rate_lo']:.5f}, {100*r['violation_rate_hi']:.5f}] | {closed} | {100*r['behavioral_teacher_fraction']:.2f}% |")
    lines += ["","DP 目标差距关闭比例为 (J教师−J策略)/(J教师−JDP)，其中 J=C+λV；v1 的 λ=0，等于成本差距。负数表示比教师更差；v2 的目标改善不能替代 SLA 可行性检查。完整区间见 CSV。"]
    if points:
        lines += ["","学习曲线展示各 checkpoint 自身的结果，预算表及成本–违约图展示四候选择优后的结果；最终时点曲线不一定等于该预算所选结果。",
                  "","![验证学习曲线](learning_curves.png)","","![验证成本与违约平面](diagnostic_pareto.png)"]
    lines += ["","配对种子差异见 `paired_differences.csv`；差值为高预算减低预算，或先验初始化减随机初始化。区间跨 0 时，不能从这些种子写出成本显著改善。",
        "区间使用预设的 Student-t 方法，可能越过零等物理边界；图中仅截断显示，表格和 CSV 保留原区间。",
        "`validation_feasible_fallback` 单独记录规则/全 FaaS/学习策略中最便宜的验证可行选项；保留规则不算 RL 改善，也没有测试 SLA 保证。","",
        f"实际完成 {len(main)} 条主训练、{len(toys)} 条 toy 训练；主训练 {summary['actual_main_samples']:,} 样本，toy {summary['actual_toy_samples']:,} 样本，总计 {summary['actual_main_samples']+summary['actual_toy_samples']:,}。",
        "样本指环境决策步，包含训练分区的重复抽样和数据增强，并非同等数量的独立云观测。",
        f"累计 fit 时间 {summary['training_seconds']/60:.2f} 分钟，checkpoint 验证时间 {summary['evaluation_seconds']/60:.2f} 分钟（并行进程累加，不是墙钟）。","",
        "## 解读边界","",
        "新增联合 categorical 动作替代旧 MultiDiscrete 因子化，并新增全 FaaS、修改残差中心；因此不能把新旧结果差异全部归因于训练时长。",
        "本次随机/先验和低/高预算之间的内部比较保持这些因素一致。基线来自冻结的训练网格，并精确重放；这是算法诊断，不是新的同等调参预算竞赛。",
        "教师保留旧分位数定义和 12 个候选网格，没有重新优化所有经济分位数、预测分布和路由参数；达到或超过这个教师不等于超过所有预测加优化方法。",
        "6 小时有限回合有明确冷启动和清空边界；γ=0.99 的训练目标与未折扣总费用并不完全相同。更多样本不自动证明收敛。",
        "校准发现的物理模型偏差、验证集重复选参、仅 5 个种子和历史测试暴露，限制了结果的外推。后续应先冻结物理假设与协议，再获取独立数据开展最终评估。","",
        "本地开发期间另有一次因 sklearn OpenMP 线程超额而中断的性能排查，目录为 Git 忽略的 `outputs/rl_diagnostics_attempt1/`。",
        "其主训练没有完成，未用于选择参数或报告主结果；上面的样本计数只统计本次冻结协议下完整完成的训练。源码在修复线程设置后重新冻结并完整重跑。","",
        "## 复查与重跑","","```bash",
        ".venv/bin/python scripts/rl_diagnostics.py --stage all --workers 3 --output outputs/rl_diagnostics_replay",
        ".venv/bin/python scripts/summarize_rl_diagnostics.py --output outputs/rl_diagnostics_replay",
        ".venv/bin/python scripts/verify_rl_diagnostics.py --require-models --output outputs/rl_diagnostics_replay","```","",
        "旧实验源码与产物保持不变；所有本次源文件、数据、模型、种子和实际样本数有哈希记录。权重保存在本地且由 Git 忽略。","",
        "方法依据：[SB3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html)、[SB3 策略源码](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/policies.html)、[MIT M/M/m 排队推导](https://web.mit.edu/urban_or_book/www/book/chapter4/4.6.1.html)。"]
    if not partial:
        overview = ["## 实测结论",""]
        for case in ("v1","v2_s05","v2_s07"):
            low = next(g for g in groups if (g["case"],g["initialization"],g["budget"])==(case,"random",65536))
            high = next(g for g in groups if (g["case"],g["initialization"],g["budget"])==(case,"random",262144))
            overview.append(f"- {case} 随机初始化：低/高预算的验证平均费用分别为 ${low['total_cost_usd']:.4f} / ${high['total_cost_usd']:.4f}；高预算平均违约率 {100*high['violation_rate']:.5f}%，教师费用 ${high['teacher_cost_usd']:.4f}。")
        prior = [g for g in groups if g["initialization"]=="rule_prior"]
        if all(abs(g["total_cost_usd"]-g["teacher_cost_usd"])<1e-9 and
               abs(g["violation_rate"]-g["teacher_violation_rate"])<1e-12 for g in prior):
            overview += ["- 三个场景中，规则先验在两档预算下选出的 checkpoint 都与教师具有相同费用和违约率；更新前已有这一能力，所选更新后策略没有额外收益。"]
        overview += ["- 独立事件模拟支持指数服务的完整 M/M/16 响应公式；固定服务时间下，旧混合近似的偏差明显，需先明确物理假设再解释云 SLA。", "",
            "本轮仍不足以支持 RL 稳定优于强规则。以下保留全部种子、失败结果、置信区间及适用边界，不能把费用更低但违约的结果计作受约束改进。", ""]
        pivot = lines.index("## 动作范围和已知未来的上限")
        lines[pivot:pivot] = overview
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(f"Summarized {len(main)} main runs, {len(toys)} toy runs; test_evaluated=False")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/rl_diagnostics")
    parser.add_argument("--partial",action="store_true")
    args = parser.parse_args()
    summarize(args.output,args.partial)
