#!/usr/bin/env python3
"""Findings-first report for the corrected physics and fixed-action audit."""
import argparse
import csv
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import physical_headroom as driver
from libra_sim.plotting import plt


def ci(values):
    values = np.asarray(values,dtype=float)
    if len(values)!=5:
        raise ValueError("five independent arrival seeds required")
    mean = float(values.mean())
    half = 2.7764451051977987*float(values.std(ddof=1))/np.sqrt(5)
    return mean,float(mean-half),float(mean+half)


def summarize(output):
    read = lambda name:json.loads((output/name).read_text())
    protocol,manifest,_ = driver.prepare(output)
    original = read("evaluation.json")
    extended = read("forecast_rule_evaluation.json")
    exact = read("exact_intervals.json")
    if len(exact["records"])!=20:
        raise ValueError("twenty exact-search intervals required")
    for path,expected in exact["source_sha256"].items():
        if driver.sha(ROOT/path)!=expected:
            raise ValueError("exact search source changed")
    if extended["source_sha256"]!=driver.sha(ROOT/"scripts/forecast_rule_controls.py"):
        raise ValueError("forecast control source changed")
    policies = ("fixed","causal_economic","perfect_quantile","certified_clairvoyant","perfect_window")
    rows = original["rows"]+extended["rows"]+[dict(policy="certified_clairvoyant",**r) for r in exact["records"]]
    groups,per_seed,certificates,paired = [],[],[],[]
    fixed_training = read("fixed_training_grid.json")
    forecast_training = read("forecast_rule_grid.json")
    for service in protocol["services_seconds"]:
        for policy in policies:
            summaries = []
            for seed in protocol["validation_arrival_seeds"]:
                selected = [r for r in rows if r["service_seconds"]==service and r["arrival_seed"]==seed and r["policy"]==policy]
                if len(selected)!=2:
                    raise ValueError("two validation windows per arrival seed required")
                aggregate = driver.aggregate([r["summary"] for r in selected])
                summaries.append(aggregate)
                per_seed.append(dict(service_seconds=service,policy=policy,arrival_seed=seed,**aggregate))
            group = dict(service_seconds=service,policy=policy,seeds=5,validation_seconds=7200)
            for metric in ("total_cost_usd","violation_rate","drain_vm_seconds","terminal_drain_vm_seconds"):
                mean,lo,hi = ci([s[metric] for s in summaries])
                group.update({metric:mean,metric+"_lo":lo,metric+"_hi":hi})
            groups.append(group)
        for comparison,a,b in (("causal_minus_fixed","fixed","causal_economic"),
                               ("perfect_minus_causal","causal_economic","perfect_quantile"),
                               ("DP_minus_perfect","perfect_quantile","certified_clairvoyant")):
            differences = []
            for seed in protocol["validation_arrival_seeds"]:
                value = lambda policy:next(r["total_cost_usd"] for r in per_seed if r["service_seconds"]==service and r["arrival_seed"]==seed and r["policy"]==policy)
                differences.append(value(b)-value(a))
            mean,lo,hi = ci(differences)
            paired.append(dict(service_seconds=service,comparison=comparison,mean_usd=mean,lo=lo,hi=hi))
        lowers = [sum(r["cost_lower_bound_usd"] for r in exact["records"] if r["service_seconds"]==service
                      and r["arrival_seed"]==seed) for seed in protocol["validation_arrival_seeds"]]
        lower = float(np.mean(lowers))
        group = lambda policy:next(g for g in groups if g["service_seconds"]==service and g["policy"]==policy)
        fixed,perfect,dp = group("fixed"),group("perfect_quantile"),group("certified_clairvoyant")
        candidates = [("fixed",fixed_training["selected"][str(service)]["summary"]["total_cost_usd"]),
                      ("causal_economic",forecast_training["selected"][f"{service}_causal_economic"]["summary"]["total_cost_usd"])]
        strong_family = min(candidates,key=lambda p:p[1])[0]
        strong = group(strong_family)
        certificates.append(dict(service_seconds=service,cost_lower_bound_usd=lower,
            feasible_offline_cost_usd=dp["total_cost_usd"],solver_cost_gap_upper_fraction=(dp["total_cost_usd"]-lower)/max(1e-12,dp["total_cost_usd"]),
            fixed_cost_usd=fixed["total_cost_usd"],fixed_mean_feasible=fixed["violation_rate"]<=protocol["epsilon"],
            any_policy_cost_improvement_upper_fraction=(fixed["total_cost_usd"]-lower)/fixed["total_cost_usd"],
            perfect_rule_cost_usd=perfect["total_cost_usd"],
            extra_offline_improvement_over_perfect_lower_usd=max(0,perfect["total_cost_usd"]-dp["total_cost_usd"]),
            extra_offline_improvement_over_perfect_upper_usd=max(0,perfect["total_cost_usd"]-lower),
            certified_windows=sum(r["exact"] for r in exact["records"] if r["service_seconds"]==service),
            strong_causal_family_selected_on_train=strong_family,strong_causal_cost_usd=strong["total_cost_usd"],
            strong_causal_violation_rate=strong["violation_rate"],
            improvement_upper_fraction_vs_strong_causal=(strong["total_cost_usd"]-lower)/strong["total_cost_usd"]))
    summary = dict(fingerprint=manifest["fingerprint"],groups=groups,per_seed=per_seed,paired_cost_differences=paired,
        certificates=certificates,test_evaluated=False,ppo_trained=False,
        constraint_unit="each one-hour cold-start episode independently; the reported two-hour cost is a sum of two independent episodes",
        statistical_unit="five subsecond-arrival/routing Monte Carlo seeds on the same two one-hour validation windows; not five training/workload seeds")
    driver.write(output/"summary.json",summary)
    for name,records in (("summary",groups),("per_seed",per_seed),("certificates",certificates),("paired_differences",paired)):
        with (output/(name+".csv")).open("w",newline="") as handle:
            writer = csv.DictWriter(handle,fieldnames=list(records[0]),lineterminator="\n")
            writer.writeheader(); writer.writerows(records)
    runtime = dict(recorded_utc=driver.datetime.now(driver.timezone.utc).isoformat(),numpy=np.__version__,accelerator=manifest["accelerator"])
    if manifest["accelerator"]=="numba":
        import numba
        runtime["numba"] = numba.__version__
    driver.write(output/"runtime.json",runtime)
    fig,axes = plt.subplots(2,2,figsize=(12,7))
    shown = ("fixed","causal_economic","perfect_quantile","certified_clairvoyant")
    labels = ("fixed offset","causal rule","perfect forecast","certified offline")
    for column,service in enumerate(protocol["services_seconds"]):
        selected = [next(g for g in groups if g["service_seconds"]==service and g["policy"]==p) for p in shown]
        for row,metric,factor in ((0,"total_cost_usd",1),(1,"violation_rate",100)):
            means = np.array([g[metric] for g in selected])*factor
            lows = np.array([g[metric+"_lo"] for g in selected])*factor
            highs = np.array([g[metric+"_hi"] for g in selected])*factor
            axes[row,column].bar(np.arange(4),means,color=["tab:blue","tab:green","tab:orange","tab:purple"],
                yerr=[means-np.maximum(0,lows),highs-means],capsize=4)
            axes[row,column].set_xticks(np.arange(4),labels,rotation=15,ha="right")
            axes[row,column].set_ylabel("cost (USD / 2 validation hours)" if row==0 else "SLA violations (%)")
        lower = next(c["cost_lower_bound_usd"] for c in certificates if c["service_seconds"]==service)
        axes[0,column].axhline(lower,color="black",ls="--",label="certified cost lower bound")
        axes[0,column].set_title(f"Deterministic service {service}s; carried queues")
        axes[0,column].legend(fontsize=8)
        axes[1,column].axhline(0.1,color="red",ls=":",label="0.1% target")
        axes[1,column].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/"controls.png",dpi=160); fig.savefig(output/"controls.svg"); plt.close(fig)
    load = read("load_statistics.json")["rows"]
    lookup = read("md16_lookup.json")
    raw_source = ROOT/"outputs/rl_diagnostics/raw_trace_seed1.csv"
    with raw_source.open() as handle:
        raw_rows = list(csv.DictReader(handle))
    raw_counts = Counter(int(r["action_delta"]) for r in raw_rows)
    driver.write(output/"raw_action_behavior.json",dict(source_sha256=driver.sha(raw_source),
        delta_counts={str(k):v for k,v in sorted(raw_counts.items())},decisions=len(raw_rows),
        meaning="deterministic greedy replay of the historical failed raw-action seed; high entropy alone does not prove a training root cause"))
    fig,axes = plt.subplots(1,2,figsize=(10,4))
    for ax,service in zip(axes,protocol["services_seconds"]):
        points = lookup["tables"][str(service)]
        means = np.array([100*r["raw_mean"] for r in points])
        low = np.array([100*max(0,r["ci95"][0]) for r in points])
        high = np.array([100*r["ci95"][1] for r in points])
        ax.errorbar([r["utilization"] for r in points],means,yerr=[means-low,high-means],fmt="o-",capsize=3)
        ax.set(title=f"M/D/16, service={service}s, stable u <= 0.99",xlabel="utilization",ylabel="P(response > 1s) (%)")
    fig.tight_layout(); fig.savefig(output/"md16_lookup.png",dpi=160); fig.savefig(output/"md16_lookup.svg"); plt.close(fig)
    findings = ["# 先把固定规则和队列模型做对：LIBRA 扩展的重新核查","",
        "## 发现","",
        "1. **旧 PPO 没有展示状态相关的额外收益。** 高预算的 15/15 个随机初始化结果，与某个固定动作的费用、违约数、VM 秒、FaaS 请求和启动 VM 秒完全相同。补扫 δ=−1 的固定规则优于旧 PPO 均值。",
        "2. **原 v2 的物理 SLA 解读撤回。** 指数等待尾拼接固定服务时间的公式偏差明显。新模型逐请求计算 FCFS 延迟；满载时积压延续到下一秒，缩容和回合结束都结算排空费用。",
        f"3. **错峰负载的波动确实下降。** 训练 CV 从 {load[0]['synchronous_cv']:.3f} 降到 {load[0]['staggered_cv']:.3f}，验证 CV 从 {load[1]['synchronous_cv']:.3f} 降到 {load[1]['staggered_cv']:.3f}。这仍是相关的错峰副本，没有冒充 16 条独立用户轨迹。",
        "4. **下面把预测收益和全局规划参照分开报告。** 固定动作、经济分位数规则、完美分位数预测规则均只在训练集选参数；离线搜索另外给出可行计划及乐观成本下界。","",
        "| 服务时间 | 策略 | 验证成本 USD，均值 [95% CI] | 违约率 %，均值 [95% CI] |","|---:|---|---|---|"]
    translated = {"fixed":"最佳固定残差","causal_economic":"因果经济分位数规则","perfect_quantile":"完美分位数预测规则","certified_clairvoyant":"已证最优的全未来计划"}
    chosen = lambda service,policy:next(g for g in groups if g["service_seconds"]==service and g["policy"]==policy)
    fixed07,causal07,perfect07 = [chosen(0.7,p)["total_cost_usd"] for p in ("fixed","causal_economic","perfect_quantile")]
    gain_causal = 1-causal07/fixed07
    gain_perfect = 1-perfect07/causal07
    findings.insert(findings.index("| 服务时间 | 策略 | 验证成本 USD，均值 [95% CI] | 违约率 %，均值 [95% CI] |"),
        f"5. **先调好规则就能获得显著的开发窗口收益。** 0.7s 下，因果经济规则比固定残差便宜 {100*gain_causal:.2f}%，完美分位数预测再便宜 {100*gain_perfect:.2f}%。0.5s 下完美预测规则反而略贵，说明指定启发式的结果并非完美信息下的最优值。\n")
    for g in groups:
        if g["policy"] not in translated:
            continue
        findings.append(f"| {g['service_seconds']} | {translated[g['policy']]} | {g['total_cost_usd']:.6f} [{g['total_cost_usd_lo']:.6f}, {g['total_cost_usd_hi']:.6f}] | {100*g['violation_rate']:.5f} [{100*max(0,g['violation_rate_lo']):.5f}, {100*min(1,g['violation_rate_hi']):.5f}] |")
    findings += ["","成本对应预先指定的两个独立 1 小时验证回合之和，每回合冷态启动、分别约束违约率 ≤0.1%；不能跨回合调剂违约预算。五个种子仅改变秒内到达时间和路由随机数，不代表五条独立云负载。区间采用五种子 t 区间；表内概率区间裁到 [0,1]，CSV 保留未裁数值。完整逐种子数据和配对差值见 CSV。","",
        "![修正模型中的对照](controls.png)","","## DP 能证明什么","",
        "| 服务时间 | 严格乐观成本下界 $ | 已找到可行离线费用 $ | 最优值证书剩余宽度 | 固定动作到下界的费用差距 |",
        "|---:|---:|---:|---:|---:|"]
    for c in certificates:
        findings.append(f"| {c['service_seconds']} | {c['cost_lower_bound_usd']:.6f} | {c['feasible_offline_cost_usd']:.6f} | {100*c['solver_cost_gap_upper_fraction']:.4f}% | {100*c['any_policy_cost_improvement_upper_fraction']:.3f}% |")
    findings += ["","再将参照换成训练集在固定残差/经济规则两家族中选出的最强因果规则：",""]
    for c in certificates:
        findings.append(f"- {c['service_seconds']}s：训练选中 `{c['strong_causal_family_selected_on_train']}`，其验证成本 ${c['strong_causal_cost_usd']:.6f}；相对全未来最优的最大费用差距为 {100*c['improvement_upper_fraction_vs_strong_causal']:.3f}%。这是同动作域的理想上限，不能全部算作可学习收益。")
    findings += ["","新队列状态包含每台 VM 的 16 个完成时刻。先用 beam 找可行计划，再用无 beam 剪枝的队列状态搜索核验；20/20 个主窗口组合均在数值容差 1e-10 内上下界闭合。证书只对同一有限动作域、指定到达/路由样本成立。",
        "下界先使用无 SLA/免费排空的精确 DP，再用每个就绪阶段重置队列的 SLA 松弛 DP 加强：重置只能加快 FCFS 完成，非负 Lagrange 乘子给出有效成本下界。最终保留未展开搜索前沿的最小下界，闭合时才标已证最优；已用独立穷举核验。原始 beam 结果另外保留，未冒充精确值。",
        "这里的 λ 仅用于数学下界，不是从验证集给 PPO 选择奖励惩罚；本轮没有训练 checkpoint。",
        "固定动作若在验证窗口不可行，其费用差距不能解释成可行策略的改善上限。即使可行，固定动作到未来 oracle 的差距仍混合了预测与规划，不是 RL 专属收益。","",
        "## 三个对照的定义","",
        "- 固定残差：因果过去 300s 分位数为中心，扫描 δ=−3…3、ρ=0.6…1.0，加全 FaaS，共 36 个候选。",
        "- 因果经济规则与完美预测规则：使用与 ρ 对应的经济临界分位数，各自扫描相同 36 个候选。前者只读已完成秒；后者读取 [t+100,t+400) 的真实需求，按实际冷启动执行，目标裁回同一残差动作域。",
        "- 全未来搜索：知道整段到达样本及路由随机数，带积压状态优化整个小时，并保留已知可行参照计划；再以精确前沿搜索证明最优性。它比完美秒级分位数预测还知道更多外部随机输入，差距不能干净地解释成纯序贯价值。",
        "",
        "额外保留了一个看未来 160s、但只优化固定目标的局部规划器（`perfect_window`）。它有时不如固定规则，说明完美预测不能补救局部规划器自身的启发式偏差；没有拿这项差距包装 RL 收益。",
        "经济分位数控制是在观察到该局部规划器偏弱后补加的开发对照；参数仍全在训练集选，验证集不承担新的盲测结论。","",
        "临界分位数使用 τρ=max(0,min(1,1−cVM/(ρ·μ·cFaaS)))，其中 μ 是每台 VM 的吞吐率、cVM 是每秒 VM 费、cFaaS 是每次 FaaS 费；ρ 改变可用容量后不能继续盲用满容量的分位数。","",
        "## 物理模型和数据","",
        "每台 VM 16 个固定服务时间的槽位；秒内到达时间条件均匀，VM 路由均匀随机。按到达顺序执行每秒配额，避免利用整秒未来请求来随机挑子集。ρ 控制可路由配额，不把 u=1 判成整秒全违约。",
        "[M/D/16 查表](md16_lookup.json)复用原独立事件仿真，并新增 0.95/0.975/0.99 利用率校准。查表是稳态参照，正式 SLA 使用跨秒事件队列；`MD16Lookup` 拒绝在饱和区外推。",
        "![M/D/16 经验查表](md16_lookup.png)","",
        "Azure 第 1–9 天只用于训练网格，第 10–11 天是开发验证；未使用第 12–14 天。16 个非循环偏移只在各自分区内部叠加，丢弃共同前缀。所有训练和验证回合都有同分区过去 1,800s 历史，决策 t 只读 t 之前已完成秒，VM 从冷态开始。",
        "固定动作在 209 个 1 小时训练回合上选，训练选择写盘后才生成验证到达样本。验证窗口和五个随机种子预先固定；这是用于检查空间的 1 小时开发试验，尚未扩成连续多天泛化实验。",
        "初版开发队列使用离线均匀挑选当秒 VM 请求，核查发现这需要偷看当秒未来；最终改为上述因果配额并完整重跑。旧开发目录独立保留并由 Git 忽略。","",
        "## 是否继续 PPO","",
        "**本轮没有再训练 PPO。** 目前只有开发窗口上的因果规则与非因果 oracle 对照，尚未证明存在强预测规则无法覆盖、且可以从历史学到的稳定收益；历史验证/测试已反复暴露，不能再拿来宣称新的泛化优势。",
        "后续若继续，先冻结强因果预测基线及独立评估数据，再使用至少 100 万步、16 个并行环境、奖励归一化和熵系数 0.001。最低验收门槛是：使用不止一种有效动作，并在验证集上满足约束且胜过训练集选出的最佳固定动作及预测规则。",
        "输入驱动 critic 可按 [Mao et al., ICLR 2019](https://people.csail.mit.edu/hongzi/var-website/index.html)使用训练期未来外部输入降低方差；actor 始终只读因果历史。该方法还没有在本项目实现或训练，不记作完成的实验。","",
        "## 已修复的报告与测试","",
        "[固定动作复核](../constant_action_audit/REPORT.md)保留全部网格；[旧 RL-study](../rl_study/REPORT.md)已改列早期训练记录，旧 v2 不再作物理 SLA 结论。`test_rl_diagnostics.py` 在缺少 Torch/SB3 时跳过对应训练测试，其余 DP 与队列测试继续执行。",
        "奖励波动复核在 `../constant_action_audit/reward_signal.json` 分训练/验证、成本/实际奖励分别记录。全轨迹的标准差不等于不可约需求噪声，单步收益也不等于优势估计方差，因此没有把这项比值当成 PPO 理论上限。","",
        f"raw 失败种子的逐步复核：{dict(sorted(raw_counts.items()))}（增量:次数）。最常见的是 +4，共 {raw_counts[4]}/{len(raw_rows)} 步；高熵下的贪心偏置可解释持续扩容行为，但尚未单独证明优化器或 critic 是唯一根因。","",
        "## 重跑","","```bash",
        ".venv/bin/python -m pip install -r requirements-physical.txt",
        ".venv/bin/python scripts/audit_constant_actions.py",
        ".venv/bin/python scripts/audit_reward_signal.py",
        ".venv/bin/python scripts/physical_headroom.py --stage all --workers 4",
        ".venv/bin/python scripts/forecast_rule_controls.py",
        ".venv/bin/python scripts/refine_physical_bounds.py",
        ".venv/bin/python scripts/summarize_physical_headroom.py",
        ".venv/bin/python scripts/verify_physical_headroom.py", "```", ""]
    (output/"REPORT.md").write_text("\n".join(findings))
    course = ["# LIBRA 仿真扩展：PPO 结果可由固定偏移解释", "",
        "## 核心发现", "",
        "**PPO 没有展示额外的状态相关控制。** 15/15 个高预算随机初始化选中结果，在费用、违约数、VM 秒、FaaS 请求和启动 VM 秒上都与某个固定残差动作相同。",
        "固定残差仍会随负载规则调整 VM，固定的是规则上的偏移和路由。补上这项廉价网格对照后，原来‘追平教师’的结论不再构成学习收益。", "",
        f"原始增量动作则会累加：旧失败种子的 {raw_counts[4]}/{len(raw_rows)} 步选择 +4 台，最终反复到达 512 台上限。", "",
        "| 旧模型场景 | 教师费用 $ | 训练集选出的固定动作费用 $ | PPO 高预算平均费用 $ |",
        "|---|---:|---:|---:|"]
    audit = json.loads((ROOT/"outputs/constant_action_audit/summary.json").read_text())
    for r in audit["comparisons"]:
        course.append(f"| {r['case']} | {r['teacher_validation']['total_cost_usd']:.3f} | {r['validation']['total_cost_usd']:.3f} | {r['ppo_high_budget_mean_cost']:.3f} |")
    course += ["","这张旧表只用于算法审计，不再作为物理 SLA 证据。原 85 次 8k–16k 步训练保留为早期探索记录。", "",
        "**旧排队模型高估固定服务时间下的延迟。** 现已改为每台 VM 16 槽位的 FCFS 事件队列，保留跨秒积压，并对缩容及回合末排空计费；满载不再直接判整秒违约。M/D/16 经验查表只用于稳态校准。", "",
        "**修正后，基线仍比扩大 PPO 训练更值得先做。** 0.7s 场景的因果经济分位数规则，比最佳固定残差便宜 "
        f"{100*gain_causal:.2f}%；完美分位数预测只再便宜 {100*gain_perfect:.2f}%。所有候选参数在训练集选，结果如下。", "",
        "| 服务时间 | 对照 | 两个验证小时的费用 $ | 违约率 % |","|---:|---|---:|---:|"]
    for service in (0.5,0.7):
        for policy in ("fixed","causal_economic","perfect_quantile","certified_clairvoyant"):
            g = chosen(service,policy)
            course.append(f"| {service} | {translated[policy]} | {g['total_cost_usd']:.6f} | {100*g['violation_rate']:.5f} |")
    course += ["","数据为 Azure 分区内 16 个非循环错峰副本，训练日 1–9、开发验证日 10–11；训练和验证均有过去 1,800s 历史，只读已完成秒。表中均值来自同两个独立 1h 冷启动回合的五个到达/路由随机种子；每回合分别约束违约率 ≤0.1%。", "",
        "## 不能从 DP 差距推导 RL 收益", "",
        "20/20 个预设服务时间/到达种子/窗口组合均已得到最优性证书：无 beam 剪枝的队列状态搜索，加上 SLA 松弛 DP 下界，使上下界在数值容差内闭合。这个最优值只针对本次有限动作域和指定到达样本。",
        "全未来搜索还知道未来到达与路由随机数；完美分位数规则只是一种启发式。两者的差距混合信息、规划和算法误差，不能直接解释成在线 RL 可学到的序贯收益。0.5s 的完美预测规则反而略贵，也说明这一点。", "",
        f"以训练集选出的最强因果规则为参照，全未来最优的费用改善上限为 {100*certificates[0]['improvement_upper_fraction_vs_strong_causal']:.2f}%（0.5s）和 {100*certificates[1]['improvement_upper_fraction_vs_strong_causal']:.2f}%（0.7s）；在线策略只能争取其中可预测、可控制的部分。", "",
        "## 结论", "",
        "本轮纠正了基线遗漏和物理模型，没有继续 PPO。下一步应先冻结强因果预测基线并取得独立评估数据；若仍有可学空间，再使用输入驱动 baseline、16 个并行环境和至少 100 万步训练。验收必须同时看非恒定有效动作、费用和 SLA。", "",
        "[详细方法、五种子区间与复跑命令](REPORT.md) · [固定动作复核](../constant_action_audit/REPORT.md) · [输入驱动方差缩减方法](https://people.csail.mit.edu/hongzi/var-website/index.html)", ""]
    (output/"COURSE_REPORT.md").write_text("\n".join(course))
    print(f"Summarized {len(groups)} controls, {len(per_seed)} paired seed records; PPO not trained")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/physical_headroom")
    args = parser.parse_args()
    summarize(args.output)
