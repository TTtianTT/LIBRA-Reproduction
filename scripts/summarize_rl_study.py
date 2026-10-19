#!/usr/bin/env python3
"""Summarize the frozen PPO study without training or selecting on test data."""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from libra_rl.training import confidence_interval
from libra_sim.plotting import plt

OUT = ROOT / "outputs/rl_study"


def read(name):
    return json.loads((OUT / name).read_text())


def verify_sources(manifest):
    for path, digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise ValueError(f"study source changed: {path}; cannot summarize stale results")


def summaries(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["kind"] == "PPO":
            j = row["job"]
            grouped[(row["dataset"], row["latency_mode"], j["version"], j["service_seconds"],
                     j["variant"], j["penalty"], j["decision_seconds"])].append(row)
    result = []
    for key, group in grouped.items():
        seeds = sorted(r["job"]["seed"] for r in group)
        if seeds != [0, 1, 2, 3, 4]:
            raise ValueError(f"expected five independent seeds: {key}, {seeds}")
        dataset, mode, version, service, variant, penalty, interval = key
        summary = {"dataset": dataset, "latency_mode": mode, "version": version,
                   "service_seconds": service, "variant": variant, "penalty": penalty,
                   "decision_seconds": interval, "seeds": seeds}
        for metric in ("total_cost_usd", "violation_rate", "expected_violations", "vm_seconds",
                       "starting_vm_seconds", "cancelled_startup_vm_seconds"):
            mean, low, high = confidence_interval([r["summary"][metric] for r in group])
            summary[metric] = mean
            summary[metric + "_ci95"] = [low, high]
        result.append(summary)
    return result


def cost_ci(row):
    low, high = row["total_cost_usd_ci95"]
    return f"{row['total_cost_usd']:.4f} ± {(high-low)/2:.4f}"


def violation_ci(row):
    low, high = row["violation_rate_ci95"]
    return f"{100*row['violation_rate']:.4f} ± {100*(high-low)/2:.4f}"


def match(rows, **criteria):
    return [r for r in rows if all(r.get(k) == v for k, v in criteria.items())]


def single(rows, **criteria):
    found = match(rows, **criteria)
    if len(found) != 1:
        raise ValueError(f"expected one row: {criteria}, got {len(found)}")
    return found[0]


def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=160)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def plot_pareto(agg, rows, dataset="Azure16x", mode="wait_budget"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, service in zip(axes, (0.5, 0.7)):
        points = match(agg, dataset=dataset, latency_mode=mode, service_seconds=service, variant="main")
        points.sort(key=lambda r: r["penalty"])
        ax.plot([100*r["violation_rate"] for r in points], [r["total_cost_usd"] for r in points],
                "--", color="tab:blue", alpha=0.5)
        for point in points:
            x, y = 100*point["violation_rate"], point["total_cost_usd"]
            lo, hi = point["violation_rate_ci95"]
            lc, hc = point["total_cost_usd_ci95"]
            ax.errorbar(x, y, xerr=[[100*(point["violation_rate"]-max(0, lo))], [100*(hi-point["violation_rate"])]],
                        yerr=[[y-lc], [hc-y]], fmt="o", color="tab:blue", capsize=3,
                        label="PPO mean / 95% CI" if point is points[0] else None)
            ax.annotate(f"λ={point['penalty']:g}", (x, y), xytext=(4, 5), textcoords="offset points", fontsize=8)
        symbols = {"LIBRA-tuned": "s", "Quantile-tuned": "^", "GBDT-tuned": "D",
                   "AUTO-adapted-tuned": "v", "SPOCK-adapted-tuned": "P", "FaaS": "*"}
        for label, symbol in symbols.items():
            group = [r for r in rows if r["kind"] == "baseline" and r["dataset"] == dataset
                     and r["latency_mode"] == mode and r["service_seconds"] == service and r["label"] == label]
            unique = {(r["summary"]["violation_rate"], r["summary"]["total_cost_usd"]) for r in group}
            if unique:
                vx, cy = zip(*sorted(unique))
                ax.scatter(np.asarray(vx)*100, cy, marker=symbol, label=label, s=42, alpha=0.8)
        ax.axvline(0.1, color="black", linestyle=":", label="SLA budget 0.1%")
        ax.set(xlabel="Expected SLA violation (%)", ylabel="Total cost (USD)", title=f"{dataset}, service={service}s")
        ax.set_xscale("symlog", linthresh=0.01)
        ax.set_xlim(left=0)
        ax.grid(alpha=0.2)
    axes[1].legend(fontsize=7, loc="best")
    save_plot(fig, "pareto" if dataset == "Azure16x" and mode == "wait_budget" else f"pareto_{dataset}_{mode}")


def plot_learning(runs):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for column, service in enumerate((1.0, 0.5, 0.7)):
        selected = [r for r in runs if r["job"]["variant"] == "main" and r["job"]["service_seconds"] == service]
        for penalty in sorted({r["job"]["penalty"] for r in selected}):
            group = [r for r in selected if r["job"]["penalty"] == penalty]
            samples = [p["samples"] for p in group[0]["learning_curve"]]
            for ax, metric, multiplier in ((axes[0, column], "total_cost_usd", 1),
                                           (axes[1, column], "violation_rate", 100)):
                values = [[r["learning_curve"][i]["summary"][metric]*multiplier for r in group]
                          for i in range(len(samples))]
                intervals = np.array([confidence_interval(v) for v in values])
                ax.plot(samples, intervals[:, 0], marker="o", label=f"λ={penalty:g}")
                ax.fill_between(samples, intervals[:, 1], intervals[:, 2], alpha=0.15)
                ax.grid(alpha=0.2)
                ax.set_xlabel("Training decision samples")
                if metric == "violation_rate":
                    ax.set_ylim(bottom=0)
        axes[0, column].set_title(f"{'v1' if service == 1 else 'v2'}, service={service}s")
        axes[0, column].set_ylabel("Validation cost (USD)")
        axes[1, column].set_ylabel("Validation violation (%)")
        axes[0, column].legend(fontsize=8)
    save_plot(fig, "learning_curves")


def plot_ablation(agg, chosen):
    variants = ["main", "no_augmentation", "no_forecast", "raw_actions", "interval120", "interval300"]
    labels = ["Residual", "No aug.", "No GBDT", "Raw actions", "K=120", "K=300"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for column, service in enumerate((0.5, 0.7)):
        points = [single(agg, dataset="Azure16x", latency_mode="wait_budget", service_seconds=service,
                         variant=v, penalty=chosen[str(service)]) for v in variants]
        for row, metric, scale in ((0, "total_cost_usd", 1), (1, "violation_rate", 100)):
            ax = axes[row, column]
            ax.bar(np.arange(len(labels)), [p[metric]*scale for p in points],
                   yerr=[(p[metric+"_ci95"][1]-p[metric+"_ci95"][0])/2*scale for p in points], capsize=3)
            ax.set(xticks=np.arange(len(labels)), xticklabels=labels,
                   ylabel="Total cost (USD)" if row == 0 else "Violation (%)", title=f"service={service}s")
            ax.tick_params(axis="x", labelrotation=20)
            ax.set_ylim(bottom=0)
            if row == 1:
                ax.axhline(0.1, color="black", linestyle=":")
            elif column == 1 and max(p[metric] for p in points) > 5*np.median([p[metric] for p in points]):
                # Preserve the unstable raw-action seed in the full panel, and
                # show the remaining variants at a readable scale in an inset.
                inset = ax.inset_axes([0.06, 0.52, 0.38, 0.4])
                subset = [p for p in points if p["variant"] != "raw_actions"]
                inset.errorbar(np.arange(5), [p[metric] for p in subset],
                               yerr=[(p[metric+"_ci95"][1]-p[metric+"_ci95"][0])/2 for p in subset],
                               fmt="o", capsize=2, markersize=3)
                inset.set_xticks(np.arange(5), ["Res.", "No aug.", "No GBDT", "120", "300"], rotation=25, fontsize=6)
                inset.tick_params(axis="y", labelsize=6)
                inset.set_title("Other variants (USD)", fontsize=7)
    save_plot(fig, "ablations")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output
    manifest, final = read("manifest.json"), read("final_results.json")
    verify_sources(manifest)
    assert manifest["fingerprint"] == final["fingerprint"]
    index = read("runs_index.json")
    runs = [read(f"runs/{name}.json") for name in index["runs"]]
    assert len(runs) == 85
    rows, agg = final["rows"], summaries(final["rows"])
    chosen = read("ablation_selection.json")["penalties"]
    baseline_rows = [r for r in rows if r["kind"] == "baseline"]
    interval_records, comparisons = [], []
    epsilon = manifest["protocol"]["epsilon"]
    for service in (0.5, 0.7):
        point = single(agg, dataset="Azure16x", latency_mode="wait_budget", service_seconds=service,
                       variant="main", penalty=chosen[str(service)])
        candidates = [r for r in baseline_rows if r["dataset"] == "Azure16x" and r["latency_mode"] == "wait_budget"
                      and r["service_seconds"] == service and r["summary"]["violation_rate"] <= epsilon]
        cheapest = min(candidates, key=lambda r: r["summary"]["total_cost_usd"])
        upper_cost = point["total_cost_usd_ci95"][1]
        upper_violation = point["violation_rate_ci95"][1]
        supported = upper_cost < cheapest["summary"]["total_cost_usd"] and upper_violation <= epsilon
        comparisons.append({"service_seconds": service, "validation_selected_penalty": chosen[str(service)],
                            "ppo": point, "cheapest_test_feasible_baseline_descriptive_only": cheapest,
                            "strict_feasible_cost_advantage_supported_by_seed_CI": supported,
                            "caveat": "test-feasible baseline minimum is descriptive; never used to select a PPO checkpoint"})
    for r in agg:
        record = {k: v for k, v in r.items() if not isinstance(v, list)}
        for key, value in r.items():
            if key.endswith("_ci95"):
                record[key + "_low"], record[key + "_high"] = value
        interval_records.append(record)
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(interval_records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(interval_records)
    learning = [{"run_id": r["job"]["id"], "variant": r["job"]["variant"], "seed": r["job"]["seed"],
                 "service_seconds": r["job"]["service_seconds"], "penalty": r["job"]["penalty"],
                 "samples": p["samples"], "validation_cost_usd": p["summary"]["total_cost_usd"],
                 "validation_violation_rate": p["summary"]["violation_rate"],
                 "training_episode_return_mean": p["training_episode_return_mean"],
                 "fit_seconds": p["fit_seconds"]} for r in runs for p in r["learning_curve"]]
    with (OUT / "learning_curves.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(learning[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(learning)
    training_rows = [{"run_id": r["job"]["id"], "version": r["job"]["version"],
                      "service_seconds": r["job"]["service_seconds"], "variant": r["job"]["variant"],
                      "penalty": r["job"]["penalty"], "seed": r["job"]["seed"],
                      "decision_seconds": r["job"]["decision_seconds"], "samples": r["samples"],
                      "selected_checkpoint_samples": r["selected_step"], "fit_seconds": r["fit_seconds"],
                      "validation_seconds": r["evaluation_seconds"],
                      "simulated_training_seconds": r["simulated_training_seconds"]} for r in runs]
    with (OUT / "training_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(training_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(training_rows)
    metrics = {"fingerprint": manifest["fingerprint"], "grouped_seed_results": agg,
               "primary_comparisons": comparisons, "training_runs": len(runs),
               "training_samples": sum(r["samples"] for r in runs),
               "fit_seconds": sum(r["fit_seconds"] for r in runs),
               "validation_evaluation_seconds": sum(r["evaluation_seconds"] for r in runs),
               "test_rows": len(rows), "dp_bounds": final["dp_bounds"],
               "selected_at_final_checkpoint_count": sum(r["selected_step"] == r["samples"] for r in runs)}
    (OUT / "summary.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    lines = ["# PPO 本地有限预算实验：RL-1–4", "",
             "本报告由真实 SB3 PPO 训练、训练集基线扫描和冻结后的测试批次生成。未把规则或随机策略当成训练结果。", "",
             "## 先看主结论", ""]
    for comparison in comparisons:
        p = comparison["ppo"]
        status = "测试均值满足" if p["violation_rate"] <= epsilon else "测试均值未满足"
        advantage = ("种子置信区间支持相对已测可行基线的成本优势。" if comparison["strict_feasible_cost_advantage_supported_by_seed_CI"]
                     else "不能据此宣称 PPO 在严格 SLA 下优于强基线。")
        lines.append(f"- 服务 {comparison['service_seconds']} 秒，验证集选 λ={comparison['validation_selected_penalty']:g}："
                     f"Azure16x 测试成本 **{cost_ci(p)} USD**，预期违约 **{violation_ci(p)}%**；"
                     f"{status} 0.1% 约束。{advantage}")
    lines += ["", "以上 ± 是五个独立 PPO 种子的 95% Student-t 区间半宽，包含训练随机性，"
              "不代表工作负载总体的不确定性。静态基线和固定 GBDT 没有额外种子置信区间。", "",
              "## 协议与真实训练量", "",
              f"共 **{len(runs)} 次训练**、**{metrics['training_samples']:,} 个决策样本**；"
              f"PPO 拟合累计 {metrics['fit_seconds']:.1f} 秒，checkpoint 验证累计 {metrics['validation_evaluation_seconds']:.1f} 秒，"
              "均不包括预测器拟合、基线扫描、最终评估和 DP。v1 每种子 8,192 样本；v2 主实验和每种消融每种子 16,384 样本。", "",
              "训练：Azure 1–9 天；验证：10–11 天；最终测试：12–14 天。主场景是明确的 16 倍调用量复制，"
              "训练再作 0.5–2 倍幅度、时间伸缩及 Poisson 增强。另测原始 Azure1x 和 WITS/Berkeley/Sin。"
              "原始 Azure 12–14 天此前已被 ML 工作使用，不是项目从未查看的盲测数据。", "",
              "GBDT 固定 100 秒 lead、300 秒标签窗口；没有为 60 秒决策另调预测窗口。"
              "预测器与策略推理的计算成本未计入云费用，训练耗时单独报告。所有策略含相同 Gateway，"
              "本报告全 FaaS 参照也保留该费用。", "",
              "每个评估分区按不重叠、至多 6 小时的回合切分；每回合资源和特征历史从空状态开始。"
              "DP 与全部基线使用相同的冷启动边界。WITS/Berkeley 因此是两个独立 6 小时回合，"
              "不能直接将本表成本与 RL-0 连续 12 小时结果相减。训练回合随机起点可使用同一训练分区前 1,800 秒历史。", "",
              "PPO 为 SB3 MlpPolicy（64×64），γ=0.99、n_steps=512、batch=128、5 epochs、学习率 3e-4、"
              "熵系数 0.01；CPU 单线程。固定有符号 log 变换，无测试集归一化统计。有限回合结束设 terminated，"
              "不对清算之后的时间进行 TimeLimit bootstrap。四次已更新 checkpoint 只按完整验证段的成本+λ×违约选取。", "",
              f"共有 {metrics['selected_at_final_checkpoint_count']}/{len(runs)} 次训练选中了最后 checkpoint。"
              "训练预算预先固定；这不证明收敛，曲线平台或较早 checkpoint 也不是收敛证据。", "",
              "## RL-1：v1 与精确下限", "",
              "| 数据 | PPO 成本 USD（均值 ± 95% 半宽） | 分位数规则 USD | DP USD | 与 DP 差距缩小比例 |",
              "|---|---:|---:|---:|---:|"]
    for dataset in ("Azure16x", "Azure1x", "WITS", "Berkeley", "Sin"):
        point = single(agg, dataset=dataset, version="v1", variant="main", latency_mode="wait_budget")
        rule = next(r for r in baseline_rows if r["dataset"] == dataset and r["version"] == "v1" and r["label"] == "Quantile-reference")["summary"]["total_cost_usd"]
        bound = next(r for r in final["dp_bounds"] if r["dataset"] == dataset and r["version"] == "v1")["total_cost_usd"]
        if rule > bound+1e-12:
            gap = (rule-point["total_cost_usd"]) / (rule-bound)
            gap_low = (rule-point["total_cost_usd_ci95"][1]) / (rule-bound)
            gap_high = (rule-point["total_cost_usd_ci95"][0]) / (rule-bound)
            gap_text = f"{100*gap:.2f}% [{100*gap_low:.2f}, {100*gap_high:.2f}]"
        else:
            gap_text = "不可定义（规则已等于 DP）"
        lines.append(f"| {dataset} | {cost_ci(point)} | {rule:.4f} | {bound:.4f} | {gap_text} |")
    lines += ["", "差距缩小为 (规则−PPO)/(规则−DP)，负数表示 PPO 比规则更差，不作零截断。"
              "DP 允许任意目标 VM 数，残差 ±3 不是同一个动作域；DP 是乐观成本下限。", "",
              "## RL-2：v2 主实验与基线", "",
              "队列主假设 `wait_budget` 是 Erlang-C 等待尾 + 固定执行时间的混合近似；不是完整 M/M/16。"
              "服务 0.5/0.7 秒同时改变吞吐和 FaaS GB-second 费用。模型逐秒准稳态，没有实际队列积压。", "",
              "| 服务秒数 | λ | PPO 成本 USD | PPO 预期违约 % | SLA 0.1% |", "|---:|---:|---:|---:|---|"]
    for service in (0.5, 0.7):
        for penalty in manifest["protocol"]["penalties_usd_per_violation"]:
            p = single(agg, dataset="Azure16x", latency_mode="wait_budget", variant="main",
                       service_seconds=service, penalty=penalty)
            lines.append(f"| {service} | {penalty:g} | {cost_ci(p)} | {violation_ci(p)} | "
                         f"{'均值满足' if p['violation_rate']<=epsilon else '均值未满足'} |")
    lines += ["", "![Cost-violation diagnostics](pareto.png)", "",
              "虚线连接预先指定 λ 的均值点，仅帮助阅读；不是已评估的混合策略。误差棒为训练种子区间。"
              "图中的非负物理坐标截去区间的负下界；原始统计区间在 CSV/JSON 中保留。", "",
              "所有基线各用 12 个预先固定候选，在训练 1–9 天按每个 λ 的目标选参数。LIBRA 扫 rho/phi，"
              "分位数扫历史窗口/利用率上限，GBDT 扫利用率上限/预测倍率；AUTO/SPOCK 从 E1 原始阈值/组大小/窗口/"
              "最小实例网格预先抽取 12 组，不是穷尽整个 E1 网格。固定论文 LIBRA、默认规则和全 FaaS 也单独报告。", "",
              "PPO 每个种子为 3 个 λ×4 个验证 checkpoint；静态基线每族 12 个训练候选。候选数上限相同，"
              "训练计算量和验证查询次数不相同，因此本次不能声称严格等计算预算。调参账本全部保留，未在测试集重选参数。", "",
              "AUTO-adapted/SPOCK-adapted 是目标控制器适配：前者观察已路由 VM 利用率，后者观察包含溢出流量的需求压力。"
              "它们与 PPO 共用同一 FaaS 溢出、计费、启动和解析 SLA 模型，没有原论文 FIFO 队列语义。"
              "LIBRA 保留论文 300 秒时钟/estimate 路由；其余默认 60 秒。不能把这种适配比较写成对原实现的逐数值击败。", "",
              "| 服务秒数 | 基线 | λ | 成本 USD | 预期违约 % | 参数 |", "|---:|---|---:|---:|---:|---|"]
    for row in baseline_rows:
        if row["dataset"] == "Azure16x" and row["version"] == "v2" and row["latency_mode"] == "wait_budget":
            s = row["summary"]
            lines.append(f"| {row['service_seconds']} | {row['label']} | {row['penalty']} | {s['total_cost_usd']:.4f} | "
                         f"{100*s['violation_rate']:.4f} | {json.dumps(row['parameters'],ensure_ascii=False)} |")
    lines += ["", "## RL-3：泛化与增强", "",
              "下表固定使用主实验验证集选出的 λ，未按 OOD 结果重选。", "",
              "| 服务秒数 | 数据 | 训练增强 | 成本 USD | 预期违约 % |", "|---:|---|---|---:|---:|"]
    for service in (0.5, 0.7):
        for dataset in ("Azure16x", "Azure1x", "WITS", "Berkeley", "Sin"):
            for variant in ("main", "no_augmentation"):
                p = single(agg, dataset=dataset, latency_mode="wait_budget", service_seconds=service,
                           variant=variant, penalty=chosen[str(service)])
                lines.append(f"| {service} | {dataset} | {'是' if variant=='main' else '否'} | {cost_ci(p)} | {violation_ci(p)} |")
    lines += ["", "## RL-4：消融", "",
              "每种配置都跑 5 个独立种子；网络、样本预算、训练数据和 λ 固定。隐藏预测时仅清零预测与 mask，"
              "规则仍为窗口分位数，因此隔离的是状态中的预测信息。", "",
              "| 服务秒数 | 变体 | 成本 USD | 预期违约 % | 取消启动 VM 秒（均值） |", "|---:|---|---:|---:|---:|"]
    for service in (0.5, 0.7):
        for variant in ("main", "no_augmentation", "no_forecast", "raw_actions", "interval120", "interval300"):
            p = single(agg, dataset="Azure16x", latency_mode="wait_budget", service_seconds=service,
                       variant=variant, penalty=chosen[str(service)])
            lines.append(f"| {service} | {variant} | {cost_ci(p)} | {violation_ci(p)} | {p['cancelled_startup_vm_seconds']:.1f} |")
    lines += ["", "60/120/300 秒保持决策样本数相同，模拟训练秒数不同；γ 固定意味着物理时间折扣也不同。"
              "不能把结果差异全归因于扩缩容频率。原始动作增减的是当前分配 VM，"
              "残差动作相对于强规则。这些差别均是消融定义的一部分。", "",
              "原始动作的失败种子全部保留：服务 0.7 秒时出现过度扩容至实例上限，费用均值及区间因此显著增大。"
              "Student-t 区间可产生负下界，那是近似统计区间，不代表可能出现负费用。图中保留完整上界，"
              "并用插图展示其他变体，未删除失败种子或调整 VM 数来美化结果。", "", "![Ablations](ablations.png)", "",
              "## 延迟假设敏感性与 DP", "",
              "完整 `mmc_response` 引入指数服务尾部；复用已冻结策略，不重新调参。它与主模型有不同的低负载违约下限。", "",
              "| 服务秒数 | Azure16x 模型 | 成本 USD | 预期违约 % |", "|---:|---|---:|---:|"]
    for service in (0.5, 0.7):
        for mode in ("wait_budget", "mmc_response"):
            p = single(agg, dataset="Azure16x", latency_mode=mode, service_seconds=service,
                       variant="main", penalty=chosen[str(service)])
            lines.append(f"| {service} | {mode} | {cost_ci(p)} | {violation_ci(p)} |")
    lines += ["", "全 FaaS 在本模型中零违约，但残差 ±3 不保证能从高规则目标直接降到零；"
              "不能因为全 FaaS 可行就假定残差动作类也能达到它。真实 FaaS 冷启动及延迟仍未建模。", "",
              "v1 使用明确读取未来的离线精确成本下限；未来信息不进入训练或策略观测。"
              "v2 用 5 个 rho 和有限 VM 域优化成本+λ×预期违约，域上限包含所有已评估策略的实际最大目标，"
              "且至少覆盖 peak/(throughput×0.6)+8；它不是任意 VM 域/连续路由下的全局成本界。"
              "不同 λ 的 DP 成本不是带 epsilon 约束的最优成本，比较目标差距时必须包含 λ×违约。", "",
              "## 学习曲线与可复查文件", "", "![Learning curves](learning_curves.png)", "",
              "- `manifest.json`：平台、依赖、协议、源码/数据指纹。",
              "- `runs/*.json`：85 次训练的真实样本数、耗时、四个 checkpoint 验证指标与权重哈希。",
              "- `baseline_training_grid.json` / `selected_baselines.json`：180 个训练候选及参数选择。",
              "- `test_freeze.json`：打开最终测试段前冻结的权重、参数、λ。",
              "- `final_results.json` / `dp_bounds.json`：全部测试及离线 DP，不经过测试调参。",
              "- `summary.csv` / `summary.json` / `learning_curves.csv`：五种子区间和学习曲线。",
              "- `training_summary.csv`：每次训练的样本数、选中 checkpoint 样本数、拟合/验证耗时。",
              "- 本地 `models/` 保存 SB3/GBDT 权重，Git 忽略；重训可生成，ZIP 字节哈希不保证跨平台一致。", "",
              "复现入口：`.venv/bin/python scripts/rl_study.py --stage all`，"
              "汇总：`.venv/bin/python scripts/summarize_rl_study.py`，"
              "校验：`.venv/bin/python scripts/verify_rl_study.py`。", "",
              "新 clone 没有被忽略的权重，应指定新目录 `--output outputs/rl_study_replay` 重训；"
              "汇总/校验也传同一 `--output`。默认目录中的已发布记录不会被缺少权重的重跑静默覆盖。", "",
              "SB3 的 [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) 与 "
              "[向量环境/终止语义](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html)"
              "使用官方接口；环境与排队模型的推导见 [RL-0 说明](../../docs/rl0.md)。", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    plot_pareto(agg, rows)
    plot_pareto(agg, rows, "WITS")
    plot_pareto(agg, rows, "Berkeley")
    plot_pareto(agg, rows, "Azure16x", "mmc_response")
    plot_learning(runs)
    plot_ablation(agg, chosen)
    print(f"Summarized {len(runs)} trained policies, {len(rows)} test rows, {len(agg)} five-seed groups.")


if __name__ == "__main__":
    main()
