#!/usr/bin/env python3
"""Validate the chronological Azure experiment and generate its report/figures."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_sim.plotting import plt

OUT = ROOT / "outputs/ml_azure2021"
POLICIES = ("paper", "fill_paper", "fill_tuned", "fill_window_quantile", "fill_previous_day_quantile",
            "fill_gbdt_lag", "fill_gbdt_calendar", "oracle")


def main():
    data = json.loads((OUT / "results.json").read_text())
    meta = data["metadata"]
    c = meta["config"]
    for path, expected in meta["source_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    assert len(data["rows"]) == 16 and len(data["daily"]) == 112
    assert len({(r["load_multiplier"], r["policy"]) for r in data["rows"]}) == 16
    stats = []
    for factor in (1, 64):
        rows = {r["policy"]: r for r in data["rows"] if r["load_multiplier"] == factor}
        assert set(rows) == set(POLICIES)
        for name, row in rows.items():
            assert row["total_requests"] == sum(meta["dataset"]["daily_requests"][7:]) * factor
            assert row["sla_violation_percent"] == 0
            assert math.isclose(row["total_cost_usd"], row["vm_cost_usd"] + row["faas_cost_usd"] + row["gateway_cost_usd"], abs_tol=1e-8)
            daily = [r for r in data["daily"] if r["policy"] == name and r["load_multiplier"] == factor]
            assert len(daily) == 7 and sorted(r["test_day"] for r in daily) == list(range(8, 15))
            assert math.isclose(sum(r["total_cost_usd"] for r in daily), row["total_cost_usd"], abs_tol=1e-8)
        for baseline in ("fill_tuned", "fill_window_quantile"):
            calendar = rows["fill_gbdt_calendar"]["total_cost_usd"]
            base = rows[baseline]["total_cost_usd"]
            days = {(r["policy"], r["test_day"]): r["total_cost_usd"] for r in data["daily"] if r["load_multiplier"] == factor}
            stats.append({"load_multiplier": factor, "baseline": baseline,
                          "calendar_gbdt_cost_change_percent": 100 * (calendar / base - 1),
                          "calendar_gbdt_cheaper_days": sum(days["fill_gbdt_calendar", day] < days[baseline, day] for day in range(8, 15))})
    (OUT / "summary.json").write_text(json.dumps({"checks_passed": True, "comparisons": stats}, indent=2) + "\n")
    lines = ["# Azure Functions 2021：两周真实调用 trace 的独立 ML 验证", "",
        "**本次结果没有支持将正弦合成 trace 上的收益推广到真实日周期负载。** "
        "固定模型与协议下，带日/周相位和前一天上下文的 GBDT 在原始及 64 倍负载上均未超过调参 EWMA 或过去 300 秒分位数规则。"
        "这不证明所有 ML 都无效；也不能把‘存在日周期’当成‘本模型必然有收益’的充分条件。", "",
        "## 数据与协议", "",
        f"- [微软官方 2021 专页]({meta['dataset']['documentation']})；数据与归属见 [SOURCE.md](../../data/azure2021/SOURCE.md)。",
        f"- 共 {meta['dataset']['source_rows']:,} 条调用、1,209,600 秒；全部函数统一聚合，没有按测试效果选应用。空闲秒保持为零。",
        "- 专页明确时间单位为秒；到达时间为 `floor(end_timestamp-duration)`。没有把计数均匀摊到分钟，也没有拉伸时间轴。",
        "- 第一周训练与调参，第二周完整留出测试；测试周不重训、不选模型超参。每个策略在测试周起点从空 VM 状态开始，连续运行七天。",
        "- 过去需求特征对模型与窗口/前一天分位数规则均可使用第一周历史；论文/调参 EWMA 按原 engine 在测试周初始化。窗口和 ML 对照共享这种可用历史。",
        "- 原始聚合均值 1.638 req/s，低于 CIP=4。因此同时预先固定 raw×1 和调用量×64 两个场景；×64 是同一到达秒的负载复制，不是真实观测到的请求量，也未根据测试结果选择倍数。",
        f"- 仅第一周的小时计数 24h 滞后相关为 {meta['dataset']['train_only_structure']['first_7_days_hourly_lag24h_correlation']:.3f}，分钟计数为 {meta['dataset']['train_only_structure']['first_7_days_minute_lag24h_correlation']:.3f}，日周期并非完美稳定。",
        "- 微软说明时间戳经过修改；时钟特征使用相对日/周相位，不声称恢复了真实生产时区或当地工作日。",
        "- Azure duration 仅用于还原到达时间。所有请求的模拟服务时间仍为 1 秒、内存 512 MB，属于 LIBRA 的统一服务假设。", "",
        "## 模型与对照", "",
        "GBDT 超参沿用 ML 扩展：300 轮、15 叶、学习率 0.05、叶最小样本 200、L2=1、seed=0；tau 根据价格固定。"
        "训练时每 300 秒取决策点，预测其后 [100,400) 秒内每 10 秒采样的需求，所有目标严格位于训练周。"
        "lag 模型使用原 21 个因果特征；calendar 模型追加相对日/周 sin/cos、前一天对应服务窗口的分位数/均值及是否已有日历史。"
        "因此 lag→calendar 对照检验的是这组季节上下文的联合作用，不单独归因于时刻特征。", "",
        "推断采用预先向量化的因果特征；已将测试周起点、中点、末点与严格历史前缀重新计算的值对照，完全一致。"
        "前一天规则只读取已经发生的前一日相位，oracle 明确读取未来，仅作参考且不构成严格下界。", "",
        "EWMA 调参网格：alpha=beta∈{0.2,0.05,0.02,0.005}、phi∈{0,0.5,1,1.5,2,3}，只最小化第一周成本。", "",
        "## 第二周模拟成本", "", "全部策略 SLA 违约为 0%，由固定服务时间和容量路由保证，不能当作 ML 的性能收益。", "",
        "| 策略 | 原始负载 USD | ×64 负载 USD |", "|---|---:|---:|"]
    for policy in POLICIES:
        row = [next(r for r in data["rows"] if r["policy"] == policy and r["load_multiplier"] == factor) for factor in (1, 64)]
        lines.append(f"| {policy} | {row[0]['total_cost_usd']:.6f} | {row[1]['total_cost_usd']:.6f} |")
    lines += ["", "| 场景 | calendar GBDT 相对基线 | 成本变化 | 更便宜的测试日数 |", "|---|---|---:|---:|"]
    for row in stats:
        lines.append(f"| ×{row['load_multiplier']} | {row['baseline']} | {row['calendar_gbdt_cost_change_percent']:+.2f}% | {row['calendar_gbdt_cheaper_days']}/7 |")
    lines += ["", "![Azure weekly cost](weekly_costs.png)", "", "![Azure trace](trace_days.png)", "",
        "## 结论边界", "",
        "本实验保留 ML 未获益的结果。与训练周相比，测试周流量分布也存在变化；没有额外归因实验，不能断言失败只来自某一个原因。"
        "只有一个聚合序列和一次两周时间切分，日级结果用于展示稳定性，不声称有独立重复样本的统计显著性。"
        "后续若设计新模型或挑选新应用，应重新保留未参与开发的测试数据，不能反复依据本次测试周调参。", "",
        "## 复现", "", "```bash", ".venv/bin/python scripts/prepare_azure_2021.py --download",
        ".venv/bin/python scripts/azure_ml_experiments.py", ".venv/bin/python scripts/summarize_azure_ml.py", "```", "",
        "[结构化结果](results.json) / [策略表](results.csv) / [逐日费用](daily_costs.csv) / [运行日志](run.log)。"
        "原始 RAR/CSV 本地保存并被 Git 忽略；归档、CSV、聚合数组的 SHA-256 见数据 manifest。"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    labels = ["Paper", "Fill EWMA", "Tuned EWMA", "Window quantile", "Previous day", "GBDT lag", "GBDT + calendar", "Oracle"]
    for ax, factor in zip(axes, (1, 64)):
        values = [next(r["total_cost_usd"] for r in data["rows"] if r["policy"] == p and r["load_multiplier"] == factor) for p in POLICIES]
        ax.barh(range(len(values)), values, color=["#8798a4"] * 5 + ["#eb6834", "#cb4f32", "#16846b"])
        ax.set(yticks=range(len(values)), yticklabels=labels, xlabel="Held-out week cost (USD)", title=f"Azure 2021: load x{factor}")
        ax.invert_yaxis()
        ax.bar_label(ax.containers[0], fmt="%.2f", padding=3, fontsize=8)
        ax.set_xlim(right=max(values) * 1.15)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"weekly_costs.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    counts = np.load(ROOT / "data/azure2021/aggregate_counts.npz")["counts"]
    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    ax.plot(np.arange(len(counts) // 3600) / 24, counts.reshape(-1, 3600).mean(axis=1), color="#3687be", lw=1)
    ax.axvline(7, color="#cb4f32", ls="--", label="Train/test boundary")
    ax.set(xlabel="Relative day", ylabel="Raw mean requests / second", title="Azure 2021 aggregate: hourly means, no load scaling")
    ax.legend()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"trace_days.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Verified 2 chronological scenarios, 16 policy runs, 112 daily costs and source fingerprints")
    print(OUT / "REPORT.md")


if __name__ == "__main__":
    main()
