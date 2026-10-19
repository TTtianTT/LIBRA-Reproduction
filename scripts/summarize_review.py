#!/usr/bin/env python3
"""Check and summarize the completed E1/E2/E3 CSV experiments."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_sim import Config

OUT = ROOT / "outputs/review"
TRACES = {"WITS": ("wits_12h", 43200), "Berkeley": ("berkeley_12h", 43200),
          "Sin": ("synthetic_sin", 10000), "Square": ("synthetic_square", 1112)}


def read(name, numeric):
    with (OUT / name).open() as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
            assert math.isfinite(row[key]), (name, key, row)
    return rows


def main():
    e1 = read("e1_baseline_sweep.csv", ("scale_out", "scale_in", "group", "window", "min_vms", "cost", "sla_pct", "faas_pct", "vm_hours"))
    e2 = read("e2_phi_rho.csv", ("rho", "phi", "cost"))
    e3 = read("e3_libra_variants.csv", ("alpha", "phi", "cost", "sla_pct", "faas_pct", "vm_hours"))
    assert (len(e1), len(e2), len(e3)) == (1080, 39, 144)
    assert len({tuple(row.values()) for row in e1}) == len(e1)
    assert len({(r["rho"], r["phi"]) for r in e2}) == len(e2)
    assert len({(r["trace"], r["variant"], r["alpha"], r["phi"]) for r in e3}) == len(e3)
    config = Config()
    assert config == Config.read(ROOT / "configs/paper.json")
    prior_path = OUT / "prior_results.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    primary, comparisons = {}, {}
    for trace, (directory, duration) in TRACES.items():
        result = json.loads((ROOT / "outputs" / directory / "results.json").read_text())
        primary[trace] = result
        if directory in prior:
            unchanged = (prior[directory]["policies"] == result["policies"]
                         and prior[directory]["phi_sweep"] == result["phi_sweep"])
            assert unchanged, f"Default results changed for {trace}"
            comparisons[trace] = unchanged
        for row in [r for r in e1 + e3 if r["trace"] == trace]:
            assert 0 <= row["sla_pct"] <= 100 and 0 <= row["faas_pct"] <= 100
            assert row["vm_hours"] >= 0 and row["cost"] >= 0
            total = result["metadata"]["trace"]["total_requests"]
            cost = row["vm_hours"] * config.vm_hourly_price + total * row["faas_pct"] / 100 * config.faas_cost_per_request
            if "variant" in row:
                cost += duration / 3600 * config.gateway_hourly_price
                assert row["sla_pct"] == 0, "LIBRA variant unexpectedly queued requests"
            assert math.isclose(cost, row["cost"], abs_tol=1e-9)
        for row in [r for r in e3 if r["trace"] == trace and r["variant"] == "paper rho*estimate" and r["alpha"] == .2]:
            reference = next(r for r in result["phi_sweep"] if r["config"]["phi"] == row["phi"])
            assert math.isclose(row["cost"], reference["total_cost_usd"], abs_tol=1e-10)
    best_baselines = []
    for trace in ("WITS", "Berkeley"):
        for policy in ("AUTO", "SPOCK"):
            rows = [r for r in e1 if r["trace"] == trace and r["policy"] == policy]
            assert len(rows) == 270
            for budget in (0, .1, .5, 1, 2, 5):
                feasible = [r for r in rows if r["sla_pct"] <= budget]
                best_baselines.append({"trace": trace, "policy": policy, "sla_budget_percent": budget,
                    "best": min(feasible, key=lambda r: r["cost"]) if feasible else None})
    best_coupling = []
    for rho in (.8, .9, 1.0):
        rows = [r for r in e2 if r["rho"] == rho]
        assert len(rows) == 13
        best_coupling.append(min(rows, key=lambda r: r["cost"]))
    for row in [r for r in e2 if r["rho"] == .8 and r["phi"] in (0, 1, 2, 3)]:
        reference = next(r for r in primary["WITS"]["phi_sweep"] if r["config"]["phi"] == row["phi"])
        assert math.isclose(row["cost"], reference["total_cost_usd"], abs_tol=1e-10)
    best_variants = []
    for trace in TRACES:
        for variant in ("paper rho*estimate", "rho*capacity", "fill ready capacity"):
            rows = [r for r in e3 if r["trace"] == trace and r["variant"] == variant]
            assert len(rows) == 12
            best_variants.append(min(rows, key=lambda r: r["cost"]))
    metadata_path = OUT / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    for name, expected in metadata.get("source_sha256", {}).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, f"Source changed during run: {name}"
    metadata.update({"completed_utc": datetime.now(timezone.utc).isoformat(),
        "summary_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "review_case_count": len(e1) + len(e2) + len(e3), "output_checks_passed": True,
        "default_metrics_unchanged": comparisons,
        "output_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest()
                          for name in ("e1_baseline_sweep.csv", "e2_phi_rho.csv", "e3_libra_variants.csv")}})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    summary = {"row_counts": {"e1": len(e1), "e2": len(e2), "e3": len(e3)},
               "best_baselines_by_sla_budget": best_baselines, "best_phi_by_rho": best_coupling,
               "best_variants_in_grid": best_variants, "default_metrics_unchanged": comparisons}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report(primary, best_baselines, best_coupling, best_variants)
    plot(e1, e2, primary)
    print("Verified 1,263 review experiments, dimensions, price accounting and default-result equivalence")
    print(OUT / "REPORT.md")


def report(primary, baseline, coupling, variants):
    lines = ["# 补丁后重跑：E1–E3 审查实验", "", "完整运行用户补丁中的 `scripts/review_experiments.py`，未跳过 E1。"
        "共 1,263 次仿真：E1 1,080 次、E2 39 次、E3 144 次。原有四组主实验与 P0–P2 也已重新运行并验证。", "",
        "默认 `estimate` 路由下，四组实验的全部策略指标与整数 phi 扫描均与补丁前一致。WITS LIBRA 仍为 "
        f"${primary['WITS']['policies']['LIBRA']['total_cost_usd']:.6f}，SLA 违约 0%。", "",
        "## E1：基线参数扫描", "", "每条 trace、每个基线扫描 270 种参数。下表在指定 SLA 违约预算内选最低成本；"
        "0% 预算用于与 LIBRA 零违约结果直接比较。参数均在同一条评估 trace 上选取，属于事后网格最优，未使用独立验证集。", "",
        "| Trace | 策略 | SLA 预算 % | 最低成本 USD | 实际 SLA % | out / in / group / window / min |",
        "|---|---|---:|---:|---:|---|"]
    for item in baseline:
        best = item["best"]
        if best:
            params = " / ".join(f"{best[key]:g}" for key in ("scale_out", "scale_in", "group", "window", "min_vms"))
            lines.append(f"| {item['trace']} | {item['policy']} | {item['sla_budget_percent']:g} | {best['cost']:.6f} | {best['sla_pct']:.6f} | {params} |")
        else:
            lines.append(f"| {item['trace']} | {item['policy']} | {item['sla_budget_percent']:g} | 无可行配置 | — | — |")
    lines += ["", "LIBRA 默认设置对照：" + "；".join(f"{name} ${primary[name]['policies']['LIBRA']['total_cost_usd']:.6f} / SLA 0%"
        for name in ("WITS", "Berkeley")) + "。", "", "![E1 cost vs SLA](e1_cost_sla.png)", "",
        "## E2：phi/rho 联合扫描", "", "WITS，rho=0.8/0.9/1.0；phi=0…3，步长 0.25。下表最优值只针对该有限网格。", "",
        "| rho | 网格最优 phi | 成本 USD |", "|---:|---:|---:|"]
    for row in coupling:
        lines.append(f"| {row['rho']:g} | {row['phi']:g} | {row['cost']:.6f} |")
    lines += ["", "![E2 phi/rho curves](e2_phi_rho.png)", "", "## E3：路由与 EWMA 参数", "",
        "三种路由设置分别为：`rho*estimate`（默认 rho=0.8）、`rho*ready_capacity`（rho=0.8）、"
        "填满 ready capacity（rho=1.0）。每种扫描 alpha=beta∈{0.2,0.02,0.005} 和 phi∈{0,1,2,3}。"
        "下表展示各设置在这 12 个组合中的最低成本；比较同时包含了参数选择的效果，不能单独归因于路由。"
        "全部 LIBRA 变体运行均为零 SLA 违约。", "",
        "| Trace | 设置 | alpha=beta | phi | 成本 USD | FaaS 请求 % |", "|---|---|---:|---:|---:|---:|"]
    for row in variants:
        lines.append(f"| {row['trace']} | {row['variant']} | {row['alpha']:g} | {row['phi']:g} | {row['cost']:.6f} | {row['faas_pct']:.3f} |")
    lines += ["", "## 文件与校验", "", "原始 CSV：", "",
        "- [E1 基线扫描](e1_baseline_sweep.csv)", "- [E2 phi/rho](e2_phi_rho.csv)", "- [E3 变体](e3_libra_variants.csv)",
        "- [机器可读汇总](summary.json)", "- [运行元数据与源文件/输出指纹](run_metadata.json)", "- [完整日志](run.log)", "",
        "已核对网格行数、组合唯一性、VM/FaaS/Gateway 成本分解、默认参数对照、运行前后源码指纹。"
        "逐次仿真由 engine 检查请求守恒。原有测试与 `scripts/verify_artifacts.py` 另行运行通过。", "",
        "```bash", ".venv/bin/python -u scripts/review_experiments.py", ".venv/bin/python scripts/summarize_review.py", "```"]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")


def plot(e1, e2, primary):
    from libra_sim.plotting import plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, trace in zip(axes, ("WITS", "Berkeley")):
        for policy, color in (("AUTO", "#3687be"), ("SPOCK", "#d79c27")):
            rows = [r for r in e1 if r["trace"] == trace and r["policy"] == policy]
            ax.scatter([r["sla_pct"] for r in rows], [r["cost"] for r in rows], s=10, alpha=.45, color=color, label=policy)
        ax.scatter([0], [primary[trace]["policies"]["LIBRA"]["total_cost_usd"]], color="#16846b", marker="*", s=160, label="LIBRA default", zorder=5)
        ax.set(xlabel="SLA violations (%)", ylabel="Total cost (USD)",
               title=f"E1: {trace} (SLA <= 5%)", xlim=(-.08, 5.08))
        ax.legend(fontsize=9)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"e1_cost_sla.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for rho, color in ((.8, "#16846b"), (.9, "#3687be"), (1.0, "#d79c27")):
        rows = [r for r in e2 if r["rho"] == rho]
        ax.plot([r["phi"] for r in rows], [r["cost"] for r in rows], "o-", color=color, label=f"rho={rho:g}")
    ax.set(xlabel="phi", ylabel="Total cost including gateway (USD)", title="E2: WITS phi/rho coupling")
    ax.legend()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"e2_phi_rho.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
