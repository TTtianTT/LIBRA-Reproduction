#!/usr/bin/env python3
"""P0 trace/MAX audit, P1 EWMA ablation, P2 upstream AUTO factorial fixes."""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_sim import Config, simulate
from libra_sim.engine import Monitor
from libra_sim.traces import load_trace
from libra_validation.upstream import load_upstream, run_auto


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def provenance(trace, config):
    source = hashlib.sha256()
    for parent in ("libra_sim", "libra_validation"):
        for path in sorted((ROOT / parent).glob("*.py")):
            source.update(str(path.relative_to(ROOT)).encode() + path.read_bytes())
    return {"created_utc": datetime.now(timezone.utc).isoformat(),
            "trace": trace, "config": asdict(config), "validation_source_sha256": source.hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "upstream_commit": subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip()}


def p0(out, config):
    full, meta = load_trace(ROOT / "data/traces/wits_logs.txt")
    prefix = [0]
    for count in full:
        prefix.append(prefix[-1] + count)
    windows = []
    for offset in range(len(full) - 43200 + 1):
        peak = max(full[offset:offset + 43200])
        machines = math.ceil(peak / config.vm_capacity)
        max_cost = machines * 12 * config.vm_hourly_price
        total = prefix[offset + 43200] - prefix[offset]
        faas_cost = total * config.faas_cost_per_request
        windows.append({"offset_seconds": offset, "duration_seconds": 43200, "peak_rps": peak,
                        "max_vms": machines, "total_requests": total, "max_cost_usd": max_cost,
                        "faas_cost_usd": faas_cost, "max_to_faas_cost_ratio": max_cost / faas_cost})
    table(out / "p0_max/all_12h_windows.csv", windows)
    samples = []
    for offset, duration in ((0, 43200), (1674, 43200), (1675, 43200), (1800, 43200), (0, 45000)):
        counts = full[offset:offset + duration]
        libra = simulate(counts, "LIBRA", config)["summary"]
        max_run = simulate(counts, "MAX", config)["summary"]
        samples.append({"offset_seconds": offset, "duration_seconds": duration, "peak_rps": max(counts),
                        "max_vms": math.ceil(max(counts) / config.vm_capacity), "max_cost_usd": max_run["total_cost_usd"],
                        "libra_cost_usd": libra["total_cost_usd"], "normalized_max_cost": max_run["total_cost_usd"] / libra["total_cost_usd"],
                        "max_sla_percent": max_run["sla_violation_percent"]})
    history = []
    revisions = subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-list", "--all"], text=True).splitlines()
    for revision in revisions:
        blobs = {}
        for name in ("configuration.py", "traces/wits_logs.txt"):
            result = subprocess.run(["git", "-C", str(ROOT / "upstream"), "rev-parse", f"{revision}:{name}"], capture_output=True, text=True)
            blobs[name] = result.stdout.strip() if result.returncode == 0 else None
        history.append({"revision": revision, "git_blobs": blobs})
    evidence = json.loads((ROOT / "references/paper_evidence.json").read_text())
    ratio = evidence["published"]["max_to_faas_cost_ratio"]
    first = windows[0]
    results = {"metadata": provenance(meta, config), "history": history, "paper_evidence": evidence,
               "window_count": len(windows), "peak_values_rps": sorted({r["peak_rps"] for r in windows}),
               "peak_epoch_zero_based": full.index(max(full)), "samples": samples,
               "all_window_max_faas_ratio_min": min(r["max_to_faas_cost_ratio"] for r in windows),
               "all_window_max_faas_ratio_max": max(r["max_to_faas_cost_ratio"] for r in windows),
               "paper_ratio_in_available_window_range": min(r["max_to_faas_cost_ratio"] for r in windows) <= ratio <= max(r["max_to_faas_cost_ratio"] for r in windows),
               "diagnostic_only": {
                   "implied_max_cost_from_paper_ratio_with_first_window_faas_usd": first["faas_cost_usd"] * ratio,
                   "implied_vm_hourly_price_if_73_vms": first["faas_cost_usd"] * ratio / (73 * 12),
                   "implied_faas_unit_price_if_73_vms": first["max_cost_usd"] / ratio / first["total_requests"],
                   "max_cost_with_extra_100s_boot_billing_usd": first["max_cost_usd"] + 73 * 100 * .1 / 3600,
                   "max_cost_with_upstream_default_292_vms_usd": 292 * 12 * .1},
               "conclusion": "Public trace/configuration and paper Figure 4a remain inconsistent. Exact WITS capture/time interval and figure-generating script are not disclosed. MAX instance sizing unchanged."}
    dump(out / "p0_max/results.json", results)
    table(out / "p0_max/sample_windows.csv", samples)
    print("P0 complete: all", len(windows), "12h windows audited; no MAX retuning", flush=True)
    return results


def p1(out, config):
    counts, meta = load_trace(ROOT / "data/traces/wits_logs.txt", 43200)
    runs, equivalent = [], []
    for mode in ("updated_mean", "previous_mean"):
        for phi in (-1, 0, 1, 2, 3, 4):
            result = simulate(counts, "LIBRA", replace(config, phi=phi), deviation_update=mode)
            summary = result["summary"]
            runs.append({"deviation_update": mode, "phi": phi, "total_cost_usd": summary["total_cost_usd"],
                         "vm_cost_usd": summary["vm_cost_usd"], "faas_cost_usd": summary["faas_cost_usd"],
                         "gateway_cost_usd": summary["gateway_cost_usd"], "sla_violation_percent": summary["sla_violation_percent"],
                         "vm_uptime_seconds": summary["vm_uptime_seconds"], "faas_requests": summary["faas_requests"]})
            if mode == "previous_mean":
                adjusted = phi / (1 - config.alpha)
                other = simulate(counts, "LIBRA", replace(config, phi=adjusted), deviation_update="updated_mean")
                equivalent.append({"previous_mean_phi": phi, "equivalent_updated_mean_phi": adjusted,
                    "cost_difference_usd": other["summary"]["total_cost_usd"] - summary["total_cost_usd"],
                    "vm_uptime_difference_seconds": other["summary"]["vm_uptime_seconds"] - summary["vm_uptime_seconds"],
                    "faas_request_difference": other["summary"]["faas_requests"] - summary["faas_requests"],
                    "allocated_vms_identical": [r["allocated_vms"] for r in result["timeline"]] == [r["allocated_vms"] for r in other["timeline"]],
                    "vm_routing_identical": [r["vm_arrivals"] for r in result["timeline"]] == [r["vm_arrivals"] for r in other["timeline"]]})
            print(f"P1 {mode} phi={phi}: ${summary['total_cost_usd']:.6f}", flush=True)
    updated = Monitor(config.alpha, config.beta)
    previous = Monitor(config.alpha, config.beta, deviation_update="previous_mean")
    error = 0.0
    for count in counts:
        _, d_new = updated.observe(count)
        _, d_old = previous.observe(count)
        error = max(error, abs(d_new - (1 - config.alpha) * d_old))
    assert error < 1e-9
    assert all(r["allocated_vms_identical"] and r["vm_routing_identical"]
               and abs(r["cost_difference_usd"]) < 1e-10 for r in equivalent)
    minima = {mode: min((r for r in runs if r["deviation_update"] == mode), key=lambda r: r["total_cost_usd"])
              for mode in ("updated_mean", "previous_mean")}
    payload = {"metadata": provenance(meta, config), "controlled_difference": "Only absolute-deviation mean reference",
               "runs": runs, "integer_grid_minima": minima, "equivalence_checks": equivalent,
               "identity": "d_updated[t] = (1-alpha) * d_previous[t] with common zero initialization and fixed alpha/beta",
               "identity_max_abs_error": error,
               "conclusion": "Both integer-grid minima are phi=2; update order alone does not reproduce the paper's phi=1 optimum."}
    dump(out / "p1_ewma/results.json", payload)
    table(out / "p1_ewma/cost_curves.csv", runs)
    return payload


def p2(out, config):
    counts, meta = load_trace(ROOT / "data/traces/wits_logs.txt", 43200)
    runs = {}
    with load_upstream(ROOT / "upstream") as (modules, hashes):
        for w, r, b in itertools.product((False, True), repeat=3):
            name = f"w{int(w)}_r{int(r)}_b{int(b)}"
            result = run_auto(modules, ROOT / "data/traces/wits_logs.txt", 43200, window=w, retirement=r, billing=b)
            assert result["total_requests"] == sum(counts), "Upstream request conservation failed"
            runs[name] = result
            print(f"P2 {name}: ${result['total_cost_usd']:.6f}, SLA={result['sla_violation_percent']:.6f}%", flush=True)
            dump(out / "p2_auto" / f"{name}.json", result)
        bridge = run_auto(modules, ROOT / "data/traces/wits_logs.txt", 43200, window=True, retirement=True, billing=True,
                          reconstructed_controller=True)
    baseline = runs["w0_r0_b0"]
    reference = json.loads((ROOT / "outputs/upstream_replay.json").read_text())["policies"]["vm_cloud"]
    assert math.isclose(baseline["total_cost_usd"], reference["total_cost_without_gateway_usd"], abs_tol=1e-10)
    assert baseline["sla_violation_percent"] == reference["sla_violation_percent"]
    reconstructed = simulate(counts, "AUTO", config)["summary"]
    assert bridge["total_requests"] == reconstructed["total_requests"]
    assert math.isclose(bridge["total_cost_usd"], reconstructed["total_cost_usd"], abs_tol=1e-10), (bridge["total_cost_usd"], reconstructed["total_cost_usd"])
    assert bridge["sla_violations"] == reconstructed["sla_violations"]
    assert bridge["response_histogram"] == reconstructed["response_histogram"]
    rows = []
    for name, run in runs.items():
        rows.append({"variant": name, **run["fixes"], "total_cost_usd": run["total_cost_usd"],
                     "sla_violation_percent": run["sla_violation_percent"],
                     "cost_delta_vs_upstream_usd": run["total_cost_usd"] - baseline["total_cost_usd"],
                     "sla_delta_vs_upstream_percentage_points": run["sla_violation_percent"] - baseline["sla_violation_percent"],
                     **run["telemetry"]})
    table(out / "p2_auto/factorial.csv", rows)
    payload = {"metadata": provenance(meta, config), "upstream_source_sha256": hashes, "runs": runs,
               "baseline_matches_historical_replay": True,
               "factor_definitions": {"window": "Only replace sum(hist[-300:])/120 with mean(hist[-120:]); preserve strict >120 gate and no-op resets",
                    "retirement": "Preserve pending removals, release at epoch>=last_used, and flush unbilled retiring VMs at end",
                    "billing": "Actual completion/release epoch minus VM creation epoch, rather than max(0,epoch-ready)+100"},
               "cumulative_path": ["w0_r0_b0", "w1_r0_b0", "w1_r1_b0", "w1_r1_b1"],
               "controller_bridge": bridge, "reconstructed_auto": reconstructed,
               "bridge_changes": ["gate >=120 instead of >120", "reset history only on an actual scaling change",
                                  "apply scale-out at next epoch boundary", "strict min/max clamps", "disable controller during final queue drain",
                                  "align all retained VM billing with final request completion"],
               "bridge_matches_reconstruction_full_histogram": True,
               "remaining_model_choice_cost_delta_usd": reconstructed["total_cost_usd"] - runs["w1_r1_b1"]["total_cost_usd"],
               "remaining_model_choice_sla_delta_percentage_points": reconstructed["sla_violation_percent"] - runs["w1_r1_b1"]["sla_violation_percent"]}
    dump(out / "p2_auto/results.json", payload)
    return payload


def plots(out):
    # Reuse the project's filesystem-safe, headless Matplotlib setup.
    from libra_sim.plotting import plt
    data = json.loads((out / "p1_ewma/results.json").read_text())
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for mode, label, color in (("updated_mean", "Deviation uses updated mean", "#16846b"),
                               ("previous_mean", "Deviation uses previous mean", "#3687be")):
        rows = [r for r in data["runs"] if r["deviation_update"] == mode]
        ax.plot([r["phi"] for r in rows], [r["total_cost_usd"] for r in rows], "o-", label=label, color=color)
    ax.set(xlabel="phi (all other settings fixed)", ylabel="Total cost including gateway (USD)",
           xticks=[-1, 0, 1, 2, 3, 4], title="P1: EWMA deviation update ablation, WITS 12h")
    ax.legend()
    for ext in ("png", "svg"):
        fig.savefig(out / f"p1_ewma/cost_curves.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    data = json.loads((out / "p2_auto/results.json").read_text())
    variants = [data["runs"][name] for name in data["cumulative_path"]] + [data["controller_bridge"]]
    labels = ["Original", "+ Window", "+ Retirement", "+ Billing", "+ Controller choices"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, key, title, ylabel in ((axes[0], "total_cost_usd", "AUTO cost", "USD"),
                                 (axes[1], "sla_violation_percent", "AUTO SLA violations", "%")):
        ax.bar(range(5), [r[key] for r in variants], color=["#8798a4", "#3687be", "#3687be", "#3687be", "#16846b"])
        ax.set(xticks=range(5), xticklabels=labels, title=title, ylabel=ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=9)
        ax.set_ylim(top=max(r[key] for r in variants) * 1.15)
    for ext in ("png", "svg"):
        fig.savefig(out / f"p2_auto/cumulative_fixes.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def report(out):
    a = json.loads((out / "p0_max/results.json").read_text())
    b = json.loads((out / "p1_ewma/results.json").read_text())
    c = json.loads((out / "p2_auto/results.json").read_text())
    lines = ["# P0–P2 验证报告", "", "所有主对照固定为 WITS 前 43,200 秒、原有定价与冷启动；未修改 MAX 的峰值定规模规则。",
        "", "## P0：MAX 成本差异", "", "核对 arXiv v1 与 OpenBU 正式发表版，MAX/FaaS/LIBRA 成本柱的相对高度相同。"
        "通过 PDF 矢量柱形高度提取，MAX/LIBRA 约 2.3291，FaaS/LIBRA 约 2.1215，故 MAX/FaaS 约 1.09785。"
        "这些是图形读数，不是作者发布的原始数值。", "",
        "公开 WITS 的独特最高峰为第 1675 行（0 基时间 1674 秒）的 1168 req/s。Git 中所有包含 trace 的版本具有同一 blob。"
        "枚举 45,000 秒文件中的全部 1,801 个 12 小时窗口后，仅出现两种峰值：1168 和 759 req/s，即 73 和 48 台 VM。", "",
        "| Offset 秒 | 时长秒 | 峰值 req/s | MAX VM | MAX USD | LIBRA USD | MAX/LIBRA |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for s in a["samples"]:
        lines.append(f"| {s['offset_seconds']} | {s['duration_seconds']} | {s['peak_rps']} | {s['max_vms']} | {s['max_cost_usd']:.4f} | {s['libra_cost_usd']:.4f} | {s['normalized_max_cost']:.4f} |")
    lines += ["", f"更强的交叉检查：MAX/FaaS 不依赖 LIBRA 的实现和归一化分母。全部窗口的该比值范围为 "
              f"[{a['all_window_max_faas_ratio_min']:.6f}, {a['all_window_max_faas_ratio_max']:.6f}]，"
              "没有精确复现图中约 1.09785。最接近的后移窗口已将巨大偏差缩小，但不能据此认定论文使用了该窗口。", "",
              "主窗口的 MAX 为 73×12×$0.1=$87.60；额外计入 100 秒启动费只会增加约 $0.203。"
              "改用按小时进位也不能降低这段完整 12 小时的成本；上游默认 292 台反而是 $350.40。"
              "结合当前 FaaS 成本和图中比值，隐含 MAX 成本约 $58.26（详见 JSON 精确值），"
              "但没有公开依据把它解释为新的实例数或价格。", "",
              "**结论：保留 73 台、不调参匹配原图。** 公开材料未给出 WITS capture 标识、原始区间起止时间或 Figure 4 生成脚本；"
              "截至本次审计，具体根因仍未确定。原始 trace 之外的区间、图表版本或未公开的计费设置只能列为待验证因素。", "",
              "来源：[arXiv v1](https://arxiv.org/pdf/2104.05491)、"
              "[OpenBU 正式发表版](https://open.bu.edu/bitstreams/86e15fd8-9c9a-436b-9d16-d9072bf4a39f/download)、"
              "[作者代码](https://github.com/aliraza0337/LIBRA)。正式版把 phi 图合并为图 4d，原仓库主复现保留 arXiv 图 3–6 编号。", "",
              "## P1：EWMA 与 phi 单变量消融", "", "仅改变 deviation 的均值引用；定价、alpha/beta、rho、冷启动、初始化、CIP、路由、缩放和计费均保持一致。", "",
              "| phi | 使用更新后均值 USD | 使用更新前均值 USD |", "|---:|---:|---:|"]
    for phi in (-1, 0, 1, 2, 3, 4):
        pair = [next(r for r in b["runs"] if r["deviation_update"] == mode and r["phi"] == phi) for mode in ("updated_mean", "previous_mean")]
        lines.append(f"| {phi} | {pair[0]['total_cost_usd']:.6f} | {pair[1]['total_cost_usd']:.6f} |")
    lines += ["", "两组的整数扫描最优点均为 **phi=2**，且全部零 SLA 违约。**更新顺序单独不足以解释 1→2 的最优点差异。**", "",
              "在固定 alpha、相同初值下有恒等式 `d_updated=(1-alpha)*d_previous`。alpha=0.2 时，"
              "previous-mean 的 phi 等效于 updated-mean 的 1.25×phi。对六个值逐一进行了重参数化对照，"
              "实际 VM 路由、VM 数和成本完全一致（浮点恒等式误差单列于 JSON）。因此这里改变的是偏差尺度，"
              "不能仅凭旧/新均值次序就断言能回到论文的整数最优点。", "", "![EWMA ablation](p1_ewma/cost_curves.png)", "",
              "## P2：AUTO 三项修复的隔离", "", "在作者原始 engine 上独立切换 W（利用率窗口）、R（缩容清理）、B（计费），运行全部 2³=8 个组合。"
              "原始 Python 2 代码只做语法适配；批量日志观察器已与原始逐请求方法做一致性测试，"
              "未修复组精确复现历史成本与 SLA。", "",
              "W 只把 `sum(hist[-300:])/120` 改为最近 120 项均值；保留严格 `>120` 触发门槛、无动作时清空历史及其他原始控制行为。"
              "R 保留待回收 VM、允许在 `epoch==last_used` 时回收，并在结束清理未结算项。"
              "B 只把费用改为实际释放/完成时刻减申请时刻。", "",
              "| W | R | B | 成本 USD | SLA % | 相对原始成本变化 USD | SLA 变化百分点 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    base = c["runs"]["w0_r0_b0"]
    for r in c["runs"].values():
        lines.append(f"| {int(r['fixes']['window'])} | {int(r['fixes']['retirement'])} | {int(r['fixes']['billing'])} | "
                     f"{r['total_cost_usd']:.6f} | {r['sla_violation_percent']:.6f} | {r['total_cost_usd']-base['total_cost_usd']:+.6f} | "
                     f"{r['sla_violation_percent']-base['sla_violation_percent']:+.6f} |")
    lines += ["", "此 trace 上，R 和 B 的单独与交互影响均为 0；没有发生待回收列表覆盖、未结算 VM 或完整冷启动前取消计费。"
              "这说明它们不能解释此次 $33.924→约 $26 的差异，不表示相关代码在所有输入上正确。"
              "另有专门边界测试触发提前取消和恰好完成时缩容，验证修复并非无效开关。", "",
              "W 修复使成本下降，但 SLA 违约增加；不能把它称为成本和性能同时改善。原平均利用率曾达到 1.00833，修复后不超过 1。", "",
              "三项全修复后为 $26.006111 / 4.327891%；主重建版本为 $26.048333 / 4.137733%。"
              f"剩余差为成本 {c['remaining_model_choice_cost_delta_usd']:+.6f} USD、SLA {c['remaining_model_choice_sla_delta_percentage_points']:+.6f} 个百分点。"
              "将原 engine 的控制器另行切到主实现的门槛、无动作历史处理、决策生效时刻、严格边界、drain 行为及最后请求完成时的统一计费终点后，"
              "成本、请求数和**完整响应时间直方图**与主重建完全一致。这将剩余差异定位到这些控制/时间模型选择的联合影响；"
              "未把它归给单独一项修复。", "", "![AUTO fixes](p2_auto/cumulative_fixes.png)", "",
              "## 重现与文件", "", "```bash", ".venv/bin/python scripts/fetch_upstream.py",
              ".venv/bin/python scripts/validate_reproduction.py", "```", "",
              "主验证无需下载论文；`references/paper_evidence.json` 保存了两份 PDF 的来源、SHA-256、页号和矢量柱形读数。"
              "如需重新提取，可运行 `scripts/extract_paper_evidence.py --arxiv ... --published ...`（可选依赖 PyMuPDF）。"
              "本报告的原始 JSON、全部窗口 CSV、消融 CSV、修复组合和 PNG/SVG 均保存在同目录。"]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("all", "p0", "p1", "p2"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/validation")
    args = parser.parse_args()
    config = Config.read(ROOT / "configs/paper.json")
    for name, run in (("p0", p0), ("p1", p1), ("p2", p2)):
        if args.part in ("all", name):
            run(args.output, config)
    if all((args.output / name / "results.json").exists() for name in ("p0_max", "p1_ewma", "p2_auto")):
        plots(args.output)
        report(args.output)
    print("Validation output:", args.output, flush=True)


if __name__ == "__main__":
    main()
