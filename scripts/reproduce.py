#!/usr/bin/env python3
"""Run all Section IV policies and phi sensitivity; write auditable outputs."""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from libra_sim import Config, simulate
from libra_sim.engine import POLICIES
from libra_sim.traces import load_trace


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "data/traces/wits_logs.txt")
    parser.add_argument("--seconds", type=int, default=43200)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/wits_12h")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    c = Config.read(args.config)
    counts, trace_meta = load_trace(args.trace, args.seconds, args.offset)
    trace_meta["path"] = str(Path(trace_meta["path"]).resolve().relative_to(ROOT)) if Path(trace_meta["path"]).resolve().is_relative_to(ROOT) else trace_meta["path"]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = {}
    for policy in POLICIES:
        result = simulate(counts, policy, c)
        results[policy] = result
        write_csv(args.output / f"{policy.lower()}_timeline.csv", result["timeline"])
        (args.output / f"{policy.lower()}_decisions.json").write_text(json.dumps(result["decisions"], indent=2) + "\n")
        s = result["summary"]
        print(f"{policy:6} cost=${s['total_cost_usd']:.6f} SLA={s['sla_violation_percent']:.4f}% "
              f"VM-hours={s['vm_uptime_seconds']/3600:.3f}", flush=True)
    sweep = []
    for phi in (-1, 0, 1, 2, 3, 4):
        s = results["LIBRA"]["summary"] if phi == c.phi else simulate(counts, "LIBRA", replace(c, phi=phi))["summary"]
        sweep.append(dict(s))
        print(f"phi={phi:2} cost=${s['total_cost_usd']:.6f}", flush=True)
    baseline = results["LIBRA"]["summary"]["total_cost_usd"]
    summaries = []
    for policy in POLICIES:
        s = results[policy]["summary"]
        s["normalized_cost"] = s["total_cost_usd"] / baseline if baseline else None
        summaries.append({k: v for k, v in s.items() if k not in ("config", "response_histogram")})
    write_csv(args.output / "summary.csv", summaries)
    write_csv(args.output / "phi_sweep.csv", [{"phi": s["config"]["phi"], **{k: v for k, v in s.items()
              if k not in ("config", "response_histogram")}} for s in sweep])
    source_hash = hashlib.sha256()
    for p in sorted((ROOT / "libra_sim").glob("*.py")):
        source_hash.update(p.name.encode() + p.read_bytes())
    revision = subprocess.run(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"],
                              capture_output=True, text=True)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                "platform": platform.platform(), "trace": trace_meta, "config": asdict(c),
                "price_derived_cip": c.price_derived_cip, "upstream_commit": revision.stdout.strip() or None,
                "simulator_source_sha256": source_hash.hexdigest(), "command": sys.argv,
                "deterministic": True, "elapsed_seconds": time.perf_counter() - started,
                "model": "paper-equations reconstruction; see docs/methodology.md"}
    payload = {"metadata": metadata, "policies": {p: results[p]["summary"] for p in POLICIES},
               "phi_sweep": sweep}
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    if not args.no_plots:
        from libra_sim.plotting import make_figures
        make_figures(results, sweep, args.output / "figures", c)
    write_report(args.output, payload)
    print(f"Results: {args.output.resolve()}", flush=True)


def write_report(out, payload):
    policies = payload["policies"]
    base = policies["LIBRA"]
    meta = payload["metadata"]
    lines = ["# LIBRA Simulation 复现结果", "", "仅运行本地仿真；未调用 AWS。",
             "", f"Trace: `{meta['trace']['path']}`；时长 {meta['trace']['duration_seconds']} 秒；"
             f"总请求 {base['total_requests']:,}；峰值 {meta['trace']['peak_requests_per_second']} req/s。",
             "", "| 策略 | 总成本 USD | 相对 LIBRA | SLA 违约 % | VM 小时 | FaaS 请求占比 % |",
             "|---|---:|---:|---:|---:|---:|"]
    for p in POLICIES:
        s = policies[p]
        lines.append(f"| {p} | {s['total_cost_usd']:.4f} | {s['normalized_cost']:.3f} | "
                     f"{s['sla_violation_percent']:.4f} | {s['vm_uptime_seconds']/3600:.3f} | "
                     f"{100*s['faas_requests']/s['total_requests'] if s['total_requests'] else 0:.2f} |")
    lines += ["", "相较各基线，LIBRA 的成本降低：", ""]
    for p in ("FAAS", "MAX", "AUTO", "SPOCK"):
        cost = policies[p]["total_cost_usd"]
        saving = 100 * (1 - base["total_cost_usd"] / cost) if cost else 0
        lines.append(f"- {p}: {saving:.2f}%")
    best = min(payload["phi_sweep"], key=lambda s: s["total_cost_usd"])
    lines += ["", f"phi 扫描的最低成本为 phi={best['config']['phi']}，${best['total_cost_usd']:.4f}。",
              "", "## 与论文结果的关系", "",
              "论文 Section IV 报告 LIBRA 零 SLA 违约、相较 AUTO/Spock 约 15% 成本节省、"
              "相较 FaaS/MAX 节省超过一半，并在其设置下 phi=1 最优。以上是本次实际运行结果，"
              "不保证数值或最优 phi 与论文一致。AUTO/SPOCK 阈值取自公开代码，Spock 根据论文描述补全；"
              "初始状态、EWMA 更新顺序和计费修正等差异见 `../../docs/methodology.md`。",
              "", "## 图表", ""]
    for name in ("fig3_trace_ewma", "fig4_policy_comparison", "fig5_uptime_cost_breakdown", "fig6_phi_sensitivity"):
        if (out / "figures" / f"{name}.png").exists():
            lines += [f"![{name}](figures/{name}.png)", ""]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
