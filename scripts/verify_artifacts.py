#!/usr/bin/env python3
"""Validate generated results against their traces, timelines and pricing."""
import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-9), (actual, expected)


def main():
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    for name, meta in manifest["files"].items():
        path = ROOT / "data/traces" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"], path
        assert len(path.read_text().splitlines()) == meta["seconds"]
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "libra_sim").glob("*.py")):
        source_hash.update(path.name.encode() + path.read_bytes())
    for name in ("wits_12h", "berkeley_12h", "synthetic_sin", "synthetic_square"):
        out = ROOT / "outputs" / name
        results = json.loads((out / "results.json").read_text())
        metadata = results["metadata"]
        assert source_hash.hexdigest() == metadata["simulator_source_sha256"], "Rerun after source changes"
        trace_path = Path(metadata["trace"]["path"])
        assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == metadata["trace"]["sha256"]
        counts = [int(n) for n in trace_path.read_text().splitlines()]
        offset = metadata["trace"]["offset_seconds"]
        duration = metadata["trace"]["duration_seconds"]
        selected = counts[offset:offset + duration]
        c = metadata["config"]
        for policy, summary in results["policies"].items():
            total = summary["total_requests"]
            assert total == sum(selected) == summary["vm_requests"] + summary["faas_requests"]
            histogram = {float(k): v for k, v in summary["response_histogram"].items()}
            assert sum(histogram.values()) == total
            assert summary["sla_violations"] == sum(v for k, v in histogram.items() if k > c["sla_seconds"])
            close(summary["sla_violation_percent"], 100 * summary["sla_violations"] / total if total else 0)
            close(summary["vm_cost_usd"], summary["vm_uptime_seconds"] * c["vm_hourly_price"] / 3600)
            close(summary["faas_cost_usd"], summary["faas_requests"] * (
                c["request_memory_mb"] / 1024 * c["faas_gb_second_price"] + c["faas_request_price"]))
            close(summary["gateway_cost_usd"], duration * c["gateway_hourly_price"] / 3600 if policy == "LIBRA" else 0)
            close(summary["total_cost_usd"], sum(summary[k] for k in ("vm_cost_usd", "faas_cost_usd", "gateway_cost_usd")))
            routed, faas, served, uptime, queue, rows = 0, 0, 0, 0, 0, 0
            with (out / f"{policy.lower()}_timeline.csv").open() as stream:
                for t, row in enumerate(csv.DictReader(stream)):
                    n, v, f, s, q, ready, starting, allocated = (int(row[k]) for k in (
                        "requests", "vm_arrivals", "faas_requests", "vm_served", "queue_requests",
                        "ready_vms", "starting_vms", "allocated_vms"))
                    assert int(row["time_s"]) == t
                    assert n == selected[t] == v + f
                    assert ready + starting == allocated
                    assert 0 <= s <= ready * (c["vm_memory_mb"] // c["request_memory_mb"])
                    assert queue + v - s == q >= 0
                    routed += v
                    faas += f
                    served += s
                    uptime += allocated
                    queue = q
                    rows += 1
            assert rows == duration
            assert routed == summary["vm_requests"]
            assert faas == summary["faas_requests"]
            assert served + queue == summary["vm_requests"]
            assert summary["vm_uptime_seconds"] >= uptime
            if summary["drain_seconds"] == 0:
                assert queue == 0
                assert summary["vm_uptime_seconds"] == uptime
        assert len(results["phi_sweep"]) == 6
        for summary in results["phi_sweep"]:
            assert summary["vm_requests"] + summary["faas_requests"] == sum(selected)
            assert summary["sla_violations"] == 0
        print(f"Verified {name}: 5 policies, 6 phi settings, trace/cost/FIFO conservation")
    figures = ROOT / "outputs/wits_12h/figures"
    for name in ("fig3_trace_ewma", "fig4_policy_comparison", "fig5_uptime_cost_breakdown", "fig6_phi_sensitivity"):
        assert (figures / f"{name}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        ET.parse(figures / f"{name}.svg")
    replay = json.loads((ROOT / "outputs/upstream_replay.json").read_text())
    assert replay["upstream_commit"] == manifest["commit"]
    for result in replay["policies"].values():
        assert result["request_conservation_ok"]
    print("Verified 4 PNG + 4 SVG figures and upstream replay provenance")


if __name__ == "__main__":
    main()
