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
        trace_path = ROOT / "data/traces" / Path(metadata["trace"]["path"]).name
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
    validation = ROOT / "outputs/validation"
    if (validation / "REPORT.md").exists():
        verify_validation(validation)


def verify_validation(out):
    a = json.loads((out / "p0_max/results.json").read_text())
    b = json.loads((out / "p1_ewma/results.json").read_text())
    c = json.loads((out / "p2_auto/results.json").read_text())
    fingerprint = hashlib.sha256()
    for parent in ("libra_sim", "libra_validation"):
        for path in sorted((ROOT / parent).glob("*.py")):
            fingerprint.update(str(path.relative_to(ROOT)).encode() + path.read_bytes())
    for result in (a, b, c):
        assert result["metadata"]["validation_source_sha256"] == fingerprint.hexdigest()
        assert result["metadata"]["runner_sha256"] == hashlib.sha256((ROOT / "scripts/validate_reproduction.py").read_bytes()).hexdigest()
    with (out / "p0_max/all_12h_windows.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1801 == a["window_count"]
    assert sorted({int(r["max_vms"]) for r in rows}) == [48, 73]
    assert a["paper_ratio_in_available_window_range"] is False
    assert len(b["runs"]) == 12
    assert all(r["sla_violation_percent"] == 0 for r in b["runs"])
    assert all(r["phi"] == 2 for r in b["integer_grid_minima"].values())
    assert b["identity_max_abs_error"] < 1e-9
    for r in b["equivalence_checks"]:
        assert r["allocated_vms_identical"] and r["vm_routing_identical"]
        close(r["cost_difference_usd"], 0)
        assert r["faas_request_difference"] == r["vm_uptime_difference_seconds"] == 0
    assert len(c["runs"]) == 8
    for name, run in c["runs"].items():
        assert run["total_requests"] == c["metadata"]["trace"]["total_requests"]
        reference = c["runs"]["w1_r0_b0" if run["fixes"]["window"] else "w0_r0_b0"]
        close(run["total_cost_usd"], reference["total_cost_usd"])
        assert run["response_histogram"] == reference["response_histogram"]
        assert run["telemetry"]["unbilled_vms"] == run["telemetry"]["pending_overwrite_events"] == 0
        assert run["telemetry"]["early_cancel_billing_calls"] == 0
    bridge, main = c["controller_bridge"], c["reconstructed_auto"]
    close(bridge["total_cost_usd"], main["total_cost_usd"])
    assert bridge["response_histogram"] == main["response_histogram"]
    for name in ("p1_ewma/cost_curves", "p2_auto/cumulative_fixes"):
        assert (out / f"{name}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        ET.parse(out / f"{name}.svg")
    print("Verified P0 1,801 windows, P1 12 runs + 6 equivalence checks, P2 8 combinations + controller bridge")


if __name__ == "__main__":
    main()
