from dataclasses import replace
import math

import pytest

from libra_sim import Config, simulate
from libra_sim.engine import Monitor, libra_target
from libra_sim.traces import load_trace


def test_ewma_equations_use_updated_average():
    m = Monitor(0.2, 0.2)
    assert m.observe(10) == (10, 0)
    avg, dev = m.observe(20)
    assert avg == 12
    assert dev == pytest.approx(1.6)


def test_cip_and_negative_phi_target():
    c = Config()
    assert libra_target(3.9, 500, c) == (0, 0)
    assert libra_target(32, 0, c) == (2, 32)
    assert libra_target(10, 50, replace(c, phi=-1)) == (0, 0)
    assert c.vm_capacity == 16
    assert c.price_derived_cip == pytest.approx(3.2545726775)


def test_faas_exact_price_and_no_idle_vms():
    c = Config()
    s = simulate([10, 0, 30], "FAAS", c)["summary"]
    assert s["total_cost_usd"] == pytest.approx(40 * 8.535e-6)
    assert s["vm_uptime_seconds"] == s["sla_violations"] == 0
    assert s["response_histogram"] == {"1": 40}


def test_max_is_peak_sized_and_prewarmed():
    s = simulate([1, 33, 2], "MAX")["summary"]
    assert s["vm_uptime_seconds"] == 3 * 3
    assert s["sla_violations"] == 0
    assert s["total_cost_usd"] == pytest.approx(9 * 0.1 / 3600)


def test_starting_vm_is_billed_and_ready_exactly_at_startup_boundary():
    c = replace(Config(), auto_min_instances=1, vm_cold_start_seconds=2,
                auto_window_seconds=100)
    result = simulate([16, 0, 16], "AUTO", c)
    assert [r["ready_vms"] for r in result["timeline"]] == [0, 0, 1]
    s = result["summary"]
    assert s["response_histogram"] == {"2": 16, "3": 16}
    assert s["sla_violations"] == 32
    assert s["drain_seconds"] == 1
    assert s["vm_uptime_seconds"] == 4


def test_fifo_and_last_epoch_queue_are_fully_accounted():
    c = replace(Config(), auto_min_instances=1, vm_cold_start_seconds=0,
                auto_window_seconds=100)
    s = simulate([32, 16], "AUTO", c)["summary"]
    assert s["response_histogram"] == {"1": 16, "2": 32}
    assert s["total_requests"] == s["vm_requests"] == 48
    assert s["sla_violation_percent"] == pytest.approx(200 / 3)
    assert s["vm_uptime_seconds"] == 3


def test_libra_keeps_old_routing_until_new_vms_ready():
    c = replace(Config(), vm_cold_start_seconds=2, scaling_interval_seconds=4,
                alpha=1, beta=1, rho=0.8)
    r = simulate([20] * 4 + [100] * 4, "LIBRA", c)
    assert [x["vm_arrivals"] for x in r["timeline"]] == [0, 0, 16, 16, 16, 16, 80, 80]
    assert r["summary"]["sla_violations"] == 0
    assert r["summary"]["vm_uptime_seconds"] == 2 * 8 + 5 * 4


def test_libra_cip_shutdown_and_scale_in_billing():
    c = replace(Config(), vm_cold_start_seconds=0, scaling_interval_seconds=2,
                alpha=1, beta=1)
    r = simulate([20, 20, 1, 1], "LIBRA", c)
    assert [x["allocated_vms"] for x in r["timeline"]] == [2, 2, 0, 0]
    assert r["summary"]["vm_uptime_seconds"] == 4
    assert r["summary"]["gateway_cost_usd"] == pytest.approx(4 * 0.0116 / 3600)


def test_spock_only_offloads_during_provisioning():
    c = replace(Config(), auto_min_instances=1, auto_scale_group=1,
                vm_cold_start_seconds=2, auto_window_seconds=100)
    r = simulate([32, 32, 32], "SPOCK", c)
    assert [x["faas_requests"] for x in r["timeline"]] == [32, 32, 0]
    assert r["summary"]["sla_violations"] == 16


def test_autoscale_observation_window_and_instance_bounds():
    c = replace(Config(), auto_min_instances=1, auto_max_instances=2,
                auto_scale_group=10, vm_cold_start_seconds=0, auto_window_seconds=2)
    r = simulate([32] * 10 + [0] * 10, "AUTO", c)
    assert r["decisions"][0] == {"time_s": 2, "target_vms": 2, "utilization": 1.0}
    assert all(1 <= row["allocated_vms"] <= 2 for row in r["timeline"])
    assert r["timeline"][-1]["allocated_vms"] == 1


def test_runner_writes_all_result_files(tmp_path):
    import json
    from pathlib import Path
    import subprocess
    import sys

    trace = tmp_path / "trace.txt"
    trace.write_text("20\n0\n100\n")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts/reproduce.py"), "--trace", str(trace),
                    "--seconds", "3", "--output", str(tmp_path / "results"), "--no-plots"],
                   check=True, capture_output=True, text=True)
    payload = json.loads((tmp_path / "results/results.json").read_text())
    assert len(payload["policies"]) == 5
    assert len(payload["phi_sweep"]) == 6
    assert (tmp_path / "results/REPORT.md").exists()
    assert (tmp_path / "results/phi_sweep.csv").exists()


@pytest.mark.parametrize("policy", ["LIBRA", "FAAS", "MAX", "AUTO", "SPOCK"])
def test_conservation_on_bursty_trace(policy):
    counts = [0] * 10 + [450] * 40 + [2] * 20 + [900] * 10 + [0] * 50
    r = simulate(counts, policy)
    s = r["summary"]
    assert s["vm_requests"] + s["faas_requests"] == sum(counts)
    assert sum(s["response_histogram"].values()) == sum(counts)
    assert all(row["vm_arrivals"] + row["faas_requests"] == row["requests"] for row in r["timeline"])
    assert s["total_cost_usd"] == pytest.approx(
        s["vm_cost_usd"] + s["faas_cost_usd"] + s["gateway_cost_usd"])
    if policy in ("LIBRA", "FAAS", "MAX"):
        assert s["sla_violations"] == 0


@pytest.mark.parametrize("policy", ["LIBRA", "FAAS", "MAX", "AUTO", "SPOCK"])
def test_zero_demand_has_finite_zero_request_metrics(policy):
    s = simulate([0] * 10, policy)["summary"]
    assert s["total_requests"] == s["sla_violations"] == 0
    assert math.isfinite(s["mean_response_seconds"])


def test_trace_bounds_and_invalid_input(tmp_path):
    path = tmp_path / "trace.txt"
    path.write_text("1\n0\n3\n")
    assert load_trace(path, 2, 1)[0] == [0, 3]
    with pytest.raises(ValueError):
        load_trace(path, 4)
    with pytest.raises(ValueError):
        simulate([1, -1])
    with pytest.raises(ValueError):
        simulate([1.5])
    with pytest.raises(ValueError):
        Config(service_seconds=2)
    with pytest.raises(ValueError):
        Config(rho=math.nan)
