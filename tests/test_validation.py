from pathlib import Path

import pytest

from libra_sim import Config, simulate
from libra_sim.engine import Monitor
from libra_validation.upstream import load_upstream, run_auto

ROOT = Path(__file__).resolve().parents[1]


def test_deviation_identity_and_first_observation():
    new = Monitor(.2, .2)
    old = Monitor(.2, .2, deviation_update="previous_mean")
    for count in [10, 20, 0, 700, 1, 25, 40] * 20:
        avg_new, d_new = new.observe(count)
        avg_old, d_old = old.observe(count)
        assert avg_new == avg_old
        assert d_new == pytest.approx(.8 * d_old)
    with pytest.raises(ValueError):
        Monitor(.2, .2, deviation_update="unrecognized")


def test_phi_rescaling_keeps_vm_allocation_and_routing_identical():
    from dataclasses import replace
    counts = ([100, 300, 0, 200, 5] * 200)[:900]
    c = Config()
    old = simulate(counts, config=c, deviation_update="previous_mean")
    new = simulate(counts, config=replace(c, phi=1.25))
    assert old["summary"] == new["summary"] | {"config": old["summary"]["config"]}
    assert [r["vm_arrivals"] for r in old["timeline"]] == [r["vm_arrivals"] for r in new["timeline"]]


@pytest.fixture
def upstream_modules():
    if not (ROOT / "upstream/vm_cloud.py").exists():
        pytest.skip("Optional author-code tests require scripts/fetch_upstream.py")
    with load_upstream(ROOT / "upstream") as (modules, _):
        yield modules


def trace_file(tmp_path, counts):
    path = tmp_path / "counts.txt"
    path.write_text("\n".join(map(str, counts)) + "\n")
    return path


def test_bulk_observer_matches_original_placement(upstream_modules, tmp_path):
    counts = [32] * 10 + [0] * 5 + [80] * 10
    path = trace_file(tmp_path, counts)
    settings = {"min_num_instance": 1, "max_num_instance": 8, "auto_scale_group": 1,
                "vm_cold_start": 2, "auto_scale_grace_period": 4}
    original = run_auto(upstream_modules, path, len(counts), bulk_observer=False, overrides=settings)
    bulk = run_auto(upstream_modules, path, len(counts), overrides=settings)
    assert bulk == original


def test_window_fix_binds_average_but_preserves_original_others(upstream_modules, tmp_path):
    counts = [32] * 10 + [0] * 5 + [80] * 10
    path = trace_file(tmp_path, counts)
    settings = {"min_num_instance": 1, "max_num_instance": 8, "auto_scale_group": 1,
                "vm_cold_start": 2, "auto_scale_grace_period": 4}
    original = run_auto(upstream_modules, path, len(counts), overrides=settings)
    window = run_auto(upstream_modules, path, len(counts), window=True, overrides=settings)
    assert original["telemetry"]["max_average_utilization"] > 1
    assert window["telemetry"]["max_average_utilization"] <= 1
    assert original["total_requests"] == window["total_requests"] == sum(counts)


def test_billing_fix_triggers_on_early_cancellation_only(upstream_modules, tmp_path):
    counts = [32] * 10 + [0] * 5 + [80] * 10
    path = trace_file(tmp_path, counts)
    settings = {"min_num_instance": 1, "max_num_instance": 8, "auto_scale_group": 1,
                "vm_cold_start": 2, "auto_scale_grace_period": 4}
    original = run_auto(upstream_modules, path, len(counts), overrides=settings)
    fixed = run_auto(upstream_modules, path, len(counts), billing=True, overrides=settings)
    assert original["telemetry"]["early_cancel_billing_calls"] > 0
    assert fixed["vm_uptime_seconds"] < original["vm_uptime_seconds"]
    assert fixed["response_histogram"] == original["response_histogram"]
    assert fixed["scale_events"] == original["scale_events"]


def test_retirement_fix_releases_at_completion_boundary(upstream_modules, tmp_path):
    path = trace_file(tmp_path, [1, 1, 17, 0, 0])
    settings = {"min_num_instance": 2, "max_num_instance": 8, "auto_scale_group": 1,
                "vm_cold_start": 0, "auto_scale_grace_period": 2}
    original = run_auto(upstream_modules, path, 5, overrides=settings)
    fixed = run_auto(upstream_modules, path, 5, retirement=True, overrides=settings)
    assert original["vm_uptime_seconds"] == 9
    assert fixed["vm_uptime_seconds"] == 8
    assert fixed["response_histogram"] == original["response_histogram"]
    assert fixed["scale_events"] == original["scale_events"]


def test_controller_bridge_matches_full_reconstruction(upstream_modules, tmp_path):
    from dataclasses import replace
    counts = [32] * 12 + [0] * 8 + [80] * 10
    path = trace_file(tmp_path, counts)
    settings = {"min_num_instance": 1, "max_num_instance": 8, "auto_scale_group": 1,
                "vm_cold_start": 2, "auto_scale_grace_period": 4}
    bridge = run_auto(upstream_modules, path, len(counts), window=True, retirement=True, billing=True,
                      reconstructed_controller=True, overrides=settings)
    c = replace(Config(), auto_min_instances=1, auto_max_instances=8, auto_scale_group=1,
                vm_cold_start_seconds=2, auto_window_seconds=4)
    main = simulate(counts, "AUTO", c)["summary"]
    assert bridge["total_cost_usd"] == pytest.approx(main["total_cost_usd"])
    assert bridge["response_histogram"] == main["response_histogram"]
