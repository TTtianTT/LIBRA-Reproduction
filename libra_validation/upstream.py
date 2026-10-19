"""Run orthogonal fixes on the upstream AUTO engine, preserving other choices.

The bulk request observer replaces only vm_instance.place_requests logging and
executes the same occupancy updates. All other execution uses upstream classes.
This is checked against the original placement method and historical metrics.
"""
from collections import Counter
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib
import io
import logging
from pathlib import Path
import re
import sys
import tempfile
from unittest.mock import patch

NAMES = ("configuration", "traffic_generator", "vm_cloud", "serverless_cloud", "manager", "simulator")


@contextmanager
def load_upstream(source):
    hashes = {}
    with tempfile.TemporaryDirectory(prefix="libra-controlled-") as tmp:
        for name in NAMES:
            raw = (Path(source) / f"{name}.py").read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            text = raw.decode().replace("itertools.count().next", "itertools.count().__next__")
            text = re.sub(r"(?m)^(\s*)print (epoch, .*?)$", r"\1print(\2)", text)
            (Path(tmp) / f"{name}.py").write_text(text)
        previous_path = sys.path[:]
        previous_modules = {name: sys.modules.pop(name) for name in NAMES if name in sys.modules}
        try:
            sys.path.insert(0, tmp)
            modules = {name: importlib.import_module(name) for name in NAMES}
            yield modules, hashes
        finally:
            sys.path[:] = previous_path
            for name in NAMES:
                sys.modules.pop(name, None)
            sys.modules.update(previous_modules)


class CostObserver(logging.Handler):
    def __init__(self):
        super().__init__()
        self.seconds = 0
        self.cost = 0.0
        self.calls = 0
        self.max_average_utilization = 0.0

    def emit(self, record):
        message = record.getMessage()
        if "vm cost |duration(s), cost|" in message:
            duration, cost = message.rsplit("|", 1)[1].split(",")
            self.seconds += int(duration)
            self.cost += float(cost)
            self.calls += 1
        elif message.startswith("ave utilization at epoch:"):
            self.max_average_utilization = max(self.max_average_utilization, float(message.rsplit(",", 1)[1]))


def run_auto(modules, trace, seconds, *, window=False, retirement=False, billing=False,
             bulk_observer=True, overrides=None, reconstructed_controller=False):
    """Three flags change only their named upstream behavior; all 8 combinations work."""
    c = modules["configuration"].configuration()
    c.policy, c.vm_auto_scaling = "vm_cloud", True
    c.end_time = seconds
    c.log_file_name = str(Path(trace).resolve())
    if reconstructed_controller and not (window and retirement and billing):
        raise ValueError("Controller bridge requires all three fixes")
    for name, value in (overrides or {}).items():
        setattr(c, name, value)
    vm_module = modules["vm_cloud"]
    Instance, Cloud = vm_module.vm_instance, vm_module.vm_cloud
    original_init = Instance.__init__
    original_place = Instance.place_requests
    original_cost = Instance.calculate_cost
    original_scale = Cloud.auto_scale
    original_kill = Cloud.kill_vms
    original_finish = Cloud.calculate_cost
    registry = []
    histogram = Counter()
    events = []
    telemetry = {"early_cancel_billing_calls": 0, "pending_overwrite_events": 0,
                 "pending_overwrite_vms": 0, "max_allocated_vms": 0,
                 "out_of_bounds_epochs": 0, "scale_in_events": 0, "scale_out_events": 0}
    observer = CostObserver()

    def init(vm, start_time, config=None):
        original_init(vm, start_time, config)
        vm.audit_created = start_time - config.vm_cold_start
        vm.audit_bill_calls = 0
        registry.append(vm)

    def place(vm, requests, epoch):
        for arrival, count in Counter(r.arrival_time for r in requests).items():
            histogram[epoch - arrival + c.req_duration] += count
        if not bulk_observer:
            return original_place(vm, requests, epoch)
        # Same assignments and Counter increments as upstream, without one INFO
        # message per request. Called only with nonempty batches by upstream.
        for i in range(epoch, epoch + c.req_duration):
            vm.handling_requests[i] += len(requests)
        vm.last_used = epoch + c.req_duration

    def cost(vm, epoch):
        telemetry["early_cancel_billing_calls"] += epoch < vm.start_time
        vm.audit_bill_calls += 1
        if not billing:
            return original_cost(vm, epoch)
        while vm.handling_requests[epoch] > 0:
            epoch += 1
        duration = max(0, epoch - vm.audit_created)
        vm_module.logger.info('vm cost |duration(s), cost|{0}, {1}'.format(
            duration, duration * vm.instance_hourly_price / 3600))

    def fixed_window_scale(cloud, epoch):
        # Isolate ONLY the mean window: keep upstream's strict > grace gate,
        # thresholds, no-op history resets, group sizes and application epoch.
        u = cloud.calculate_utilization(epoch)
        cloud.utilization_hist.append(u)
        gate = len(cloud.utilization_hist) >= c.auto_scale_grace_period if reconstructed_controller else len(cloud.utilization_hist) > c.auto_scale_grace_period
        if gate:
            recent = cloud.utilization_hist[-c.auto_scale_grace_period:]
            average = sum(recent) / len(recent)
            vm_module.logger.info("ave utilization at epoch: {0}, {1}".format(epoch, average))
            if reconstructed_controller:
                before = len(cloud.active_vms)
                target = before
                if average > c.auto_scale_out_thresh:
                    target = min(c.max_num_instance, before + c.auto_scale_group)
                elif average < c.auto_scale_in_thresh:
                    target = max(c.min_num_instance, before - c.auto_scale_group)
                if target > before:
                    cloud.init_vms(epoch + 1, target - before)
                elif target < before:
                    cloud.vms_to_kill.extend(cloud.active_vms[target:])
                    cloud.active_vms = cloud.active_vms[:target]
                if target != before:
                    cloud.utilization_hist = []
                else:
                    cloud.utilization_hist = cloud.utilization_hist[-(c.auto_scale_grace_period - 1):] if c.auto_scale_grace_period > 1 else []
            elif average > c.auto_scale_out_thresh:
                if len(cloud.active_vms) < c.max_num_instance:
                    cloud.init_vms(epoch, c.auto_scale_group)
                cloud.utilization_hist = []
            if not reconstructed_controller and average < c.auto_scale_in_thresh:
                if len(cloud.active_vms) > c.auto_scale_group:
                    cloud.vms_to_kill = cloud.active_vms[-c.auto_scale_group:]
                    cloud.active_vms = cloud.active_vms[:-c.auto_scale_group]
                cloud.utilization_hist = []

    def scale(cloud, epoch):
        if reconstructed_controller and epoch > c.end_time:
            return
        previous_pending = list(cloud.vms_to_kill)
        before = len(cloud.active_vms)
        (fixed_window_scale if window else original_scale)(cloud, epoch)
        missing = [v for v in previous_pending if v not in cloud.vms_to_kill]
        if missing:
            telemetry["pending_overwrite_events"] += 1
            telemetry["pending_overwrite_vms"] += len(missing)
        if retirement:
            cloud.vms_to_kill = previous_pending + [v for v in cloud.vms_to_kill if v not in previous_pending]
        after = len(cloud.active_vms)
        if after != before:
            events.append({"epoch": epoch, "before": before, "after": after})
            telemetry["scale_in_events" if after < before else "scale_out_events"] += 1
        telemetry["max_allocated_vms"] = max(telemetry["max_allocated_vms"], after)
        telemetry["out_of_bounds_epochs"] += not (c.min_num_instance <= after <= c.max_num_instance)

    def kill(cloud, epoch):
        if not retirement:
            return original_kill(cloud, epoch)
        pending = []
        for vm in cloud.vms_to_kill:
            if epoch >= vm.last_used:
                vm.calculate_cost(epoch)
            else:
                pending.append(vm)
        cloud.vms_to_kill = pending

    def finish(cloud, epoch):
        if reconstructed_controller:
            # The original simulator stops when its queue becomes empty, possibly
            # while the last batch is still executing. Align every retained VM's
            # measurement end with the final completion, as the main model does.
            while any(vm.handling_requests[epoch] > 0 for vm in cloud.active_vms + cloud.vms_to_kill):
                epoch += 1
        original_finish(cloud, epoch)
        if retirement:
            for vm in cloud.vms_to_kill:
                if not vm.audit_bill_calls:
                    vm.calculate_cost(max(epoch, vm.last_used))

    vm_logger = vm_module.logger
    old_level, old_propagate, old_handlers = vm_logger.level, vm_logger.propagate, vm_logger.handlers[:]
    vm_logger.setLevel(logging.INFO)
    vm_logger.propagate = False
    vm_logger.handlers[:] = [observer]
    try:
        with patch.object(Instance, "__init__", init), patch.object(Instance, "place_requests", place), \
             patch.object(Instance, "calculate_cost", cost), patch.object(Cloud, "auto_scale", scale), \
             patch.object(Cloud, "kill_vms", kill), patch.object(Cloud, "calculate_cost", finish):
            with redirect_stdout(io.StringIO()):
                sim = modules["simulator"].simulator(c)
                sim.run()
    finally:
        vm_logger.setLevel(old_level)
        vm_logger.propagate = old_propagate
        vm_logger.handlers[:] = old_handlers
    total = sum(histogram.values())
    violations = sum(n for latency, n in histogram.items() if latency > 1)
    return {"fixes": {"window": window, "retirement": retirement, "billing": billing},
            "reconstructed_controller": reconstructed_controller,
            "total_requests": total, "sla_violations": violations,
            "sla_violation_percent": 100 * violations / total if total else 0,
            "vm_uptime_seconds": observer.seconds, "total_cost_usd": observer.cost,
            "response_histogram": {str(k): v for k, v in sorted(histogram.items())},
            "telemetry": {**telemetry, "max_average_utilization": observer.max_average_utilization,
                          "created_vms": len(registry), "billed_vms": observer.calls,
                          "unbilled_vms": sum(not vm.audit_bill_calls for vm in registry),
                          "multiply_billed_vms": sum(vm.audit_bill_calls > 1 for vm in registry),
                          "pending_retirement_at_end": len(sim.manager.vm_cloud.vms_to_kill)},
            "scale_events": events,
            "scale_events_sha256": hashlib.sha256(repr(events).encode()).hexdigest()}
