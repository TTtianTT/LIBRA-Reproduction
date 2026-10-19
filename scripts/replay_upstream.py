#!/usr/bin/env python3
"""Replay the author's Python 2 simulator after syntax-only porting.

No AWS directory is loaded. Does not fix historical algorithm/accounting bugs.
Spock is absent upstream and is deliberately not reported by this script.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_sim.traces import load_trace


class Metrics(logging.Handler):
    def __init__(self):
        super().__init__()
        self.vm_requests = 0
        self.faas_requests = 0
        self.violations = 0
        self.vm_cost = 0.0
        self.faas_cost = 0.0
        self.vm_seconds = 0

    def emit(self, record):
        msg = record.getMessage()
        if "request served at |id,time,duration,queue_delay|" in msg:
            values = msg.rsplit("|", 1)[1].split(",")
            self.vm_requests += 1
            self.violations += float(values[2]) + float(values[3]) > 1
        elif "requests served by serverless at |requests,time,duration,cost|" in msg:
            values = msg.rsplit("|", 1)[1].split(",")
            self.faas_requests += int(values[0])
            self.faas_cost += float(values[3])
        elif "vm cost |duration(s), cost|" in msg:
            duration, cost = msg.rsplit("|", 1)[1].split(",")
            self.vm_seconds += int(duration)
            self.vm_cost += float(cost)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=43200)
    parser.add_argument("--trace", type=Path, default=ROOT / "data/traces/wits_logs.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/upstream_replay.json")
    args = parser.parse_args()
    counts, trace = load_trace(args.trace, args.seconds)
    source = ROOT / "upstream"
    if not (source / "simulator.py").exists():
        parser.error("Missing upstream checkout; run python scripts/fetch_upstream.py")
    source_hashes = {}
    results = {}
    with tempfile.TemporaryDirectory(prefix="libra-python3-") as tmp:
        for name in ("configuration", "traffic_generator", "vm_cloud", "serverless_cloud", "manager", "simulator"):
            original = (source / f"{name}.py").read_text()
            source_hashes[name] = hashlib.sha256(original.encode()).hexdigest()
            ported = original.replace("itertools.count().next", "itertools.count().__next__")
            ported = re.sub(r"(?m)^(\s*)print (epoch, .*?)$", r"\1print(\2)", ported)
            (Path(tmp) / f"{name}.py").write_text(ported)
        sys.path.insert(0, tmp)
        configuration = importlib.import_module("configuration")
        simulator = importlib.import_module("simulator")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        for policy in ("hybrid", "serverless", "vm_cloud_max", "vm_cloud"):
            config = configuration.configuration()
            config.policy = policy
            config.end_time = args.seconds
            config.log_file_name = str(args.trace.resolve())
            config.vm_auto_scaling = policy == "vm_cloud"
            if policy == "vm_cloud_max":
                # Explicit necessary experiment override: defaults use 292, but
                # Section IV requires provisioning for this trace's actual peak.
                config.max_num_instance = math.ceil(max(counts) / 16)
            handler = Metrics()
            root_logger.addHandler(handler)
            # Preserve upstream's verbose print output in a file, while keeping
            # the terminal focused on the four experiment summaries.
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.with_suffix(".stdout.log").open("a") as stdout_log:
                with redirect_stdout(stdout_log):
                    simulator.simulator(config).run()
            root_logger.removeHandler(handler)
            total = handler.vm_requests + handler.faas_requests
            results[policy] = {"total_requests": total, "vm_requests": handler.vm_requests,
                               "faas_requests": handler.faas_requests,
                               "sla_violation_percent": 100 * handler.violations / total if total else 0,
                               "vm_cost_usd": handler.vm_cost, "faas_cost_usd": handler.faas_cost,
                               "total_cost_without_gateway_usd": handler.vm_cost + handler.faas_cost,
                               "vm_uptime_seconds": handler.vm_seconds,
                               "request_conservation_ok": total == sum(counts)}
            print(policy, json.dumps(results[policy]), flush=True)
    revision = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
    payload = {"upstream_commit": revision.stdout.strip(), "source_sha256": source_hashes,
               "trace": trace, "port_changes": ["print statements to print()", "iterator .next to .__next__"],
               "experiment_overrides": ["end_time=selected duration", "absolute trace path",
                    "policy and vm_auto_scaling", "MAX instance count=ceil(trace peak/16)"],
               "caveat": "Historical bugs retained; gateway cost not added; MAX starts cold; Spock unavailable",
               "policies": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
