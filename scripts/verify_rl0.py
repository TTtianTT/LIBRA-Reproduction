#!/usr/bin/env python3
"""Check RL-0 source/data fingerprints and independently replay stored DP plans."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_rl.dp import evaluate_schedule
from libra_sim.traces import load_trace


def main():
    data = json.loads((ROOT / "outputs/rl0/results.json").read_text())
    for path, digest in data["source_sha256"].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != digest:
            raise AssertionError(f"RL-0 source changed; regenerate results: {path}")
    counts = {}
    for name, trace in data["traces"].items():
        x, info = load_trace(ROOT / trace["path"], seconds=trace["duration_seconds"])
        assert info == trace, f"trace changed: {name}"
        counts[name] = x
    for row in data["bounds"]:
        dp = row["dp"]
        replay = evaluate_schedule(counts[row["trace"]], dp["targets"],
                                   decision_seconds=row["decision_seconds"], rhos=dp["rhos"])
        assert abs(replay["total_cost_usd"] - row["dp_cost_usd"]) <= 1e-9
        assert row["dp_cost_usd"] <= row["rule_cost_usd"] + 1e-9
        assert abs(row["recoverable_fraction"] - (1 - row["dp_cost_usd"] / row["rule_cost_usd"])) <= 1e-12
    assert len(data["parity"]) == 8
    for row in data["parity"]:
        assert max(row["absolute_errors_usd"].values()) <= 1e-9
    for row in data["random_reference"]:
        assert len(row["costs_usd"]) == 5
        assert abs(float(np.mean(row["costs_usd"])) - row["mean_usd"]) <= 1e-9
    assert data["azure"].get("test_evaluated", False) is False
    print("RL-0 verified: source/trace fingerprints, 6 DP replays, 8 parity cases, random references; no Azure test evaluation.")


if __name__ == "__main__":
    main()
