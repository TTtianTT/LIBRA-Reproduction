#!/usr/bin/env python3
"""Refine every physical oracle interval without changing any policy or data."""
import json
from pathlib import Path

import physical_headroom as driver
from libra_rl.physical import ArrivalStream
from libra_rl.physical_exact import optimal_interval


def main():
    output = driver.ROOT/"outputs/physical_headroom"
    protocol,manifest,data = driver.prepare(output)
    limits = json.loads((driver.ROOT/"configs/physical_exact.json").read_text())
    sources = {name:driver.sha(driver.ROOT/name) for name in
               ("libra_rl/physical_exact.py","libra_rl/physical_dual.py","scripts/refine_physical_bounds.py","configs/physical_exact.json")}
    extended = json.loads((output/"forecast_rule_evaluation.json").read_text())["rows"]
    records = []
    for name in sorted((output/"runs").glob("*.json")):
        old = json.loads(name.read_text())
        s,seed,w = old["service_seconds"],old["arrival_seed"],old["window"]
        incumbent = next(r for r in extended if r["policy"]=="refined_clairvoyant" and
                         (r["service_seconds"],r["arrival_seed"],r["window"])==(s,seed,w))
        stream = ArrivalStream.generate(data["validation"][old["start"]:old["start"]+3600],seed=old["stream_seed"])
        result = optimal_interval(stream,old["bases"],service_seconds=s,incumbent_targets=incumbent["targets"],
            incumbent_rhos=incumbent["rhos"],epsilon=protocol["epsilon"],
            max_seconds=limits["max_seconds_per_window"],max_expansions=limits["max_expansions_per_window"],
            max_live_nodes=limits["max_frontier_entries"])
        records.append(dict(service_seconds=s,arrival_seed=seed,window=w,**result))
        driver.write(output/"exact_intervals.json",dict(physical_fingerprint=manifest["fingerprint"],source_sha256=sources,
            limits=limits,records=records,test_evaluated=False,ppo_trained=False))
        print(f"s={s} seed={seed} w={w}: [{result['cost_lower_bound_usd']:.7f}, "
              f"{result['feasible_upper_bound_usd']:.7f}] {result['stopped']}, expanded={result['expanded_states']}",flush=True)


if __name__=="__main__":
    main()
