"""Extra experiments from the code review (not part of the original reproduction).

E1  Baseline strength: sweep AUTO/SPOCK thresholds, group size, window and minimum
    instances; report the cheapest setting under several SLA-violation budgets.
E2  phi/rho coupling: fine phi sweep at rho = 0.8, 0.9, 1.0 (WITS).
E3  LIBRA variants: routing = rho*estimate (paper) vs rho*ready-capacity, and the
    EWMA horizon (alpha = beta = 0.2 / 0.02 / 0.005), on all four traces.

Usage:  .venv/bin/python scripts/review_experiments.py [--skip-sweep]
Results are printed and written to outputs/review/*.csv.
"""
import argparse
import csv
import itertools
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_sim.config import Config          # noqa: E402
from libra_sim.engine import simulate        # noqa: E402
from libra_sim.traces import load_trace      # noqa: E402

TRACES = [("WITS", "wits_logs.txt", 43200), ("Berkeley", "berkeley_logs.txt", 43200),
          ("Sin", "sin_logs.txt", 10000), ("Square", "square_load.txt", 1112)]
OUT = ROOT / "outputs" / "review"


def run(counts, policy, cfg, **kw):
    s = simulate(counts, policy, cfg, **kw)["summary"]
    return {"cost": s["total_cost_usd"], "sla_pct": s["sla_violation_percent"],
            "faas_pct": 100 * s["faas_requests"] / s["total_requests"],
            "vm_hours": s["vm_uptime_seconds"] / 3600}


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def e1_baseline_sweep(base):
    rows = []
    for tname, fname, sec in TRACES[:2]:
        counts, _ = load_trace(ROOT / "data/traces" / fname, seconds=sec)
        libra = run(counts, "LIBRA", base)
        print(f"\n[E1] {tname}: LIBRA (paper config) cost={libra['cost']:.3f} SLA={libra['sla_pct']:.2f}%")
        grid = itertools.product((0.4, 0.5, 0.6, 0.7, 0.8), (0.1, 0.2, 0.3), (5, 10, 20),
                                 (30, 60, 120), (5, 10))
        for out, inn, group, window, mn in grid:
            if inn >= out:
                continue
            cfg = replace(base, auto_scale_out_threshold=out, auto_scale_in_threshold=inn,
                          auto_scale_group=group, auto_window_seconds=window, auto_min_instances=mn)
            for pol in ("AUTO", "SPOCK"):
                rows.append({"trace": tname, "policy": pol, "scale_out": out, "scale_in": inn,
                             "group": group, "window": window, "min_vms": mn, **run(counts, pol, cfg)})
        for pol in ("AUTO", "SPOCK"):
            mine = [r for r in rows if r["trace"] == tname and r["policy"] == pol]
            for cap in (5, 2, 1, 0.5, 0.1):
                ok = [r for r in mine if r["sla_pct"] <= cap]
                if not ok:
                    print(f"  {pol:5s} SLA<={cap:>4}%: no setting in grid")
                    continue
                b = min(ok, key=lambda r: r["cost"])
                print(f"  {pol:5s} SLA<={cap:>4}%: cost={b['cost']:.3f} (SLA {b['sla_pct']:.3f}%) "
                      f"out={b['scale_out']} in={b['scale_in']} group={b['group']} "
                      f"window={b['window']} min={b['min_vms']}")
    write("e1_baseline_sweep.csv", rows)


def e2_phi_rho(base):
    counts, _ = load_trace(ROOT / "data/traces/wits_logs.txt", seconds=43200)
    rows = []
    phis = [x / 4 for x in range(13)]
    print("\n[E2] WITS phi/rho coupling")
    for rho in (0.8, 0.9, 1.0):
        curve = [run(counts, "LIBRA", replace(base, rho=rho, phi=p))["cost"] for p in phis]
        rows += [{"rho": rho, "phi": p, "cost": v} for p, v in zip(phis, curve)]
        i = min(range(len(phis)), key=curve.__getitem__)
        print(f"  rho={rho}: best phi={phis[i]} cost={curve[i]:.3f}; phi=1 -> {curve[4]:.3f} "
              f"(+{100 * (curve[4] / curve[i] - 1):.1f}%)")
    write("e2_phi_rho.csv", rows)


def e3_variants(base):
    rows = []
    for tname, fname, sec in TRACES:
        counts, _ = load_trace(ROOT / "data/traces" / fname, seconds=sec)
        print(f"\n[E3] {tname}")
        for label, routing, rho in (("paper rho*estimate", "estimate", 0.8),
                                    ("rho*capacity", "capacity", 0.8),
                                    ("fill ready capacity", "capacity", 1.0)):
            for a in (0.2, 0.02, 0.005):
                for phi in (0, 1, 2, 3):
                    r = run(counts, "LIBRA", replace(base, rho=rho, alpha=a, beta=a, phi=phi),
                            routing=routing)
                    rows.append({"trace": tname, "variant": label, "alpha": a, "phi": phi, **r})
                    if phi == 1:
                        print(f"  {label:20s} alpha={a:<5} phi=1 cost={r['cost']:.3f} "
                              f"SLA={r['sla_pct']:.2f}% FaaS={r['faas_pct']:.1f}%")
    write("e3_libra_variants.csv", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true", help="skip E1 (about 5 minutes)")
    args = ap.parse_args()
    cfg = Config()
    if not args.skip_sweep:
        e1_baseline_sweep(cfg)
    e2_phi_rho(cfg)
    e3_variants(cfg)
    print(f"\nCSV written to {OUT}")
