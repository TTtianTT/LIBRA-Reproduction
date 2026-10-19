#!/usr/bin/env python3
"""LIBRA-ML: replace avg + phi*std with a learned tau-quantile of future demand.

Protocols (no test data is used for training or tuning):
  LOTO      train on the other traces, evaluate on the main window of the held-out one
  temporal  train on the first half of a trace's main window, evaluate on the second
            half with a fresh start (WITS/Berkeley: 6 h + 6 h; Sin: 5,000 s + 5,000 s)

Policies on each evaluation segment (all share configs/paper.json prices/timing):
  paper          LIBRA as in the paper (rho*estimate routing, EWMA alpha=0.2, phi=1)
  fill_paper     fill ready VM capacity first (rho=1), paper EWMA
  fill_tuned     fill routing, EWMA (alpha=beta, phi) picked by cost on the training data
  fill_quantile  fill routing, tau-quantile of the last 300 s (no training)
  fill_linear    fill routing, linear quantile regression on causal features
  fill_gbdt      fill routing, gradient-boosted quantile regression
  fill_gbdt_resid  ablation: GBDT predicts a correction factor on the window quantile
  oracle         fill routing, knows the next window (offline reference, not deployable)
  spock / auto   default threshold baselines, for context only

Usage: .venv/bin/python scripts/ml_experiments.py [--output outputs/ml]
"""
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from libra_sim import Config, simulate                                   # noqa: E402
from libra_sim.traces import load_trace                                  # noqa: E402
from libra_ml import (FEATURE_NAMES, newsvendor_tau, training_set,      # noqa: E402
                      model_estimator, oracle_estimator, window_quantile_estimator)
from libra_ml.models import GBDT_PARAMS, fit_gbdt, fit_linear            # noqa: E402
from libra_ml.policies import pinball                                    # noqa: E402

TRACES = {"WITS": ("wits_logs.txt", 43200), "Berkeley": ("berkeley_logs.txt", 43200),
          "Sin": ("sin_logs.txt", 10000)}
EWMA_GRID = [(a, p) for a in (0.2, 0.05, 0.02, 0.005) for p in (0, 0.5, 1, 1.5, 2, 3)]
POLICIES = ("paper", "fill_paper", "fill_tuned", "fill_quantile", "fill_linear",
            "fill_gbdt", "fill_gbdt_resid", "oracle", "spock", "auto")
LEARNED = ("fill_tuned", "fill_quantile", "fill_linear", "fill_gbdt", "fill_gbdt_resid")


def folds():
    full = {name: load_trace(ROOT / "data/traces" / f)[0] for name, (f, _) in TRACES.items()}
    for name, (_, seconds) in TRACES.items():
        yield {"fold": f"LOTO:{name}", "protocol": "leave-one-trace-out", "test_trace": name,
               "train": {k: v for k, v in full.items() if k != name},
               "test": full[name][:seconds], "test_range": [0, seconds]}
    for name, (_, seconds) in TRACES.items():
        half = seconds // 2
        yield {"fold": f"temporal:{name}", "protocol": "temporal", "test_trace": name,
               "train": {f"{name}[0,{half})": full[name][:half]},
               "test": full[name][half:seconds], "test_range": [half, seconds]}


def tune_ewma(train, base):
    """Pick (alpha=beta, phi) with fill routing by mean cost ratio over training traces."""
    fill = replace(base, rho=1.0)
    ref = {k: simulate(v, "LIBRA", fill, routing="capacity")["summary"]["total_cost_usd"]
           for k, v in train.items()}
    scores = []
    for alpha, phi in EWMA_GRID:
        cfg = replace(fill, alpha=alpha, beta=alpha, phi=phi)
        ratio = np.mean([simulate(v, "LIBRA", cfg, routing="capacity")["summary"]["total_cost_usd"]
                         / ref[k] for k, v in train.items()])
        scores.append((ratio, alpha, phi))
    ratio, alpha, phi = min(scores)
    return alpha, phi, ratio


def run_fold(fold, base, tau):
    t0 = time.time()
    parts = [training_set(v) for v in fold["train"].values()]
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    gbdt = fit_gbdt(X, y, tau)
    linear = fit_linear(X, y, tau)
    parts = [training_set(v, scale_quantile=tau) for v in fold["train"].values()]
    gbdt_resid = fit_gbdt(np.vstack([p[0] for p in parts]),
                          np.concatenate([p[1] for p in parts]), tau)
    alpha, phi, train_ratio = tune_ewma(fold["train"], base)
    fill = replace(base, rho=1.0)
    test = fold["test"]
    runs = {
        "paper": dict(policy="LIBRA", config=base),
        "fill_paper": dict(policy="LIBRA", config=fill, routing="capacity"),
        "fill_tuned": dict(policy="LIBRA", config=replace(fill, alpha=alpha, beta=alpha, phi=phi),
                           routing="capacity"),
        "fill_quantile": dict(policy="LIBRA", config=fill, routing="capacity",
                              estimator=window_quantile_estimator(tau)),
        "fill_linear": dict(policy="LIBRA", config=fill, routing="capacity",
                            estimator=model_estimator(linear)),
        "fill_gbdt": dict(policy="LIBRA", config=fill, routing="capacity",
                          estimator=model_estimator(gbdt)),
        "fill_gbdt_resid": dict(policy="LIBRA", config=fill, routing="capacity",
                                estimator=model_estimator(gbdt_resid, scale_quantile=tau)),
        "oracle": dict(policy="LIBRA", config=fill, routing="capacity",
                       estimator=oracle_estimator(test, base)),
        "spock": dict(policy="SPOCK", config=base),
        "auto": dict(policy="AUTO", config=base),
    }
    x = np.asarray(test, dtype=float)
    rows = []
    for name in POLICIES:
        kw = dict(runs[name])
        result = simulate(test, kw.pop("policy"), kw.pop("config"), **kw)
        s = result["summary"]
        row = {"fold": fold["fold"], "protocol": fold["protocol"], "test_trace": fold["test_trace"],
               "test_start_s": fold["test_range"][0], "test_end_s": fold["test_range"][1],
               "policy": name, "total_cost_usd": s["total_cost_usd"], "vm_cost_usd": s["vm_cost_usd"],
               "faas_cost_usd": s["faas_cost_usd"], "gateway_cost_usd": s["gateway_cost_usd"],
               "sla_violation_percent": s["sla_violation_percent"],
               "faas_request_percent": 100 * s["faas_requests"] / s["total_requests"],
               "vm_hours": s["vm_uptime_seconds"] / 3600, "capacity_pinball": "",
               "forecast_coverage": ""}
        if name not in ("spock", "auto"):
            # Pinball loss of provisioned slots vs demand over each decision's window,
            # normalised by mean demand: the ML-side view of provisioning cost.
            losses = [pinball(x[d["time_s"] + 100: d["time_s"] + 400], 16 * d["target_vms"], tau)
                      for d in result["decisions"] if d["time_s"] + 100 < len(x)]
            row["capacity_pinball"] = float(np.mean(losses) / x.mean())
            # Share of seconds whose demand is <= the decision's provisioning level
            # (before rounding to VMs); a calibrated tau-quantile forecast gives ~tau.
            row["forecast_coverage"] = float(np.mean(np.concatenate(
                [x[d["time_s"] + 100: d["time_s"] + 400] <= d["estimate"]
                 for d in result["decisions"] if d["time_s"] + 100 < len(x)])))
        rows.append(row)
    cost = {r["policy"]: r["total_cost_usd"] for r in rows}
    gap = cost["fill_paper"] - cost["oracle"]
    for r in rows:
        r["vs_paper_percent"] = 100 * (r["total_cost_usd"] / cost["paper"] - 1)
        r["headroom_captured"] = ((cost["fill_paper"] - r["total_cost_usd"]) / gap
                                  if r["policy"] in LEARNED and gap > 0 else "")
    meta = {"fold": fold["fold"], "train_series": {k: len(v) for k, v in fold["train"].items()},
            "train_rows": int(len(y)), "tuned_ewma": {"alpha_beta": alpha, "phi": phi,
                                                      "train_cost_ratio": train_ratio},
            "linear_coef": dict(zip(FEATURE_NAMES, map(float, linear.coef_))),
            "seconds": round(time.time() - t0, 1)}
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/ml")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base = Config.read(ROOT / "configs/paper.json")
    tau = newsvendor_tau(base)
    all_rows, metas = [], []
    for fold in folds():
        rows, meta = run_fold(fold, base, tau)
        all_rows += rows
        metas.append(meta)
        best = {r["policy"]: r for r in rows}
        print(f"{fold['fold']:18s} paper={best['paper']['total_cost_usd']:.3f} "
              f"fill={best['fill_paper']['total_cost_usd']:.3f} "
              f"tuned={best['fill_tuned']['total_cost_usd']:.3f} "
              f"quant={best['fill_quantile']['total_cost_usd']:.3f} "
              f"lin={best['fill_linear']['total_cost_usd']:.3f} "
              f"gbdt={best['fill_gbdt']['total_cost_usd']:.3f} "
              f"oracle={best['oracle']['total_cost_usd']:.3f}  ({meta['seconds']}s)", flush=True)
    with (args.output / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(all_rows)
    import sklearn
    sources = sorted([*ROOT.glob("libra_ml/*.py"), *ROOT.glob("libra_sim/*.py"),
                      ROOT / "scripts/ml_experiments.py"])
    payload = {"metadata": {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
        "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "tau": tau, "config": asdict(base), "gbdt_params": GBDT_PARAMS, "ewma_grid": EWMA_GRID,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources}},
        "folds": metas, "rows": all_rows}
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output.relative_to(ROOT) if args.output.is_relative_to(ROOT) else args.output}")


if __name__ == "__main__":
    main()
