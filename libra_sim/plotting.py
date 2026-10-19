from pathlib import Path
import os
import tempfile

# Keep plotting usable under a restricted filesystem, including on macOS.
_cache = Path(tempfile.gettempdir()) / "libra-matplotlib"
_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import POLICIES

COLORS = {"LIBRA": "#16846b", "FAAS": "#d79c27", "MAX": "#8574ad",
          "SPOCK": "#3687be", "AUTO": "#cb6464"}


def make_figures(results, sweep, output, config):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140})
    timeline = results["LIBRA"]["timeline"]
    t = np.array([r["time_s"] for r in timeline])
    demand = np.array([r["requests"] for r in timeline])
    avg = np.array([r["ewma"] for r in timeline])
    baseline = results["LIBRA"]["summary"]["total_cost_usd"]
    base_uptime = results["LIBRA"]["summary"]["vm_uptime_seconds"]
    denominator = baseline if baseline else 1

    def save(fig, name):
        fig.savefig(out / f"{name}.png", bbox_inches="tight")
        fig.savefig(out / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3.5), layout="constrained")
    ax.plot(t, demand, color="#87a7bd", lw=0.5, label="Requests / second", rasterized=True)
    ax.plot(t, avg, color=COLORS["LIBRA"], lw=0.7, label=f"EWMA (alpha={config.alpha:g})", rasterized=True)
    ax.set(xlabel="Time (s)", ylabel="Requests / second", title="Figure 3 reproduction: demand and EWMA")
    ax.legend(loc="upper right")
    save(fig, "fig3_trace_ewma")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), layout="constrained")
    ax = axes[0]
    x = np.arange(len(POLICIES))
    ax.bar(x, [results[p]["summary"]["total_cost_usd"] / denominator for p in POLICIES],
           color=[COLORS[p] for p in POLICIES], width=0.6)
    ax.set(xticks=x, xticklabels=POLICIES, ylabel="Cost / LIBRA cost", title="(a) Cost and SLA violations")
    right = ax.twinx()
    right.spines["right"].set_visible(True)
    right.plot(x, [results[p]["summary"]["sla_violation_percent"] for p in POLICIES],
               color="#303a47", marker="o", label="SLA violations")
    right.set_ylabel("SLA violations (%)")
    right.set_ylim(bottom=0)
    right.legend(loc="upper left", fontsize=8)
    decisions = results["LIBRA"]["decisions"]
    times = [r["time_s"] for r in decisions]
    # Offline reference: whole NEXT scaling window's mean, without cold start.
    # The paper does not publish an exact oracle algorithm; this is labeled here.
    oracle = [np.ceil(np.mean(demand[s:s + config.scaling_interval_seconds]) / config.vm_capacity)
              for s in times]
    axes[1].step(times, oracle, where="post", ls="--", color="#3687be", label="Offline window-mean reference")
    axes[1].step(times, [r["target_vms"] for r in decisions], where="post", color=COLORS["LIBRA"], label="LIBRA allocated")
    axes[1].set(xlabel="Time (s)", ylabel="VMs", title="(b) VM provisioning")
    axes[1].legend(fontsize=8)
    window = config.scaling_interval_seconds
    starts = list(range(0, len(demand), window))
    for key, label, color in [("requests", "Total", "#36424d"), ("vm_arrivals", "IaaS", COLORS["LIBRA"]),
                              ("faas_requests", "FaaS", COLORS["FAAS"])]:
        values = [np.mean([r[key] for r in timeline[s:s + window]]) for s in starts]
        axes[2].plot(starts, values, label=label, color=color)
    axes[2].set(xlabel="Time (s)", ylabel="Mean requests / second", title="(c) Request distribution")
    axes[2].legend(fontsize=8)
    save(fig, "fig4_policy_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    policies = ("LIBRA", "AUTO", "SPOCK")
    axes[0].bar(policies, [results[p]["summary"]["vm_uptime_seconds"] / (base_uptime or 1)
                          for p in policies], color=[COLORS[p] for p in policies])
    axes[0].set(ylabel="VM seconds / LIBRA VM seconds", title="(a) Aggregate VM uptime")
    bottom = np.zeros(3)
    for key, label, color in [("vm_cost_usd", "IaaS", "#6d9fbc"), ("gateway_cost_usd", "Gateway", "#d3d8de"),
                              ("faas_cost_usd", "FaaS", "#e2b24e")]:
        vals = np.array([results[p]["summary"][key] / denominator for p in policies])
        axes[1].bar(policies, vals, bottom=bottom, label=label, color=color)
        bottom += vals
    axes[1].set(ylabel="Cost / LIBRA total cost", title="(b) Cost breakdown")
    axes[1].set_ylim(0, max(bottom) * 1.25 if max(bottom) else 1)
    axes[1].legend(loc="upper center", ncol=3, fontsize=9)
    save(fig, "fig5_uptime_cost_breakdown")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    phis = [s["config"]["phi"] for s in sweep]
    axes[0].plot(phis, [s["total_cost_usd"] for s in sweep], "o-", color=COLORS["LIBRA"])
    axes[0].set(xlabel="phi", ylabel="Total cost including gateway (USD)", title="(a) Total cost", xticks=phis)
    bottom = np.zeros(len(sweep))
    for key, label, color in [("vm_cost_usd", "IaaS", "#6d9fbc"), ("gateway_cost_usd", "Gateway", "#d3d8de"),
                              ("faas_cost_usd", "FaaS", "#e2b24e")]:
        vals = np.array([s[key] for s in sweep])
        axes[1].bar(phis, vals, bottom=bottom, label=label, color=color, width=0.65)
        bottom += vals
    axes[1].set(xlabel="phi", ylabel="Cost (USD)", title="(b) Cost breakdown", xticks=phis)
    axes[1].legend()
    save(fig, "fig6_phi_sensitivity")
