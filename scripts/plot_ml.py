#!/usr/bin/env python3
"""Figure + Markdown tables for outputs/ml (reads results.csv; no simulation)."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROWS = [("fill_paper", "Fill + paper EWMA", False), ("fill_tuned", "Fill + tuned EWMA", False),
        ("fill_quantile", "Fill + 300 s quantile", False), ("fill_linear", "Fill + linear QR (ML)", True),
        ("fill_gbdt", "Fill + GBDT QR (ML)", True)]
FOLDS = [("LOTO:WITS", "WITS"), ("LOTO:Berkeley", "Berkeley"), ("LOTO:Sin", "Sin"),
         ("temporal:WITS", "WITS"), ("temporal:Berkeley", "Berkeley"), ("temporal:Sin", "Sin")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NON_ML, ML = "#2a78d6", "#eb6834"


def load(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {(r["fold"], r["policy"]): r for r in rows}


def plot(data, out):
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), sharey=True, facecolor=SURFACE)
    for ax, (fold, title) in zip(axes.flat, FOLDS):
        paper = float(data[fold, "paper"]["total_cost_usd"])
        oracle = float(data[fold, "oracle"]["total_cost_usd"]) / paper
        ax.set_facecolor(SURFACE)
        ax.axvline(1.0, color=MUTED, lw=1.2)
        ax.axvline(oracle, color=MUTED, lw=1.2, ls=(0, (4, 3)))
        for y, (key, label, is_ml) in enumerate(ROWS):
            v = float(data[fold, key]["total_cost_usd"]) / paper
            ax.plot([oracle, v], [y, y], color=GRID, lw=2, zorder=1)
            ax.scatter(v, y, s=64, color=ML if is_ml else NON_ML, edgecolor=SURFACE,
                       linewidth=2, zorder=3)
        protocol = "leave-one-trace-out" if fold.startswith("LOTO") else "temporal split"
        ax.set_title(f"{title} · {protocol}", fontsize=10, color=INK, loc="left")
        ax.set_xlim(0.5, 1.05)
        ax.set_yticks(range(len(ROWS)), [r[1] for r in ROWS], fontsize=9, color=INK)
        ax.set_ylim(len(ROWS) + 0.15, -0.6)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        for x, word in ((1.0, "paper"), (oracle, "oracle")):
            ax.text(x, len(ROWS) - 0.15, word, fontsize=8, color=MUTED, ha="center", va="center",
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1))
    for ax in axes[1]:
        ax.set_xlabel("Total cost / paper LIBRA cost (lower is better)", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], marker="o", ls="", color=NON_ML, markersize=8, label="Rule / tuned rule"),
               plt.Line2D([], [], marker="o", ls="", color=ML, markersize=8, label="Learned model (ML)"),
               plt.Line2D([], [], color=MUTED, ls=(0, (4, 3)), label="Oracle (knows the future)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.0), labelcolor=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("png", "svg"):
        fig.savefig(out / f"ml_cost.{ext}", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def tables(data):
    cols = ["paper", "fill_paper", "fill_tuned", "fill_quantile", "fill_linear", "fill_gbdt", "oracle"]
    head = "| 折 | 论文 LIBRA | 填满+论文EWMA | 填满+调参EWMA | 填满+300s分位数 | 填满+线性QR | 填满+GBDT QR | Oracle |"
    lines = [head, "|---|" + "---:|" * len(cols)]
    for fold, _ in FOLDS:
        lines.append(f"| {fold} | " + " | ".join(f"{float(data[fold, c]['total_cost_usd']):.3f}"
                                                 for c in cols) + " |")
    lines += ["", "| 折 | 调参EWMA | 300s分位数 | 线性QR | GBDT QR | GBDT 残差版 |", "|---|---:|---:|---:|---:|---:|"]
    for fold, _ in FOLDS:
        lines.append(f"| {fold} | " + " | ".join(
            f"{float(data[fold, c]['headroom_captured']):+.0%}"
            for c in ("fill_tuned", "fill_quantile", "fill_linear", "fill_gbdt", "fill_gbdt_resid")) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/ml")
    args = ap.parse_args()
    data = load(args.output / "results.csv")
    plot(data, args.output)
    print(tables(data))
