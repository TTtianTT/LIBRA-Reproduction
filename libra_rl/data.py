"""Partition boundaries for later RL training; RL-0 does not consume test days."""
from pathlib import Path

import numpy as np

DAY = 86400
AZURE_SPLITS = {"train": (0, 9 * DAY), "validation": (9 * DAY, 11 * DAY),
                "test": (11 * DAY, 14 * DAY)}


def load_azure_partition(split, *, root=Path("data/azure2021"), multiplier=1):
    if split not in AZURE_SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if not isinstance(multiplier, int) or multiplier < 1:
        raise ValueError("multiplier must be a positive integer")
    with np.load(Path(root) / "aggregate_counts.npz") as archive:
        counts = archive["counts"]
    if len(counts) != 14 * DAY:
        raise ValueError("expected exactly fourteen days of per-second counts")
    start, end = AZURE_SPLITS[split]
    return counts[start:end].astype(np.int64) * multiplier
