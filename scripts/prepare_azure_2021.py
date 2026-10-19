#!/usr/bin/env python3
"""Fetch/aggregate Microsoft's 2021 per-invocation trace into 14 daily seconds.

Official schema units are seconds. arrival = end_timestamp - duration. No
interpolation, smoothing, load scaling or cherry-picking of applications here.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/azure2021"
NAME = "AzureFunctionsInvocationTraceForTwoWeeksJan2021"
URL = f"https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/{NAME}.rar"
DOC = "https://github.com/Azure/AzurePublicDataset/blob/master/AzureFunctionsInvocationTrace2021.md"
HORIZON = 14 * 86400


def aggregate(path):
    counts = np.zeros(HORIZON, dtype=np.int64)
    total = excluded = invalid = 0
    earliest, latest = float("inf"), -float("inf")
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["app", "func", "end_timestamp", "duration"]:
            raise ValueError(f"Unexpected official CSV schema: {reader.fieldnames}")
        for row in reader:
            total += 1
            end, duration = float(row["end_timestamp"]), float(row["duration"])
            if not (math.isfinite(end) and math.isfinite(duration) and duration >= 0):
                invalid += 1
                continue
            arrival = end - duration
            earliest, latest = min(earliest, arrival), max(latest, arrival)
            if 0 <= arrival < HORIZON:
                counts[math.floor(arrival)] += 1
            else:
                excluded += 1
    assert int(counts.sum()) + excluded + invalid == total
    return counts, {"source_rows": total, "included_arrivals": int(counts.sum()),
                    "outside_14_day_arrivals": excluded, "invalid_rows": invalid,
                    "earliest_arrival_seconds": earliest, "latest_arrival_seconds": latest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing official RAR and unpack using unrar")
    args = parser.parse_args()
    raw = DIRECTORY / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive, text = raw / f"{NAME}.rar", raw / f"{NAME}.txt"
    if args.download and not archive.exists():
        subprocess.run(["curl", "-L", "--fail", "--retry", "3", URL, "-o", str(archive)], check=True)
    if args.download and not text.exists():
        subprocess.run(["unrar", "x", "-o-", str(archive), str(raw) + "/"], check=True)
    if not text.exists():
        parser.error("Missing official data; run with --download (requires curl and unrar)")
    counts, details = aggregate(text)
    np.savez_compressed(DIRECTORY / "aggregate_counts.npz", counts=counts)
    train = counts[:7 * 86400]
    minute = train.reshape(-1, 60).sum(axis=1).astype(float)
    hourly = train.reshape(-1, 3600).mean(axis=1)
    def corr(x, lag):
        return float(np.corrcoef(x[lag:], x[:-lag])[0, 1]) if x.std() else None
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_url": URL, "documentation": DOC,
        "attribution": "Yanqi Zhang, Inigo Goiri, Gohar Irfan Chaudhry, Rodrigo Fonseca, Sameh Elnikety, Christina Delimitrou, Ricardo Bianchini. Faster and Cheaper Serverless Computing on Harvested Resources. SOSP 2021.",
        "license": "CC BY; https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE",
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_csv_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
        "aggregate_sha256": hashlib.sha256(counts.astype('<i8').tobytes()).hexdigest(),
        "time_unit": "seconds, per dataset-specific documentation; do not divide by 1000",
        "arrival_rule": "floor(end_timestamp - duration), with 0 <= arrival < 1209600",
        "start_date_as_documented": "2021-01-31", "timestamp_caveat": "Microsoft modified timestamps; calendar phase is relative, not verified original wall-clock time",
        "duration_seconds": len(counts), "zero_seconds_retained": int((counts == 0).sum()),
        "mean_rps": float(counts.mean()), "peak_rps": int(counts.max()),
        "daily_requests": counts.reshape(14, 86400).sum(axis=1).tolist(),
        "train_only_structure": {"first_7_days_minute_lag24h_correlation": corr(minute, 1440),
                                 "first_7_days_hourly_lag24h_correlation": corr(hourly, 24)}, **details}
    (DIRECTORY / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    (DIRECTORY / "SOURCE.md").write_text(
        "# Azure Functions Invocation Trace 2021\n\n"
        f"[Microsoft dataset documentation]({DOC}). Download: [official RAR]({URL}).\n\n"
        f"Attribution: {manifest['attribution']}\n\n"
        "License: [CC BY](https://github.com/Azure/AzurePublicDataset/blob/master/LICENSE).\n\n"
        "The dedicated schema specifies seconds. Arrival is end_timestamp minus duration; all zero-demand seconds are retained. "
        "The aggregate covers 14 days, with out-of-range arrivals counted explicitly in manifest.json. "
        "No applications were selected and no timestamps were stretched or smoothed. "
        "Original request durations are used only to reconstruct arrivals; downstream LIBRA simulation retains its fixed one-second service model.\n\n"
        "Raw files are local under raw/ and ignored by Git. aggregate_counts.npz stores the unscaled per-second counts; manifest.json stores hashes.\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
