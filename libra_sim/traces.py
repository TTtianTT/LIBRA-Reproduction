import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def display_path(path):
    """Repository-relative path for provenance, so results do not embed a home directory."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved)


def load_trace(path, seconds=None, offset=0):
    path = Path(path)
    if offset < 0 or (seconds is not None and seconds <= 0):
        raise ValueError("Offset must be nonnegative and duration positive")
    counts = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        try:
            n = int(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: expected integer request count") from exc
        if n < 0:
            raise ValueError(f"{path}:{line_no}: negative demand")
        counts.append(n)
    stop = len(counts) if seconds is None else offset + seconds
    if offset >= len(counts) or stop > len(counts):
        raise ValueError(f"Trace has {len(counts)} seconds; requested [{offset}, {stop})")
    selected = counts[offset:stop]
    return selected, {
        "path": display_path(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "available_seconds": len(counts), "offset_seconds": offset,
        "duration_seconds": len(selected), "total_requests": sum(selected),
        "peak_requests_per_second": max(selected),
    }
