#!/usr/bin/env python3
"""Download pinned research source for optional historical replay. No cloud SDK."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = "53a804412ca24c6eca7fa226fcccc0220b17fb0f"
path = ROOT / "upstream"
if not path.exists():
    subprocess.run(["git", "clone", "https://github.com/aliraza0337/LIBRA.git", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "--detach", REVISION], check=True)
actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
if actual != REVISION:
    raise SystemExit(f"Existing upstream is {actual}, expected {REVISION}; left untouched")
print(f"Verified upstream {actual}")
