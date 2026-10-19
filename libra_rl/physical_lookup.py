"""Empirical steady M/D/16 interpolation, never a transient/saturation model."""
import json
from pathlib import Path
import numpy as np


class MD16Lookup:
    def __init__(self,tables):
        self.tables = tables

    @classmethod
    def load(cls,path):
        return cls(json.loads(Path(path).read_text())["tables"])

    def probability(self,utilization,service_seconds):
        points = self.tables[str(float(service_seconds))]
        grid = np.array([p["utilization"] for p in points])
        rates = np.array([p["monotone_estimate"] for p in points])
        u = np.asarray(utilization,dtype=float)
        if np.any(~np.isfinite(u)) or np.any(u<0) or np.any(u>grid[-1]) or np.any(u>=1):
            raise ValueError("outside measured stable range; use the transient queue, not extrapolation")
        return np.interp(u,grid,rates)
