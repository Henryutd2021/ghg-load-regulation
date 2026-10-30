"""Result tables of a run, and checks of the constraints recomputed from the solution.

Emissions are in tonnes of CO2. "Baseline" is the same run without regulation (x_t = 0), so the
reduction is the effect of load regulation alone. "Reference" uses the region's unchanged demand and
intensity; it differs from the baseline only in cases that change the inputs (demand, solar, wind).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .scenarios import ModelData

KG_PER_T = 1000.0
#: |x_t| above this counts as an adjusted hour.
ADJUSTED_TOL = 1e-6


@dataclass
class RunResult:
    region: str
    case: dict
    hourly: pd.DataFrame
    windows: pd.DataFrame
    totals: dict

    def to_csv(self, directory) -> Path:
        """Write hourly.csv, windows.csv and totals.csv (one row: case settings and totals)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.hourly.to_csv(d / "hourly.csv", index=False)
        self.windows.to_csv(d / "windows.csv", index=False)
        pd.DataFrame([{"region": self.region, **self.case, **self.totals}]).to_csv(d / "totals.csv", index=False)
        return d


def collect(data: ModelData, solutions) -> RunResult:
    """Assemble the result tables from the window solutions (any object with label, hours, x, z, objective, termination)."""
    n = len(data.demand)
    x, z = np.full(n, np.nan), np.full(n, np.nan)
    rows = []
    for s in solutions:
        x[s.hours], z[s.hours] = s.x, s.z
        d, e = data.demand[s.hours], data.intensity[s.hours]
        base = float(np.sum(d * e) / KG_PER_T)
        opt = float(np.sum((1 + s.x) * d * e) / KG_PER_T)
        shift = s.x * d
        rows.append({
            "window": s.label,
            "hours": int(s.hours.size),
            "baseline_t": base,
            "optimized_t": opt,
            "reduction_t": base - opt,
            "reduction_pct": 100 * (base - opt) / base if base > 0 else np.nan,
            "adjusted_hours": int(np.sum(np.abs(s.x) > ADJUSTED_TOL)),
            "added_mwh": float(np.sum(shift[shift > 0])),
            "removed_mwh": float(-np.sum(shift[shift < 0])),
            "objective_t": s.objective,
            "termination": s.termination,
        })
    windows = pd.DataFrame(rows)
    hourly = pd.DataFrame({
        "time": data.time,
        "window": data.window,
        "demand_mwh": data.demand,
        "x": x,
        "adjusted_demand_mwh": (1 + x) * data.demand,
        "adjusted": np.abs(x) > ADJUSTED_TOL,
        "z": z,
        "intensity_kg_per_mwh": data.intensity,
        "baseline_emissions_t": data.demand * data.intensity / KG_PER_T,
        "emissions_t": (1 + x) * data.demand * data.intensity / KG_PER_T,
    })
    base, opt = float(windows["baseline_t"].sum()), float(windows["optimized_t"].sum())
    reference = float(np.sum(data.reference_demand * data.reference_intensity) / KG_PER_T)
    totals = {
        "hours": n,
        "windows": len(windows),
        "baseline_t": base,
        "optimized_t": opt,
        "reduction_t": base - opt,
        "reduction_pct": 100 * (base - opt) / base if base > 0 else np.nan,
        "adjusted_hours": int(windows["adjusted_hours"].sum()),
        "added_mwh": float(windows["added_mwh"].sum()),
        "removed_mwh": float(windows["removed_mwh"].sum()),
        "reference_t": reference,
        "all_optimal": bool((windows["termination"] == "optimal").all()),
    }
    return RunResult(region=data.region, case=data.case.as_dict(), hourly=hourly, windows=windows, totals=totals)


def constraint_residuals(result: RunResult, data: ModelData) -> dict:
    """Largest violation of each constraint, recomputed from the solution (0 = satisfied).

    eq3: per window, sum_t d_t x_t >= 0; eq4: x_t d_t <= c_max_t; eq5: (1+x_t) d_t >= c_min_t;
    range: |x_t| <= a; eq6_7: |x_t| in hours with z_t = 0 (must be 0 up to tolerance).
    Values are in MWh except range and eq6_7 (fractions of demand).
    """
    x = result.hourly["x"].to_numpy(float)
    z = result.hourly["z"].to_numpy(float)
    d = data.demand
    per_window = pd.Series(d * x).groupby(data.window).sum()
    return {
        "eq3": float(max(0.0, -per_window.min())),
        "eq4": float(max(0.0, np.max(x * d - data.c_max))),
        "eq5": float(max(0.0, np.max(data.c_min - (1 + x) * d))),
        "range": float(max(0.0, np.max(np.abs(x)) - data.case.adjustment_limit)),
        "eq6_7": float(np.max(np.abs(x[z < 0.5]), initial=0.0)),
    }
