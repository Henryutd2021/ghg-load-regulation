"""Result tables of a run, and checks of the constraints recomputed from the solution.

Emissions are in tonnes of CO2. "Baseline" is the same run without regulation (x_t = 0), so the
reduction is the effect of load regulation alone. "Reference" uses the region's unchanged demand and
intensity; it differs from the baseline only in cases that change the inputs (demand, solar, wind).

Two counts of adjusted hours are reported:

- ``indicator_hours``: the sum of the binary indicators z_t, the quantity penalized in Eq. 1;
- ``adjusted_hours``: hours with |x_t| > ADJUSTED_TOL, i.e. a load change of practical size.

They are equal unless the solver keeps an indicator at 1 for a negligible x_t (possible only when it
does not raise the objective, e.g. with λ = 0).
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


def _window_row(data: ModelData, s) -> dict:
    d, e = data.demand[s.hours], data.intensity[s.hours]
    base = float(np.sum(d * e) / KG_PER_T)
    solved = bool(np.all(np.isfinite(s.x)))
    row = {"window": s.label, "hours": int(s.hours.size), "baseline_t": base, "solved": solved,
           "termination": s.termination}
    if solved:
        opt = float(np.sum((1 + s.x) * d * e) / KG_PER_T)
        shift = s.x * d
        row.update({
            "optimized_t": opt,
            "reduction_t": base - opt,
            "reduction_pct": 100 * (base - opt) / base if base > 0 else np.nan,
            "adjusted_hours": int(np.sum(np.abs(s.x) > ADJUSTED_TOL)),
            "indicator_hours": int(np.sum(s.z)),
            "added_mwh": float(np.sum(shift[shift > 0])),
            "removed_mwh": float(-np.sum(shift[shift < 0])),
            "objective_t": s.objective,
        })
    else:
        row.update({k: np.nan for k in ("optimized_t", "reduction_t", "reduction_pct", "adjusted_hours",
                                        "indicator_hours", "added_mwh", "removed_mwh", "objective_t")})
    return row


def collect(data: ModelData, solutions) -> RunResult:
    """Assemble the result tables from the window solutions (objects with label, hours, x, z, objective, termination).

    A window without a solution (NaN values) gets NaN results; then every run total except the
    baseline and reference is NaN, because a partial total would not describe the whole period.
    """
    n = len(data.demand)
    x, z = np.full(n, np.nan), np.full(n, np.nan)
    rows = []
    for s in solutions:
        x[s.hours], z[s.hours] = s.x, s.z
        rows.append(_window_row(data, s))
    cols = ["window", "hours", "baseline_t", "optimized_t", "reduction_t", "reduction_pct", "adjusted_hours",
            "indicator_hours", "added_mwh", "removed_mwh", "objective_t", "solved", "termination"]
    windows = pd.DataFrame(rows)[cols]
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
    all_solved = bool(windows["solved"].all())
    base = float(windows["baseline_t"].sum())

    def total(col):
        return float(windows[col].sum()) if all_solved else np.nan

    opt = total("optimized_t")
    totals = {
        "hours": n,
        "windows": len(windows),
        "windows_solved": int(windows["solved"].sum()),
        "baseline_t": base,
        "optimized_t": opt,
        "reduction_t": base - opt,
        "reduction_pct": 100 * (base - opt) / base if base > 0 else np.nan,
        "adjusted_hours": total("adjusted_hours"),
        "indicator_hours": total("indicator_hours"),
        "added_mwh": total("added_mwh"),
        "removed_mwh": total("removed_mwh"),
        "reference_t": float(np.sum(data.reference_demand * data.reference_intensity) / KG_PER_T),
        "all_optimal": bool(all_solved and (windows["termination"] == "optimal").all()),
    }
    if all_solved:
        totals["adjusted_hours"] = int(totals["adjusted_hours"])
        totals["indicator_hours"] = int(totals["indicator_hours"])
    return RunResult(region=data.region, case=data.case.as_dict(), hourly=hourly, windows=windows, totals=totals)


def constraint_residuals(result: RunResult, data: ModelData, big_m: float = None) -> dict:
    """Largest violation of each constraint, recomputed from the solution (0 = satisfied).

    eq3: per window, sum_t d_t x_t >= 0 (MWh); eq4: x_t d_t <= c_max_t (MWh); eq5: (1+x_t) d_t >= c_min_t (MWh);
    range: |x_t| <= a; eq6_7: |x_t|/M - z_t (M = a unless given). Raises ValueError if the result has
    windows without a solution, since their constraints cannot be checked.
    """
    x = result.hourly["x"].to_numpy(float)
    z = result.hourly["z"].to_numpy(float)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z))):
        raise ValueError("the result has windows without a solution; residuals are undefined")
    a = data.case.adjustment_limit
    big_m = a if big_m is None else big_m
    d = data.demand
    per_window = pd.Series(d * x).groupby(data.window).sum()
    return {
        "eq3": float(max(0.0, -per_window.min())),
        "eq4": float(max(0.0, np.max(x * d - data.c_max))),
        "eq5": float(max(0.0, np.max(data.c_min - (1 + x) * d))),
        "range": float(max(0.0, np.max(np.abs(x)) - a)),
        "eq6_7": float(max(0.0, np.max(np.abs(x) / big_m - z))),
    }
