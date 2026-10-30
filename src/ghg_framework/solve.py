"""Solve the windows of a run with HiGHS (default) or Gurobi through linopy, and verify each solution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .inputs import RegionData
from .model import KG_PER_T, ModelOptions, WindowModel, build_window_model
from .results import RunResult, collect
from .scenarios import Case, ModelData, feasibility_problems, model_data

# The objective is the change in emissions plus the penalty (Eq. 1 without its constant), so the
# relative MIP gap applies to that change. Feasibility and integrality tolerances are tightened
# (1e-8; HiGHS 1.15 can stop with an internal error at 1e-9) because the indicator constraints
# (Eqs. 6-7) must hold for very small adjustments too. Any solver option can be passed instead.
HIGHS_DEFAULTS = {"mip_rel_gap": 1e-6, "mip_feasibility_tolerance": 1e-8, "primal_feasibility_tolerance": 1e-8,
                  "output_flag": False}
GUROBI_DEFAULTS = {"MIPGap": 1e-6, "IntFeasTol": 1e-8, "FeasibilityTol": 1e-8, "OutputFlag": 0}

#: Tolerances of the check that every returned solution passes (see verify_window).
VERIFY_TOL = {"integrality": 1e-6, "indicator": 1e-7, "constraints": 1e-7, "objective": 1e-6}


class SolveError(RuntimeError):
    """The solver did not return an optimal, verified solution, or the data leave no feasible adjustment."""


@dataclass
class WindowSolution:
    label: str
    hours: np.ndarray
    x: np.ndarray
    z: np.ndarray               # indicators, rounded to 0/1 after the integrality check
    objective: float            # t CO2: change in emissions + λ · indicator hours, recomputed from x and z
    termination: str            # "optimal", or the solver's condition, or "verification_failed: ..."


def verify_window(wm: WindowModel, x: np.ndarray, z_raw: np.ndarray, solver_objective: float) -> list:
    """Problems of a returned solution (empty list = verified).

    Checks, on the unrounded solver values: integrality of z; Eqs. 6-7 with the rounded z (on the
    scale of x/M); Eqs. 3-5 relative to the window's demand; and the objective recomputed from x and
    the rounded z against the solver's objective.
    """
    d = wm.data.demand[wm.hours]
    e = wm.data.intensity[wm.hours]
    c_max, c_min = wm.data.c_max[wm.hours], wm.data.c_min[wm.hours]
    lam = wm.data.case.lam
    problems = []
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(z_raw))):
        return ["non-finite solution values"]
    z = np.round(z_raw)
    if np.max(np.abs(z_raw - z)) > VERIFY_TOL["integrality"]:
        problems.append(f"z not integral (max deviation {np.max(np.abs(z_raw - z)):.2e})")
    ind = np.max(np.abs(x) / wm.big_m - z)
    if ind > VERIFY_TOL["indicator"]:
        problems.append(f"Eqs. 6-7 violated: |x|/M exceeds z by {ind:.2e}")
    scale = float(np.max(d))
    checks = {
        "Eq. 3": -float(np.sum(d * x)) / float(np.sum(d)),
        "Eq. 4": float(np.max(x * d - c_max)) / scale,
        "Eq. 5": float(np.max(c_min - (1 + x) * d)) / scale,
        "range": float(np.max(np.abs(x)) - wm.data.case.adjustment_limit),
    }
    for name, v in checks.items():
        if v > VERIFY_TOL["constraints"]:
            problems.append(f"{name} violated by {v:.2e}")
    recomputed = float(np.sum(d * e * x) / KG_PER_T + lam * np.sum(z))
    obj_scale = float(np.sum(d * e) / KG_PER_T) * wm.data.case.adjustment_limit + lam * len(d)
    if abs(recomputed - solver_objective) > VERIFY_TOL["objective"] * max(1.0, obj_scale):
        problems.append(f"objective {solver_objective:.6g} differs from the recomputed {recomputed:.6g}")
    return problems


def solve_window(wm: WindowModel, solver: str = "highs", options: Optional[Mapping[str, object]] = None,
                 require_optimal: bool = True) -> WindowSolution:
    """Solve one window model in memory and verify the solution (``verify_window``)."""
    defaults = {"highs": HIGHS_DEFAULTS, "gurobi": GUROBI_DEFAULTS}.get(solver, {})
    opts = {**defaults, **(options or {})}
    status, termination = wm.model.solve(solver_name=solver, io_api="direct", progress=False, **opts)
    termination = str(termination)
    nan = np.full(wm.hours.size, np.nan)
    if termination != "optimal":
        if require_optimal:
            raise SolveError(f"window {wm.label}: solver {solver} returned status={status}, termination={termination}")
        return WindowSolution(wm.label, wm.hours, nan, nan, float("nan"), termination)
    x = wm.x.solution.to_numpy().astype(float)
    z_raw = wm.z.solution.to_numpy().astype(float)
    problems = verify_window(wm, x, z_raw, float(wm.model.objective.value))
    if problems:
        message = f"verification_failed: {'; '.join(problems)}"
        if require_optimal:
            raise SolveError(f"window {wm.label}: {message}")
        return WindowSolution(wm.label, wm.hours, nan, nan, float("nan"), message)
    z = np.round(z_raw)
    d, e = wm.data.demand[wm.hours], wm.data.intensity[wm.hours]
    objective = float(np.sum(d * e * x) / KG_PER_T + wm.data.case.lam * np.sum(z))
    return WindowSolution(wm.label, wm.hours, x, z, objective, termination)


def solve_data(data: ModelData, solver: str = "highs", options: Optional[Mapping[str, object]] = None,
               model_options: Optional[ModelOptions] = None, require_optimal: bool = True) -> RunResult:
    """Build and solve every window of ``data`` in time order; windows are independent.

    With ``require_optimal=False`` a window without a verified optimum is kept as missing: its rows
    and the run totals are NaN, and ``totals["all_optimal"]`` is False.
    """
    bad = feasibility_problems(data)
    if len(bad):
        raise SolveError(f"{data.region}: {len(bad)} hour(s) have no feasible adjustment under Eqs. 4-5 and the "
                         f"range ±{data.case.adjustment_limit}; first: {bad.iloc[0].to_dict()}")
    sols = []
    for label in data.windows:
        wm = build_window_model(data, label, model_options)
        sols.append(solve_window(wm, solver, options, require_optimal))
    return collect(data, sols)


def run(region: RegionData, case: Optional[Case] = None, solver: str = "highs",
        options: Optional[Mapping[str, object]] = None, model_options: Optional[ModelOptions] = None,
        intensity=None, require_optimal: bool = True) -> RunResult:
    """One run: :func:`scenarios.model_data` for ``region`` and ``case``, then :func:`solve_data`."""
    return solve_data(model_data(region, case, intensity), solver, options, model_options, require_optimal)
