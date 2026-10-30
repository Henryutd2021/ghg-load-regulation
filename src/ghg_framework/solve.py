"""Solve the windows of a run with HiGHS (default) or Gurobi through linopy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from .inputs import RegionData
from .model import ModelOptions, WindowModel, build_window_model
from .results import RunResult, collect
from .scenarios import Case, ModelData, feasibility_problems, model_data

# The objective is the change in emissions (Eq. 1 without its constant), so a relative MIP gap
# applies to the reduction itself. Any solver option can be passed instead.
HIGHS_DEFAULTS = {"mip_rel_gap": 1e-6, "output_flag": False}
GUROBI_DEFAULTS = {"MIPGap": 1e-6, "OutputFlag": 0}


class SolveError(RuntimeError):
    """The solver did not return an optimal solution, or the data leave no feasible adjustment."""


@dataclass
class WindowSolution:
    label: str
    hours: np.ndarray
    x: np.ndarray
    z: np.ndarray
    objective: float            # t CO2 (change in emissions + λ · adjusted hours)
    termination: str


def solve_window(wm: WindowModel, solver: str = "highs", options: Optional[Mapping[str, object]] = None,
                 require_optimal: bool = True) -> WindowSolution:
    """Solve one window model in memory."""
    defaults = {"highs": HIGHS_DEFAULTS, "gurobi": GUROBI_DEFAULTS}.get(solver, {})
    opts = {**defaults, **(options or {})}
    status, termination = wm.model.solve(solver_name=solver, io_api="direct", progress=False, **opts)
    termination = str(termination)
    if termination != "optimal":
        if require_optimal:
            raise SolveError(f"window {wm.label}: solver {solver} returned status={status}, termination={termination}")
        nan = np.full(wm.hours.size, np.nan)
        return WindowSolution(wm.label, wm.hours, nan, nan, float("nan"), termination)
    return WindowSolution(
        label=wm.label,
        hours=wm.hours,
        x=wm.x.solution.to_numpy().astype(float),
        z=np.round(wm.z.solution.to_numpy().astype(float)),
        objective=float(wm.model.objective.value),
        termination=termination,
    )


def solve_data(data: ModelData, solver: str = "highs", options: Optional[Mapping[str, object]] = None,
               model_options: Optional[ModelOptions] = None, require_optimal: bool = True) -> RunResult:
    """Build and solve every window of ``data`` in time order; windows are independent."""
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
