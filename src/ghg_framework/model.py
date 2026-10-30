"""The load-regulation model of one optimization window (Eqs. 1-7 of docs/formulation.md), in linopy.

Variables (dimension ``t`` = hours of the window):

    x[t]   load adjustment, a fraction of d_t, -a <= x_t <= a
    z[t]   binary indicator of an adjusted hour

E_t = (1 + x_t) d_t e_t (Eq. 2) is a linear expression of x_t, not a separate variable. The objective
is Eq. 1 without its constant term sum_t d_t e_t, which does not change the optimum (see
formulation.md, conventions). Emissions are in tonnes: d (MWh) · e (kg/MWh) / 1000.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import linopy
import numpy as np
import pandas as pd

from .scenarios import ModelData

KG_PER_T = 1000.0


@dataclass(frozen=True)
class ModelOptions:
    """Options of the model build.

    big_m: M of Eqs. 6-7. None (default) uses M = a, the smallest value that is valid because |x_t| <= a.
    """

    big_m: Optional[float] = None


@dataclass
class WindowModel:
    """A built model of one window, with the data it was built from."""

    label: str
    hours: np.ndarray             # positions of the window's hours in the ModelData arrays
    model: linopy.Model
    data: ModelData
    big_m: float

    @property
    def x(self) -> linopy.Variable:
        return self.model.variables["x"]

    @property
    def z(self) -> linopy.Variable:
        return self.model.variables["z"]


def build_window_model(data: ModelData, label: str, options: Optional[ModelOptions] = None) -> WindowModel:
    """Build the model of window ``label`` of ``data``."""
    options = options or ModelOptions()
    a = data.case.adjustment_limit
    big_m = a if options.big_m is None else float(options.big_m)
    if not np.isfinite(big_m) or big_m < a:
        raise ValueError(f"big_m must be >= the adjustment limit a = {a} (|x_t| <= a), not {big_m}")
    idx = np.flatnonzero(data.window == label)
    if idx.size == 0:
        raise ValueError(f"no hours in window {label!r}")

    t = pd.RangeIndex(idx.size, name="t")
    d = pd.Series(data.demand[idx], index=t)
    e = pd.Series(data.intensity[idx], index=t)
    de_t = d * e / KG_PER_T                                   # t CO2 per unit of x_t
    c_max = pd.Series(data.c_max[idx], index=t)
    c_min = pd.Series(data.c_min[idx], index=t)

    m = linopy.Model()
    x = m.add_variables(lower=-a, upper=a, coords=[t], name="x")
    z = m.add_variables(binary=True, coords=[t], name="z")

    # Eq. 1 minus the constant sum_t d_t e_t:  sum_t d_t e_t x_t + λ sum_t z_t
    objective = (de_t * x).sum()
    if data.case.lam > 0:
        objective = objective + data.case.lam * z.sum()
    m.add_objective(objective)

    m.add_constraints((d * x).sum() >= 0, name="eq3_total_demand")        # sum (1+x) d >= sum d
    m.add_constraints(d * x <= c_max, name="eq4_max_increase")            # x d <= c_max
    m.add_constraints(d * x >= c_min - d, name="eq5_min_load")            # (1+x) d >= c_min
    m.add_constraints(big_m * z - x >= 0, name="eq6_indicator_up")        # z >= x / M
    m.add_constraints(big_m * z + x >= 0, name="eq7_indicator_down")      # z >= -x / M
    return WindowModel(label=label, hours=idx, model=m, data=data, big_m=big_m)
