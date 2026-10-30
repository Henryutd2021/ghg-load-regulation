"""Cases (strategy settings and input changes) and the data of one model run.

A :class:`Case` holds the three regulation strategies of the article (window, adjustment range,
penalty λ) and the input changes used in sensitivity studies (curtailment share, scaled demand,
solar and wind). :func:`model_data` turns a region and a case into the arrays of the model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .inputs import BASELOAD_FUELS, FUELS, REGULATION_BASE_FUELS, RegionData

#: Window lengths understood by :func:`window_labels`.
WINDOWS: Tuple[str, ...] = ("day", "week", "month", "quarter", "year", "all")


@dataclass(frozen=True)
class Case:
    """Settings of one run. Defaults: quarterly window, ±5 % range, c_max = 10 % of total generation, λ = 0."""

    name: str = "benchmark"
    window: str = "quarter"                     # optimization window (cycle T of Eq. 1)
    adjustment_limit: float = 0.05              # a: -a <= x_t <= a (fraction of demand)
    curtailment_share: float = 0.10             # κ: c_max_t = κ · output of regulation_base_fuels
    lam: float = 0.0                            # λ: penalty per adjusted hour (t CO2)
    demand_scale: float = 1.0                   # multiplies d_t
    solar_scale: float = 1.0                    # multiplies solar output
    wind_scale: float = 1.0                     # multiplies wind output
    regulation_base_fuels: Tuple[str, ...] = REGULATION_BASE_FUELS
    baseload_fuels: Tuple[str, ...] = BASELOAD_FUELS

    def __post_init__(self):
        if self.window not in WINDOWS:
            raise ValueError(f"window must be one of {list(WINDOWS)}, not {self.window!r}")
        for label in ("adjustment_limit", "curtailment_share", "lam", "demand_scale", "solar_scale", "wind_scale"):
            v = getattr(self, label)
            if not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f"{label} must be a finite number >= 0, not {v!r}")
        if not 0 < self.adjustment_limit < 1:
            raise ValueError("adjustment_limit must be between 0 and 1 (a fraction of demand)")
        if self.curtailment_share > 1:
            raise ValueError("curtailment_share must be between 0 and 1")
        if self.demand_scale == 0:
            raise ValueError("demand_scale must be > 0")
        for label in ("regulation_base_fuels", "baseload_fuels"):
            fuels = tuple(getattr(self, label))
            unknown = set(fuels) - set(FUELS)
            if unknown:
                raise ValueError(f"{label}: unknown fuels {sorted(unknown)}; choose from {list(FUELS)}")
            if len(set(fuels)) != len(fuels):
                raise ValueError(f"{label}: each fuel may appear only once")
            object.__setattr__(self, label, fuels)

    def with_(self, **changes) -> "Case":
        """A copy with some settings changed (the name too, if given)."""
        return replace(self, **changes)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["regulation_base_fuels"] = "+".join(self.regulation_base_fuels)
        d["baseload_fuels"] = "+".join(self.baseload_fuels)
        return d


@dataclass
class ModelData:
    """Arrays of one run (all of length = hours of the region)."""

    region: str
    case: Case
    time: pd.DatetimeIndex
    window: np.ndarray            # window label of every hour
    demand: np.ndarray            # d_t, MWh
    intensity: np.ndarray         # e_t, kg CO2/MWh
    c_max: np.ndarray             # Eq. 4 right-hand side, MWh
    c_min: np.ndarray             # Eq. 5 right-hand side, MWh
    reference_intensity: np.ndarray = field(repr=False)   # e_t of the unchanged inputs
    reference_demand: np.ndarray = field(repr=False)      # d_t of the unchanged inputs

    @property
    def windows(self):
        """Window labels in time order."""
        return list(dict.fromkeys(self.window.tolist()))


def window_labels(time: pd.DatetimeIndex, window: str) -> np.ndarray:
    """Label of the optimization window of every hour (calendar windows of the timestamps)."""
    if window == "day":
        labels = time.strftime("%Y-%m-%d")
    elif window == "week":
        iso = time.isocalendar()
        labels = [f"{y}-W{w:02d}" for y, w in zip(iso.year, iso.week)]
    elif window == "month":
        labels = time.strftime("%Y-%m")
    elif window == "quarter":
        labels = [f"{y}-Q{q}" for y, q in zip(time.year, time.quarter)]
    elif window == "year":
        labels = time.strftime("%Y")
    elif window == "all":
        labels = ["all"] * len(time)
    else:
        raise ValueError(f"window must be one of {list(WINDOWS)}, not {window!r}")
    return np.asarray(labels, dtype=object)


def scaled_intensity(intensity: np.ndarray, demand: np.ndarray, added: np.ndarray, curtailment_share: float) -> np.ndarray:
    """Intensity after a change ``added`` (MWh) in solar and wind output.

    Framework rule: the added output, less the curtailment share κ, displaces (or, if negative,
    is replaced by) supply at the hour's average intensity:  e'_t = e_t · max(0, 1 - (1-κ)·Δg_t / d_t).
    Users with a dispatch model of their own can pass its intensity series instead.
    """
    return intensity * np.clip(1.0 - (1.0 - curtailment_share) * added / demand, 0.0, None)


def model_data(region: RegionData, case: Optional[Case] = None, intensity: Optional[np.ndarray] = None) -> ModelData:
    """Arrays of the model for ``region`` under ``case``.

    - d_t = demand_scale · demand.
    - Solar and wind output are scaled; c_max_t = κ · output of ``case.regulation_base_fuels`` (Eq. 4),
      c_min_t = output of ``case.baseload_fuels`` (Eq. 5).
    - e_t is the region's intensity, changed by :func:`scaled_intensity` when solar or wind are scaled,
      unless ``intensity`` (kg CO2/MWh, one value per hour) is given.
    """
    case = case or Case()
    gen = region.generation.copy()
    gen["solar"] = gen["solar"] * case.solar_scale
    gen["wind"] = gen["wind"] * case.wind_scale
    added = (gen["solar"] - region.generation["solar"] + gen["wind"] - region.generation["wind"]).to_numpy(float)
    demand = region.demand * case.demand_scale
    if intensity is not None:
        e = np.asarray(intensity, float)
        if e.shape != demand.shape or not np.all(np.isfinite(e)) or np.any(e < 0):
            raise ValueError("intensity must have one finite value >= 0 per hour")
    elif case.solar_scale != 1 or case.wind_scale != 1:
        e = scaled_intensity(region.intensity, demand, added, case.curtailment_share)
    else:
        e = region.intensity.copy()
    return ModelData(
        region=region.name,
        case=case,
        time=region.time,
        window=window_labels(region.time, case.window),
        demand=demand,
        intensity=e,
        c_max=case.curtailment_share * gen[list(case.regulation_base_fuels)].sum(axis=1).to_numpy(float),
        c_min=gen[list(case.baseload_fuels)].sum(axis=1).to_numpy(float),
        reference_intensity=region.intensity.copy(),
        reference_demand=region.demand.copy(),
    )


def feasibility_problems(data: ModelData) -> pd.DataFrame:
    """Hours where Eqs. 4-5 and the adjustment range leave no feasible x_t.

    x_t must satisfy max(-a, c_min_t/d_t - 1) <= x_t <= min(a, c_max_t/d_t). Returns those hours
    with both limits; an empty frame means every hour has a feasible range.
    """
    a = data.case.adjustment_limit
    lower = np.maximum(-a, data.c_min / data.demand - 1.0)
    upper = np.minimum(a, data.c_max / data.demand)
    bad = lower > upper + 1e-12
    return pd.DataFrame({"time": data.time[bad], "lower": lower[bad], "upper": upper[bad]})
