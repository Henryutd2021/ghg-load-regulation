"""Hourly data of one region, as read by the model (schemas in ``docs/data_pipeline.md``).

Units are part of the format and are not converted here:

    demand       MWh per hour   (d_t)
    intensity    kg CO2 per MWh (e_t, average intensity of the region's electricity)
    generation   MWh per hour, one column per fuel in FUELS
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

#: Fuel categories of the generation table (EIA Hourly Electric Grid Monitor categories).
FUELS: Tuple[str, ...] = ("coal", "gas", "nuclear", "oil", "hydro", "solar", "wind", "other")

#: Fuels whose output counts as renewable in the penetration metric (solar, wind, hydro).
RENEWABLE_FUELS: Tuple[str, ...] = ("hydro", "solar", "wind")

#: Default fuels of the two hourly bounds (Eqs. 4-5); see docs/formulation.md.
CURTAILABLE_FUELS: Tuple[str, ...] = ("hydro", "solar", "wind")
BASELOAD_FUELS: Tuple[str, ...] = ("coal", "nuclear")


@dataclass(frozen=True)
class RegionData:
    """Hourly demand, emission intensity and generation by fuel of one region."""

    name: str
    time: pd.DatetimeIndex
    demand: np.ndarray            # MWh per hour
    intensity: np.ndarray         # kg CO2 / MWh
    generation: pd.DataFrame      # MWh per hour, columns = FUELS, index = time

    def __post_init__(self):
        validate_region(self)

    @property
    def hours(self) -> int:
        return len(self.time)

    def fuel_sum(self, fuels: Iterable[str]) -> np.ndarray:
        """Hourly sum of the output of ``fuels`` (MWh)."""
        fuels = tuple(fuels)
        unknown = set(fuels) - set(FUELS)
        if unknown:
            raise ValueError(f"unknown fuels {sorted(unknown)}; choose from {list(FUELS)}")
        return self.generation[list(fuels)].sum(axis=1).to_numpy(float)

    def slice(self, start, end) -> "RegionData":
        """The hours with start <= time < end (timestamps or strings)."""
        mask = (self.time >= pd.Timestamp(start)) & (self.time < pd.Timestamp(end))
        if not mask.any():
            raise ValueError(f"{self.name}: no hours between {start} and {end}")
        return RegionData(self.name, self.time[mask], self.demand[mask], self.intensity[mask],
                          self.generation.loc[mask])


def validate_region(r: RegionData) -> None:
    """Raise ValueError (naming the region) if the data break the format rules."""
    n = len(r.time)
    if n == 0:
        raise ValueError(f"{r.name}: no hours")
    if not isinstance(r.time, pd.DatetimeIndex):
        raise ValueError(f"{r.name}: time must be a DatetimeIndex")
    if r.time.has_duplicates or not r.time.is_monotonic_increasing:
        raise ValueError(f"{r.name}: time must be strictly increasing without duplicates")
    for label, arr in (("demand", r.demand), ("intensity", r.intensity)):
        if np.shape(arr) != (n,):
            raise ValueError(f"{r.name}: {label} has shape {np.shape(arr)}, expected ({n},)")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{r.name}: {label} has missing or non-finite values")
    if np.any(r.demand <= 0):
        raise ValueError(f"{r.name}: demand must be > 0 in every hour")
    if np.any(r.intensity < 0):
        raise ValueError(f"{r.name}: intensity must be >= 0")
    if list(r.generation.columns) != list(FUELS):
        raise ValueError(f"{r.name}: generation columns must be {list(FUELS)} in this order")
    if len(r.generation) != n:
        raise ValueError(f"{r.name}: generation has {len(r.generation)} rows, expected {n}")
    g = r.generation.to_numpy(float)
    if not np.all(np.isfinite(g)):
        raise ValueError(f"{r.name}: generation has missing or non-finite values")
    if np.any(g < 0):
        raise ValueError(f"{r.name}: generation must be >= 0 (clip net consumption in the data pipeline)")


def make_region(name: str, time, demand, intensity, generation) -> RegionData:
    """Build a RegionData from array-likes; ``generation`` is a frame or mapping with the FUELS columns."""
    time = pd.DatetimeIndex(pd.to_datetime(time))
    gen = pd.DataFrame(generation).copy()
    missing = [f for f in FUELS if f not in gen.columns]
    if missing:
        raise ValueError(f"{name}: generation lacks fuels {missing}")
    gen = gen[list(FUELS)].astype(float)
    gen.index = time
    return RegionData(str(name), time, np.asarray(demand, float), np.asarray(intensity, float), gen)
