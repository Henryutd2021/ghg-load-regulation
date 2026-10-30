"""Synthetic example regions.

Everything here is invented: four regions with different generation mixes ("Sunny", "Windy",
"Mixed", "Thermal"), smooth hourly shapes with random noise, and round illustrative emission
factors. The data show how the framework works and how to read its output; they say nothing about
any real power system.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from .inputs import FUELS, RegionData, make_region
from .preprocess import EIA_COLUMNS as RAW_COLUMNS

#: Annual output of each fuel as a share of annual demand (gas fills the rest, hour by hour).
PROFILES: Dict[str, Dict[str, float]] = {
    "Sunny": {"solar": 0.30, "wind": 0.08, "hydro": 0.07, "nuclear": 0.08, "coal": 0.02, "oil": 0.0, "other": 0.03},
    "Windy": {"solar": 0.05, "wind": 0.30, "hydro": 0.01, "nuclear": 0.10, "coal": 0.12, "oil": 0.0, "other": 0.02},
    "Mixed": {"solar": 0.06, "wind": 0.14, "hydro": 0.06, "nuclear": 0.17, "coal": 0.17, "oil": 0.01, "other": 0.03},
    "Thermal": {"solar": 0.03, "wind": 0.02, "hydro": 0.02, "nuclear": 0.22, "coal": 0.26, "oil": 0.01, "other": 0.02},
}

#: Illustrative emission factors, kg CO2 per MWh of output (round numbers, not data).
EMISSION_FACTORS = {"coal": 1000.0, "gas": 400.0, "oil": 800.0}

#: Minimum gas output as a share of demand in every hour.
GAS_MIN_SHARE = 0.05


def _shape(values: np.ndarray, annual_energy: float) -> np.ndarray:
    values = np.clip(values, 0.0, None)
    total = values.sum()
    return values * (annual_energy / total) if total > 0 else values


def synthetic_region(name: str = "Mixed", hours: int = 8760, start: str = "2025-01-01", seed: int = 0,
                     mean_demand_mwh: float = 20000.0, profile: Optional[Dict[str, float]] = None) -> RegionData:
    """One synthetic region; ``profile`` defaults to PROFILES[name]."""
    profile = dict(PROFILES[name] if profile is None else profile)
    rng = np.random.default_rng(seed)
    time = pd.date_range(start, periods=hours, freq="h")
    doy = time.day_of_year.to_numpy(float)
    hod = time.hour.to_numpy(float)
    season = np.cos(2 * np.pi * (doy - 200) / 365)                      # +1 in mid-July, -1 in mid-January

    daily = 0.10 * np.sin(np.pi * (hod - 7) / 14) * ((hod >= 7) & (hod <= 21)) - 0.06 * ((hod < 6) | (hod > 22))
    demand = mean_demand_mwh * (1 + 0.12 * season + daily + 0.02 * rng.standard_normal(hours))
    annual = demand.sum()

    sun = np.clip(np.sin(np.pi * (hod - 6) / 12), 0, None) * (1 + 0.35 * season)
    clouds = np.repeat(rng.uniform(0.45, 1.0, size=hours // 24 + 1), 24)[:hours]
    wind = np.empty(hours)
    w = 0.0
    for i, eps in enumerate(rng.standard_normal(hours)):
        w = 0.97 * w + 0.25 * eps
        wind[i] = w
    wind = 1 / (1 + np.exp(-(wind - 0.2))) * (1 - 0.25 * season) * (1 + 0.15 * np.cos(np.pi * (hod - 2) / 12))
    hydro = 1 + 0.4 * np.cos(2 * np.pi * (doy - 130) / 365) + 0.1 * np.sin(np.pi * (hod - 8) / 14)
    nuclear = np.where((doy > 90) & (doy < 115), 0.75, 1.0)
    coal = 1 + 0.25 * season + 0.05 * rng.standard_normal(hours)
    flat = np.ones(hours)

    gen = {
        "solar": _shape(sun * clouds, profile.get("solar", 0) * annual),
        "wind": _shape(wind, profile.get("wind", 0) * annual),
        "hydro": _shape(hydro, profile.get("hydro", 0) * annual),
        "nuclear": _shape(nuclear, profile.get("nuclear", 0) * annual),
        "coal": _shape(coal, profile.get("coal", 0) * annual),
        "oil": _shape(flat, profile.get("oil", 0) * annual),
        "other": _shape(flat, profile.get("other", 0) * annual),
    }
    others = sum(gen.values())
    gen["gas"] = np.maximum(demand - others, GAS_MIN_SHARE * demand)
    gen = pd.DataFrame({f: gen[f] for f in FUELS}, index=time)
    emissions = sum(EMISSION_FACTORS.get(f, 0.0) * gen[f] for f in FUELS)
    intensity = (emissions / gen.sum(axis=1)).to_numpy(float)
    return make_region(name, time, demand, intensity, gen)


def synthetic_inputs(names: Iterable[str] = tuple(PROFILES), hours: int = 8760, start: str = "2025-01-01",
                     seed: int = 0) -> Dict[str, RegionData]:
    """All synthetic regions (one seed per region, derived from ``seed``)."""
    return {n: synthetic_region(n, hours=hours, start=start, seed=seed + i) for i, n in enumerate(names)}


def synthetic_raw_records(region: RegionData, seed: int = 0, gap_share: float = 0.002) -> pd.DataFrame:
    """The region as raw hourly records in the layout of EIA's regional files, with the defects that
    the pipeline must handle: intensity in lb CO2/kWh, short gaps, a few negative net-generation values,
    and some zero-consumption hours."""
    rng = np.random.default_rng(seed)
    n = region.hours
    df = pd.DataFrame({RAW_COLUMNS["time"]: region.time})
    df[RAW_COLUMNS["demand"]] = region.demand
    df[RAW_COLUMNS["intensity"]] = region.intensity / 453.59237          # kg/MWh -> lb/kWh
    for f in FUELS:
        df[RAW_COLUMNS[f]] = region.generation[f].to_numpy()
    for col in (RAW_COLUMNS["demand"], RAW_COLUMNS["intensity"], RAW_COLUMNS["solar"]):
        idx = rng.choice(np.arange(1, n - 1), size=max(1, int(gap_share * n)), replace=False)
        df.loc[idx, col] = np.nan
    neg = rng.choice(n, size=max(1, int(gap_share * n)), replace=False)
    df.loc[neg, RAW_COLUMNS["hydro"]] = -rng.uniform(1, 50, size=neg.size)
    zero = rng.choice(np.arange(1, n - 1), size=max(1, int(gap_share * n / 2)), replace=False)
    df.loc[zero, RAW_COLUMNS["demand"]] = 0.0
    return df
