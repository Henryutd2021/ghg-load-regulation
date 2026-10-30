"""Generic steps of the data pipeline (docs/data_pipeline.md): from raw hourly records of a region to
:class:`RegionData`.

The default column names follow the layout of EIA's regional hourly files (Hourly Electric Grid
Monitor); pass ``columns`` for another layout.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .inputs import FUELS, RENEWABLE_FUELS, RegionData, make_region

#: Raw column for each field: time, demand, intensity and net generation by fuel.
EIA_COLUMNS: Dict[str, str] = {
    "time": "Local time",
    "demand": "Consumed Electricity",
    "intensity": "CO2 Emissions Intensity for Generated Electricity",
    "coal": "NG: COL", "gas": "NG: NG", "nuclear": "NG: NUC", "oil": "NG: OIL", "hydro": "NG: WAT",
    "solar": "NG: SUN", "wind": "NG: WND", "other": "NG: OTH",
}

#: Factors that convert an intensity to kg CO2 per MWh.
INTENSITY_UNITS: Dict[str, float] = {
    "kg/MWh": 1.0, "g/kWh": 1.0, "t/MWh": 1000.0, "lb/kWh": 453.59237, "lb/MWh": 0.45359237,
}


def convert_intensity(values, unit: str) -> np.ndarray:
    """Intensity in kg CO2/MWh from ``unit`` (one of INTENSITY_UNITS)."""
    if unit not in INTENSITY_UNITS:
        raise ValueError(f"unknown intensity unit {unit!r}; choose from {list(INTENSITY_UNITS)}")
    return np.asarray(values, float) * INTENSITY_UNITS[unit]


def select_period(df: pd.DataFrame, time_col: str, start=None, end=None) -> pd.DataFrame:
    """Rows with start <= time < end, sorted by time, with ``time_col`` parsed to timestamps."""
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col])
    if start is not None:
        out = out[out[time_col] >= pd.Timestamp(start)]
    if end is not None:
        out = out[out[time_col] < pd.Timestamp(end)]
    return out.sort_values(time_col).reset_index(drop=True)


def fill_gaps(series: pd.Series, max_gap: int = 3) -> Tuple[pd.Series, int]:
    """Fill interior runs of missing values of at most ``max_gap`` hours by linear interpolation.

    Returns the series and the number of values still missing (longer runs and edges are left for a
    decision: a longer source window, another source, or dropping the period)."""
    s = series.astype(float)
    missing = s.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    keep = missing & (run_len > max_gap)
    filled[keep] = np.nan
    return filled, int(filled.isna().sum())


def region_from_records(name: str, df: pd.DataFrame, columns: Optional[Mapping[str, str]] = None,
                        intensity_unit: str = "lb/kWh", start=None, end=None, max_gap: int = 3,
                        zero_demand_as_missing: bool = True) -> Tuple[RegionData, dict]:
    """Raw hourly records of one region -> (RegionData, report).

    Steps: select the period; treat zero demand as missing (optional); fill short gaps in demand,
    intensity and each fuel; clip negative net generation to 0; convert the intensity to kg CO2/MWh.
    Raises ValueError if values are still missing after gap filling; the report lists what was done.
    """
    cols = dict(EIA_COLUMNS if columns is None else columns)
    absent = [c for c in cols.values() if c not in df.columns]
    if absent:
        raise ValueError(f"{name}: raw records lack columns {absent}")
    raw = select_period(df, cols["time"], start, end)
    report = {"region": name, "hours": len(raw)}

    demand = raw[cols["demand"]].astype(float)
    if zero_demand_as_missing:
        report["zero_demand_hours"] = int((demand == 0).sum())
        demand = demand.mask(demand == 0)
    series = {"demand": demand, "intensity": raw[cols["intensity"]]}
    series.update({f: raw[cols[f]] for f in FUELS})
    filled, still_missing = {}, {}
    for key, s in series.items():
        report[f"missing_{key}"] = int(s.isna().sum())
        filled[key], still_missing[key] = fill_gaps(s, max_gap)
    left = {k: v for k, v in still_missing.items() if v}
    if left:
        raise ValueError(f"{name}: values still missing after filling gaps of up to {max_gap} h: {left}")

    gen = pd.DataFrame({f: filled[f] for f in FUELS})
    report["negative_generation_values"] = int((gen < 0).to_numpy().sum())
    gen = gen.clip(lower=0.0)
    intensity = convert_intensity(filled["intensity"], intensity_unit)
    region = make_region(name, raw[cols["time"]], filled["demand"].to_numpy(float), intensity, gen)
    return region, report


def qa_summary(region: RegionData) -> dict:
    """Quantities to compare with published totals before the data are used."""
    gen_total = region.generation.sum()
    total = float(gen_total.sum())
    return {
        "region": region.name,
        "first_hour": str(region.time[0]),
        "last_hour": str(region.time[-1]),
        "hours": region.hours,
        "demand_twh": float(region.demand.sum() / 1e6),
        "generation_twh": total / 1e6,
        "generation_to_demand": total / float(region.demand.sum()),
        "renewable_share": float(gen_total[list(RENEWABLE_FUELS)].sum() / total) if total > 0 else np.nan,
        "intensity_min": float(region.intensity.min()),
        "intensity_mean": float(region.intensity.mean()),
        "intensity_max": float(region.intensity.max()),
    }
