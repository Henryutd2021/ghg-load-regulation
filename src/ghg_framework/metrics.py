"""Region metrics for the classification of grids (docs/scenarios.md):

- renewable penetration: share of solar, wind and hydro output in total output over the period;
- generation diversity: the Shannon-Wiener index H = -sum_r P_r ln P_r (Eq. 8), P_r = share of fuel r;
- intensity fluctuation: spread statistics of the hourly emission intensity.

Shares use total output over the whole period of the data (a year in the article).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from .inputs import FUELS, RENEWABLE_FUELS, RegionData

FLUCTUATION_STATS = ("iqr", "std", "range", "p90_p10")


def fuel_shares(region: RegionData, fuels: Iterable[str] = FUELS) -> pd.Series:
    """Share of each fuel in the total output of ``fuels`` over the period."""
    fuels = list(fuels)
    totals = region.generation[fuels].sum()
    total = float(totals.sum())
    if total <= 0:
        raise ValueError(f"{region.name}: total output is zero")
    return totals / total


def renewable_share(region: RegionData, renewable: Iterable[str] = RENEWABLE_FUELS) -> float:
    """Output of ``renewable`` fuels divided by total output (all fuels)."""
    return float(fuel_shares(region)[list(renewable)].sum())


def diversity_index(region: RegionData, fuels: Iterable[str] = FUELS) -> float:
    """Shannon-Wiener index of the generation mix, Eq. 8 (natural log; fuels with zero output are skipped)."""
    p = fuel_shares(region, fuels).to_numpy(float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def intensity_fluctuation(region: RegionData) -> dict:
    """Spread of the hourly intensity (kg CO2/MWh): interquartile range, standard deviation, range, p90-p10."""
    e = region.intensity
    q10, q25, q75, q90 = np.percentile(e, [10, 25, 75, 90])
    return {"iqr": float(q75 - q25), "std": float(np.std(e)), "range": float(np.max(e) - np.min(e)),
            "p90_p10": float(q90 - q10), "median": float(np.median(e))}


def region_metrics(regions: Mapping[str, RegionData]) -> pd.DataFrame:
    """One row per region: annual demand, mean intensity, renewable share, diversity index, fluctuation statistics."""
    rows = []
    for name, r in regions.items():
        rows.append({
            "region": name,
            "demand_twh": float(r.demand.sum() / 1e6),
            "mean_intensity_kg_per_mwh": float(np.sum(r.demand * r.intensity) / np.sum(r.demand)),
            "renewable_share": renewable_share(r),
            "diversity_index": diversity_index(r),
            **{f"intensity_{k}": v for k, v in intensity_fluctuation(r).items()},
        })
    return pd.DataFrame(rows).set_index("region")


def split_high_low(values: pd.Series, threshold: Optional[float] = None) -> pd.Series:
    """'high' / 'low' label per region: above the threshold (default: the median of ``values``) is 'high'."""
    threshold = float(values.median()) if threshold is None else float(threshold)
    return pd.Series(np.where(values > threshold, "high", "low"), index=values.index, name=f"{values.name}_group")


def classify(metrics: pd.DataFrame, fluctuation_stat: str = "iqr") -> pd.DataFrame:
    """The three classification dimensions, each split into high and low at its median."""
    if fluctuation_stat not in FLUCTUATION_STATS:
        raise ValueError(f"fluctuation_stat must be one of {list(FLUCTUATION_STATS)}")
    return pd.DataFrame({
        "penetration": split_high_low(metrics["renewable_share"]),
        "diversity": split_high_low(metrics["diversity_index"]),
        "fluctuation": split_high_low(metrics[f"intensity_{fluctuation_stat}"]),
    })
