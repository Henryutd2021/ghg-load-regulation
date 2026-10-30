"""Read and write the input directory format (schemas in ``docs/data_pipeline.md``).

An input directory holds:

    regions.csv            region, label
    hourly/<region>.csv    time, demand_mwh, intensity_kg_per_mwh, gen_<fuel>_mwh for every fuel in FUELS

Units are part of the column names; a file with other columns is refused, not converted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd

from .inputs import FUELS, RegionData, make_region

HOURLY_COLUMNS = ["time", "demand_mwh", "intensity_kg_per_mwh"] + [f"gen_{f}_mwh" for f in FUELS]


def region_frame(region: RegionData) -> pd.DataFrame:
    """The region in the hourly file layout."""
    df = pd.DataFrame({"time": region.time, "demand_mwh": region.demand, "intensity_kg_per_mwh": region.intensity})
    for f in FUELS:
        df[f"gen_{f}_mwh"] = region.generation[f].to_numpy()
    return df


def read_region(path, name: Optional[str] = None) -> RegionData:
    """Read one hourly file (validated)."""
    path = Path(path)
    df = pd.read_csv(path)
    cols = list(df.columns)
    missing = [c for c in HOURLY_COLUMNS if c not in cols]
    extra = [c for c in cols if c not in HOURLY_COLUMNS]
    if missing or extra:
        raise ValueError(f"{path.name}: columns must be {HOURLY_COLUMNS} (missing {missing}, unexpected {extra})")
    try:
        time = pd.to_datetime(df["time"])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{path.name}: column 'time' is not a timestamp: {exc}") from exc
    gen = {f: df[f"gen_{f}_mwh"].to_numpy(float) for f in FUELS}
    try:
        return make_region(name or path.stem, time, df["demand_mwh"].to_numpy(float),
                           df["intensity_kg_per_mwh"].to_numpy(float), gen)
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from exc


def read_inputs(directory) -> Dict[str, RegionData]:
    """Read every region listed in ``regions.csv``; returns {region: RegionData}."""
    d = Path(directory)
    index = pd.read_csv(d / "regions.csv")
    if "region" not in index.columns:
        raise ValueError("regions.csv: missing column 'region'")
    names = [str(r) for r in index["region"]]
    if len(set(names)) != len(names):
        raise ValueError("regions.csv: duplicate region names")
    return {n: read_region(d / "hourly" / f"{n}.csv", n) for n in names}


def write_inputs(directory, regions: Mapping[str, RegionData], labels: Optional[Mapping[str, str]] = None,
                 description: Optional[str] = None) -> Path:
    """Write ``regions`` as an input directory (and an optional README.txt with ``description``)."""
    d = Path(directory)
    (d / "hourly").mkdir(parents=True, exist_ok=True)
    labels = labels or {}
    pd.DataFrame({"region": list(regions), "label": [labels.get(n, "") for n in regions]}).to_csv(d / "regions.csv", index=False)
    for name, r in regions.items():
        region_frame(r).to_csv(d / "hourly" / f"{name}.csv", index=False)
    if description:
        (d / "README.txt").write_text(description.rstrip() + "\n")
    return d
