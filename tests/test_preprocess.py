import numpy as np
import pandas as pd
import pytest

from ghg_framework import EIA_COLUMNS, convert_intensity, fill_gaps, qa_summary, region_from_records, synthetic_raw_records


def test_convert_intensity_units():
    assert convert_intensity([1.0], "lb/kWh")[0] == pytest.approx(453.59237)
    assert convert_intensity([1.0], "t/MWh")[0] == pytest.approx(1000.0)
    assert convert_intensity([500.0], "g/kWh")[0] == pytest.approx(500.0)
    with pytest.raises(ValueError, match="unknown intensity unit"):
        convert_intensity([1.0], "lb/MMBtu")


def test_fill_gaps_short_and_long():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan, 8.0, np.nan])
    filled, left = fill_gaps(s, max_gap=3)
    assert filled[1] == pytest.approx(2.0)
    assert filled[3:7].isna().all()                  # a 4-hour run is left for a decision
    assert np.isnan(filled[8])                       # trailing edge is not extrapolated
    assert left == 5


def test_region_from_records_restores_the_region(week):
    raw = synthetic_raw_records(week, seed=1)
    region, report = region_from_records("Mixed", raw)
    assert report["hours"] == week.hours and report["negative_generation_values"] >= 1
    assert report["zero_demand_hours"] >= 1 and report["missing_demand"] >= 1
    np.testing.assert_allclose(region.intensity, week.intensity, rtol=0.05)          # gaps filled by interpolation
    untouched = raw[EIA_COLUMNS["demand"]].notna() & (raw[EIA_COLUMNS["demand"]] > 0)
    np.testing.assert_allclose(region.demand[untouched], week.demand[untouched])
    assert (region.generation >= 0).all().all()
    qa = qa_summary(region)
    assert qa["hours"] == week.hours and 0.9 < qa["generation_to_demand"] < 1.5


def test_region_from_records_refuses_long_gaps(week):
    raw = synthetic_raw_records(week, seed=1)
    raw.loc[20:30, EIA_COLUMNS["intensity"]] = np.nan
    with pytest.raises(ValueError, match="still missing"):
        region_from_records("Mixed", raw, max_gap=3)


def test_period_selection_and_missing_columns(week):
    raw = synthetic_raw_records(week, seed=2)
    region, _ = region_from_records("Mixed", raw, start="2025-06-03", end="2025-06-05")
    assert region.hours == 48
    with pytest.raises(ValueError, match="lack columns"):
        region_from_records("Mixed", raw.drop(columns=[EIA_COLUMNS["wind"]]))
