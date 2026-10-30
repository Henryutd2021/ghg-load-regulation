import numpy as np
import pandas as pd
import pytest

from ghg_framework import EIA_COLUMNS, complete_hours, convert_intensity, fill_gaps, qa_summary, region_from_records, synthetic_raw_records


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
    assert len(raw) == week.hours - 2                                                 # two hours without a record
    region, report = region_from_records("Mixed", raw)
    assert report["hours"] == week.hours and report["missing_rows"] == 2
    assert report["negative_generation_values"] >= 1
    assert report["zero_demand_hours"] >= 1 and report["missing_demand"] >= 3
    assert (region.time == week.time).all()
    np.testing.assert_allclose(region.intensity, week.intensity, rtol=0.05)          # gaps filled by interpolation
    present = pd.Series(week.time).isin(pd.to_datetime(raw[EIA_COLUMNS["time"]])).to_numpy()
    raw_demand = pd.Series(np.nan, index=week.time)
    raw_demand[pd.to_datetime(raw[EIA_COLUMNS["time"]])] = raw[EIA_COLUMNS["demand"]].to_numpy()
    untouched = present & raw_demand.notna().to_numpy() & (raw_demand.to_numpy() > 0)
    np.testing.assert_allclose(region.demand[untouched], week.demand[untouched])
    assert (region.generation >= 0).all().all()
    qa = qa_summary(region)
    assert qa["hours"] == week.hours and 0.9 < qa["generation_to_demand"] < 1.5


def test_missing_rows_beyond_the_gap_limit_are_refused(week):
    raw = synthetic_raw_records(week, seed=1, dropped_rows=0)
    raw = raw.drop(index=range(40, 50)).reset_index(drop=True)                        # 10 hours without a record
    with pytest.raises(ValueError, match="still missing"):
        region_from_records("Mixed", raw, max_gap=3)


def test_complete_hours_rejects_repeats_and_sub_hourly_records():
    t = pd.date_range("2025-11-02 00:00", periods=4, freq="h")
    repeated = pd.DataFrame({"time": [t[0], t[1], t[1], t[2]], "v": range(4)})
    with pytest.raises(ValueError, match="repeated timestamps"):
        complete_hours(repeated, "time")
    half = pd.DataFrame({"time": pd.date_range("2025-01-01", periods=4, freq="30min"), "v": range(4)})
    with pytest.raises(ValueError, match="on the hour"):
        complete_hours(half, "time")
    edges = pd.DataFrame({"time": t[1:3], "v": [1.0, 2.0]})
    out, added = complete_hours(edges, "time", start=t[0], end=t[3] + pd.Timedelta(hours=1))
    assert added == 2 and len(out) == 4 and out["v"].isna().sum() == 2
    four = pd.DataFrame({"time": t, "v": [0.0, 1.0, 2.0, 3.0]})
    out, added = complete_hours(four, "time", end=t[3] + pd.Timedelta(minutes=30))   # [.., 03:30) keeps 03:00
    assert added == 0 and len(out) == 4 and out["v"].tolist() == [0.0, 1.0, 2.0, 3.0]
    out, added = complete_hours(four.iloc[::-1], "time")                            # unsorted input
    assert added == 0 and out["time"].is_monotonic_increasing
    with pytest.raises(ValueError, match="no whole hour"):
        complete_hours(four, "time", start=t[0] + pd.Timedelta(minutes=10), end=t[0] + pd.Timedelta(minutes=50))


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


def test_synthetic_raw_records_on_short_inputs(week):
    short = week.slice("2025-06-02 00:00", "2025-06-02 04:00")
    raw = synthetic_raw_records(short, seed=0)
    assert len(raw) == 2                                                             # first and last row kept
    with pytest.raises(ValueError, match="need at least"):
        synthetic_raw_records(short, dropped_rows=3)
    with pytest.raises(ValueError, match="non-negative integer"):
        synthetic_raw_records(short, dropped_rows=-1)
    with pytest.raises(ValueError, match="at least 3 hours"):
        synthetic_raw_records(week.slice("2025-06-02 00:00", "2025-06-02 02:00"), dropped_rows=0)
