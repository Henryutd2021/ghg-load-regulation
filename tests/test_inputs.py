import numpy as np
import pandas as pd
import pytest

from ghg_framework import FUELS, make_region, read_inputs, read_region, write_inputs
from ghg_framework.io import HOURLY_COLUMNS, region_frame


def _parts(region):
    return region.time, region.demand.copy(), region.intensity.copy(), region.generation.copy()


def test_fuel_columns_and_hours(week):
    assert list(week.generation.columns) == list(FUELS)
    assert week.hours == 168
    assert np.all(week.demand > 0) and np.all(week.intensity >= 0)


@pytest.mark.parametrize("breaker, message", [
    (lambda t, d, e, g: (t[::-1], d, e, g), "strictly increasing"),
    (lambda t, d, e, g: (t, np.where(np.arange(d.size) == 3, np.nan, d), e, g), "non-finite"),
    (lambda t, d, e, g: (t, np.where(np.arange(d.size) == 3, 0.0, d), e, g), "demand must be > 0"),
    (lambda t, d, e, g: (t, d, -e, g), "intensity must be >= 0"),
    (lambda t, d, e, g: (t, d, e, g.assign(coal=-1.0)), "generation must be >= 0"),
    (lambda t, d, e, g: (t, d, e, g.assign(wind=np.inf)), "non-finite"),
    (lambda t, d, e, g: (t, d, e, g.drop(columns="oil")), "lacks fuels"),
    (lambda t, d, e, g: (t, d[:-1], e, g), "shape"),
])
def test_validation_errors(week, breaker, message):
    with pytest.raises(ValueError, match=message):
        make_region("bad", *breaker(*_parts(week)))


def test_slice(spring):
    april = spring.slice("2025-04-01", "2025-05-01")
    assert april.hours == 30 * 24 and april.time[0] == pd.Timestamp("2025-04-01")
    with pytest.raises(ValueError, match="no hours"):
        spring.slice("2030-01-01", "2030-02-01")


def test_input_directory_round_trip(tmp_path, week, spring):
    write_inputs(tmp_path, {"A": week, "B": spring}, labels={"A": "one week"}, description="test")
    back = read_inputs(tmp_path)
    assert list(back) == ["A", "B"]
    for name, original in (("A", week), ("B", spring)):
        r = back[name]
        assert r.name == name
        np.testing.assert_allclose(r.demand, original.demand)
        np.testing.assert_allclose(r.intensity, original.intensity)
        np.testing.assert_allclose(r.generation.to_numpy(), original.generation.to_numpy())
        assert (r.time == original.time).all()
    assert (tmp_path / "README.txt").read_text().strip() == "test"


def test_wrong_unit_column_is_refused(tmp_path, week):
    df = region_frame(week).rename(columns={"intensity_kg_per_mwh": "intensity_lb_per_kwh"})
    df.to_csv(tmp_path / "X.csv", index=False)
    with pytest.raises(ValueError, match="unexpected \\['intensity_lb_per_kwh'\\]"):
        read_region(tmp_path / "X.csv")


def test_bad_values_name_the_file(tmp_path, week):
    df = region_frame(week)
    df.loc[5, "gen_solar_mwh"] = np.nan
    df.to_csv(tmp_path / "X.csv", index=False)
    with pytest.raises(ValueError, match="X.csv: .*non-finite"):
        read_region(tmp_path / "X.csv")


def test_duplicate_regions_refused(tmp_path, week):
    write_inputs(tmp_path, {"A": week})
    pd.DataFrame({"region": ["A", "A"], "label": ["", ""]}).to_csv(tmp_path / "regions.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        read_inputs(tmp_path)


def test_hourly_columns_cover_all_fuels():
    assert HOURLY_COLUMNS[:3] == ["time", "demand_mwh", "intensity_kg_per_mwh"]
    assert len(HOURLY_COLUMNS) == 3 + len(FUELS)
