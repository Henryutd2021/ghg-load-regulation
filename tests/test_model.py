import numpy as np
import pytest

from ghg_framework import (Case, ModelOptions, SolveError, build_window_model, constraint_residuals, make_region,
                           model_data, run, scaled_intensity, window_labels)
from ghg_framework.results import ADJUSTED_TOL


def _residuals_ok(result, data, tol=1e-6):
    res = constraint_residuals(result, data)
    scale = float(data.demand.max())
    assert res["eq3"] <= tol * scale * len(data.demand)
    assert res["eq4"] <= tol * scale and res["eq5"] <= tol * scale
    assert res["range"] <= 1e-9 and res["eq6_7"] <= ADJUSTED_TOL


def test_window_labels(spring):
    assert len(set(window_labels(spring.time, "month"))) == 3
    assert list(dict.fromkeys(window_labels(spring.time, "quarter"))) == ["2025-Q1", "2025-Q2"]
    assert set(window_labels(spring.time, "all")) == {"all"}
    assert len(set(window_labels(spring.time, "day"))) == 92
    with pytest.raises(ValueError):
        window_labels(spring.time, "fortnight")


def test_solve_optimal_and_constraints_hold(week):
    case = Case(window="day", lam=2.0)
    data = model_data(week, case)
    result = run(week, case)
    assert result.totals["all_optimal"] and len(result.windows) == 7
    _residuals_ok(result, data)
    w = result.windows
    np.testing.assert_allclose(w["reduction_t"], w["baseline_t"] - w["optimized_t"])
    np.testing.assert_allclose(w["objective_t"], -w["reduction_t"] + case.lam * w["adjusted_hours"], atol=1e-6)
    h = result.hourly
    np.testing.assert_allclose(h["emissions_t"].sum(), result.totals["optimized_t"])
    assert (w["added_mwh"] >= w["removed_mwh"] - 1e-6).all()          # Eq. 3: consumption not reduced


def test_reduction_is_never_negative(week):
    for lam in (0.0, 5.0, 1e6):
        assert run(week, Case(window="day", lam=lam)).totals["reduction_t"] >= -1e-6


def test_larger_windows_reduce_at_least_as_much(spring):
    red = {w: run(spring, Case(window=w)).totals["reduction_t"] for w in ("month", "quarter", "all")}
    assert red["month"] <= red["quarter"] + 1e-6 <= red["all"] + 2e-6


def test_wider_range_reduces_at_least_as_much(week):
    red = [run(week, Case(window="all", adjustment_limit=a)).totals["reduction_t"] for a in (0.02, 0.05, 0.08)]
    assert red[0] <= red[1] + 1e-6 <= red[2] + 2e-6


def test_penalty_reduces_adjusted_hours(week):
    hours = [run(week, Case(window="all", lam=lam)).totals["adjusted_hours"] for lam in (0.0, 1.0, 20.0, 1e6)]
    assert hours == sorted(hours, reverse=True)
    assert hours[-1] == 0


def test_no_curtailment_means_no_regulation(week):
    result = run(week, Case(window="day", curtailment_share=0.0))
    assert abs(result.totals["reduction_t"]) < 1e-6
    assert np.allclose(result.hourly["x"], 0.0, atol=1e-7)


def test_big_m_choice_does_not_change_the_optimum(week):
    case = Case(window="all", lam=3.0)
    a = run(week, case).totals
    b = run(week, case, model_options=ModelOptions(big_m=100.0)).totals
    assert a["adjusted_hours"] == b["adjusted_hours"]
    assert a["reduction_t"] == pytest.approx(b["reduction_t"], rel=1e-5, abs=1e-6)
    with pytest.raises(ValueError, match="big_m"):
        build_window_model(model_data(week, case), "all", ModelOptions(big_m=0.01))


def test_infeasible_hours_are_reported(week):
    g = week.generation.copy()
    g.loc[g.index[10], "nuclear"] = 2.0 * week.demand[10]           # baseload above (1 + a) d_t
    bad = make_region("bad", week.time, week.demand, week.intensity, g)
    with pytest.raises(SolveError, match="no feasible adjustment"):
        run(bad, Case(window="day"))


def test_model_data_bounds_and_scaling(week):
    case = Case(curtailment_share=0.2, solar_scale=1.5, wind_scale=0.5, demand_scale=1.1)
    data = model_data(week, case)
    solar, wind = week.generation["solar"].to_numpy(), week.generation["wind"].to_numpy()
    hydro = week.generation["hydro"].to_numpy()
    np.testing.assert_allclose(data.c_max, 0.2 * (1.5 * solar + 0.5 * wind + hydro))
    np.testing.assert_allclose(data.c_min, week.generation[["coal", "nuclear"]].sum(axis=1))
    np.testing.assert_allclose(data.demand, 1.1 * week.demand)
    added = 0.5 * solar - 0.5 * wind
    np.testing.assert_allclose(data.intensity, scaled_intensity(week.intensity, 1.1 * week.demand, added, 0.2))
    np.testing.assert_allclose(data.reference_intensity, week.intensity)


def test_scaled_intensity_rule():
    e = scaled_intensity(np.array([400.0, 400.0, 400.0]), np.array([100.0, 100.0, 100.0]),
                         np.array([0.0, 50.0, 500.0]), 0.1)
    np.testing.assert_allclose(e, [400.0, 400.0 * (1 - 0.9 * 0.5), 0.0])


def test_user_intensity_and_fuel_sets(week):
    flat = np.full(week.hours, 300.0)
    data = model_data(week, Case(curtailable_fuels=("solar", "wind"), baseload_fuels=("nuclear",)), intensity=flat)
    np.testing.assert_allclose(data.intensity, flat)
    np.testing.assert_allclose(data.c_min, week.generation["nuclear"])
    result = run(week, Case(window="day"), intensity=flat)
    assert abs(result.totals["reduction_t"]) < 1e-6        # a flat intensity leaves nothing to shift toward


@pytest.mark.parametrize("bad", [dict(window="fortnight"), dict(adjustment_limit=0.0), dict(adjustment_limit=1.5),
                                 dict(lam=-1.0), dict(curtailment_share=float("nan")), dict(demand_scale=0.0),
                                 dict(baseload_fuels=("coal", "peat"))])
def test_case_validation(bad):
    with pytest.raises(ValueError):
        Case(**bad)


def test_gurobi_matches_highs(week):
    gurobipy = pytest.importorskip("gurobipy")
    try:
        gurobipy.Model().dispose()
    except gurobipy.GurobiError:
        pytest.skip("no Gurobi licence")
    case = Case(window="day", lam=2.0)
    small = week.slice("2025-06-02", "2025-06-04")              # 48 h: within the size-limited licence
    a = run(small, case).totals
    b = run(small, case, solver="gurobi").totals
    assert a["reduction_t"] == pytest.approx(b["reduction_t"], rel=1e-5)
    assert a["adjusted_hours"] == b["adjusted_hours"]
