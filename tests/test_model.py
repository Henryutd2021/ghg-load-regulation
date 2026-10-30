import numpy as np
import pytest

import pandas as pd

from ghg_framework import (FUELS, Case, ModelOptions, SolveError, build_window_model, constraint_residuals, make_region,
                           model_data, run, scaled_intensity, window_labels)
from ghg_framework.results import collect
from ghg_framework.solve import WindowSolution


def _residuals_ok(result, data, tol=1e-7):
    res = constraint_residuals(result, data)
    scale = float(data.demand.max())
    assert res["eq3"] <= tol * scale * len(data.demand)
    assert res["eq4"] <= tol * scale and res["eq5"] <= tol * scale
    assert res["range"] <= 1e-9 and res["eq6_7"] <= tol


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
    np.testing.assert_allclose(w["objective_t"], -w["reduction_t"] + case.lam * w["indicator_hours"], atol=1e-6)
    assert (w["indicator_hours"] == w["adjusted_hours"]).all()
    h = result.hourly
    np.testing.assert_allclose(h["emissions_t"].sum(), result.totals["optimized_t"])
    assert (w["added_mwh"] >= w["removed_mwh"] - 1e-6).all()          # Eq. 3: consumption not reduced


def test_reduction_is_never_negative_when_no_adjustment_is_feasible(week):
    # x = 0 is feasible in every hour of this fixture (baseload below demand), so the optimum cannot
    # increase emissions; with baseload above demand Eq. 5 can force an increase (see the next test).
    assert np.all(model_data(week).c_min <= week.demand)
    for lam in (0.0, 5.0, 1e6):
        assert run(week, Case(window="day", lam=lam)).totals["reduction_t"] >= -1e-6


def test_baseload_above_demand_forces_an_increase():
    time = pd.date_range("2025-01-01", periods=4, freq="h")
    gen = {f: np.zeros(4) for f in FUELS}
    gen["nuclear"] = np.full(4, 101.0)
    gen["wind"] = np.full(4, 50.0)
    r = make_region("base", time, np.full(4, 100.0), np.full(4, 300.0), gen)
    result = run(r, Case(window="all"))
    np.testing.assert_allclose(result.hourly["x"], 0.01, atol=1e-9)        # Eq. 5: (1 + x) 100 >= 101
    assert result.totals["reduction_t"] < 0


def test_tiny_adjustments_pay_the_penalty():
    """Regression: an adjustment far below the solver's default tolerances must not escape Eqs. 6-7."""
    time = pd.date_range("2025-01-01", periods=2, freq="h")
    gen = {f: np.zeros(2) for f in FUELS}
    gen["gas"] = np.full(2, 1e5)
    r = make_region("tiny", time, np.full(2, 1e5), np.array([0.0, 1000.0]), gen)
    case = Case(window="all", curtailment_share=5e-7, lam=10.0)        # c_max = 0.05 MWh: x <= 5e-7
    result = run(r, case)
    # shifting 0.05 MWh saves 0.05 t but costs 2 x 10 t of penalty: the optimum is no adjustment
    np.testing.assert_allclose(result.hourly["x"], 0.0, atol=1e-12)
    assert result.totals["indicator_hours"] == 0 and result.totals["reduction_t"] == pytest.approx(0.0, abs=1e-9)
    _residuals_ok(result, model_data(r, case))
    big_m = run(r, case, model_options=ModelOptions(big_m=100.0))          # a large M must not change this
    np.testing.assert_allclose(big_m.hourly["x"], 0.0, atol=1e-12)
    assert big_m.totals["indicator_hours"] == 0
    cheap = run(r, case.with_(lam=0.001))                                   # now the shift pays off
    assert cheap.totals["indicator_hours"] == 2
    assert cheap.totals["adjusted_hours"] == 0                              # |x| = 5e-7 is below ADJUSTED_TOL
    assert cheap.totals["reduction_t"] == pytest.approx(0.05, rel=1e-6)


def test_missing_window_makes_totals_unavailable(week):
    data = model_data(week, Case(window="day"))
    labels = data.windows
    sols = []
    for i, label in enumerate(labels):
        hours = np.flatnonzero(data.window == label)
        nan = np.full(hours.size, np.nan)
        zero = np.zeros(hours.size)
        sols.append(WindowSolution(label, hours, nan if i == 3 else zero, nan if i == 3 else zero,
                                   float("nan") if i == 3 else 0.0, "time_limit" if i == 3 else "optimal"))
    result = collect(data, sols)
    t = result.totals
    assert t["windows_solved"] == 6 and not t["all_optimal"]
    for key in ("optimized_t", "reduction_t", "reduction_pct", "adjusted_hours", "added_mwh"):
        assert np.isnan(t[key])
    assert np.isfinite(t["baseline_t"])
    assert not result.windows.loc[3, "solved"] and np.isnan(result.windows.loc[3, "reduction_t"])
    unsolved = data.window == labels[3]
    assert result.hourly.loc[unsolved, "adjusted"].isna().all()
    assert not result.hourly.loc[~unsolved, "adjusted"].isna().any()
    with pytest.raises(ValueError, match="without a solution"):
        constraint_residuals(result, data)


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
    total = week.generation.sum(axis=1).to_numpy()
    np.testing.assert_allclose(data.c_max, 0.2 * (total + 0.5 * solar - 0.5 * wind))   # default: total generation
    hydro = week.generation["hydro"].to_numpy()
    renewable = model_data(week, case.with_(regulation_base_fuels=("hydro", "solar", "wind")))
    np.testing.assert_allclose(renewable.c_max, 0.2 * (1.5 * solar + 0.5 * wind + hydro))
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
    data = model_data(week, Case(regulation_base_fuels=("solar", "wind"), baseload_fuels=("nuclear",)), intensity=flat)
    np.testing.assert_allclose(data.intensity, flat)
    np.testing.assert_allclose(data.c_min, week.generation["nuclear"])
    result = run(week, Case(window="day"), intensity=flat)
    assert abs(result.totals["reduction_t"]) < 1e-6        # a flat intensity leaves nothing to shift toward


@pytest.mark.parametrize("bad", [dict(window="fortnight"), dict(adjustment_limit=0.0), dict(adjustment_limit=1.5),
                                 dict(lam=-1.0), dict(curtailment_share=float("nan")), dict(curtailment_share=1.5),
                                 dict(demand_scale=0.0), dict(baseload_fuels=("coal", "peat")),
                                 dict(regulation_base_fuels=("solar", "solar"))])
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


def test_verification_rejects_indicator_and_constraint_violations(week):
    from ghg_framework import verify_window
    case = Case(window="all", lam=2.0)
    wm = build_window_model(model_data(week, case), "all")
    n = wm.hours.size
    zero = np.zeros(n)
    assert verify_window(wm, zero, zero, 0.0) == []
    x = zero.copy()
    x[0] = 1e-5                                                     # adjusted, but z = 0
    assert any("Eqs. 6-7" in p for p in verify_window(wm, x, zero, 0.0))
    assert any("integral" in p for p in verify_window(wm, zero, np.full(n, 0.4), 0.0))
    assert any("range" in p for p in verify_window(wm, np.full(n, 0.06), np.ones(n), 0.0))
    assert any("objective" in p for p in verify_window(wm, zero, zero, 123.0))
    assert verify_window(wm, zero, zero, float("nan")) == ["non-finite solution values or objective"]
    assert any("outside {0, 1}" in p for p in verify_window(wm, zero, np.full(n, 2.0), 2.0 * 2 * n))
    big = build_window_model(model_data(week, case), "all", ModelOptions(big_m=100.0))
    assert any("Eqs. 6-7" in p for p in verify_window(big, x, zero, 0.0))   # checked on x/a, not x/M
