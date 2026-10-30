# Study design: grid types, regulation strategies and sensitivity cases

## Three regulation strategies

The model has three adjustable dimensions, which the article calls the three types of optimization strategy:

| Strategy | Parameter | Code |
|---|---|---|
| size of the time window for load regulation | the optimization cycle $T$: month, quarter or year | `Case(window="month" \| "quarter" \| "year")` (also `"day"`, `"week"`, `"all"`) |
| magnitude of load regulation | the adjustment range $a$ | `Case(adjustment_limit=...)` |
| sensitivity (difficulty) of regulation | the weighting factor $\lambda$ of adjusted hours | `Case(lam=...)` |

The article's benchmark setting is an adjustment range of ±5 % with a quarterly window, and a regulation capacity bounded by a 10 % curtailment share. `Case()` uses these settings, with $\lambda = 0$; set `lam` for each study.

## Grid types

The regions are classified along three dimensions, each split into a high and a low group:

| Dimension | Metric | Code |
|---|---|---|
| clean-energy penetration | share of solar, wind and hydro output in total annual output | `metrics.renewable_share` |
| generation-resource richness (diversity) | Shannon–Wiener index, Eq. (8) | `metrics.diversity_index` |
| emission-intensity fluctuation | spread of the hourly intensity | `metrics.intensity_fluctuation` (interquartile range, standard deviation, range, p90–p10) |

- `metrics.region_metrics(regions)` computes all three for a set of regions.
- `metrics.classify(table)` splits each dimension at its median into "high" and "low". `split_high_low(values, threshold)` takes a threshold instead.
- Penetration and diversity can overlap. When a region ranks high on both, assign it to the dimension where it is more distinctive, and record the choice.

## Case studies

Each grid dimension is paired with one strategy:

| Case study | Regions compared | Strategy varied |
|---|---|---|
| 1 | high vs low renewable penetration | window size: monthly, quarterly, yearly |
| 2 | high vs low generation diversity | adjustment range, from ±2 % to ±10 % |
| 3 | high vs low intensity fluctuation | penalty $\lambda$; reported as adjusted hours per month |

Results to read for each run (`RunResult`):

- `totals["reduction_t"]`, `totals["reduction_pct"]`: the reduction of attributed emissions against the same case without regulation, in tonnes and in % of the baseline;
- `windows`: the same per window, with adjusted hours and the energy moved;
- `hourly`: $x_t$, adjusted load, $z_t$ and emissions for every hour.

## Sensitivity families

| Family | What changes | Code |
|---|---|---|
| future renewable share | hourly solar and wind output scaled by factors | `Case(solar_scale=..., wind_scale=...)` |
| energy available for regulation | the curtailment share $\kappa$ of Eq. (4); the article explores 2 % to 18 % | `Case(curtailment_share=...)` |
| difficulty of regulation | $\lambda$ | `Case(lam=...)` |
| demand level | hourly demand scaled | `Case(demand_scale=...)` |

When solar or wind output is scaled:

- $c^{max}_t$ is recomputed from the scaled output.
- The intensity changes by a framework rule: the added output, less the curtailment share, displaces supply at the hour's average intensity, $e'_t = e_t \max\!\big(0,\, 1 - (1-\kappa)\,\Delta g_t / d_t\big)$ (`scenarios.scaled_intensity`).
- To use intensities from a dispatch model instead, pass them to `run(region, case, intensity=...)`.
- `totals["baseline_t"]` is the case's own emissions without regulation, so `reduction_t` isolates the effect of load regulation. `totals["reference_t"]` gives the emissions of the unchanged inputs, for comparisons across cases.

## In code

```python
from ghg_framework import Case, run, read_inputs, region_metrics, classify

regions = read_inputs("my_inputs/")
groups = classify(region_metrics(regions))

base = Case()                                             # quarterly window, ±5 %, κ = 10 %
by_window = {w: run(regions["A"], base.with_(window=w)).totals for w in ("month", "quarter", "year")}
by_range = {a: run(regions["B"], base.with_(adjustment_limit=a)).totals for a in (0.02, 0.04, 0.06, 0.08, 0.10)}
by_lambda = {l: run(regions["C"], base.with_(window="month", lam=l)).windows["adjusted_hours"].mean() for l in (0, 1, 5, 20)}
```

[`examples/strategies.py`](../examples/strategies.py) runs all of these on the synthetic regions.
