# Study design: grid types, regulation strategies and sensitivity cases

This document separates three things:

- **the article's study design** (Methods and Results sections of the article);
- **the tools of this software** that set up such a design for any data;
- **the synthetic example settings** used in `examples/`.

## The article's study design

### Data and regions

- Hourly data for the year 2023 from the U.S. Energy Information Administration, for twelve regions of the contiguous United States ([data_pipeline.md](data_pipeline.md)).

### Three regulation strategies

The model has three adjustable dimensions, which the article calls the three types of optimization strategy:

| Strategy | Parameter | Code |
|---|---|---|
| size of the time window for load regulation | the optimization cycle $T$: month, quarter or year | `Case(window="month" \| "quarter" \| "year")` |
| magnitude of load regulation | the adjustment range $a$ | `Case(adjustment_limit=...)` |
| sensitivity (difficulty) of regulation | the weighting factor $\lambda$ of adjusted hours | `Case(lam=...)` |

### Three grid dimensions

| Dimension | Metric | Groups |
|---|---|---|
| clean-energy penetration | share of solar, wind and hydro output in total annual output | high and low, three regions each (the article's Table 1) |
| generation-resource richness (diversity) | Shannon–Wiener index, Eq. (8) | high and low, three regions each (Table 1) |
| emission-intensity fluctuation | the range of the hourly intensity fluctuation | all twelve regions, regrouped into a high and a low group |

Penetration and diversity can overlap. The article assigns a region that ranks high on both to the dimension where it is more distinctive.

### Case studies

Each grid dimension is paired with one strategy:

| Case study | Regions compared | Strategy varied |
|---|---|---|
| 1 | high vs low renewable penetration | window size: monthly, quarterly, yearly |
| 2 | high vs low generation diversity | adjustment range, from ±2 % to ±10 % |
| 3 | high vs low intensity fluctuation | penalty $\lambda$; adjusted hours per month |

### Sensitivity analysis

- **Benchmark:** an adjustment range of ±5 % with a quarterly window; the regulation capacity bounded at 10 % (formulation.md, Eq. 4).
- **Families:**
  1. a higher future share of solar and wind: the hourly output of each is scaled by percentages;
  2. the energy available for regulation: the 10 % bound varied between 2 % and 18 %;
  3. the difficulty of regulation: $\lambda$ varied.
- $\lambda$ is set per study. It is in the unit of the objective, so a value from another study must be converted to this framework's unit (t CO2 per adjusted hour) before use (formulation.md, conventions).

## Tools of this software

| Task | Code |
|---|---|
| metrics of all regions | `metrics.region_metrics(regions)`: annual demand, mean intensity, renewable share, diversity index, and the spread statistics range (max − min), interquartile range, standard deviation, p90–p10 |
| high/low groups | `metrics.classify(table, fluctuation_stat="range")` splits each dimension at its median; `split_high_low(values, threshold)` takes a threshold instead. The median split, and the choice among spread statistics, are tools of this software; choose groups that match your study |
| one run | `run(region, Case(...))` |
| windows beyond the article's | `Case(window="day" \| "week" \| "all")` |
| input changes | see the table below |

| Sensitivity family | What changes | Code |
|---|---|---|
| future renewable share | hourly solar and wind output scaled by factors | `Case(solar_scale=..., wind_scale=...)` |
| energy available for regulation | the share $\kappa$ of Eq. (4) | `Case(curtailment_share=...)`; `Case(regulation_base_fuels=...)` sets the fuels it applies to |
| difficulty of regulation | $\lambda$ | `Case(lam=...)` |
| demand level | hourly demand scaled | `Case(demand_scale=...)` |

`Case()` holds the benchmark settings above ($a$ = 5 %, quarterly window, $\kappa$ = 10 % of total generation) with $\lambda = 0$; set `lam` for each study.

When solar or wind output is scaled:

- $c^{max}_t$ is recomputed from the scaled output.
- The intensity changes by a rule of this framework: the added output, less the share $\kappa$, displaces supply at the hour's average intensity, $e'_t = e_t \max\!\big(0,\, 1 - (1-\kappa)\,\Delta g_t / d_t\big)$ (`scenarios.scaled_intensity`).
- To use intensities from a dispatch model instead, pass them to `run(region, case, intensity=...)`.
- `totals["baseline_t"]` is the case's own emissions without regulation, so `reduction_t` isolates the effect of load regulation. `totals["reference_t"]` gives the emissions of the unchanged inputs, for comparisons across cases.

Results to read for each run (`RunResult`):

- `totals["reduction_t"]`, `totals["reduction_pct"]`: the reduction of attributed emissions against the same case without regulation, in tonnes and in % of the baseline;
- `totals["adjusted_hours"]`, `totals["indicator_hours"]` and the per-window values in `windows` (case study 3 reads adjusted hours per month);
- `hourly`: $x_t$, adjusted load, $z_t$ and emissions for every hour.

## Synthetic example settings

`examples/strategies.py` applies the design to the four synthetic regions of `ghg_framework.synthetic`: windows month/quarter/year; ranges ±2 % to ±10 %; $\lambda$ = 0, 1, 5 and 20 t CO2 per adjusted hour (chosen for the synthetic data only); $\kappa$ from 2 % to 18 %; solar or wind scaled by 1.5. With four regions, each median split puts two regions in each group.

```python
from ghg_framework import Case, run, read_inputs, region_metrics, classify

regions = read_inputs("my_inputs/")
groups = classify(region_metrics(regions))

base = Case()                                             # quarterly window, ±5 %, κ = 10 % of total generation
by_window = {w: run(regions["A"], base.with_(window=w)).totals for w in ("month", "quarter", "year")}
by_range = {a: run(regions["B"], base.with_(adjustment_limit=a)).totals for a in (0.02, 0.04, 0.06, 0.08, 0.10)}
by_lambda = {l: run(regions["C"], base.with_(window="month", lam=l)).windows["adjusted_hours"].mean() for l in (0, 1, 5, 20)}
```
