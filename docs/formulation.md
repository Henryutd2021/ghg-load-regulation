# Formulation

This document states the load-regulation model and the grid-classification metric of

> H. Li, G. Shen, S. Senemmar, A. Mehmani and J. Zhang, "Real-time greenhouse gas emission intensity informed demand-side load regulation for power grid decarbonization," *Cell Reports Sustainability* 2 (2025) 100367. [doi:10.1016/j.crsus.2025.100367](https://doi.org/10.1016/j.crsus.2025.100367)

- Equation numbers (1)–(8) follow the article (Methods), so that each part can be matched to it.
- [Conventions of this implementation](#conventions-of-this-implementation) lists the definitions that the framework makes explicit.

## Idea

A region's hourly electricity demand is re-timed toward hours in which the region's electricity has a lower greenhouse-gas emission intensity. The total energy consumed in each optimization cycle stays at least at its baseline. A penalty on the number of adjusted hours balances the emission objective against the disruption caused by frequent adjustments.

## Sets and indices

| Symbol | Meaning | Code |
|---|---|---|
| $`t = 1,\dots,T`$ | hours of one optimization cycle (window) | dimension `t` of a window model |
| $`w`$ | optimization window: a calendar day, week, month, quarter or year, or all hours | `Case.window`, `window_labels` |
| $`r = 1,\dots,R`$ | generation resource types (fuels) | `FUELS`: coal, gas, nuclear, oil, hydro, solar, wind, other |

## Parameters

| Symbol | Meaning | Unit | Code |
|---|---|---|---|
| $`d_t`$ | baseline electricity demand in hour $`t`$ | MWh | `ModelData.demand` |
| $`e_t`$ | real-time emission intensity of the region's electricity in hour $`t`$ | kg CO2/MWh | `ModelData.intensity` |
| $`D_t`$ | total demand of the cycle before adjustment | MWh | $`\sum_t d_t`$ of the window |
| $`c^{max}_t`$ | maximum regulation capacity in hour $`t`$ (limit for re-integrating curtailed renewable generation) | MWh | `ModelData.c_max` |
| $`c^{min}_t`$ | minimum capacity in hour $`t`$ (baseline of unregulatable sources such as coal and nuclear) | MWh | `ModelData.c_min` |
| $`a`$ | adjustment range: the predefined limit of $`\lvert x_t\rvert`$, $`0 < a < 1`$ | fraction of $`d_t`$ | `Case.adjustment_limit` |
| $`\kappa`$ | share of generation available for regulation in Eq. (4), $`0 \le \kappa \le 1`$ (see conventions) | – | `Case.curtailment_share` |
| $`\lambda`$ | weighting factor of the adjusted hours, $`\lambda \ge 0`$ | t CO2 per adjusted hour | `Case.lam` |
| $`M`$ | big-M constant of Eqs. (6)–(7) | – | `ModelOptions.big_m` |
| $`P_r`$ | share of total generation from resource $`r`$ | – | `metrics.fuel_shares` |

## Variables

| Symbol | Meaning | Domain | Code |
|---|---|---|---|
| $`x_t`$ | load adjustment in hour $`t`$, as a fraction of $`d_t`$ (negative: reduction, positive: increase) | $`-a \le x_t \le a`$ | `x` |
| $`z_t`$ | 1 if a non-zero adjustment occurs in hour $`t`$, 0 otherwise | $`\{0, 1\}`$ | `z` |
| $`E_t`$ | emissions in hour $`t`$ after adjustment | kg CO2 with the units above; reported in t | expression of $`x_t`$ (Eq. 2) |

## Load-regulation model

For each optimization cycle:

```math
\min \; \sum_{t=1}^{T} E_t \;+\; \lambda \sum_{t=1}^{T} z_t \qquad (1)
```

```math
E_t = (1 + x_t)\, d_t\, e_t \qquad (2)
```

subject to

```math
\sum_{t=1}^{T} (1 + x_t)\, d_t \;\ge\; D_t \qquad (3)
```

```math
x_t\, d_t \;\le\; c^{max}_t \qquad (4)
```

```math
(1 + x_t)\, d_t \;\ge\; c^{min}_t \qquad (5)
```

```math
z_t \;\ge\; \frac{x_t}{M} \qquad (6)
```

```math
z_t \;\ge\; -\frac{x_t}{M} \qquad (7)
```

with $`-a \le x_t \le a`$ and $`z_t \in \{0, 1\}`$.

With $`d_t`$ in MWh and $`e_t`$ in kg CO2/MWh, Eq. (2) gives kg CO2. The framework divides by 1000 and works in tonnes throughout: $`E_t = (1 + x_t)\,d_t\,e_t / 1000`$ (conventions below).

- **Eq. (1)** minimizes the emissions of the cycle plus a penalty proportional to the number of adjusted hours.
- **Eq. (2)** gives the emissions of hour $`t`$ for the adjusted load.
- **Eq. (3)** keeps the total consumption of the cycle at least at its baseline: load is moved in time, not removed.
- **Eqs. (4)–(5)** bound each hour: the added load cannot exceed the regulation capacity $`c^{max}_t`$, and the adjusted load cannot fall below $`c^{min}_t`$.
- **Eqs. (6)–(7)** force $`z_t = 1`$ whenever $`x_t \neq 0`$. When $`x_t = 0`$, minimizing Eq. (1) with $`\lambda > 0`$ sets $`z_t = 0`$.

The model is a mixed-integer linear program with $`T`$ continuous and $`T`$ binary variables per cycle.

## Grid-classification metric

The regions are classified by renewable penetration, generation diversity and intensity fluctuation ([scenarios.md](scenarios.md)). Generation diversity is the Shannon–Wiener index:

```math
H = -\sum_{r}^{R} P_r \ln P_r \qquad (8)
```

A higher $`H`$ means a more diverse mix of generation resources.

## Mapping to the code

| Equation | Code (`src/ghg_framework/`) |
|---|---|
| (1) | objective of `model.build_window_model` |
| (2) | `results.collect` (hourly `emissions_t`) and the objective coefficients |
| (3) | constraint `eq3_total_demand` |
| (4) | constraint `eq4_max_increase` |
| (5) | constraint `eq5_min_load` |
| (6) | constraint `eq6_indicator_up` |
| (7) | constraint `eq7_indicator_down` |
| range of $`x_t`$ | bounds of variable `x` |
| (8) | `metrics.diversity_index` |

`results.constraint_residuals` recomputes Eqs. (3)–(7) and the range from a solution.

## Conventions of this implementation

- **Units:** demand and generation in MWh per hour, intensity in kg CO2/MWh, emissions in tonnes: $`E_t = (1 + x_t)\,d_t\,e_t / 1000`$.
  - $`\lambda`$ is in the unit of Eq. (1), here tonnes of CO2 per adjusted hour. A value of $`\lambda`$ taken from another study must be converted to this unit first: its effect depends on the unit in which emissions are measured.
- **Cycles:** each window is a separate model, solved independently and in time order (`solve.solve_data`). Calendar windows follow the timestamps of the data.
- **$`D_t`$ (Eq. 3):** the total baseline demand of the window, $`\sum_{t \in w} d_t`$. In code, Eq. (3) is written as $`\sum_t d_t x_t \ge 0`$.
- **$`c^{max}_t`$ (Eq. 4):** the article describes $`c^{max}_t`$ as the limit for re-integrating curtailed renewable generation, capped at 10 % of total generation. The framework builds it as $`c^{max}_t = \kappa \sum_{r \in F^{reg}} g_{r,t}`$.
  - $`g_{r,t}`$ is the hourly output of resource $`r`$; $`\kappa`$ is `Case.curtailment_share` (default 10 %).
  - $`F^{reg}`$ is `Case.regulation_base_fuels`. The default is all fuels, so $`c^{max}_t`$ is $`\kappa`$ times total generation.
  - To tie the bound to the output that would otherwise be curtailed, restrict $`F^{reg}`$, for example to solar, wind and hydro.
- **$`c^{min}_t`$ (Eq. 5):** the hourly output of the unregulatable sources, $`c^{min}_t = \sum_{r \in F^{base}} g_{r,t}`$, with $`F^{base}`$ = coal and nuclear by default (`Case.baseload_fuels`).
- **$`M`$ (Eqs. 6–7):** any $`M \ge a`$ is valid, because $`\lvert x_t\rvert \le a`$. The default is $`M = a`$; `ModelOptions(big_m=...)` sets another value.
  - The constraints are written as in the article, $`z_t - x_t/M \ge 0`$ and $`z_t + x_t/M \ge 0`$, so that with $`M = a`$ the solver's tolerances apply on the scale of $`z_t`$.
  - With $`M > a`$ the model also keeps $`z_t - x_t/a \ge 0`$ and $`z_t + x_t/a \ge 0`$. For binary $`z_t`$ these follow from Eqs. (6)–(7) and the range, so the optimum is the same; they keep a large $`M`$ from letting very small adjustments pass within the solver's tolerances.
- **Objective constant:** in tonnes, $`\sum_t E_t = \sum_t d_t e_t/1000 + \sum_t d_t e_t x_t/1000`$. The first sum does not depend on the decisions and is left out of the solver's objective. The solver therefore minimizes the change in emissions plus the penalty, and a relative MIP gap applies to that change (default $`10^{-6}`$).
- **Tolerances and verification:** the solvers run with feasibility and integrality tolerances of $`10^{-8}`$. Every returned solution is then checked (`solve.verify_window`): a finite objective; $`z_t`$ integral and in $`\{0, 1\}`$; Eqs. (6)–(7) with the rounded $`z_t`$, on the scale $`x_t/a`$ whatever $`M`$ is; Eqs. (3)–(5) and the range; and the objective recomputed from $`x_t`$ and $`z_t`$. A window that fails is reported as not solved; by default `run` stops with an error.
- **Adjusted hours:** two counts are reported. `indicator_hours` is $`\sum_t z_t`$, the quantity penalized in Eq. (1). `adjusted_hours` counts the hours with $`\lvert x_t\rvert > 10^{-6}`$, i.e. load changes of practical size. The two differ when an indicator is 1 for an adjustment below $`10^{-6}`$: with $`\lambda = 0`$ (the objective does not involve $`z_t`$, so only `adjusted_hours` is meaningful), or when a very small adjustment is worth its penalty.
- **Feasibility:** every hour needs $`\max(-a,\; c^{min}_t/d_t - 1) \le \min(a,\; c^{max}_t/d_t)`$. `scenarios.feasibility_problems` lists the hours that break this, and `solve` stops with an error naming them.
- **Eq. (8):** $`P_r`$ is the fraction of each fuel in total output over the period of the data (a year), with the natural logarithm. Fuels with zero output are skipped.

## Modelling notes

- $`e_t`$ is an input: an hourly average intensity of the region's electricity. $`E_t`$ is the emission attributed to the region's consumption in hour $`t`$. The generation mix does not respond to the load change inside the model.
- Within a cycle, demand and intensity are known for every hour.
- Each region is modelled on its own.
- Hours are linked only through the cycle total of Eq. (3). There are no ramping, storage or comfort constraints. Such constraints can be added to the window model (`WindowModel.model` is a linopy model).
