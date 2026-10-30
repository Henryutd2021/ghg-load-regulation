"""Minimal example: one synthetic region, one case, HiGHS. Runs in a few seconds.

The region is invented (ghg_framework.synthetic); the numbers say nothing about any real grid.
"""

from ghg_framework import Case, run, synthetic_region

region = synthetic_region("Sunny")                              # one synthetic year, hourly
case = Case(window="quarter", adjustment_limit=0.05, curtailment_share=0.10, lam=0.0)
result = run(region, case)                                      # HiGHS; run(..., solver="gurobi") also works

print(f"region {result.region}, case: window={case.window}, range=±{case.adjustment_limit:.0%}, λ={case.lam}")
print(result.windows[["window", "hours", "baseline_t", "optimized_t", "reduction_t", "reduction_pct", "adjusted_hours"]]
      .round(2).to_string(index=False))
t = result.totals
print(f"year: reduction {t['reduction_t']:,.0f} t CO2 ({t['reduction_pct']:.2f} % of attributed emissions), "
      f"{t['added_mwh']:,.0f} MWh moved to lower-intensity hours, all windows optimal: {t['all_optimal']}")
