"""The three regulation strategies and the sensitivity families of docs/scenarios.md, on synthetic regions.

1. window size (month, quarter, year)        -> regions split by renewable penetration
2. adjustment range (±2 % ... ±10 %)         -> regions split by generation diversity
3. penalty λ per adjusted hour                -> regions split by intensity fluctuation
4. sensitivity: curtailment share κ, solar and wind growth

All data are synthetic (ghg_framework.synthetic); the numbers only illustrate the workflow.
Takes about a minute with HiGHS.
"""

import pandas as pd

from ghg_framework import Case, classify, region_metrics, run, synthetic_inputs

regions = synthetic_inputs()
metrics = region_metrics(regions)
groups = classify(metrics)
print("Region metrics and groups (median split):")
print(metrics[["renewable_share", "diversity_index", "intensity_iqr"]].join(groups).round(3).to_string(), "\n")

base = Case()                                               # quarterly window, ±5 %, κ = 10 %, λ = 0


def pct(region, case):
    return run(regions[region], case).totals["reduction_pct"]


print("1. Window size, reduction in % of attributed emissions")
rows = [{"region": r, "group": groups.loc[r, "penetration"],
         **{w: pct(r, base.with_(window=w)) for w in ("month", "quarter", "year")}} for r in regions]
print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

print("2. Adjustment range, reduction in %")
ranges = (0.02, 0.04, 0.06, 0.08, 0.10)
rows = [{"region": r, "group": groups.loc[r, "diversity"],
         **{f"±{a:.0%}": pct(r, base.with_(adjustment_limit=a)) for a in ranges}} for r in regions]
print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

print("3. Penalty λ (t CO2 per adjusted hour), mean adjusted hours per month")
lams = (0.0, 1.0, 5.0, 20.0)
rows = []
for r in regions:
    row = {"region": r, "group": groups.loc[r, "fluctuation"]}
    for lam in lams:
        res = run(regions[r], base.with_(window="month", lam=lam))
        row[f"λ={lam:g}"] = res.windows["adjusted_hours"].mean()
    rows.append(row)
print(pd.DataFrame(rows).round(1).to_string(index=False), "\n")

print("4a. Curtailment share κ, reduction in %")
shares = (0.02, 0.06, 0.10, 0.14, 0.18)
rows = [{"region": r, **{f"κ={k:.0%}": pct(r, base.with_(curtailment_share=k)) for k in shares}} for r in ("Sunny", "Windy")]
print(pd.DataFrame(rows).round(3).to_string(index=False), "\n")

print("4b. Solar and wind growth: reduction by load regulation (vs the same case without regulation)")
rows = []
for r in ("Sunny", "Windy"):
    for solar, wind in ((1.0, 1.0), (1.5, 1.0), (1.0, 1.5)):
        t = run(regions[r], base.with_(solar_scale=solar, wind_scale=wind)).totals
        rows.append({"region": r, "solar×": solar, "wind×": wind, "baseline_t": t["baseline_t"],
                     "reduction_pct": t["reduction_pct"], "reference_t": t["reference_t"]})
print(pd.DataFrame(rows).round(2).to_string(index=False))
