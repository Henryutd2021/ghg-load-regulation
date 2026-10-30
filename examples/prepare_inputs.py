"""From raw hourly records to an input directory (the generic steps of docs/data_pipeline.md).

The raw records here are synthetic, laid out like EIA's regional hourly files and seeded with the
defects the pipeline must handle (gaps, negative net generation, zero-consumption hours, intensity
in lb CO2/kWh). With real files, replace `synthetic_raw_records(...)` by reading the downloaded data.

Usage:  python examples/prepare_inputs.py [OUTPUT_DIR]      (default: outputs/prepared_inputs)
"""

import sys

import pandas as pd

from ghg_framework import qa_summary, read_inputs, region_from_records, region_metrics, synthetic_inputs, synthetic_raw_records, write_inputs

out_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/prepared_inputs"

regions, reports, qa = {}, [], []
for i, (name, truth) in enumerate(synthetic_inputs().items()):
    raw = synthetic_raw_records(truth, seed=i)                         # stands in for a downloaded file
    region, report = region_from_records(name, raw, intensity_unit="lb/kWh", start="2025-01-01", end="2026-01-01")
    regions[name] = region
    reports.append(report)
    qa.append(qa_summary(region))

print("What the pipeline did:")
print(pd.DataFrame(reports).to_string(index=False), "\n")
print("QA quantities to compare with published totals:")
print(pd.DataFrame(qa).round(3).to_string(index=False), "\n")

write_inputs(out_dir, regions, description="Synthetic regions prepared by examples/prepare_inputs.py. Invented data.")
back = read_inputs(out_dir)                                            # applies the format checks
print(f"wrote and re-read {len(back)} regions in {out_dir}")
print(region_metrics(back)[["demand_twh", "renewable_share", "diversity_index"]].round(3).to_string())
