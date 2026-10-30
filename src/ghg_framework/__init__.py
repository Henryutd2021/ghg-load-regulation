"""ghg_framework: emission-intensity-informed demand-side load regulation for a power-grid region.

Implements the load-regulation model of Li, Shen, Senemmar, Mehmani & Zhang, "Real-time greenhouse
gas emission intensity informed demand-side load regulation for power grid decarbonization",
Cell Reports Sustainability 2 (2025) 100367, doi:10.1016/j.crsus.2025.100367. See docs/formulation.md.

Typical use::

    from ghg_framework import synthetic_region, Case, run
    region = synthetic_region("Sunny")              # or io.read_inputs("my_inputs/")["<region>"]
    result = run(region, Case(window="quarter", adjustment_limit=0.05))
    print(result.totals["reduction_pct"])
"""

from .inputs import BASELOAD_FUELS, CURTAILABLE_FUELS, FUELS, RENEWABLE_FUELS, RegionData, make_region
from .io import read_inputs, read_region, write_inputs
from .metrics import classify, diversity_index, intensity_fluctuation, region_metrics, renewable_share
from .model import ModelOptions, WindowModel, build_window_model
from .preprocess import EIA_COLUMNS, convert_intensity, fill_gaps, qa_summary, region_from_records
from .results import RunResult, constraint_residuals
from .scenarios import WINDOWS, Case, ModelData, feasibility_problems, model_data, scaled_intensity, window_labels
from .solve import SolveError, run, solve_data, solve_window
from .synthetic import PROFILES, synthetic_inputs, synthetic_raw_records, synthetic_region

__version__ = "1.0.0"

__all__ = [
    "BASELOAD_FUELS", "CURTAILABLE_FUELS", "FUELS", "RENEWABLE_FUELS", "RegionData", "make_region",
    "read_inputs", "read_region", "write_inputs", "classify", "diversity_index", "intensity_fluctuation",
    "region_metrics", "renewable_share", "ModelOptions", "WindowModel", "build_window_model", "EIA_COLUMNS",
    "convert_intensity", "fill_gaps", "qa_summary", "region_from_records", "RunResult", "constraint_residuals",
    "WINDOWS", "Case", "ModelData", "feasibility_problems", "model_data", "scaled_intensity", "window_labels",
    "SolveError", "run", "solve_data", "solve_window", "PROFILES", "synthetic_inputs", "synthetic_raw_records",
    "synthetic_region", "__version__",
]
