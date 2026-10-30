# GHG load-regulation framework: re-timing electricity demand by real-time emission intensity

A documented, runnable implementation of the demand-side load-regulation framework in:

> Honglin Li, Gulai Shen, Soroush Senemmar, Ali Mehmani and Jie Zhang. **Real-time greenhouse gas emission intensity informed demand-side load regulation for power grid decarbonization.** *Cell Reports Sustainability* 2 (2025) 100367. [doi:10.1016/j.crsus.2025.100367](https://doi.org/10.1016/j.crsus.2025.100367)

The framework shifts a region's hourly electricity demand toward hours in which the region's electricity has a lower greenhouse-gas emission intensity. It does this within an optimization window, keeping the window's total consumption at least at its baseline, and within hourly limits set by the energy available for regulation and by unregulatable baseload. A penalty on adjusted hours trades emissions against the disruption of frequent adjustments. The framework also classifies grid regions by renewable penetration, generation diversity and emission-intensity fluctuation, which is how the article compares regulation strategies across grid types.

## Research question

How should demand-side load regulation guided by real-time emission intensity be designed for power grids with different characteristics? The design choices are the size of the time window, the magnitude of the allowed adjustment, and the penalty on frequent adjustment. The grid characteristics are clean-energy penetration, diversity of generation resources and fluctuation of emission intensity.

## System and scope

| Aspect | In the article | In this framework |
|---|---|---|
| Regions | U.S. grid regions (EIA's regional aggregations of balancing authorities), one calendar year of hourly data | any set of regions, any hourly period |
| Decision | hourly load adjustment $x_t$, a fraction of demand, within a predefined range | same |
| Objective | emissions of the optimization cycle plus a penalty $\lambda$ per adjusted hour (Eq. 1) | same |
| Constraints | total consumption of the cycle at least at its baseline (Eq. 3); hourly bounds from regulation capacity and unregulatable sources (Eqs. 4–5); indicators of adjusted hours (Eqs. 6–7) | same; each solution is verified against Eqs. (3)–(7) |
| Optimization cycle | month, quarter or year, each solved separately | also day, week or the whole period |
| Grid classification | renewable penetration, Shannon–Wiener diversity (Eq. 8), intensity fluctuation | same, with a choice of fluctuation statistic |
| Model type | mixed-integer linear program per cycle | same (linopy; HiGHS by default, Gurobi optional) |

The complete mathematics, with the article's equation numbers, is in [docs/formulation.md](docs/formulation.md).

## What is in this repository

| Path | Content |
|---|---|
| [`docs/formulation.md`](docs/formulation.md) | sets, parameters, variables, objective and every constraint, Eqs. (1)–(8); implementation conventions; modelling notes |
| [`docs/data_pipeline.md`](docs/data_pipeline.md) | how inputs are built from EIA's hourly regional data, the data-quality rules, and the input file schemas |
| [`docs/scenarios.md`](docs/scenarios.md) | the article's study design (grid types, regulation strategies, case studies, sensitivity families) and how to set each up |
| [`docs/architecture.md`](docs/architecture.md) | modules, data flow, solvers, problem size, output tables, tests |
| [`src/ghg_framework/`](src/ghg_framework/) | the implementation: data interface and validation, preprocessing, model, solver interface, results, region metrics, command line |
| [`examples/`](examples/) | `minimal.py` (one run, a few seconds), `strategies.py` (the three strategies and the sensitivity families on four regions), `prepare_inputs.py` (raw hourly records to an input directory); all on synthetic data |
| [`tests/`](tests/) | pytest suite: data interface, model and constraint checks, solver agreement, metrics, preprocessing, command line |

## Quick start

```bash
git clone https://github.com/Henryutd2021/ghg-load-regulation.git
cd ghg-load-regulation
python -m venv .venv && source .venv/bin/activate      # Python 3.10 or newer (tested with 3.12)
pip install -e ".[test]"
python examples/minimal.py                              # synthetic region, HiGHS, a few seconds
pytest
```

With conda instead: `conda env create -f environment.yml && conda activate ghg-framework && pip install -e .`

The command line works on input directories:

```bash
ghg-framework synthetic my_inputs/                                   # write four synthetic regions
ghg-framework metrics my_inputs/                                     # penetration, diversity, fluctuation, groups
ghg-framework run my_inputs/ --region Sunny --window year --out-dir outputs/sunny_year
ghg-framework run my_inputs/ --region Mixed --limit 0.08 --lam 5 --out-dir outputs/mixed
```

In Python:

```python
from ghg_framework import Case, read_inputs, run, region_metrics, classify

regions = read_inputs("my_inputs/")                     # your own data, see docs/data_pipeline.md
groups = classify(region_metrics(regions))
case = Case(window="quarter", adjustment_limit=0.05, curtailment_share=0.10, lam=0.0)   # c_max = 10 % of total generation
result = run(regions["Sunny"], case)                    # or run(..., solver="gurobi")
print(result.totals["reduction_pct"], result.totals["adjusted_hours"])
result.to_csv("outputs/sunny_quarter")
```

**About the example data:** `ghg_framework.synthetic` generates four invented regions ("Sunny", "Windy", "Mixed", "Thermal") with smooth hourly shapes, random noise and round illustrative emission factors. They show how the model behaves and how to read its output. Their numbers say nothing about any real power system.

## Using your own data

1. Build one hourly file per region as described in [docs/data_pipeline.md](docs/data_pipeline.md). `ghg_framework.preprocess` and `examples/prepare_inputs.py` cover the generic steps from EIA's regional hourly files. The format is validated when read: units are part of the column names, timestamps must be strictly increasing, and values must be finite.
2. Classify the regions (`region_metrics`, `classify`) and choose the cases ([docs/scenarios.md](docs/scenarios.md)).
3. Run the cases and read the result tables ([docs/architecture.md](docs/architecture.md#output-tables)).

## What this repository does not contain

This repository presents the modelling framework. It does not contain the code, the input data or the results used for the published article. For the article's results, see the article.

## Citation

If you use this framework, please cite the article above. Citation metadata for both the article and this software are in [`CITATION.cff`](CITATION.cff).

## Licence

- Code: [MIT](LICENSE).
- Documentation (`README.md`, `docs/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Source data are not included. EIA data are U.S. government data; see [docs/data_pipeline.md](docs/data_pipeline.md#terms-of-use-of-the-source).
