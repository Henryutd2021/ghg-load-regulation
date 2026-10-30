"""Command line: ``ghg-framework synthetic|run|metrics`` (see README)."""

from __future__ import annotations

import argparse
import json
import sys

from .io import read_inputs, write_inputs
from .metrics import FLUCTUATION_STATS, classify, region_metrics
from .scenarios import WINDOWS, Case
from .solve import run
from .synthetic import synthetic_inputs


def _synthetic(args) -> int:
    regions = synthetic_inputs(hours=args.hours, seed=args.seed)
    write_inputs(args.directory, regions, description="Synthetic example regions made by ghg_framework.synthetic. Invented data.")
    print(f"wrote {len(regions)} synthetic regions to {args.directory}")
    return 0


def _run(args) -> int:
    regions = read_inputs(args.directory)
    if args.region not in regions:
        print(f"unknown region {args.region!r}; regions: {list(regions)}", file=sys.stderr)
        return 2
    case = Case(name=args.name, window=args.window, adjustment_limit=args.limit, curtailment_share=args.curtailment_share,
                lam=args.lam, demand_scale=args.demand_scale, solar_scale=args.solar_scale, wind_scale=args.wind_scale)
    result = run(regions[args.region], case, solver=args.solver)
    print(json.dumps({"region": result.region, **result.totals}, indent=2))
    if args.out_dir:
        result.to_csv(args.out_dir)
        print(f"wrote result tables to {args.out_dir}")
    return 0


def _metrics(args) -> int:
    m = region_metrics(read_inputs(args.directory))
    table = m.join(classify(m, args.fluctuation_stat))
    print(table.round(4).to_string())
    if args.out:
        table.to_csv(args.out)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ghg-framework", description="Emission-intensity-informed load regulation (framework).")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synthetic", help="write a synthetic input directory")
    s.add_argument("directory")
    s.add_argument("--hours", type=int, default=8760)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=_synthetic)

    r = sub.add_parser("run", help="solve one region and case")
    r.add_argument("directory")
    r.add_argument("--region", required=True)
    r.add_argument("--name", default="cli")
    r.add_argument("--window", choices=WINDOWS, default="quarter")
    r.add_argument("--limit", type=float, default=0.05, help="adjustment range a (fraction of demand)")
    r.add_argument("--curtailment-share", type=float, default=0.10, help="κ of Eq. 4")
    r.add_argument("--lam", type=float, default=0.0, help="penalty λ per adjusted hour (t CO2)")
    r.add_argument("--demand-scale", type=float, default=1.0)
    r.add_argument("--solar-scale", type=float, default=1.0)
    r.add_argument("--wind-scale", type=float, default=1.0)
    r.add_argument("--solver", default="highs")
    r.add_argument("--out-dir")
    r.set_defaults(func=_run)

    m = sub.add_parser("metrics", help="classification metrics of all regions")
    m.add_argument("directory")
    m.add_argument("--fluctuation-stat", choices=FLUCTUATION_STATS, default="iqr")
    m.add_argument("--out")
    m.set_defaults(func=_metrics)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
