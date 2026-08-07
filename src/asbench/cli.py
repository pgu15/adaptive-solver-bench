"""Command line entry point: `asbench run`, `asbench report`."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from .arms import DEFAULT_ARMS
from .measure import CostTable, build_cost_table
from .metrics import summarise
from .policies import (
    UCB1,
    DiscountedUCB,
    EpsilonGreedy,
    Fixed,
    best_fixed_in_hindsight,
    oracle_per_step,
    replay,
)
from .problems import regime_change_sequence, synthetic_sequence

FIELDS = [
    "policy",
    "total_cost",
    "oracle_cost",
    "best_fixed_cost",
    "regret_vs_oracle",
    "oracle_fraction",
    "speedup_vs_best_fixed",
    "beat_best_fixed",
    "switches",
]


def _policies(n_arms: int, arm_names: list[str], seed: int):
    ps = [EpsilonGreedy(n_arms, seed=seed), UCB1(n_arms, seed=seed),
          DiscountedUCB(n_arms, seed=seed)]
    ps += [Fixed(n_arms, i, f"fixed:{name}", seed) for i, name in enumerate(arm_names)]
    return ps


def cmd_run(args) -> int:
    if args.problem == "regime-change":
        seq = regime_change_sequence(n=args.grid, seed=args.seed)
    else:
        seq = synthetic_sequence(n=args.grid, steps=args.steps, seed=args.seed)
    print(f"measuring {len(seq)} steps x {len(DEFAULT_ARMS)} arms on {seq.name}")
    table = build_cost_table(seq, DEFAULT_ARMS, repeats=args.repeats,
                             verbose=args.verbose)
    out = Path(args.out)
    table.save(out / "costs.json")

    costs = table.cost(metric=args.metric)
    rows = [summarise(r, costs) for r in
            [oracle_per_step(costs), best_fixed_in_hindsight(costs)]
            + [replay(p, costs) for p in
               _policies(table.n_arms, table.arm_names, args.seed)]]

    with (out / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    _print_table(rows)
    print(f"\nwrote {out / 'costs.json'} and {out / 'results.csv'}")
    return 0


def cmd_report(args) -> int:
    table = CostTable.load(Path(args.costs))
    costs = table.cost(metric=args.metric)
    rows = [summarise(r, costs) for r in
            [oracle_per_step(costs), best_fixed_in_hindsight(costs)]
            + [replay(p, costs) for p in
               _policies(table.n_arms, table.arm_names, args.seed)]]
    _print_table(rows)
    return 0


def _print_table(rows) -> None:
    hdr = f"{'policy':<24}{'cost':>10}{'% of oracle':>13}{'vs best-fixed':>15}{'switches':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: r["total_cost"]):
        print(f"{r['policy']:<24}{r['total_cost']:>10.3f}"
              f"{100 * r['oracle_fraction']:>12.1f}%"
              f"{r['speedup_vs_best_fixed']:>14.2f}x"
              f"{r['switches']:>10d}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="asbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="measure a sequence and evaluate policies")
    r.add_argument(
        "--problem", choices=["regime-change", "drift"], default="regime-change"
    )
    r.add_argument("--grid", type=int, default=64)
    r.add_argument("--steps", type=int, default=24)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--metric", choices=["seconds", "iterations"], default="seconds")
    r.add_argument("--out", default="results")
    r.add_argument("--verbose", action="store_true")
    r.set_defaults(func=cmd_run)

    q = sub.add_parser("report", help="re-evaluate policies on a saved cost table")
    q.add_argument("costs")
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--metric", choices=["seconds", "iterations"], default="seconds")
    q.set_defaults(func=cmd_report)

    args = p.parse_args(argv)
    Path(getattr(args, "out", "results")).mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
