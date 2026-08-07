"""Generate the figures the README leads with."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from asbench.measure import CostTable
from asbench.metrics import cumulative_regret
from asbench.policies import (
    UCB1,
    DiscountedUCB,
    EpsilonGreedy,
    best_fixed_in_hindsight,
    replay,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("costs")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--metric", default="seconds")
    args = ap.parse_args()

    table = CostTable.load(Path(args.costs))
    costs = table.cost(args.metric)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. per-arm cost over the sequence -- shows whether a crossover exists
    fig, ax = plt.subplots(figsize=(7, 4))
    for j, name in enumerate(table.arm_names):
        ax.semilogy(costs[:, j], label=name, lw=1.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel(args.metric)
    ax.set_title(f"per-arm cost: {table.sequence}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "arm_costs.png", dpi=150)

    # 2. cumulative regret vs the per-step oracle
    fig, ax = plt.subplots(figsize=(7, 4))
    policies = [
        EpsilonGreedy(table.n_arms),
        UCB1(table.n_arms),
        DiscountedUCB(table.n_arms),
    ]
    for p in policies:
        r = replay(p, costs)
        ax.plot(cumulative_regret(r, costs), label=r.policy, lw=1.5)
    bf = best_fixed_in_hindsight(costs)
    ax.plot(cumulative_regret(bf, costs), "k--", label=bf.policy, lw=1.5)
    ax.set_xlabel("timestep")
    ax.set_ylabel("cumulative regret vs per-step oracle")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "regret.png", dpi=150)
    print(f"wrote {out}/arm_costs.png and {out}/regret.png")


if __name__ == "__main__":
    main()
