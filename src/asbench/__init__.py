"""adaptive-solver-bench: benchmarking adaptive preconditioner selection."""

__version__ = "0.1.0"

from .arms import DEFAULT_ARMS, Arm, solve
from .measure import CostTable, build_cost_table
from .metrics import cumulative_regret, summarise
from .policies import (
    UCB1,
    DiscountedUCB,
    EpsilonGreedy,
    Fixed,
    best_fixed_in_hindsight,
    oracle_per_step,
    replay,
)
from .problems import ProblemSequence, synthetic_sequence

__all__ = [
    "DEFAULT_ARMS",
    "UCB1",
    "Arm",
    "CostTable",
    "DiscountedUCB",
    "EpsilonGreedy",
    "Fixed",
    "ProblemSequence",
    "best_fixed_in_hindsight",
    "build_cost_table",
    "cumulative_regret",
    "oracle_per_step",
    "replay",
    "solve",
    "summarise",
    "synthetic_sequence",
]
