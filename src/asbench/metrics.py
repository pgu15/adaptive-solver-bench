"""Regret and speedup metrics."""

from __future__ import annotations

import numpy as np

from .policies import Rollout, best_fixed_in_hindsight, oracle_per_step


def summarise(rollout: Rollout, costs: np.ndarray) -> dict:
    """Compare one rollout against both oracles.

    `oracle_fraction` is the headline number: 1.0 means the policy matched a
    clairvoyant per-step chooser. `beat_best_fixed` is the honest question --
    does adaptivity buy anything over just picking one good preconditioner?
    """
    oracle = oracle_per_step(costs).total
    fixed = best_fixed_in_hindsight(costs).total
    total = rollout.total
    return {
        "policy": rollout.policy,
        "total_cost": total,
        "oracle_cost": oracle,
        "best_fixed_cost": fixed,
        "regret_vs_oracle": total - oracle,
        "oracle_fraction": oracle / total if total > 0 else float("nan"),
        "speedup_vs_best_fixed": fixed / total if total > 0 else float("nan"),
        "beat_best_fixed": bool(total < fixed),
        "switches": int((np.diff(rollout.choices) != 0).sum()),
    }


def cumulative_regret(rollout: Rollout, costs: np.ndarray) -> np.ndarray:
    return np.cumsum(rollout.realised - costs.min(axis=1))
