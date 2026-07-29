"""Measurement layer.

Key design decision: measurement is separated from policy simulation.

We solve *every* arm on *every* step once, producing a dense cost table
C[step, arm]. Policies are then replayed against that table offline.

Why this matters:
  * The per-step oracle is computable at all (it needs every arm's cost).
  * Comparing 6 policies costs the same wall-clock as comparing 1.
  * Policy comparisons are exactly reproducible -- no timing noise between runs.
  * Re-running a policy after a bug fix takes milliseconds, not hours.

The cost is that this only works for policies whose choice does not change the
*state* of the system being solved. That holds here (each solve starts from the
same zero guess), and the limitation is documented rather than assumed away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .arms import Arm, solve
from .problems import ProblemSequence


@dataclass
class CostTable:
    sequence: str
    arm_names: list[str]
    seconds: np.ndarray  # shape (steps, arms)
    iterations: np.ndarray  # shape (steps, arms)
    converged: np.ndarray  # bool, shape (steps, arms)

    @property
    def n_steps(self) -> int:
        return self.seconds.shape[0]

    @property
    def n_arms(self) -> int:
        return self.seconds.shape[1]

    def cost(self, metric: str = "seconds") -> np.ndarray:
        """Cost matrix with non-convergence penalised, not silently dropped."""
        raw = self.seconds if metric == "seconds" else self.iterations
        raw = raw.astype(float).copy()
        penalty = 2.0 * np.nanmax(raw)
        raw[~self.converged] = penalty
        return raw

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sequence": self.sequence,
            "arm_names": self.arm_names,
            "seconds": self.seconds.tolist(),
            "iterations": self.iterations.tolist(),
            "converged": self.converged.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> CostTable:
        d = json.loads(Path(path).read_text())
        return cls(
            sequence=d["sequence"],
            arm_names=d["arm_names"],
            seconds=np.array(d["seconds"]),
            iterations=np.array(d["iterations"]),
            converged=np.array(d["converged"], dtype=bool),
        )


def build_cost_table(
    sequence: ProblemSequence,
    arms: list[Arm],
    repeats: int = 1,
    verbose: bool = False,
) -> CostTable:
    """Solve every (step, arm) pair; take the min over repeats for timing."""
    S, K = len(sequence), len(arms)
    secs = np.zeros((S, K))
    iters = np.zeros((S, K), dtype=int)
    conv = np.zeros((S, K), dtype=bool)

    for i, system in enumerate(sequence.systems):
        for j, arm in enumerate(arms):
            best = None
            for _ in range(max(1, repeats)):
                r = solve(system, arm)
                if best is None or r.seconds < best.seconds:
                    best = r
            secs[i, j] = best.seconds
            iters[i, j] = best.iterations
            conv[i, j] = best.converged
        if verbose:
            print(f"  step {i + 1}/{S} done", flush=True)

    return CostTable(
        sequence=sequence.name,
        arm_names=[a.name for a in arms],
        seconds=secs,
        iterations=iters,
        converged=conv,
    )
