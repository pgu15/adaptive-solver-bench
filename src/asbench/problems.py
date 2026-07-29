"""Problem sequences: sequences of related sparse linear systems.

A *single* linear solve is not a bandit problem. The bandit framing only makes
sense when you solve many *related* systems in a row -- e.g. one per timestep of
a transient PDE simulation -- because that is what lets a policy learn from
earlier solves and pay off on later ones.

Two sources are supported:
  * `synthetic`  -- variable-coefficient 2D Poisson, drifting over timesteps.
                    Runs offline, deterministic, good for CI.
  * `suitesparse` -- real matrices from the SuiteSparse Matrix Collection,
                    with the sequence formed by perturbing the RHS/operator.
                    Requires network access on first use (results are cached).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CACHE_DIR = Path.home() / ".cache" / "asbench"


@dataclass(frozen=True)
class System:
    """One linear system A x = b."""

    A: sp.csr_matrix
    b: np.ndarray
    step: int


@dataclass(frozen=True)
class ProblemSequence:
    """A named sequence of related systems, solved in order."""

    name: str
    systems: list[System]

    def __len__(self) -> int:
        return len(self.systems)


def _diffusion_2d(coeff: np.ndarray) -> sp.csr_matrix:
    """Variable-coefficient diffusion operator on an n x n grid, Dirichlet BCs.

    Face coefficients are harmonic means of the two adjacent cell values and the
    diagonal is the row sum of the faces. That construction is an M-matrix and
    is symmetric positive definite by design -- important, because CG silently
    breaks down on indefinite operators and a benchmark that quietly hands the
    solver a bad matrix measures nothing.
    """
    n = coeff.shape[0]
    N = n * n
    idx = np.arange(N).reshape(n, n)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    diag = np.zeros(N)

    def add_faces(a, b, ia, ib):
        f = 2.0 * a * b / (a + b)  # harmonic mean
        rows.extend([ia, ib])
        cols.extend([ib, ia])
        vals.extend([-f, -f])
        np.add.at(diag, ia, f)
        np.add.at(diag, ib, f)

    add_faces(
        coeff[:, :-1].ravel(), coeff[:, 1:].ravel(),
        idx[:, :-1].ravel(), idx[:, 1:].ravel(),
    )
    add_faces(
        coeff[:-1, :].ravel(), coeff[1:, :].ravel(),
        idx[:-1, :].ravel(), idx[1:, :].ravel(),
    )
    # Dirichlet boundary contributions keep the operator nonsingular
    diag[idx[0, :]] += coeff[0, :]
    diag[idx[-1, :]] += coeff[-1, :]
    diag[idx[:, 0]] += coeff[:, 0]
    diag[idx[:, -1]] += coeff[:, -1]

    rows.append(np.arange(N))
    cols.append(np.arange(N))
    vals.append(diag)

    rows_a = np.concatenate(rows)
    cols_a = np.concatenate(cols)
    vals_a = np.concatenate(vals)
    A = sp.coo_matrix((vals_a, (rows_a, cols_a)), shape=(N, N)).tocsr()
    A.sum_duplicates()
    return A


def synthetic_sequence(
    name: str = "poisson-drift",
    n: int = 48,
    steps: int = 30,
    seed: int = 0,
    drift: float = 0.5,
) -> ProblemSequence:
    """Variable-coefficient Poisson whose conditioning drifts across timesteps.

    The coefficient field starts smooth and grows increasingly heterogeneous,
    so the *best* preconditioner changes partway through the sequence. That
    non-stationarity is the whole point: a fixed choice cannot win everywhere.
    """
    rng = np.random.default_rng(seed)
    N = n * n
    field = rng.normal(size=(n, n))
    # smooth the field so the coefficient has spatial structure, not white noise
    for _ in range(3):
        field = 0.5 * field + 0.125 * (
            np.roll(field, 1, 0) + np.roll(field, -1, 0)
            + np.roll(field, 1, 1) + np.roll(field, -1, 1)
        )
    field /= np.abs(field).max()

    systems: list[System] = []
    for t in range(steps):
        contrast = drift * t  # orders of magnitude of coefficient variation
        coeff = np.exp(contrast * field)
        A = _diffusion_2d(coeff)
        b = rng.normal(size=N)
        b /= np.linalg.norm(b)
        systems.append(System(A=A, b=b, step=t))
    return ProblemSequence(name=f"{name}-n{n}", systems=systems)


def suitesparse_sequence(
    group: str,
    matrix: str,
    steps: int = 20,
    seed: int = 0,
) -> ProblemSequence:
    """Build a sequence from one SuiteSparse matrix by perturbing it per step.

    Requires `ssgetpy` and network access on first call; downloads are cached
    under ~/.cache/asbench. Kept out of CI for that reason.
    """
    try:
        import ssgetpy
        from scipy.io import mmread
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "suitesparse problems need `pip install adaptive-solver-bench[real]`"
        ) from exc

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result = ssgetpy.search(group=group, name=matrix, limit=1)
    if not result:
        raise ValueError(f"no SuiteSparse matrix {group}/{matrix}")
    mm_path, _ = result[0].download(destpath=str(CACHE_DIR), extract=True)
    A0 = sp.csr_matrix(mmread(Path(mm_path)))

    rng = np.random.default_rng(seed)
    N = A0.shape[0]
    diag_shift = sp.eye(N, format="csr")

    systems: list[System] = []
    for t in range(steps):
        # A mild diagonal shift stands in for a timestep term (I/dt + A).
        A = A0 + (0.05 * t) * diag_shift
        b = rng.normal(size=N)
        b /= np.linalg.norm(b)
        systems.append(System(A=A.tocsr(), b=b, step=t))
    return ProblemSequence(name=f"{group}/{matrix}", systems=systems)
