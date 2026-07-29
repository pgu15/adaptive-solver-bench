"""Arms = (preconditioner, solver) configurations the policy chooses between."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .problems import System


@dataclass(frozen=True)
class Arm:
    """One configuration the bandit can pull."""

    name: str
    build: Callable[[sp.csr_matrix], spla.LinearOperator | None]


@dataclass(frozen=True)
class SolveResult:
    arm: str
    seconds: float
    iterations: int
    converged: bool
    residual: float


def _none(A):
    return None


def _jacobi(A):
    d = A.diagonal()
    d = np.where(np.abs(d) < 1e-14, 1.0, d)
    inv = 1.0 / d
    return spla.LinearOperator(A.shape, matvec=lambda x: inv * x)


def _ilu(drop_tol: float, fill_factor: float):
    def build(A):
        try:
            ilu = spla.spilu(
                A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor
            )
        except RuntimeError:
            return None  # singular pivot; treat as unpreconditioned
        return spla.LinearOperator(A.shape, matvec=ilu.solve)

    return build


def _amg(A):
    import pyamg

    ml = pyamg.smoothed_aggregation_solver(A.tocsr())
    return ml.aspreconditioner()


DEFAULT_ARMS: list[Arm] = [
    Arm("none", _none),
    Arm("jacobi", _jacobi),
    Arm("ilu-loose", _ilu(drop_tol=1e-2, fill_factor=3.0)),
    Arm("ilu-tight", _ilu(drop_tol=1e-4, fill_factor=10.0)),
    Arm("amg", _amg),
]


def solve(
    system: System,
    arm: Arm,
    rtol: float = 1e-8,
    maxiter: int = 2000,
) -> SolveResult:
    """Run one CG solve and record cost.

    Setup time is included deliberately: an expensive preconditioner that halves
    the iteration count is not free, and a benchmark that only counts iterations
    hides exactly the tradeoff the policy is supposed to learn.
    """
    n_iter = 0

    def count(_xk):
        nonlocal n_iter
        n_iter += 1

    t0 = time.perf_counter()
    M = arm.build(system.A)
    x, info = spla.cg(
        system.A, system.b, rtol=rtol, maxiter=maxiter, M=M, callback=count
    )
    elapsed = time.perf_counter() - t0

    resid = float(np.linalg.norm(system.b - system.A @ x))
    return SolveResult(
        arm=arm.name,
        seconds=elapsed,
        iterations=n_iter,
        converged=(info == 0),
        residual=resid,
    )
