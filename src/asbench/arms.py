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


def _symmetric_gauss_seidel(sweeps: int):
    """Symmetric Gauss-Seidel as a preconditioner.

    The symmetric (forward-then-backward) sweep matters: a one-directional
    sweep gives a NONSYMMETRIC operator, and CG requires its preconditioner to
    be symmetric positive definite. See the note on ILU below.
    """

    def build(A):
        from pyamg.relaxation.relaxation import gauss_seidel

        A = A.tocsr()

        def matvec(x):
            x = np.asarray(x, dtype=A.dtype).ravel()
            y = np.zeros_like(x)
            gauss_seidel(A, y, x, iterations=sweeps, sweep="symmetric")
            return y

        return spla.LinearOperator(A.shape, matvec=matvec, dtype=A.dtype)

    return build


def _amg(max_coarse: int, theta: float, sweeps: int = 1):
    """Smoothed-aggregation AMG. `theta` is the strength-of-connection
    threshold: 0.0 treats all couplings as strong (good for isotropic
    operators), higher values only coarsen along strong couplings (better when
    the operator is anisotropic)."""

    def build(A):
        import pyamg

        ml = pyamg.smoothed_aggregation_solver(
            A.tocsr(),
            max_coarse=max_coarse,
            strength=("symmetric", {"theta": theta}),
            presmoother=("gauss_seidel", {"sweep": "symmetric", "iterations": sweeps}),
            postsmoother=("gauss_seidel", {"sweep": "symmetric", "iterations": sweeps}),
        )
        return ml.aspreconditioner()

    return build


# NOTE ON ILU
# -----------
# Earlier versions of this benchmark included scipy `spilu` arms. That was a
# correctness bug, not a tuning problem: an incomplete LU factorisation of an
# SPD matrix is not itself symmetric, and CG's convergence theory requires an
# SPD preconditioner. Those arms hit the iteration cap on essentially every
# problem, which looked like "ILU is a bad preconditioner here" but was really
# CG breaking down on an invalid operator. They inflated the arm cost spread by
# ~1000x and drove the original negative result. Incomplete Cholesky would be
# the correct SPD analogue; scipy does not ship one, so the symmetric
# Gauss-Seidel arm above stands in.


DEFAULT_ARMS: list[Arm] = [
    Arm("none", _none),
    Arm("jacobi", _jacobi),
    Arm("sym-gauss-seidel", _symmetric_gauss_seidel(1)),
    Arm("amg", _amg(max_coarse=50, theta=0.0)),
    Arm("amg-aniso", _amg(max_coarse=50, theta=0.25)),
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
