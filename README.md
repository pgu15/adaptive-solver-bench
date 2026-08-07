# adaptive-solver-bench

Does adaptively choosing a preconditioner actually beat picking one good one?

Iterative sparse solvers (CG, GMRES) need a preconditioner, and the best choice
depends on the matrix. In a transient simulation you solve thousands of
*related* systems in a row while the operator slowly changes, so the best choice
drifts. That makes preconditioner selection a non-stationary multi-armed bandit:
each arm is a preconditioner, the cost is wall-clock time, and you only observe
the arm you pulled.

This repo measures whether that framing pays off, against honest baselines.

## Result so far

On a 1600-DOF variable-coefficient diffusion sequence whose coefficient contrast
grows over 20 timesteps (`asbench run --grid 40 --steps 20`):

| policy | total seconds | % of per-step oracle | vs. best fixed arm |
|---|---|---|---|
| oracle-per-step | 0.097 | 100% | 1.11x |
| best-fixed-in-hindsight (jacobi) | 0.108 | 90% | 1.00x |
| eps-greedy | 1.347 | 7% | 0.08x |
| d-ucb | 1.401 | 7% | 0.08x |
| ucb1 | 1.407 | 7% | 0.08x |

**The bandits lose badly, and that is the interesting part.** The arm spread is
roughly three orders of magnitude — a single pull of `ilu-loose` costs more than
the entire oracle run — so the exploration cost swamps any adaptive gain. The
per-step oracle only beats the best fixed arm by 11%, meaning there is barely
any adaptive headroom to win on this problem in the first place.

Open questions this raises, in the order I'm working on them:

1. **Cheap feedback.** Iteration count correlates with wall-clock but costs
   nothing extra to observe. Does switching the reward signal help?
2. **Bounded exploration.** Capping a trial solve at *k* iterations turns an
   unboundedly expensive pull into a bounded one. Standard bandit regret bounds
   assume bounded rewards; wall-clock is not bounded.
3. **Harder sequences.** Does adaptive headroom appear on real SuiteSparse
   matrices, or on sequences with a sharper regime change?

## Install

```bash
git clone https://github.com/<you>/adaptive-solver-bench
cd adaptive-solver-bench
pip install -e ".[dev]"
```

## Use

```bash
asbench run --grid 40 --steps 20 --metric seconds --out results
asbench report results/costs.json --metric iterations   # replay, no re-solving
python scripts/plot.py results/costs.json --out figures
```

## Design

**Measurement is separated from policy simulation.** `asbench run` solves every
arm on every step once and writes a dense cost table `C[step, arm]`. Policies
are then replayed against that table offline.

This buys three things:

- The per-step oracle becomes computable at all — it needs every arm's cost.
- Comparing ten policies costs the same wall-clock as comparing one.
- Policy comparisons are exactly reproducible; no timing noise between runs.

The limitation is that it only works for policies whose choice doesn't change
the state of the next solve. That holds here — every solve starts from a zero
initial guess. Warm-starting from the previous solution would break it, and
would need a different harness.

**Setup time is charged to the arm.** An AMG hierarchy that halves the iteration
count is not free. A benchmark that counts only iterations hides exactly the
tradeoff the policy is supposed to learn.

**Non-convergence is penalised, not dropped.** A diverged solve is recorded at
2x the worst observed cost rather than silently excluded, so a policy cannot
look good by choosing arms that fail fast.

## Layout

```
src/asbench/
  problems.py   sequences of related systems (synthetic + SuiteSparse)
  arms.py       preconditioner configurations and the CG solve wrapper
  measure.py    builds and caches the cost table
  policies.py   eps-greedy, UCB1, D-UCB, oracles, offline replay
  metrics.py    regret and speedup summaries
  cli.py        asbench run / asbench report
```

## Caveats

- The synthetic problem is small (1600 DOF). Timing at this size is dominated by
  Python and scipy overhead; conclusions about wall-clock should not be trusted
  until the SuiteSparse path is running at realistic sizes.
- `scipy.sparse.linalg.cg` is not a serious production solver. PETSc via petsc4py
  is the right backend and is the next infrastructure change.
- Only symmetric positive definite systems and CG so far; GMRES and nonsymmetric
  problems are not implemented.

## License

MIT
