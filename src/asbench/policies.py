"""Arm-selection policies, plus offline replay against a measured cost table.

All policies see only the cost of the arm they actually chose -- standard bandit
feedback. The oracles below are baselines for regret, not runnable policies.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Policy:
    """Choose an arm each step; observe only the chosen arm's cost."""

    name: str = "base"

    def __init__(self, n_arms: int, seed: int = 0):
        self.n_arms = n_arms
        self.rng = np.random.default_rng(seed)

    def select(self, step: int) -> int:
        raise NotImplementedError

    def update(self, arm: int, cost: float) -> None:
        pass


class Fixed(Policy):
    def __init__(self, n_arms: int, arm: int, name: str, seed: int = 0):
        super().__init__(n_arms, seed)
        self.arm = arm
        self.name = name

    def select(self, step: int) -> int:
        return self.arm


class EpsilonGreedy(Policy):
    name = "eps-greedy"

    def __init__(self, n_arms: int, eps: float = 0.1, seed: int = 0):
        super().__init__(n_arms, seed)
        self.eps = eps
        self.counts = np.zeros(n_arms)
        self.means = np.zeros(n_arms)

    def select(self, step: int) -> int:
        if step < self.n_arms:
            return step  # one forced pull per arm
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmin(self.means))

    def update(self, arm: int, cost: float) -> None:
        self.counts[arm] += 1
        self.means[arm] += (cost - self.means[arm]) / self.counts[arm]


class UCB1(Policy):
    """UCB1 adapted for cost minimisation.

    Means are kept in raw cost units and the exploration bonus is scaled by the
    largest cost seen so far. Normalising the *bonus* rather than the *costs*
    avoids retroactively rescaling means every time a new maximum appears.
    """

    name = "ucb1"

    def __init__(self, n_arms: int, c: float = 0.1, seed: int = 0):
        super().__init__(n_arms, seed)
        self.c = c
        self.counts = np.zeros(n_arms)
        self.means = np.zeros(n_arms)
        self.scale = 1e-12
        self.t = 0

    def select(self, step: int) -> int:
        self.t += 1
        if step < self.n_arms:
            return step  # one forced pull per arm
        bonus = (
            self.c
            * self.scale
            * np.sqrt(2.0 * np.log(self.t) / np.maximum(self.counts, 1e-12))
        )
        return int(np.argmin(self.means - bonus))

    def update(self, arm: int, cost: float) -> None:
        self.scale = max(self.scale, cost)
        self.counts[arm] += 1
        self.means[arm] += (cost - self.means[arm]) / self.counts[arm]


class DiscountedUCB(UCB1):
    """D-UCB: geometrically discounted sums, so old observations decay.

    This is the version that matters here. The best preconditioner drifts as the
    simulation evolves, and a stationary estimator averages the good early
    regime together with the bad late one and never switches.
    """

    name = "d-ucb"

    def __init__(
        self, n_arms: int, c: float = 0.1, gamma: float = 0.9, seed: int = 0
    ):
        super().__init__(n_arms, c, seed)
        self.gamma = gamma
        self._n = np.zeros(n_arms)  # discounted pull counts
        self._s = np.zeros(n_arms)  # discounted cost sums

    def select(self, step: int) -> int:
        self.t += 1
        if step < self.n_arms:
            return step
        # Horizon term is the *discounted* total, not t. Using log(t) here lets
        # the bonus grow without bound while counts stay capped at 1/(1-gamma),
        # which turns the policy into pure exploration.
        horizon = max(self._n.sum(), np.e)
        bonus = (
            self.c
            * self.scale
            * np.sqrt(2.0 * np.log(horizon) / np.maximum(self.counts, 1e-12))
        )
        return int(np.argmin(self.means - bonus))

    def update(self, arm: int, cost: float) -> None:
        self.scale = max(self.scale, cost)
        self._n *= self.gamma
        self._s *= self.gamma
        self._n[arm] += 1.0
        self._s[arm] += cost
        if self._n.max() < 1e-6:  # guard against float underflow on long runs
            self._n /= self._n.max()
            self._s /= self._n.max()
        # Unpulled arms decay in numerator and denominator alike, so the ratio
        # is preserved; only the exploration bonus grows for them.
        self.counts = np.maximum(self._n, 1e-12)
        self.means = self._s / self.counts


@dataclass
class Rollout:
    policy: str
    choices: np.ndarray
    realised: np.ndarray

    @property
    def total(self) -> float:
        return float(self.realised.sum())


def replay(policy: Policy, costs: np.ndarray) -> Rollout:
    """Run a policy against a precomputed cost table (steps x arms)."""
    n_steps = costs.shape[0]
    choices = np.zeros(n_steps, dtype=int)
    realised = np.zeros(n_steps)
    for t in range(n_steps):
        a = policy.select(t)
        c = float(costs[t, a])
        policy.update(a, c)
        choices[t] = a
        realised[t] = c
    return Rollout(policy=policy.name, choices=choices, realised=realised)


def oracle_per_step(costs: np.ndarray) -> Rollout:
    choices = np.argmin(costs, axis=1)
    return Rollout("oracle-per-step", choices, costs[np.arange(len(choices)), choices])


def best_fixed_in_hindsight(costs: np.ndarray) -> Rollout:
    arm = int(np.argmin(costs.sum(axis=0)))
    return Rollout(
        "best-fixed-hindsight", np.full(costs.shape[0], arm), costs[:, arm].copy()
    )
