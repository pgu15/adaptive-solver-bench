import numpy as np
import pytest

from asbench.metrics import summarise
from asbench.policies import (
    UCB1,
    DiscountedUCB,
    EpsilonGreedy,
    Fixed,
    best_fixed_in_hindsight,
    oracle_per_step,
    replay,
)


@pytest.fixture
def costs():
    """Arm 0 is best for the first half, arm 1 for the second: non-stationary."""
    c = np.ones((40, 3))
    c[:20, 0] = 0.1
    c[20:, 1] = 0.1
    c[:, 2] = 1.5
    return c


def test_oracle_is_a_lower_bound(costs):
    oracle = oracle_per_step(costs)
    for p in (EpsilonGreedy(3, seed=1), UCB1(3, seed=1), DiscountedUCB(3, seed=1)):
        assert replay(p, costs).total >= oracle.total - 1e-12


def test_best_fixed_matches_a_fixed_policy(costs):
    bf = best_fixed_in_hindsight(costs)
    fixed = [replay(Fixed(3, i, f"f{i}"), costs).total for i in range(3)]
    assert bf.total == pytest.approx(min(fixed))


def test_discounting_beats_plain_ucb_under_drift(costs):
    """The point of the whole benchmark: a stationary policy anchors on the
    arm that was good early and fails to switch."""
    ducb = np.mean([replay(DiscountedUCB(3, seed=s), costs).total for s in range(5)])
    ucb = np.mean([replay(UCB1(3, seed=s), costs).total for s in range(5)])
    assert ducb < ucb


def test_replay_is_deterministic(costs):
    a = replay(UCB1(3, seed=7), costs)
    b = replay(UCB1(3, seed=7), costs)
    assert np.array_equal(a.choices, b.choices)


def test_summary_fractions(costs):
    s = summarise(oracle_per_step(costs), costs)
    assert s["oracle_fraction"] == pytest.approx(1.0)
    assert s["regret_vs_oracle"] == pytest.approx(0.0)
