import numpy as np

from asbench.arms import DEFAULT_ARMS
from asbench.measure import CostTable, build_cost_table
from asbench.problems import synthetic_sequence


def test_small_end_to_end(tmp_path):
    seq = synthetic_sequence(n=12, steps=3, seed=0)
    table = build_cost_table(seq, DEFAULT_ARMS[:3])
    assert table.seconds.shape == (3, 3)
    assert (table.seconds > 0).all()

    p = tmp_path / "costs.json"
    table.save(p)
    assert np.allclose(CostTable.load(p).seconds, table.seconds)


def test_nonconvergence_is_penalised():
    t = CostTable("x", ["a", "b"], np.array([[1.0, 2.0]]),
                  np.array([[10, 20]]), np.array([[True, False]]))
    c = t.cost("seconds")
    assert c[0, 1] > c[0, 0]
