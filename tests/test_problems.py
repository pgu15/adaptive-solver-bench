import numpy as np

from asbench.problems import synthetic_sequence


def test_sequence_shape_and_symmetry():
    seq = synthetic_sequence(n=10, steps=4, seed=0)
    assert len(seq) == 4
    A = seq.systems[0].A
    assert A.shape == (100, 100)
    assert abs(A - A.T).max() < 1e-12


def test_conditioning_drifts():
    seq = synthetic_sequence(n=10, steps=6, seed=0)
    d0 = seq.systems[0].A.diagonal()
    d5 = seq.systems[5].A.diagonal()
    assert np.std(d5) > np.std(d0)
