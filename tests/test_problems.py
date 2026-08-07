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


def test_regime_change_switches_operator():
    from asbench.problems import regime_change_sequence

    seq = regime_change_sequence(n=12, blocks=((3, 0.5), (2, 12.0), (3, 0.5)))
    assert len(seq) == 8
    easy = seq.systems[0].A.diagonal()
    hard = seq.systems[3].A.diagonal()
    back = seq.systems[5].A.diagonal()
    # the hard block is far more heterogeneous, and the sequence returns
    assert np.std(hard) > 10 * np.std(easy)
    assert np.allclose(back, easy)


def test_regime_change_rhs_varies_within_a_block():
    from asbench.problems import regime_change_sequence

    seq = regime_change_sequence(n=10, blocks=((3, 1.0),))
    assert not np.allclose(seq.systems[0].b, seq.systems[1].b)
