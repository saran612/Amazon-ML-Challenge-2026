import numpy as np

from er import metrics


def test_worked_example():
    # predict [a, b, c] for S1 #0, truth [a, c]  ->  P = 2/3, R = 1  ->  F0.5 = 0.714
    owner = np.array([0, 1, 0])
    f = metrics.per_entity_f05(np.array([0, 0, 0]), np.array([0, 1, 2]), owner, np.array([0]))
    assert abs(f[0] - 0.7143) < 1e-3


def test_singletons_and_empty_predictions():
    owner = np.array([0, -1])  # record 0 belongs to S1 #0, record 1 to nobody
    none = np.array([], dtype=int)
    assert metrics.per_entity_f05(none, none, owner, np.array([2]))[0] == 1.0       # singleton, left empty
    assert metrics.per_entity_f05(np.array([2]), np.array([1]), owner, np.array([2]))[0] == 0.0  # singleton, false merge
    assert metrics.per_entity_f05(none, none, owner, np.array([0]))[0] == 0.0       # missed its only match


def test_self_check_runs():
    metrics.self_check()
