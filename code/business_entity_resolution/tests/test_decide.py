import numpy as np
import polars as pl

from er import decide


def test_each_record_goes_to_its_best_s1():
    best = decide.argmax_per_query(np.array([0, 0, 1]), np.array([5, 6, 5]), np.array([0.3, 0.9, 0.4]))
    assert sorted(best.select("q", "s").rows()) == [(0, 6), (1, 5)]


def test_global_threshold():
    best = pl.DataFrame({"q": [0, 1], "s": [5, 5], "p": [0.9, 0.4]})
    assert decide.select_global(best, 0.5).rows() == [(5, 0)]


def test_expected_f05_keeps_the_best_prefix():
    best = pl.DataFrame({"q": [0, 1, 2], "s": [7, 7, 7], "p": [0.99, 0.98, 0.10]})
    out = decide.select_expected_f(best, t_assign=0.02, miss_mass=0.0)
    assert sorted(out["q"].to_list()) == [0, 1]


def test_expected_f05_prefers_empty_for_a_likely_singleton():
    best = pl.DataFrame({"q": [0], "s": [7], "p": [0.05]})
    assert decide.select_expected_f(best, t_assign=0.02, miss_mass=0.0).height == 0
