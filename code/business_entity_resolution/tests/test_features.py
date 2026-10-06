import numpy as np
import polars as pl

from er.features import competitor_features, number_distance


def test_number_distance():
    Q = pl.DataFrame({"num1": [305, 2026, -1], "num1_str": ["305", "2026", ""]})
    S = pl.DataFrame({"num1": [304, 2005, 12], "num1_str": ["304", "2005", "12"]})
    out = number_distance(Q, S)
    assert out["num1_close2"].tolist() == [1.0, 0.0, -1.0]      # off by one / clearly different / missing
    assert out["num1_lev"][0] == 1.0 and out["num1_absdiff"][2] == -1.0


def test_competitors_among_same_name_candidates():
    q = np.array([0, 0, 0])
    core = pl.Series(["acme", "acme", "other"])
    out = competitor_features(q, core, np.array([90, 60, 95], np.float32), np.array([0, 2, 0], np.float32))
    assert out["core_addr_gap"].tolist() == [0.0, -30.0, 0.0]    # best same-name address vs the runner-up
    assert out["addr_rank"].tolist() == [2.0, 3.0, 1.0]
