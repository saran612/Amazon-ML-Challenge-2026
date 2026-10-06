import score


def test_scorer_self_test():
    score.self_test()


def test_scorer_macro_average():
    pred = {"a": {"x", "y", "z"}, "b": set()}
    truth = {"a": {"x", "z"}, "b": set()}
    summary, per = score.score(pred, truth)
    assert abs(per["a"] - 0.7143) < 1e-3 and per["b"] == 1.0
    assert abs(summary["macro_f05"] - (0.7143 + 1.0) / 2) < 1e-3
