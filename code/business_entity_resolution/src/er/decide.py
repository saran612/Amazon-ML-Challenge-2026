"""Decision layer: from pair probabilities to one set of matches per S1.

1. Assignment: each S2/S3 record goes to its highest-probability S1 (it can belong to at most
   one), and only if that probability is at least t_assign.
2. Per-S1 set selection: sort the S1's assigned records by probability and keep the prefix of
   size k that maximizes expected F0.5, approximated with expected counts:
       E[F0.5 | top k] ~= 1.25 * sum(p_1..p_k) / (0.25 * (sum(p_all) + miss_mass) + k)
   The empty set is chosen when P(no match) = prod(1 - p_i) is higher than every prefix.
"""
from __future__ import annotations

import numpy as np
import polars as pl


def argmax_per_query(q: np.ndarray, s: np.ndarray, p: np.ndarray) -> pl.DataFrame:
    df = pl.DataFrame({"q": q, "s": s, "p": p})
    return df.sort(["q", "p"], descending=[False, True]).unique("q", keep="first", maintain_order=True)


def select_expected_f(best: pl.DataFrame, t_assign: float, miss_mass: float) -> pl.DataFrame:
    a = best.filter(pl.col("p") >= t_assign).sort(["s", "p"], descending=[False, True])
    if a.height == 0:
        return a.select("s", "q")
    pc = pl.col("p").clip(1e-6, 1 - 1e-6)
    a = a.with_columns(
        k=pl.int_range(1, pl.len() + 1).over("s"),
        tp=pl.col("p").cum_sum().over("s"),
        g=pl.col("p").sum().over("s") + miss_mass,
        f0=(1 - pc).log().sum().over("s").exp(),
    ).with_columns(f=1.25 * pl.col("tp") / (0.25 * pl.col("g") + pl.col("k")))
    choice = a.group_by("s").agg(
        k_best=pl.col("k").sort_by("f", descending=True).first(),
        f_best=pl.col("f").max(),
        f0=pl.col("f0").first(),
    )
    a = a.join(choice, on="s")
    return a.filter((pl.col("k") <= pl.col("k_best")) & (pl.col("f_best") > pl.col("f0"))).select("s", "q")


def select_global(best: pl.DataFrame, t: float) -> pl.DataFrame:
    return best.filter(pl.col("p") >= t).select("s", "q")


def decide(best: pl.DataFrame, params: dict) -> pl.DataFrame:
    if params["method"] == "expected_f":
        return select_expected_f(best, params["t_assign"], params["miss_mass"])
    return select_global(best, params["t"])
