"""Consistency between records of the same business.

Records of one business (its S2 and S3 records, and repeated records within a source) should
end up with the same S1. When record r is confidently matched to S1 s ("anchor", p >= 0.5)
and record q, also a candidate of s, is a near-duplicate of r, q's probability for s is raised:

    p'(q, s) = max( p(q, s),  lambda * p(r, s) * sim(q, r) )   for sim(q, r) >= sim_min

sim is the token-set similarity of the two records' names (and addresses when both have one).
lambda = 0 switches the step off; lambda and sim_min are tuned on validation half A together
with the "off" setting, so the step is only kept when it measurably helps.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process


def build_triples(q: np.ndarray, s: np.ndarray, p: np.ndarray, qframe: pl.DataFrame,
                  anchor_min: float, target_min: float) -> pl.DataFrame:
    """(row, pa, sim): pair row index, the anchor's probability, and the q-r similarity."""
    df = pl.DataFrame({"row": np.arange(len(q), dtype=np.int64), "q": q, "s": s, "p": p})
    anchors = df.filter(pl.col("p") >= anchor_min).select("s", pl.col("q").alias("r"), pl.col("p").alias("pa"))
    t = (df.filter(pl.col("p") >= target_min).select("row", "q", "s")
         .join(anchors, on="s").filter(pl.col("q") != pl.col("r")))
    if t.height == 0:
        return pl.DataFrame({"row": [], "pa": [], "sim": []},
                            schema={"row": pl.Int64, "pa": pl.Float32, "sim": pl.Float32})
    qi, ri = t["q"].to_numpy(), t["r"].to_numpy()
    names = qframe["name_norm"]
    addrs = qframe["addr_norm"]
    n_sim = process.cpdist(names.gather(qi).to_list(), names.gather(ri).to_list(),
                           scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    a_q, a_r = addrs.gather(qi), addrs.gather(ri)
    a_sim = process.cpdist(a_q.to_list(), a_r.to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    both = ((a_q.str.len_chars() > 0) & (a_r.str.len_chars() > 0)).to_numpy()
    sim = np.where(both, 0.5 * n_sim + 0.5 * a_sim, n_sim) / 100
    return pl.DataFrame({"row": t["row"], "pa": t["pa"].cast(pl.Float32), "sim": pl.Series(sim, dtype=pl.Float32)})


def apply(p: np.ndarray, triples: pl.DataFrame, lam: float, sim_min: float) -> np.ndarray:
    if lam <= 0 or triples.height == 0:
        return p
    boost = (triples.filter(pl.col("sim") >= sim_min)
             .group_by("row").agg((lam * pl.col("pa") * pl.col("sim")).max().alias("b")))
    out = p.copy()
    rows = boost["row"].to_numpy()
    out[rows] = np.maximum(out[rows], boost["b"].to_numpy().astype(np.float32))
    return out
