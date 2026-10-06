"""Error analysis on validation half B.

Every false positive (a record wrongly matched to an S1) and false negative (a true match that
was not predicted) is tagged with the slices it belongs to. For each slice, the "headroom" is the
macro F0.5 gain if every error in that slice were fixed (an oracle fix): an upper bound on what
an improvement targeting that slice could be worth.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .metrics import macro_f05


def _slices(qi: np.ndarray, si: np.ndarray, q: pl.DataFrame, s1: pl.DataFrame) -> dict[str, np.ndarray]:
    name_sim = process.cpdist(q["core"].gather(qi).to_list(), s1["core"].gather(si).to_list(),
                              scorer=fuzz.token_set_ratio, workers=-1)
    return {
        "non_latin_name": q["nonascii"].to_numpy()[qi],
        "empty_address": (q["addr_norm"].str.len_chars() == 0).to_numpy()[qi],
        "domain_name": q["is_domain"].to_numpy()[qi],
        "trade_name (name sim < 50)": name_sim < 50,
        "generic_name (>= 5 S1 share it)": s1["core_cnt"].to_numpy()[si] >= 5,
    }


def error_analysis(pred_s: np.ndarray, pred_q: np.ndarray, owner: np.ndarray, eval_s: np.ndarray,
                   cand_q: np.ndarray, cand_s: np.ndarray, q: pl.DataFrame, s1: pl.DataFrame,
                   examples_path: Path | None = None) -> dict:
    in_eval = np.zeros(s1.height, bool)
    in_eval[eval_s] = True
    m = in_eval[pred_s]
    ps, pq = pred_s[m], pred_q[m]
    correct = owner[pq] == ps
    fp_s, fp_q = ps[~correct], pq[~correct]

    true_q = np.flatnonzero((owner >= 0) & in_eval[np.maximum(owner, 0)])
    predicted = np.zeros(len(owner), bool)
    predicted[pq[correct]] = True
    fn_q = true_q[~predicted[true_q]]
    fn_s = owner[fn_q]
    in_cand = np.zeros(len(owner), bool)
    in_cand[cand_q[owner[cand_q] == cand_s]] = True

    base = macro_f05(ps, pq, owner, eval_s)
    fp_sl = _slices(fp_q, fp_s, q, s1)
    fn_sl = _slices(fn_q, fn_s, q, s1)
    fn_sl["blocking_miss"] = ~in_cand[fn_q]
    fp_sl["blocking_miss"] = np.zeros(len(fp_q), bool)

    out = {"macro_f05": base, "n_false_pos": int(len(fp_q)), "n_false_neg": int(len(fn_q)), "slices": {}}
    for name in fn_sl:
        fpm, fnm = fp_sl[name], fn_sl[name]
        keep = np.ones(len(ps), bool)
        wrong_idx = np.flatnonzero(~correct)
        keep[wrong_idx[fpm]] = False
        fixed_s = np.concatenate([ps[keep], fn_s[fnm]])
        fixed_q = np.concatenate([pq[keep], fn_q[fnm]])
        out["slices"][name] = {
            "false_pos": int(fpm.sum()), "false_neg": int(fnm.sum()),
            "headroom_macro_f05": float(macro_f05(fixed_s, fixed_q, owner, eval_s) - base),
        }
    if examples_path is not None:
        _write_examples(examples_path, fp_q, fp_s, fn_q, fn_s, q, s1)
    return out


def _write_examples(path: Path, fp_q, fp_s, fn_q, fn_s, q, s1, n: int = 200) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for kind, qi, si in (("false_positive", fp_q, fp_s), ("false_negative", fn_q, fn_s)):
        if len(qi) == 0:
            continue
        pick = rng.choice(len(qi), min(n, len(qi)), replace=False)
        for j in pick:
            rows.append({"kind": kind, "query_id": q["entity_id"][int(qi[j])], "s1_id": s1["entity_id"][int(si[j])],
                         "query_name": q["name_norm"][int(qi[j])], "s1_name": s1["name_norm"][int(si[j])],
                         "query_address": q["addr_norm"][int(qi[j])], "s1_address": s1["addr_norm"][int(si[j])]})
    if rows:
        pl.DataFrame(rows).write_csv(path, separator="\t")
