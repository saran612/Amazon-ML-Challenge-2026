"""Candidate generation, run per country.

Direction is flipped: every S2/S3 record is a query, and the S1 records of the same country
are the index. Each S2/S3 record belongs to at most one S1, so the matcher later picks at
most one S1 per query.

Passes:
  A  char 3-grams of the core name with spaces removed (typos, reordering, domains, scripts)
  B  address word tokens (trade names: different name, same address)
  C  joint name + address word tokens (generic names disambiguated by address)
  D  nearest names in a multilingual embedding space (cross-script names)
A-C are sparse TF-IDF top-k searches; D is a FAISS search (embed.py).
The union is capped to `max_candidates` per query by a combined score, always keeping the
top `keep_top_per_pass` of every pass.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import embed
from .config import Config
from .log import log

SPARSE_PASSES = ("char", "addr", "joint")
PASSES = SPARSE_PASSES + ("embed",)  # r_embed is 99 when embeddings are switched off
COS_COLS = ("cos_char", "cos_nword", "cos_addr", "cos_joint")

# ---------------------------------------------------------------------------- worker side
_ST: sp.csr_matrix | None = None


def _init_worker(folder: str, shape: tuple[int, int]) -> None:
    global _ST
    arrs = [np.load(os.path.join(folder, f"{k}.npy"), mmap_mode="r") for k in ("data", "indices", "indptr")]
    _ST = sp.csr_matrix(tuple(arrs), shape=shape, copy=False)


def _topk_task(args):
    data, indices, indptr, shape, k, row0 = args
    Q = sp.csr_matrix((data, indices, indptr), shape=shape)
    C = (Q @ _ST).tocsr()
    ip, cd, ci = C.indptr, C.data, C.indices
    rows, cols, vals = [], [], []
    for r in range(C.shape[0]):
        a, b = ip[r], ip[r + 1]
        if a == b:
            continue
        d = cd[a:b]
        sel = np.argpartition(-d, k)[:k] if b - a > k else np.arange(b - a)
        rows.append(np.full(len(sel), row0 + r, np.int32))
        cols.append(ci[a:b][sel].astype(np.int32))
        vals.append(d[sel].astype(np.float32))
    if not rows:
        return np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


# ---------------------------------------------------------------------------- main side
def _prune_for_retrieval(S: sp.csr_matrix, cfg: Config) -> sp.csr_matrix:
    """Drop very frequent columns from the index (retrieval only; they carry little signal)."""
    df = np.bincount(S.indices, minlength=S.shape[1])
    cap = max(cfg.retrieval_min_df_cap, int(cfg.retrieval_max_df * S.shape[0]))
    keep = (df <= cap).astype(np.float32)
    Sr = (S @ sp.diags(keep)).tocsr()
    Sr.eliminate_zeros()
    return Sr


def sparse_topk(Q: sp.csr_matrix, S: sp.csr_matrix, k: int, cfg: Config, pool_dir: str):
    """Top-k index rows of S for every row of Q by (pruned) dot product, in parallel."""
    ST = _prune_for_retrieval(S, cfg).T.tocsr()
    folder = tempfile.mkdtemp(dir=pool_dir)
    try:
        for key in ("data", "indices", "indptr"):
            np.save(os.path.join(folder, f"{key}.npy"), getattr(ST, key))
        tasks = []
        for st in range(0, Q.shape[0], cfg.query_chunk):
            sub = Q[st:st + cfg.query_chunk]
            tasks.append((sub.data, sub.indices, sub.indptr, sub.shape, k, st))
        out_r, out_c, out_v = [], [], []
        with ProcessPoolExecutor(cfg.n_jobs, mp_context=get_context("spawn"),
                                 initializer=_init_worker, initargs=(folder, ST.shape)) as ex:
            for r, c, v in ex.map(_topk_task, tasks, chunksize=4):
                out_r.append(r), out_c.append(c), out_v.append(v)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    return np.concatenate(out_r), np.concatenate(out_c), np.concatenate(out_v)


def rowwise_dot(A: sp.csr_matrix, B: sp.csr_matrix, ai: np.ndarray, bi: np.ndarray,
                chunk: int = 1_000_000) -> np.ndarray:
    out = np.empty(len(ai), np.float32)
    for st in range(0, len(ai), chunk):
        x = A[ai[st:st + chunk]]
        y = B[bi[st:st + chunk]]
        out[st:st + chunk] = np.asarray(x.multiply(y).sum(axis=1)).ravel()
    return out


def _tfidf(**kw) -> TfidfVectorizer:
    return TfidfVectorizer(sublinear_tf=True, dtype=np.float32, lowercase=False, **kw)


def rowwise_max(A: sp.csr_matrix, B: sp.csr_matrix, ai: np.ndarray, bi: np.ndarray,
                chunk: int = 1_000_000) -> np.ndarray:
    """For binary-IDF rows: IDF of the rarest token the two rows share (0 when none)."""
    out = np.empty(len(ai), np.float32)
    for st in range(0, len(ai), chunk):
        M = A[ai[st:st + chunk]].multiply(B[bi[st:st + chunk]]).tocsr()
        out[st:st + chunk] = np.sqrt(M.max(axis=1).toarray().ravel())
    return out


def _binary_idf(M: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    B = M.copy()
    B.data = idf[B.indices].astype(np.float32)
    return B


def build_matrices(s1c: pl.DataFrame, qc: pl.DataFrame, cfg: Config) -> dict:
    """TF-IDF matrices fitted on this country's records of the dataset being processed."""
    ns = s1c.height
    char_text = pl.concat([s1c["core_ns"], qc["core_ns"]])
    char_text = (" " + char_text + " ").to_list()
    char = _tfidf(analyzer="char", ngram_range=(cfg.char_ngram, cfg.char_ngram), min_df=2).fit_transform(char_text)
    vn = _tfidf(token_pattern=r"[^ ]+")
    nword = vn.fit_transform(pl.concat([s1c["core"], qc["core"]]).to_list()).tocsr()
    va = _tfidf(token_pattern=r"[^ ]+")
    addr = va.fit_transform(pl.concat([s1c["addr_norm"], qc["addr_norm"]]).to_list()).tocsr()
    joint = normalize(sp.hstack([nword, addr]).tocsr())
    mats = {}
    for key, M in (("char", char), ("nword", nword), ("addr", addr), ("joint", joint),
                   ("nword_idf", _binary_idf(nword, vn.idf_)), ("addr_idf", _binary_idf(addr, va.idf_))):
        M = M.tocsr()
        mats[key] = (M[:ns], M[ns:])
    return mats


def block_country(s1c: pl.DataFrame, qc: pl.DataFrame, cfg: Config, pool_dir: str,
                  emb_path=None) -> pl.DataFrame:
    """Candidate pairs for one country with blocking scores and query-side context."""
    mats = build_matrices(s1c, qc, cfg)
    ks = {"char": cfg.topk_char, "addr": cfg.topk_addr, "joint": cfg.topk_joint}
    found = []

    def add_pass(p, r, c, v):
        found.append(pl.DataFrame({"lq": r, "ls": c, "v": v})
                     .with_columns(pl.col("v").rank("ordinal", descending=True).over("lq")
                                   .cast(pl.Int16).alias(f"r_{p}"))
                     .drop("v"))
        log(f"   pass {p}: {len(r):,} hits")

    for p in SPARSE_PASSES:
        S, Q = mats[p]
        add_pass(p, *sparse_topk(Q, S, ks[p], cfg, pool_dir))
    if emb_path is not None:  # pass D: nearest names in the multilingual embedding space
        add_pass("embed", *embed.ann_topk(emb_path, s1c["eid"].to_numpy(), qc["eid"].to_numpy(), cfg))
    pairs = found[0]
    for f in found[1:]:
        pairs = pairs.join(f, on=["lq", "ls"], how="full", coalesce=True)
    if "r_embed" not in pairs.columns:
        pairs = pairs.with_columns(r_embed=pl.lit(None, pl.Int16))
    lq = pairs["lq"].to_numpy()
    ls = pairs["ls"].to_numpy()
    cos = {}
    for col, key in zip(COS_COLS, ("char", "nword", "addr", "joint")):
        S, Q = mats[key]
        cos[col] = rowwise_dot(Q, S, lq, ls)
    for col, key in (("name_max_idf", "nword_idf"), ("addr_max_idf", "addr_idf")):
        S, Q = mats[key]
        cos[col] = rowwise_max(Q, S, lq, ls)
    if emb_path is not None:
        E = np.load(emb_path, mmap_mode="r")
        cos["cos_embed"] = embed.rowwise_cos(E, qc["eid"].to_numpy()[lq], s1c["eid"].to_numpy()[ls])
    else:
        cos["cos_embed"] = np.full(len(lq), -1, np.float32)
    del mats
    pairs = pairs.with_columns(**{k: pl.Series(v) for k, v in cos.items()})
    pairs = pairs.with_columns(
        [pl.col(f"r_{p}").fill_null(99) for p in PASSES]
    ).with_columns(
        comb=(0.4 * pl.col("cos_char") + 0.4 * pl.col("cos_joint") + 0.2 * pl.col("cos_addr")),
        r_min=pl.min_horizontal([pl.col(f"r_{p}") for p in PASSES]),
    ).with_columns(
        comb_rank=pl.col("comb").rank("ordinal", descending=True).over("lq").cast(pl.Int16),
    )
    pairs = pairs.filter((pl.col("comb_rank") <= cfg.max_candidates) | (pl.col("r_min") <= cfg.keep_top_per_pass))
    pairs = add_query_context(pairs, "comb", "lq")
    # compact dtypes: the pair table is the largest object in memory (~45 bytes per pair)
    return pairs.with_columns(
        q=pl.Series(qc["idx"].to_numpy()[pairs["lq"].to_numpy()], dtype=pl.Int32),
        s=pl.Series(s1c["idx"].to_numpy()[pairs["ls"].to_numpy()], dtype=pl.Int32),
        comb=pl.col("comb").cast(pl.Float32),
        comb_other=pl.col("comb_other").cast(pl.Float32),
        comb_margin=pl.col("comb_margin").cast(pl.Float32),
    ).drop("lq", "ls", "r_min")


def add_query_context(pairs: pl.DataFrame, score: str, key: str) -> pl.DataFrame:
    """Per-query context of a score: rank, best other candidate, margin, candidate count."""
    top1 = pl.col(score).max().over(key)
    top2 = pl.col(score).top_k(2).min().over(key)
    n = pl.len().over(key)
    best_other = pl.when(n == 1).then(0.0).when(pl.col(score) >= top1).then(top2).otherwise(top1)
    return pairs.with_columns(
        (best_other).alias(f"{score}_other"),
        (pl.col(score) - best_other).alias(f"{score}_margin"),
        n.cast(pl.Int16).alias("n_cand"),
    )


def recall_report(pairs: pl.DataFrame, owner: np.ndarray, s1: pl.DataFrame, q: pl.DataFrame,
                  s_mask: np.ndarray) -> dict:
    """Blocking quality on the true pairs whose S1 is in s_mask.

    Overall and per-pass pair recall, recall on hard slices, candidates per query / per S1,
    and reduction ratio (share of same-country S1 x query comparisons avoided).
    """
    from rapidfuzz import fuzz, process

    pq, ps = pairs["q"].to_numpy(), pairs["s"].to_numpy()
    is_true = owner[pq] == ps
    true_q = np.flatnonzero(owner >= 0)
    true_q = true_q[s_mask[owner[true_q]]]

    def recall(hit_rows: np.ndarray, subset: np.ndarray) -> float | None:
        found = np.zeros(len(owner), bool)
        found[pq[hit_rows]] = True
        return float(found[subset].mean()) if len(subset) else None

    out = {"true_pairs": int(len(true_q)), "pair_recall": recall(is_true, true_q)}
    for p in PASSES:
        out[f"recall_pass_{p}"] = recall(is_true & (pairs[f"r_{p}"].to_numpy() < 99), true_q)

    qn = q["core"].to_numpy()[true_q]
    sn = s1["core"].to_numpy()[owner[true_q]]
    name_sim = process.cpdist(qn.tolist(), sn.tolist(), scorer=fuzz.token_set_ratio, workers=-1)
    slices = {
        "non_latin_name": q["nonascii"].to_numpy()[true_q],
        "empty_address": (q["addr_norm"].str.len_chars() == 0).to_numpy()[true_q],
        "domain_name": q["is_domain"].to_numpy()[true_q],
        "trade_name (name sim < 50)": name_sim < 50,
    }
    out["slices"] = {k: {"share": float(m.mean()), "recall": recall(is_true, true_q[m])}
                     for k, m in slices.items()}

    n_s1 = s1.group_by("country").len("n_s1")
    n_q = q.group_by("country").len("n_q")
    all_cmp = n_s1.join(n_q, on="country").select((pl.col("n_s1").cast(pl.Float64) * pl.col("n_q")).sum()).item()
    out["cand_per_query"] = float(len(pq) / max(1, len(np.unique(pq))))
    out["cand_per_s1"] = float(len(pq) / max(1, s1.height))
    out["reduction_ratio"] = float(1 - len(pq) / max(1.0, all_cmp))
    return out
