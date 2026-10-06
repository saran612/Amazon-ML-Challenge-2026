"""Pairwise features for (query, S1) candidate pairs.

There is deliberately no country feature, so the model applies to countries unseen in training.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .config import Config

REC_COLS = ["name_norm", "core", "core_ns", "skel", "addr_norm", "place_str", "place_list", "street_str",
            "nums_str", "core_cnt", "local_cnt", "nonascii", "is_domain", "num1", "num1_str"]

# (feature name, record column, scorer)
FUZZY = [
    ("n_ratio", "name_norm", fuzz.ratio),
    ("n_tset", "name_norm", fuzz.token_set_ratio),
    ("n_tsort", "name_norm", fuzz.token_sort_ratio),
    ("n_partial", "name_norm", fuzz.partial_ratio),
    ("c_ratio", "core", fuzz.ratio),
    ("c_tset", "core", fuzz.token_set_ratio),
    ("c_jw", "core", JaroWinkler.normalized_similarity),
    ("ns_ratio", "core_ns", fuzz.ratio),
    ("ns_partial", "core_ns", fuzz.partial_ratio),
    ("sk_ratio", "skel", fuzz.ratio),
    ("a_ratio", "addr_norm", fuzz.ratio),
    ("a_tset", "addr_norm", fuzz.token_set_ratio),
    ("a_tsort", "addr_norm", fuzz.token_sort_ratio),
    ("pl_tset", "place_str", fuzz.token_set_ratio),
    ("st_tset", "street_str", fuzz.token_set_ratio),
    ("st_tsort", "street_str", fuzz.token_sort_ratio),
    ("num_tset", "nums_str", fuzz.token_set_ratio),
]
SOFT = ["soft_nq", "soft_ns", "soft_stq", "soft_sts", "num_shared", "num_fuzzy", "num_first_eq",
        "plink_q", "plink_s",
        "num_min_diff"]                                   # closest pair of house numbers
PAIR_COLS = ["cos_char", "cos_nword", "cos_addr", "cos_joint", "comb", "comb_rank", "comb_other",
             "comb_margin", "n_cand", "r_char", "r_addr", "r_joint",
             "r_embed", "cos_embed",                   # multilingual name embeddings
             "name_max_idf", "addr_max_idf"]           # rarest token the two records share
FLAGS = ["first_tok_eq", "q_src3", "q_nonascii", "q_domain", "s_domain", "q_addr_empty", "s_addr_empty",
         "q_ntok", "s_ntok", "len_ratio", "q_core_cnt", "s_core_cnt", "q_nnum", "s_nnum",
         "q_npl", "s_npl",
         "q_local_cnt", "s_local_cnt", "cand_same_core"]   # local name rarity
# house-number distance (noise like 305 vs 304 vs a different address like 2026 vs 2005)
NUMDIST = ["num1_absdiff", "num1_reldiff", "num1_close2", "num1_lev"]
# the S1's other records ("strong siblings": records whose best blocking match is this S1)
SIB = ["sib_n", "sib_max_name", "sib_max_addr", "sib_max_joint", "sib_num_agree", "sib_num_close"]
# competitors among the query's candidates, overall and with the same core name
COMP = ["core_addr_gap", "core_addr_rank", "addr_rank", "addr_gap_other", "core_num_gap"]
FEATURES = PAIR_COLS + [f for f, _, _ in FUZZY] + SOFT + FLAGS + NUMDIST + SIB + COMP
GROUPS = {"blocking+cosines": PAIR_COLS, "fuzzy text": [f for f, _, _ in FUZZY], "soft/number/place": SOFT,
          "flags+rarity": FLAGS, "number distance": NUMDIST, "siblings": SIB, "competitors": COMP}


# ---------------------------------------------------------------------------- soft matching
_LINKS: frozenset = frozenset()


def init_worker(links: list[str]) -> None:
    """Soft-feature workers receive the learned place links once, at start-up."""
    global _LINKS
    _LINKS = frozenset(links)


def _place_link(ta: list[str], tb: list[str]) -> float:
    """Share of places in ta that equal, or co-occur in the data with, a place in tb."""
    if not ta or not tb:
        return -1.0
    sb = set(tb)
    hit = 0
    for x in ta:
        if x in sb or any((f"{x}|{y}" if x < y else f"{y}|{x}") in _LINKS for y in tb):
            hit += 1
    return hit / len(ta)


def _abbrev(short: str, long: str) -> bool:
    if len(short) < 2 or short[0] != long[0]:
        return False
    it = iter(long)
    return all(c in it for c in short)


def _tok_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if _abbrev(short, long):
        return 0.8
    j = JaroWinkler.normalized_similarity(a, b)
    return j if j >= 0.85 else 0.0


def _soft(ta: list[str], tb: list[str]) -> float:
    """Share of tokens in ta with an exact, abbreviation or close-spelling partner in tb."""
    if not ta or not tb:
        return -1.0
    tot = 0.0
    for x in ta:
        best = 0.0
        for y in tb:
            v = _tok_sim(x, y)
            if v > best:
                best = v
                if best == 1.0:
                    break
        tot += best
    return tot / len(ta)


def _num_match(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 2 and (a.startswith(b) or b.startswith(a)
                                                    or a.endswith(b) or b.endswith(a)))


def _soft_task(args) -> np.ndarray:
    qc, sc, qs, ss, qn, sn, qp, sp_ = args
    out = np.empty((len(qc), len(SOFT)), np.float32)
    for i in range(len(qc)):
        a, b = qc[i].split(), sc[i].split()
        c, d = qs[i].split(), ss[i].split()
        na, nb = qn[i].split(), sn[i].split()
        if na and nb:
            shared = len(set(na) & set(nb))
            fz = float(any(_num_match(x, y) for x in na for y in nb))
            first = float(na[0] == nb[0])
            mind = float(np.log1p(min(abs(int(x[:15]) - int(y[:15])) for x in na for y in nb)))
        else:
            shared, fz, first, mind = -1, -1.0, -1.0, -1.0
        pa = [x for x in qp[i].split("|") if x]
        pb = [x for x in sp_[i].split("|") if x]
        out[i] = (_soft(a, b), _soft(b, a), _soft(c, d), _soft(d, c), shared, fz, first,
                  _place_link(pa, pb), _place_link(pb, pa), mind)
    return out


def soft_features(pool: ProcessPoolExecutor, cols: list[list[str]], task: int) -> np.ndarray:
    n = len(cols[0])
    tasks = [tuple(c[st:st + task] for c in cols) for st in range(0, n, task)]
    return np.vstack(list(pool.map(_soft_task, tasks))) if tasks else np.empty((0, len(SOFT)), np.float32)


# ---------------------------------------------------------------------------- main
def compute_features(pairs: pl.DataFrame, s1: pl.DataFrame, q: pl.DataFrame,
                     pool: ProcessPoolExecutor, cfg: Config, anchors: pl.DataFrame) -> np.ndarray:
    """Feature matrix (float32, columns in FEATURES order) for one chunk of pairs.

    The chunk must contain whole queries (all candidates of each query it touches).
    `anchors` (columns s, r) lists each S1's strong siblings, see build_anchors.
    """
    qi = pairs["q"].to_numpy()
    si = pairs["s"].to_numpy()
    Q = q.select(REC_COLS + ["source"])[qi]
    S = s1.select(REC_COLS)[si]

    cols = {c: pairs[c].cast(pl.Float32).to_numpy() for c in PAIR_COLS}
    str_cache: dict[str, tuple[list, list]] = {}

    def strs(c):
        if c not in str_cache:
            str_cache[c] = (Q[c].to_list(), S[c].to_list())
        return str_cache[c]

    for name, col, scorer in FUZZY:
        a, b = strs(col)
        v = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
        if scorer is JaroWinkler.normalized_similarity:
            v = v * 100
        # empty strings on either side carry no information: mark as missing
        empty = (Q[col].str.len_chars() == 0) | (S[col].str.len_chars() == 0)
        v[empty.to_numpy()] = -1
        cols[name] = v

    soft = soft_features(pool, [*strs("core"), *strs("street_str"), *strs("nums_str"), *strs("place_list")],
                         cfg.soft_task_size)
    for j, name in enumerate(SOFT):
        cols[name] = soft[:, j]

    q_ntok = Q["core"].str.count_matches(" ") + 1
    s_ntok = S["core"].str.count_matches(" ") + 1
    q_len = Q["core_ns"].str.len_chars().cast(pl.Float32)
    s_len = S["core_ns"].str.len_chars().cast(pl.Float32)
    flags = {
        "first_tok_eq": Q["core"].str.split(" ").list.first() == S["core"].str.split(" ").list.first(),
        "q_src3": (Q["source"] == 3),
        "q_nonascii": Q["nonascii"],
        "q_domain": Q["is_domain"],
        "s_domain": S["is_domain"],
        "q_addr_empty": Q["addr_norm"].str.len_chars() == 0,
        "s_addr_empty": S["addr_norm"].str.len_chars() == 0,
        "q_ntok": q_ntok,
        "s_ntok": s_ntok,
        "len_ratio": pl.Series(np.minimum(q_len, s_len) / np.maximum(np.maximum(q_len, s_len), 1)),
        "q_core_cnt": np.log1p(Q["core_cnt"].cast(pl.Float32)),
        "s_core_cnt": np.log1p(S["core_cnt"].cast(pl.Float32)),
        "q_nnum": Q["nums_str"].str.count_matches(r"\d+"),
        "s_nnum": S["nums_str"].str.count_matches(r"\d+"),
        "q_npl": Q["place_str"].str.count_matches(r"[a-z0-9]+"),
        "s_npl": S["place_str"].str.count_matches(r"[a-z0-9]+"),
    }
    # how many of this query's candidates carry the same core name as this S1
    same = pl.DataFrame({"q": qi, "core": S["core"]}).select(pl.len().over("q", "core")).to_series()
    flags["q_local_cnt"] = Q["local_cnt"]
    flags["s_local_cnt"] = S["local_cnt"]
    flags["cand_same_core"] = same
    for k, v in flags.items():
        cols[k] = pl.Series(v).cast(pl.Float32).to_numpy()

    cols.update(number_distance(Q, S))
    cols.update(sibling_features(qi, si, q, anchors))
    cols.update(competitor_features(qi, S["core"], cols["a_tset"], cols["num1_absdiff"]))
    return np.column_stack([cols[f] for f in FEATURES]).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------- number / sibling / competitor
def number_distance(Q: pl.DataFrame, S: pl.DataFrame) -> dict:
    """Distance between the first house numbers (-1 when either record has none)."""
    a = Q["num1"].to_numpy().astype(np.float64)
    b = S["num1"].to_numpy().astype(np.float64)
    ok = (a >= 0) & (b >= 0)
    d = np.abs(a - b)
    lev = process.cpdist(Q["num1_str"].to_list(), S["num1_str"].to_list(), scorer=Levenshtein.distance,
                         workers=-1, dtype=np.float32)
    return {
        "num1_absdiff": np.where(ok, np.log1p(d), -1).astype(np.float32),
        "num1_reldiff": np.where(ok, d / np.maximum(np.maximum(a, b), 1), -1).astype(np.float32),
        "num1_close2": np.where(ok, (d <= 2).astype(np.float32), -1).astype(np.float32),
        "num1_lev": np.where(ok, lev, -1).astype(np.float32),
    }


def sibling_features(qi: np.ndarray, si: np.ndarray, q: pl.DataFrame, anchors: pl.DataFrame) -> dict:
    """How this record compares with the S1's other strong records (its "siblings").

    A true match usually resembles the business's other records even when it differs from the
    S1 record itself (house number noise); a same-name neighbour usually does not.
    """
    n = len(qi)
    out = {k: np.full(n, -1, np.float32) for k in SIB}
    out["sib_n"] = np.zeros(n, np.float32)
    t = (pl.DataFrame({"row": np.arange(n, dtype=np.int64), "q": qi, "s": si})
         .join(anchors, on="s", how="inner").filter(pl.col("r") != pl.col("q")))
    if t.height == 0:
        return out
    rq, rr = t["q"].to_numpy(), t["r"].to_numpy()
    names, addrs = q["name_norm"], q["addr_norm"]
    nm = process.cpdist(names.gather(rq).to_list(), names.gather(rr).to_list(),
                        scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    aq, ar = addrs.gather(rq), addrs.gather(rr)
    ad = process.cpdist(aq.to_list(), ar.to_list(), scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32)
    has_addr = ((aq.str.len_chars() > 0) & (ar.str.len_chars() > 0)).to_numpy()
    num = q["num1"].to_numpy()
    n1, n2 = num[rq], num[rr]
    has_num = (n1 >= 0) & (n2 >= 0)
    def masked(v, ok):  # missing -> null, which max/mean skip (NaN would poison the mean)
        return pl.Series(np.where(ok, v, np.nan).astype(np.float32)).fill_nan(None)

    t = t.with_columns(
        nm=pl.Series(nm),
        ad=masked(ad, has_addr),
        jt=pl.Series(np.where(has_addr, 0.5 * nm + 0.5 * ad, nm).astype(np.float32)),
        agree=masked((n1 == n2).astype(np.float32), has_num),
        close=masked((np.abs(n1 - n2) <= 2).astype(np.float32), has_num),
    )
    g = t.group_by("row").agg(
        pl.len().alias("sib_n"), pl.col("nm").max().alias("sib_max_name"),
        pl.col("ad").max().alias("sib_max_addr"), pl.col("jt").max().alias("sib_max_joint"),
        pl.col("agree").mean().alias("sib_num_agree"), pl.col("close").mean().alias("sib_num_close"),
    )
    rows = g["row"].to_numpy()
    for k in SIB:
        out[k][rows] = g[k].cast(pl.Float32).fill_null(-1).fill_nan(-1).to_numpy()
    return out


def competitor_features(qi: np.ndarray, core: pl.Series, a_tset: np.ndarray, num1_absdiff: np.ndarray) -> dict:
    """Is this S1 the best address match among the query's candidates (overall / with the same name)?"""
    df = pl.DataFrame({"q": qi, "core": core, "a": a_tset,
                       "d": np.where(num1_absdiff < 0, 99.0, num1_absdiff).astype(np.float32)})
    n = pl.len().over("q")
    top1 = pl.col("a").max().over("q")
    top2 = pl.col("a").top_k(2).min().over("q")
    other = pl.when(n == 1).then(-1.0).when(pl.col("a") >= top1).then(top2).otherwise(top1)
    r = df.select(
        (pl.col("a") - pl.col("a").max().over("q", "core")).alias("core_addr_gap"),
        pl.col("a").rank("ordinal", descending=True).over("q", "core").alias("core_addr_rank"),
        pl.col("a").rank("ordinal", descending=True).over("q").alias("addr_rank"),
        (pl.col("a") - other).alias("addr_gap_other"),
        (pl.col("d") - pl.col("d").min().over("q", "core")).alias("core_num_gap"),
    )
    return {k: r[k].cast(pl.Float32).to_numpy() for k in COMP}


def build_anchors(pairs_path, cfg: Config) -> pl.DataFrame:
    """Each S1's strong siblings: queries whose best blocking candidate it is (score >= sib_min_comb),
    at most sib_max_per_s per S1, best first. Uses blocking output only (no labels), so it is built
    the same way on training and test data."""
    a = (pl.scan_parquet(pairs_path)
         .filter((pl.col("comb_rank") == 1) & (pl.col("comb") >= cfg.sib_min_comb))
         .select("s", pl.col("q").alias("r"), "comb").collect()
         .sort(["s", "comb"], descending=[False, True]))
    return (a.with_columns(k=pl.int_range(pl.len()).over("s"))
            .filter(pl.col("k") < cfg.sib_max_per_s).select("s", "r"))
