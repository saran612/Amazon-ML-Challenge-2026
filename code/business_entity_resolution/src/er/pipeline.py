"""End-to-end stages shared by train.py and test.py."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import fields
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from . import analysis, blocking, consistency, crossenc, decide, embed, features, metrics, vocab
from .config import Config
from .io import read_ground_truth, read_split, write_id_lists
from .log import log, stage
from .model import Matcher, build_dataset, fit_calibration, fit_matcher, save_json
from .normalize import normalize_frame
from .split import OTHER, VAL_A, VAL_B, s1_roles, sample_universe

FEATURE_NAMES = features.FEATURES
NO_CONSISTENCY = {"lambda": 0.0, "sim_min": 1.0}

# Bump a stage's version when its code changes in a way that invalidates cached outputs.
STAGE_VERSION = {"norm": 1, "maps": 1, "derive": 1, "finetune": 1, "emb": 1, "pairs": 2, "xtrain": 3, "scores": 2,
                 "ce": 1}
# Stages whose outputs from before per-stage cache keys existed may be adopted (see Cache).
LEGACY_OK = {"norm", "maps", "derive", "finetune", "emb"}


# ---------------------------------------------------------------------------- caching
class Cache:
    """Per-stage cache: each stage's outputs are reused only if its own key still matches.

    A key covers the settings the stage depends on and the keys of the stages it builds on, so
    changing, say, the number of candidates recomputes blocking and later stages but keeps the
    normalized records, the fine-tuned encoder and the embeddings.

    Run folders written before per-stage keys existed carry a meta.json; for the stages in
    LEGACY_OK their outputs are adopted once (they were produced with unchanged settings), and
    meta.json is removed at the end of the first successful run.
    """

    def __init__(self, run_dir: Path, force: bool):
        if force and run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.dir = run_dir
        self.legacy = (run_dir / "meta.json").exists()

    @staticmethod
    def key(stage_name: str, *parts) -> str:
        blob = json.dumps([stage_name, STAGE_VERSION[stage_name], *parts], sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def hit(self, stage_name: str, key: str, *files: str) -> bool:
        if not all((self.dir / f).exists() for f in files):
            return False
        kf = self.dir / f"{stage_name}.key"
        if kf.exists():
            return kf.read_text() == key
        if self.legacy and stage_name in LEGACY_OK:
            kf.write_text(key)
            log(f"   reusing cached {stage_name} outputs (written before per-stage cache keys)")
            return True
        return False

    def done(self, stage_name: str, key: str) -> None:
        (self.dir / f"{stage_name}.key").write_text(key)

    def finish(self) -> None:
        (self.dir / "meta.json").unlink(missing_ok=True)


def _hash_json(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def soft_pool(cfg: Config, links: list[str]) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(cfg.n_jobs, mp_context=get_context("spawn"),
                               initializer=features.init_worker, initargs=(links,))


# ---------------------------------------------------------------------------- records
def load_records(data_dir: Path, split: str, cache: Cache, key: str, cfg: Config, sample: float | None):
    """Ingest + generic normalization (cached). Returns s1, q, owner (None for test)."""
    d = cache.dir
    files = ["norm_s1.parquet", "norm_q.parquet"] + (["owner.npy"] if split == "train" else [])
    if cache.hit("norm", key, *files):
        owner = np.load(d / "owner.npy") if split == "train" else None
        return pl.read_parquet(d / "norm_s1.parquet"), pl.read_parquet(d / "norm_q.parquet"), owner
    s1, q = read_split(data_dir, split)
    s1 = s1.with_columns(idx=pl.int_range(pl.len(), dtype=pl.Int64))
    q = q.with_columns(idx=pl.int_range(pl.len(), dtype=pl.Int64))
    owner = read_ground_truth(data_dir, s1, q) if split == "train" else None
    if sample:
        s1, q, owner = sample_universe(s1, q, owner, sample, cfg.seed)
        log(f"sampled {sample:.0%}: S1={s1.height:,} queries={q.height:,}")
    s1 = normalize_frame(s1)
    q = normalize_frame(q)
    s1.write_parquet(d / "norm_s1.parquet")
    q.write_parquet(d / "norm_q.parquet")
    if owner is not None:
        np.save(d / "owner.npy", owner)
    cache.done("norm", key)
    return s1, q, owner


def mine_maps(s1: pl.DataFrame, q: pl.DataFrame, owner: np.ndarray, keep_s: np.ndarray,
              cfg: Config) -> dict:
    """Per-country token maps from true pairs whose S1 is outside the validation set.

    The shared map ("*") keeps only variants every training country agrees on; it is what an
    country unseen in training receives.
    """
    qi, si = vocab.sample_pairs(owner, keep_s, cfg.abbrev_max_pairs, cfg.seed)
    s_country = s1["country"].to_numpy()[si]
    out = {"name": {}, "addr": {}}
    for key, expr in (("name", pl.col("name_norm")), ("addr", pl.col("addr_segs").list.join(" "))):
        t1 = s1.select(expr.alias("t"))["t"]
        tq = q.select(expr.alias("t"))["t"]
        for c in np.unique(s_country):
            sel = s_country == c
            a = t1.gather(si[sel]).to_list()
            b = tq.gather(qi[sel]).to_list()
            df_c = vocab.token_doc_freq(pl.concat([t1.filter(s1["country"] == c), tq.filter(q["country"] == c)]))
            out[key][c] = vocab.mine_token_map(a, b, df_c, cfg)
            log(f"   {c}: mined {len(out[key][c])} {key} variants, e.g. {list(out[key][c].items())[:15]}")
        out[key][vocab.SHARED] = vocab.shared_map(out[key])
        log(f"   shared {key} variants (for unseen countries): {len(out[key][vocab.SHARED])}, "
            f"e.g. {list(out[key][vocab.SHARED].items())[:15]}")
    return out


def derive_records(s1: pl.DataFrame, q: pl.DataFrame, maps: dict, cfg: Config, cache: Cache, key: str):
    """Apply token maps, learn frequency statistics on these files, add derived columns.

    Also assigns every distinct original-script name an embedding id (`eid`), learns the place
    links and adds local name rarity. Returns s1, q, place links.
    """
    d = cache.dir
    files = ("s1.parquet", "q.parquet", "place_links.json", "emb_texts.parquet")
    if cache.hit("derive", key, *files):
        s1, q = pl.read_parquet(d / "s1.parquet"), pl.read_parquet(d / "q.parquet")
        links = json.loads((d / "place_links.json").read_text())
    else:
        s1 = vocab.apply_maps(s1, maps)
        q = vocab.apply_maps(q, maps)
        both = pl.concat([s1.select("country", "name_norm", "addr_segs"),
                          q.select("country", "name_norm", "addr_segs")])
        stop = vocab.learn_stopwords(both, cfg)
        places = vocab.learn_places(both, cfg)
        del both
        for c in stop:
            log(f"   {c}: {len(stop[c])} low-information name tokens (e.g. {stop[c][:12]}), "
                f"{len(places.get(c, []))} place segments")
        s1 = vocab.derive(s1, stop, places)
        q = vocab.derive(q, stop, places)
        s1, q = vocab.core_counts(s1, q)

        links = vocab.learn_place_links(pl.concat([s1.select("country", "place_list"),
                                                   q.select("country", "place_list")]), cfg)
        log(f"   {len(links):,} place links learned, e.g. {links[:6]}")

        texts = pl.DataFrame({"text": pl.concat([s1["name_raw"], q["name_raw"]]).unique(maintain_order=True)})
        texts = texts.with_columns(eid=pl.int_range(pl.len(), dtype=pl.Int32))
        s1 = s1.join(texts, left_on="name_raw", right_on="text", how="left").sort("idx")
        q = q.join(texts, left_on="name_raw", right_on="text", how="left").sort("idx")
        texts.select("text").write_parquet(d / "emb_texts.parquet")
        log(f"   {texts.height:,} distinct original-script names")
        s1.write_parquet(d / "s1.parquet")
        q.write_parquet(d / "q.parquet")
        (d / "place_links.json").write_text(json.dumps(links))
        cache.done("derive", key)

    s1, q = vocab.add_local_counts(s1, q)  # cheap, recomputed every run
    s1, q = vocab.add_numbers(s1), vocab.add_numbers(q)  # first house number
    # later stages index rows by position: row i must hold idx == i
    for frame in (s1, q):
        assert (frame["idx"].to_numpy() == np.arange(frame.height)).all(), "row order broken"
    return s1, q, links


# ---------------------------------------------------------------------------- name embeddings
def finetune_embedder(s1: pl.DataFrame, q: pl.DataFrame, owner: np.ndarray, keep_s: np.ndarray,
                      cfg: Config, cache: Cache, key: str) -> Path:
    """Fine-tune the encoder on one true pair per S1 outside the validation set."""
    out = cache.dir / "embed_model"
    if cache.hit("finetune", key, "embed_model/DONE"):
        return out
    shutil.rmtree(out, ignore_errors=True)
    qi = np.flatnonzero(owner >= 0)
    qi = qi[keep_s[owner[qi]]]
    df = pl.DataFrame({"s": owner[qi], "a": s1["name_raw"].gather(owner[qi]), "b": q["name_raw"].gather(qi)})
    df = (df.filter(pl.col("a") != pl.col("b"))           # identical strings teach nothing
          .sample(fraction=1.0, shuffle=True, seed=cfg.seed)
          .unique("s", keep="first")                      # one pair per S1: fewer in-batch false negatives
          .head(cfg.finetune_pairs))
    pairs_path = cache.dir / "finetune_pairs.parquet"
    df.select("a", "b").write_parquet(pairs_path)
    log(f"   fine-tuning {cfg.embed_model} on {df.height:,} training pairs")
    embed.finetune(pairs_path, out, cfg)
    (out / "DONE").write_text("ok")
    cache.done("finetune", key)
    return out


def embed_names(cache: Cache, key: str, model_path: str, cfg: Config) -> Path:
    out = cache.dir / "emb.npy"
    if not cache.hit("emb", key, "emb.npy", "emb.DONE"):
        embed.encode_texts(cache.dir / "emb_texts.parquet", out, model_path, cfg)
        (cache.dir / "emb.DONE").write_text(model_path)
        cache.done("emb", key)
    return out


def emb_key(k_norm: str, identity: str, cfg: Config) -> str:
    return Cache.key("emb", k_norm, identity, cfg.embed_max_len, cfg.embed_dim, os.environ.get("ER_EMBED_BACKEND", ""))


# ---------------------------------------------------------------------------- blocking
def pairs_key(k_derive: str, k_emb: str | None, cfg: Config, *extra) -> str:
    return Cache.key("pairs", k_derive, k_emb or "none", cfg.char_ngram, cfg.retrieval_max_df,
                     cfg.retrieval_min_df_cap, cfg.topk_char, cfg.topk_addr, cfg.topk_joint, cfg.topk_embed,
                     cfg.ann_nprobe, cfg.max_candidates, cfg.keep_top_per_pass, *extra)


def simulate_density(s1: pl.DataFrame, q: pl.DataFrame, owner: np.ndarray, role: np.ndarray, cfg: Config):
    """Remove a share of the non-validation S1 records from the index (training only).

    Their S2/S3 records stay, now matching nothing: they become distractors that look like real
    businesses, as the test set's extra records do. Validation S1 records are never removed, so
    validation keeps the same entities, only in a test-like crowd. Returns s1, q, owner, role.
    """
    rng = np.random.default_rng(cfg.seed + 7)
    before = float((owner < 0).mean())
    drop = (role == OTHER) & (rng.random(len(role)) < cfg.sim_drop_s1_frac)
    keep = ~drop
    new = np.full(len(role), -1, np.int64)
    new[keep] = np.arange(int(keep.sum()))
    owner = np.where(owner >= 0, new[np.maximum(owner, 0)], -1)
    s1 = s1.filter(pl.Series(keep)).with_columns(idx=pl.int_range(pl.len(), dtype=pl.Int64))
    # S1-dependent counts must describe the reduced index (as the test index is described by its own)
    cols = ["core_cnt", "local_cnt", "place_key"]
    s1, q = s1.drop(cols, strict=False), q.drop(cols, strict=False)
    s1, q = vocab.core_counts(s1, q)
    s1, q = vocab.add_local_counts(s1, q)
    log(f"   removed {int(drop.sum()):,} non-validation S1 records ({cfg.sim_drop_s1_frac:.0%}); "
        f"distractor share {before:.1%} -> {float((owner < 0).mean()):.1%}; "
        f"records per S1 {q.height / len(role):.2f} -> {q.height / s1.height:.2f}")
    return s1, q, owner, role[keep]


def run_blocking(s1: pl.DataFrame, q: pl.DataFrame, cfg: Config, cache: Cache, key: str, emb_path) -> Path:
    """Candidate pairs (sorted by query) written to pairs.parquet; returns the path."""
    out = cache.dir / "pairs.parquet"
    if cache.hit("pairs", key, "pairs.parquet"):
        return out
    pool_dir = cache.dir / "tmp"
    pool_dir.mkdir(exist_ok=True)
    part_files = []
    countries = sorted(set(s1["country"].unique().to_list()) & set(q["country"].unique().to_list()))
    for i, c in enumerate(countries):
        s1c = s1.filter(pl.col("country") == c)
        qc = q.filter(pl.col("country") == c)
        log(f"   {c or '(blank)'}: {s1c.height:,} S1 x {qc.height:,} queries")
        f = pool_dir / f"part_{i}.parquet"
        blocking.block_country(s1c, qc, cfg, str(pool_dir), emb_path).sort(["q", "comb_rank"]).write_parquet(f)
        part_files.append(f)
    # Stream the per-country parts into one file: never holds every country's pairs in memory.
    # Each query's candidates stay contiguous (a query belongs to one country).
    pl.concat([pl.scan_parquet(f) for f in part_files]).sink_parquet(out)
    shutil.rmtree(pool_dir, ignore_errors=True)
    cache.done("pairs", key)
    return out


def query_chunks(q: np.ndarray, rows: int):
    """Row ranges of roughly `rows` pairs that never split one query's candidates.

    Only requires each query's pairs to be contiguous (not globally sorted)."""
    n = len(q)
    if n == 0:
        return
    starts = np.flatnonzero(np.r_[True, q[1:] != q[:-1]])
    start = 0
    while start < n:
        target = start + rows
        if target >= n:
            end = n
        else:
            i = int(np.searchsorted(starts, target, side="left"))
            end = int(starts[i]) if i < len(starts) else n
        yield start, end
        start = end


def featurize(pairs: pl.DataFrame, s1, q, pool, cfg: Config, anchors: pl.DataFrame, out_path: Path, label=None):
    """Features for all pairs (grouped by query), chunk by chunk, written straight to a
    memory-mapped .npy file so the full matrix never has to sit in RAM."""
    qs = pairs["q"].to_numpy()
    X = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=(pairs.height, len(FEATURE_NAMES)))
    for a, b in query_chunks(qs, cfg.feature_chunk):
        X[a:b] = features.compute_features(pairs[a:b], s1, q, pool, cfg, anchors)
        log(f"   {label or 'features'}: {b:,}/{pairs.height:,} pairs")
    X.flush()
    del X
    return np.load(out_path, mmap_mode="r")


def stream_scores(pairs_path: Path, s1, q, pool, cfg: Config, matchers: dict, label: str,
                  anchors: pl.DataFrame, masks: dict | None = None):
    """Features -> probabilities chunk by chunk, reading the pairs from disk.

    Only the query column is held in memory; each chunk of pairs is read from the Parquet file,
    scored and dropped. `masks` (name -> boolean per row, whole queries) limits a matcher to the
    rows it needs; other rows stay NaN. Stage-1 scores are kept for the "main" matcher only.
    """
    masks = masks or {}
    qs = pl.read_parquet(pairs_path, columns=["q"])["q"].to_numpy()
    lazy = pl.scan_parquet(pairs_path)
    out = {name: np.full(len(qs), np.nan, np.float32) for name in matchers}
    if "main" in matchers:
        out["main_p1"] = np.full(len(qs), np.nan, np.float32)
    for a, b in query_chunks(qs, cfg.feature_chunk):
        X = features.compute_features(lazy.slice(a, b - a).collect(), s1, q, pool, cfg, anchors)
        for name, m in matchers.items():
            mask = masks.get(name)
            idx = np.arange(b - a) if mask is None else np.flatnonzero(mask[a:b])
            if len(idx) == 0:
                continue
            p1, p2 = m.predict(X[idx], qs[a:b][idx])
            out[name][a + idx] = p2
            if name == "main":
                out["main_p1"][a + idx] = p1
        log(f"   {label}: {b:,}/{len(qs):,} pairs scored")
    return out


# ---------------------------------------------------------------------------- evaluation
def evaluate(best: pl.DataFrame, params: dict, owner, eval_s) -> float:
    pred = decide.decide(best, params)
    return metrics.macro_f05(pred["s"].to_numpy(), pred["q"].to_numpy(), owner, eval_s)


def tune(best: pl.DataFrame, owner, eval_s, cfg: Config, methods=("global", "expected_f")):
    """Grid-search decision parameters on eval_s; returns (best params, best score)."""
    grid = []
    if "global" in methods:
        grid += [{"method": "global", "t": t} for t in cfg.global_t_grid]
    if "expected_f" in methods:
        grid += [{"method": "expected_f", "t_assign": t, "miss_mass": m}
                 for t in cfg.t_assign_grid for m in cfg.miss_mass_grid]
    scored = [(evaluate(best, p, owner, eval_s), p) for p in grid]
    score, params = max(scored, key=lambda x: x[0])
    return params, score


def best_per_query(q, s, p):
    return decide.argmax_per_query(q, s, p)


def _pred(best: pl.DataFrame, params: dict):
    pred = decide.decide(best, params)
    return pred["s"].to_numpy(), pred["q"].to_numpy()


def _pred_p(best: pl.DataFrame, params: dict) -> pl.DataFrame:
    """Predicted (s, q) pairs with the probability that selected them."""
    return decide.decide(best, params).join(best.select("q", "p"), on="q", how="left")


def _min_p(pred: pl.DataFrame, t: float):
    kept = pred.filter(pl.col("p") >= t) if t > 0 else pred
    return kept["s"].to_numpy(), kept["q"].to_numpy()


# ============================================================================ TRAIN
def run_train(data_dir: Path, work_dir: Path, cfg: Config, sample: float | None, force: bool) -> None:
    metrics.self_check()
    t_start = time.time()
    suffix = f"_sample{sample}" if sample else ""
    cache = Cache(work_dir / f"train{suffix}", force)
    run_dir = cache.dir
    model_dir = work_dir / f"model{suffix}"
    report: dict = {"config": cfg.to_dict(), "sample": sample}

    k_norm = Cache.key("norm", sample, cfg.seed)
    with stage("1/9 ingest + normalize"):
        s1, q, owner = load_records(data_dir, "train", cache, k_norm, cfg, sample)
    role = s1_roles(s1.height, cfg.val_frac, cfg.seed)
    is_val = role != OTHER
    s_country = s1["country"].to_numpy()
    q_country = q["country"].to_numpy()

    with stage("2/9 learn vocabulary"):
        k_maps = Cache.key("maps", k_norm, cfg.val_frac, cfg.seed, cfg.abbrev_min_support,
                           cfg.abbrev_min_confidence, cfg.abbrev_max_pairs)
        if cache.hit("maps", k_maps, "token_maps.json"):
            maps = json.loads((run_dir / "token_maps.json").read_text())
        else:
            maps = mine_maps(s1, q, owner, ~is_val, cfg)
            (run_dir / "token_maps.json").write_text(json.dumps(maps))
            cache.done("maps", k_maps)
        k_derive = derive_key(k_norm, maps, cfg)
        s1, q, links = derive_records(s1, q, maps, cfg, cache, k_derive)

    emb_path, embed_model_dir, k_emb, embed_identity = None, None, None, None
    if cfg.use_embeddings:
        with stage("3/9 multilingual name embeddings"):
            model_path, embed_identity = cfg.embed_model, cfg.embed_model
            if cfg.finetune_embed:
                k_ft = Cache.key("finetune", k_norm, cfg.val_frac, cfg.seed, cfg.embed_model, cfg.embed_max_len,
                                 cfg.finetune_pairs, cfg.finetune_batch, cfg.finetune_lr, cfg.finetune_temperature)
                embed_model_dir = finetune_embedder(s1, q, owner, ~is_val, cfg, cache, k_ft)
                if (embed_model_dir / "config.json").exists():
                    model_path, embed_identity = str(embed_model_dir), k_ft
            k_emb = emb_key(k_norm, embed_identity, cfg)
            emb_path = embed_names(cache, k_emb, model_path, cfg)

    if cfg.sim_drop_s1_frac > 0:
        with stage("3b/9 simulate test-like distractor density"):
            s1, q, owner, role = simulate_density(s1, q, owner, role, cfg)
            is_val = role != OTHER
            s_country = s1["country"].to_numpy()

    with stage("4/9 blocking"):
        extra = (cfg.sim_drop_s1_frac, cfg.seed) if cfg.sim_drop_s1_frac > 0 else ()
        k_pairs = pairs_key(k_derive, k_emb, cfg, *extra)
        pairs_path = run_blocking(s1, q, cfg, cache, k_pairs, emb_path)
        pairs = pl.read_parquet(pairs_path)
        rec_val = blocking.recall_report(pairs, owner, s1, q, is_val)
        log(f"   blocking: {pairs.height:,} pairs, {rec_val['cand_per_query']:.1f}/query, "
            f"validation pair recall {rec_val['pair_recall']:.4f}")
        report["blocking"] = {"val": rec_val}
        anchors = features.build_anchors(pairs_path, cfg)
        log(f"   {anchors.height:,} strong sibling links for {anchors['s'].n_unique():,} S1 records")

    # validation queries: any query whose true owner or any candidate is a validation S1
    pq = pairs["q"].to_numpy()
    ps = pairs["s"].to_numpy()
    q_val = np.zeros(q.height, bool)
    q_val[pq[is_val[ps]]] = True
    q_val[(owner >= 0) & is_val[np.maximum(owner, 0)]] = True
    rng = np.random.default_rng(cfg.seed)
    has_cand = np.zeros(q.height, bool)
    has_cand[pq] = True
    cand_q = np.flatnonzero(~q_val & has_cand)
    per_q = pairs.height / max(1, int(has_cand.sum()))
    n_train_q = min(len(cand_q), int(cfg.max_train_pairs / max(per_q, 1)))
    q_train = np.zeros(q.height, bool)
    q_train[rng.choice(cand_q, n_train_q, replace=False)] = True
    train_pairs = pairs.filter(pl.Series(q_train[pq]))
    val_path = run_dir / "val_pairs.parquet"
    pairs.filter(pl.Series(q_val[pq])).write_parquet(val_path)
    upper = blocking_upper_bound(pq, ps, owner, np.flatnonzero(role == VAL_B))
    del pairs, pq, ps  # the full pair table is not needed any more (keeps peak memory down)
    log(f"   training queries {q_train.sum():,} ({train_pairs.height:,} pairs); "
        f"validation queries {q_val.sum():,}")

    pool = soft_pool(cfg, links)
    ce_dir = None
    try:
        with stage("5/9 features (training rows)"):
            k_x = Cache.key("xtrain", k_pairs, cfg.val_frac, cfg.max_train_pairs, cfg.seed, FEATURE_NAMES,
                            cfg.sib_min_comb, cfg.sib_max_per_s)
            if cache.hit("xtrain", k_x, "X_train.npy"):
                X = np.load(run_dir / "X_train.npy", mmap_mode="r")
            else:
                X = featurize(train_pairs, s1, q, pool, cfg, anchors, run_dir / "X_train.npy", "train features")
                cache.done("xtrain", k_x)
            tq = train_pairs["q"].to_numpy()
            ts = train_pairs["s"].to_numpy()
            del train_pairs
            y = (owner[tq] == ts).astype(np.float32)
            groups = np.where(owner[tq] >= 0, owner[tq], s1.height + tq)
            log(f"   {len(y):,} rows, {int(y.sum()):,} positives, {X.shape[1]} features")

        with stage("6/9 train matcher"):
            D = build_dataset(X, y, FEATURE_NAMES, cfg)
            main, oof_main = fit_matcher(D, X, y, tq, groups, np.arange(len(y)), FEATURE_NAMES, cfg)
            matchers = {"main": main}
            if cfg.use_proxy:
                qc = q_country[tq]
                for src in proxy_countries(s_country):
                    rows = np.flatnonzero(qc == src)
                    if len(rows):
                        log(f"   proxy model trained on {src} only")
                        matchers[f"proxy_{src}"], _ = fit_matcher(D, X, y, tq, groups, rows, FEATURE_NAMES, cfg)
            del D
            if cfg.use_cross_encoder:
                ce_dir = train_cross_encoder(tq, ts, y, oof_main, s1, q, cfg, cache, k_x)
            del X

        with stage("7/9 score validation queries"):
            vqs = pl.read_parquet(val_path, columns=["q", "s"])
            vq, vs = vqs["q"].to_numpy(), vqs["s"].to_numpy()
            del vqs
            touches_A = np.zeros(q.height, bool)
            touches_A[vq[role[vs] == VAL_A]] = True
            rc = q_country[vq]
            masks = {}  # each proxy model scores only its home tuning half and the other country
            for src, tgt in proxy_directions(s_country):
                if f"proxy_{src}" in matchers:
                    masks[f"proxy_{src}"] = (rc == tgt) | ((rc == src) & touches_A[vq])
            sc = stream_scores(val_path, s1, q, pool, cfg, matchers, "validation", anchors, masks)
    finally:
        pool.shutdown()

    with stage("8/9 calibrate, tune decision + cross-encoder + consistency + unseen-country rule, errors"):
        comb = pl.read_parquet(val_path, columns=["comb"])["comb"].to_numpy()
        vy = (owner[vq] == vs).astype(np.float32)
        sA = np.flatnonzero(role == VAL_A)
        sB = np.flatnonzero(role == VAL_B)
        in_A = role[vs] == VAL_A

        results = {}
        # baseline: blocking score only, one global threshold
        best = best_per_query(vq, vs, comb)
        params, _ = tune(best, owner, sA, cfg, methods=("global",))
        results["baseline_blocking_score"] = metrics.report(*_pred(best, params), owner, sB, s_country) | {"params": params}

        m = matchers["main"]
        best1 = best_per_query(vq, vs, sc["main_p1"])
        params, _ = tune(best1, owner, sA, cfg, methods=("global",))
        results["stage1_global_threshold"] = metrics.report(*_pred(best1, params), owner, sB, s_country) | {"params": params}

        m.calib_x, m.calib_y = fit_calibration(sc["main"][in_A], vy[in_A])
        p = m.calibrate(sc["main"])
        best2 = best_per_query(vq, vs, p)
        pg, sg = tune(best2, owner, sA, cfg, methods=("global",))
        pe, se = tune(best2, owner, sA, cfg, methods=("expected_f",))
        results["stage2_global_threshold"] = metrics.report(*_pred(best2, pg), owner, sB, s_country) | {"params": pg}
        results["stage2_expected_f05"] = metrics.report(*_pred(best2, pe), owner, sB, s_country) | {"params": pe}
        final, final_score = (pe, se) if se >= sg else (pg, sg)

        ce_cfg = None
        if ce_dir is not None:  # tuned together with "off" (weight 1.0)
            lo, hi = cfg.ce_band
            band = (p >= lo) & (p <= hi)
            ce = crossenc.score(crossenc.pair_texts(vq[band], vs[band], q, s1), ce_dir, run_dir, cfg, "val")
            trials = [(evaluate(best_per_query(vq, vs, crossenc.blend(p, band, ce, w)), final, owner, sA), w)
                      for w in cfg.ce_weight_grid]
            score_w, w = max(trials, key=lambda t: (t[0], t[1]))
            log(f"   cross-encoder weight chosen on VAL_A: {w} ({score_w:.4f} vs {final_score:.4f} off)")
            if w < 1.0 and score_w > final_score:
                p = crossenc.blend(p, band, ce, w)
                best2 = best_per_query(vq, vs, p)
                final, final_score = tune(best2, owner, sA, cfg)
                ce_cfg = {"weight": w, "band": [lo, hi]}
            results["stage2_plus_cross_encoder"] = (metrics.report(*_pred(best2, final), owner, sB, s_country)
                                                    | {"params": final, "cross_encoder": ce_cfg})

        cons = dict(NO_CONSISTENCY)
        best_final = best2
        if cfg.use_consistency:  # tuned together with "off" (lambda = 0)
            tri = consistency.build_triples(vq, vs, p, q, cfg.consistency_anchor_min, cfg.consistency_target_min)
            log(f"   consistency: {tri.height:,} (record, near-duplicate anchor) triples")
            trials = []
            for lam in cfg.consistency_lambda_grid:
                for sm in cfg.consistency_sim_grid:
                    b = best_per_query(vq, vs, consistency.apply(p, tri, lam, sm))
                    trials.append((evaluate(b, final, owner, sA), lam, sm))
            score_c, lam, sm = max(trials, key=lambda t: t[0])
            log(f"   consistency chosen on VAL_A: lambda={lam}, sim_min={sm} ({score_c:.4f} vs {final_score:.4f} off)")
            if lam > 0 and score_c > final_score:
                cons = {"lambda": lam, "sim_min": sm}
                best_final = best_per_query(vq, vs, consistency.apply(p, tri, lam, sm))
                final, final_score = tune(best_final, owner, sA, cfg)
            results["stage2_plus_consistency"] = (metrics.report(*_pred(best_final, final), owner, sB, s_country)
                                                     | {"params": final, "consistency": cons})
        log(f"   final decision layer (chosen on VAL_A): {final}, consistency {cons}, cross-encoder {ce_cfg}")

        fs, fq = _pred(best_final, final)
        export_validation(run_dir, s1, q, owner, sB, fs, fq)
        report["error_analysis_valB"] = analysis.error_analysis(
            fs, fq, owner, sB, vq, vs, q, s1, run_dir / "error_examples.tsv")
        report["blocking"]["upper_bound_macro_f05_valB"] = upper
        report["validation_valB"] = results
        report["feature_importance"] = importance_report(m.importance)

        rule = None
        if cfg.use_proxy:
            report["proxy"], rule, report["unseen_rule_search"] = proxy_report(
                matchers, sc, masks, vq, vs, vy, owner, role, s_country, cfg)
        report["unseen_country_rule"] = rule
        report["final"] = {"decision": final, "consistency": cons, "cross_encoder": ce_cfg,
                           "unseen_country_rule": rule}

    with stage("9/9 save model"):
        if model_dir.exists():
            shutil.rmtree(model_dir)
        m.save(model_dir)
        (model_dir / "token_maps.json").write_text(json.dumps(maps))
        save_json(cfg.to_dict(), model_dir / "config.json")
        save_json({"params": final, "consistency": cons, "unseen_country_rule": rule, "cross_encoder": ce_cfg,
                   "features": FEATURE_NAMES}, model_dir / "decision.json")
        if embed_model_dir is not None and (embed_model_dir / "config.json").exists():
            shutil.copytree(embed_model_dir, model_dir / "embed_model")
        if embed_identity is not None:
            (model_dir / "embed_key.txt").write_text(embed_identity)
        if ce_cfg is not None:
            shutil.copytree(ce_dir, model_dir / "ce_model")
        for name in ("error_examples.tsv", "val_matching_results.tsv", "val_ground_truth.tsv"):
            if (run_dir / name).exists():
                shutil.copy(run_dir / name, model_dir / name)
        report["train_minutes"] = round((time.time() - t_start) / 60, 1)
        save_json(report, model_dir / "train_report.json")
        (model_dir / "model_id.txt").write_text(f"{time.time():.0f}")
    cache.finish()

    print_report(report)


def train_cross_encoder(tq, ts, y, oof, s1, q, cfg: Config, cache: Cache, k_x: str) -> Path:
    """Fine-tune the cross-encoder on training pairs stage 1 found hard (cached)."""
    out = cache.dir / "ce_model"
    k_ce = Cache.key("ce", k_x, cfg.ce_base_model, cfg.ce_train_pairs, cfg.ce_train_band, cfg.ce_max_len,
                     cfg.ce_batch, cfg.ce_lr)
    if cache.hit("ce", k_ce, "ce_model/DONE"):
        return out
    shutil.rmtree(out, ignore_errors=True)
    lo, hi = cfg.ce_train_band
    rows = np.flatnonzero((oof >= lo) & (oof <= hi))
    rng = np.random.default_rng(cfg.seed)
    if len(rows) > cfg.ce_train_pairs:
        rows = rng.choice(rows, cfg.ce_train_pairs, replace=False)
    texts = crossenc.pair_texts(tq[rows], ts[rows], q, s1).with_columns(y=pl.Series(y[rows]))
    crossenc.train(texts, out, cache.dir, cfg)
    (out / "DONE").write_text("ok")
    cache.done("ce", k_ce)
    return out


def importance_report(imp: np.ndarray) -> dict:
    order = np.argsort(-imp)
    top = [(FEATURE_NAMES[i], float(imp[i])) for i in order[:20]]
    groups = {g: float(sum(imp[FEATURE_NAMES.index(f)] for f in fs)) for g, fs in features.GROUPS.items()}
    return {"top": top, "groups": groups}


def derive_key(k_norm: str, maps: dict, cfg: Config) -> str:
    return Cache.key("derive", k_norm, _hash_json(maps), cfg.stop_df_ratio, cfg.place_min_ratio,
                     cfg.place_min_count, cfg.place_link_min_ratio, cfg.place_link_sample, cfg.seed)


def export_validation(run_dir: Path, s1: pl.DataFrame, q: pl.DataFrame, owner: np.ndarray,
                      eval_s: np.ndarray, pred_s: np.ndarray, pred_q: np.ndarray) -> None:
    """Final predictions and ground truth for validation half B, in the official file format,
    so the independent scorer (scripts/score.py) can re-check the score."""
    pos = np.full(s1.height, -1, np.int64)
    pos[eval_s] = np.arange(len(eval_s))
    ids = s1["entity_id"].gather(eval_s)
    q_ids = q["entity_id"].to_numpy()
    keep = pos[pred_s] >= 0
    write_id_lists(run_dir / "val_matching_results.tsv", ids, pos[pred_s[keep]], q_ids[pred_q[keep]],
                   "matched_entity_ids")
    tq = np.flatnonzero(owner >= 0)
    tq = tq[pos[owner[tq]] >= 0]
    write_id_lists(run_dir / "val_ground_truth.tsv", ids, pos[owner[tq]], q_ids[tq], "matched_entity_ids")


def blocking_upper_bound(q: np.ndarray, s: np.ndarray, owner, eval_s) -> float:
    """Macro F0.5 of a perfect matcher restricted to the blocking candidates."""
    hit = owner[q] == s
    return metrics.macro_f05(s[hit], q[hit], owner, eval_s)


def proxy_countries(s_country: np.ndarray) -> list[str]:
    """The two most common countries of the training S1 records (the proxy needs two)."""
    vals, counts = np.unique(s_country, return_counts=True)
    return [str(v) for v in vals[np.argsort(-counts)][:2]] if len(vals) >= 2 else []


def proxy_directions(s_country: np.ndarray) -> list[tuple[str, str]]:
    c = proxy_countries(s_country)
    return [(c[0], c[1]), (c[1], c[0])] if len(c) == 2 else []


def proxy_report(matchers, sc, masks, vq, vs, vy, owner, role, s_country, cfg: Config):
    """Leave-one-country-out: train + tune on one country, report on the other.

    Unseen-country rule: search ONE extra minimum match probability that maximizes
    the worse of the two directions' gains on the target countries' half A, then keep it only if
    it improves both directions on half B.

    Approximations: the mined token maps are per country; the fine-tuned encoder
    saw pairs from both countries; the consistency step and cross-encoder are not applied.
    """
    out, runs = {}, {}
    for src, tgt in proxy_directions(s_country):
        name = f"proxy_{src}"
        if name not in matchers:
            continue
        idx = np.flatnonzero(masks[name])
        mq, ms, my, mp = vq[idx], vs[idx], vy[idx], sc[name][idx]
        m = matchers[name]
        src_A = (role[ms] == VAL_A) & (s_country[ms] == src)
        m.calib_x, m.calib_y = fit_calibration(mp[src_A], my[src_A])
        best = best_per_query(mq, ms, m.calibrate(mp))
        tune_s = np.flatnonzero((role == VAL_A) & (s_country == src))
        params, _ = tune(best, owner, tune_s, cfg)
        pred = _pred_p(best, params)
        tgt_all = np.flatnonzero((role != OTHER) & (s_country == tgt))
        tgt_A = np.flatnonzero((role == VAL_A) & (s_country == tgt))
        tgt_B = np.flatnonzero((role == VAL_B) & (s_country == tgt))
        out[f"{src}->{tgt}"] = metrics.report(*_min_p(pred, 0.0), owner, tgt_all, s_country) | {"params": params}
        base_A = metrics.macro_f05(*_min_p(pred, 0.0), owner, tgt_A)
        gains = {t: metrics.macro_f05(*_min_p(pred, t), owner, tgt_A) - base_A for t in cfg.unseen_min_p_grid}
        runs[f"{src}->{tgt}"] = {"pred": pred, "tgt_B": tgt_B, "gains_A": gains}

    if len(runs) < 2:
        return out, None, None
    grid = list(cfg.unseen_min_p_grid)
    worst = {t: min(r["gains_A"][t] for r in runs.values()) for t in grid}
    t_star = max(grid, key=lambda t: (worst[t], -t))
    search = {"min_p": t_star, "worst_gain_A": worst[t_star],
              "own_best": {d: max(grid, key=lambda t: r["gains_A"][t]) for d, r in runs.items()}}
    helps = True
    for d, r in runs.items():
        base = metrics.macro_f05(*_min_p(r["pred"], 0.0), owner, r["tgt_B"])
        ruled = metrics.macro_f05(*_min_p(r["pred"], t_star), owner, r["tgt_B"])
        out[d]["unseen_rule_check"] = {"min_p": t_star, "valB_without_rule": base, "valB_with_rule": ruled}
        log(f"   unseen-country rule {d}: common min_p {t_star} -> half B {base:.4f} -> {ruled:.4f}")
        helps &= ruled > base
    if helps and t_star > 0:
        rule = {"min_p": float(t_star), "train_countries": sorted(np.unique(s_country).tolist())}
        log(f"   unseen-country rule KEPT: min_p = {t_star} for countries not in {rule['train_countries']}")
    else:
        rule = None
        log("   unseen-country rule not kept (no common value improved both proxy directions)")
    return out, rule, search


def print_report(report: dict) -> None:
    b = report["blocking"]
    v = b["val"]
    print("\n" + "=" * 78)
    print("TRAINING REPORT (validation half B, never used for tuning)")
    print("=" * 78)
    print(f"blocking pair recall (validation): {v['pair_recall']:.4f}   "
          f"candidates/query: {v['cand_per_query']:.1f}   candidates/S1: {v['cand_per_s1']:.1f}   "
          f"reduction ratio: {v['reduction_ratio']:.6f}")
    print("   per pass: " + "  ".join(f"{p}={v[f'recall_pass_{p}']:.4f}" for p in blocking.PASSES
                                     if v[f"recall_pass_{p}"]))
    print("   slices:   " + "  ".join(f"{k} ({s['share']:.1%} of pairs)={s['recall']:.4f}"
                                     for k, s in v["slices"].items() if s["recall"] is not None))
    print(f"   perfect-matcher ceiling (macro F0.5): {b['upper_bound_macro_f05_valB']:.4f}")
    print(f"{'method':32s} {'macro F0.5':>10s} {'singleton':>10s} {'precision':>10s} {'recall':>8s}  by country")
    for name, r in report["validation_valB"].items():
        ctry = "  ".join(f"{k[4:]}={val:.4f}" for k, val in r.items() if k.startswith("f05_"))
        print(f"{name:32s} {r['macro_f05']:10.4f} {r['singleton_f05'] or 0:10.4f} "
              f"{r['pair_precision']:10.4f} {r['pair_recall']:8.4f}  {ctry}")
    last = list(report["validation_valB"].values())[-1]["by_true_matches"]
    print("   final method by true matches per S1: " + "  ".join(
        f"{k} ({x['share']:.1%})={x['f05']:.4f}" for k, x in last.items() if x["f05"] is not None))
    proxy = report.get("proxy") or {}
    for k, r in proxy.items():
        print(f"proxy {k:26s} {r['macro_f05']:10.4f} {r['singleton_f05'] or 0:10.4f} "
              f"{r['pair_precision']:10.4f} {r['pair_recall']:8.4f}")
        c = r.get("unseen_rule_check")
        if c:
            print(f"   rule check on target half B: min_p {c['min_p']}: "
                  f"{c['valB_without_rule']:.4f} -> {c['valB_with_rule']:.4f}")
    if proxy:
        print(f"worst proxy direction: {min(r['macro_f05'] for r in proxy.values()):.4f}")
        print(f"unseen-rule search: {report.get('unseen_rule_search')}")
    print(f"unseen-country rule: {report.get('unseen_country_rule')}")
    fi = report.get("feature_importance")
    if fi:
        print("feature importance (stage-1 gain share) by group: " + "  ".join(
            f"{g}={x:.3f}" for g, x in sorted(fi["groups"].items(), key=lambda kv: -kv[1])))
        print("   top features: " + ", ".join(f"{n} {x:.3f}" for n, x in fi["top"][:12]))
    ea = report["error_analysis_valB"]
    print(f"error analysis (final method): {ea['n_false_pos']:,} false positives, {ea['n_false_neg']:,} false negatives")
    for name, sl in sorted(ea["slices"].items(), key=lambda kv: -kv[1]["headroom_macro_f05"]):
        print(f"   {name:34s} FP={sl['false_pos']:7,}  FN={sl['false_neg']:7,}  "
              f"headroom if fixed: +{sl['headroom_macro_f05']:.4f}")
    print(f"final: {report['final']}")
    print(f"total training time: {report['train_minutes']} min")
    print("=" * 78)


# ============================================================================ TEST
def run_test(data_dir: Path, work_dir: Path, output_dir: Path, sample: float | None, force: bool,
             n_jobs: int | None) -> None:
    suffix = f"_sample{sample}" if sample else ""
    model_dir = work_dir / f"model{suffix}"
    if not (model_dir / "model_id.txt").exists():
        raise SystemExit(f"No trained model in {model_dir}. Run train.py{' --sample ' + str(sample) if sample else ''} first.")
    saved = json.loads((model_dir / "config.json").read_text())
    cfg = Config(**{f.name: saved[f.name] for f in fields(Config) if f.name in saved})
    if n_jobs:
        cfg.n_jobs = n_jobs
    decision = json.loads((model_dir / "decision.json").read_text())
    params, cons = decision["params"], decision.get("consistency", NO_CONSISTENCY)
    rule, ce_cfg = decision.get("unseen_country_rule"), decision.get("cross_encoder")
    maps = json.loads((model_dir / "token_maps.json").read_text())
    model_id = (model_dir / "model_id.txt").read_text()
    matcher = Matcher.load(model_dir, cfg.n_folds)
    cache = Cache(work_dir / f"test{suffix}", force)
    out_dir = Path(output_dir) / "sample" if sample else Path(output_dir)

    k_norm = Cache.key("norm", sample, cfg.seed)
    with stage("1/6 ingest + normalize"):
        s1, q, _ = load_records(data_dir, "test", cache, k_norm, cfg, sample)
        log(f"   countries (S1): {s1['country'].value_counts().sort('count', descending=True).rows()}")
    with stage("2/6 apply vocabulary + test-file statistics"):
        k_derive = derive_key(k_norm, maps, cfg)
        s1, q, links = derive_records(s1, q, maps, cfg, cache, k_derive)
    emb_path, k_emb = None, None
    if cfg.use_embeddings:
        with stage("3/6 multilingual name embeddings"):
            tuned = model_dir / "embed_model"
            model_path = str(tuned) if (tuned / "config.json").exists() else cfg.embed_model
            key_file = model_dir / "embed_key.txt"
            identity = key_file.read_text() if key_file.exists() else model_path
            k_emb = emb_key(k_norm, identity, cfg)
            emb_path = embed_names(cache, k_emb, model_path, cfg)
    with stage("4/6 blocking"):
        k_pairs = pairs_key(k_derive, k_emb, cfg)
        pairs_path = run_blocking(s1, q, cfg, cache, k_pairs, emb_path)
        qs = pl.read_parquet(pairs_path, columns=["q", "s"])
        pq, ps = qs["q"].to_numpy(), qs["s"].to_numpy()
        del qs
        log(f"   {len(pq):,} candidate pairs, {len(np.unique(pq)):,} queries with candidates")
        cand_path = out_dir / "candidate_pairs.tsv"
        write_id_lists(cand_path, s1["entity_id"], ps, q["entity_id"].to_numpy()[pq], "candidate_entity_ids")
    with stage("5/6 score candidates"):
        k_sc = Cache.key("scores", k_pairs, model_id, FEATURE_NAMES)
        if cache.hit("scores", k_sc, "scores.npy"):
            raw = np.load(cache.dir / "scores.npy")
            log("   reusing cached test probabilities")
        else:
            anchors = features.build_anchors(pairs_path, cfg)
            log(f"   {anchors.height:,} strong sibling links")
            pool = soft_pool(cfg, links)
            try:
                raw = stream_scores(pairs_path, s1, q, pool, cfg, {"main": matcher}, "test", anchors)["main"]
            finally:
                pool.shutdown()
            np.save(cache.dir / "scores.npy", raw)
            cache.done("scores", k_sc)
        p = matcher.calibrate(raw)
        if ce_cfg:
            lo, hi = ce_cfg["band"]
            band = (p >= lo) & (p <= hi)
            ce = crossenc.score(crossenc.pair_texts(pq[band], ps[band], q, s1), model_dir / "ce_model",
                                cache.dir, cfg, "test")
            p = crossenc.blend(p, band, ce, ce_cfg["weight"])
            log(f"   cross-encoder applied to {int(band.sum()):,} uncertain pairs (weight {ce_cfg['weight']})")
        if cons["lambda"] > 0:
            tri = consistency.build_triples(pq, ps, p, q, cfg.consistency_anchor_min, cfg.consistency_target_min)
            p = consistency.apply(p, tri, cons["lambda"], cons["sim_min"])
            log(f"   consistency applied (lambda={cons['lambda']}, sim_min={cons['sim_min']}, {tri.height:,} triples)")
    with stage("6/6 decide + write outputs"):
        best = best_per_query(pq, ps, p)
        pred = _pred_p(best, params)
        if rule:  # stricter minimum probability for countries absent from training
            country = s1["country"].to_numpy()[pred["s"].to_numpy()]
            unseen = ~np.isin(country, rule["train_countries"])
            keep = ~unseen | (pred["p"].to_numpy() >= rule["min_p"])
            log(f"   unseen-country rule (min_p={rule['min_p']:.3f}): removed {int((~keep).sum()):,} of "
                f"{int(unseen.sum()):,} predicted pairs in countries {sorted(set(country[unseen].tolist()))}")
            pred = pred.filter(pl.Series(keep))
        pred = pred.select("s", "q")
        cand = pl.DataFrame({"q": pq, "s": ps})
        n_outside = pred.join(cand, on=["q", "s"], how="anti").height
        if n_outside:
            raise RuntimeError(f"{n_outside} predicted matches are not in the candidate set")
        log("   check: every predicted match is in the candidate set")
        match_path = out_dir / "matching_results.tsv"
        write_id_lists(match_path, s1["entity_id"], pred["s"].to_numpy(),
                       q["entity_id"].to_numpy()[pred["q"].to_numpy()], "matched_entity_ids")
        summarize(s1, pred)
    cache.finish()

    if sample:
        log(f"sample run: outputs in {out_dir}")
        return
    log(f"outputs: {match_path} and {cand_path}")


def summarize(s1: pl.DataFrame, pred: pl.DataFrame) -> None:
    n = np.bincount(pred["s"].to_numpy(), minlength=s1.height)
    df = pl.DataFrame({"country": s1["country"], "n": n})
    summary = df.group_by("country").agg(
        s1=pl.len(), empty_share=(pl.col("n") == 0).mean(), mean_matches=pl.col("n").mean()
    ).sort("s1", descending=True)
    log(f"   predictions per country:\n{summary}")
