"""Two-stage LightGBM matcher with isotonic calibration.

Stage 1 scores each pair from pair features. Stage 2 adds query-side context computed from
stage-1 scores (rank among the query's candidates, margin to the best other candidate), which
tells the model when a query has two plausible owners. Stage-1 scores used to train stage 2
are out-of-fold, so stage 2 never sees optimistic in-sample scores.

Memory (30M training rows on a 24 GB Mac): the feature matrix stays memory-mapped on
disk; it is binned once into a LightGBM Dataset that every fold uses through subsets (no copies);
stage 2 sees the stage-1 score, its context and the top-N base features by stage-1 importance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from .config import Config
from .log import log

CTX = ["p1", "p1_rank", "p1_other", "p1_margin", "p1_n_hi", "p1_sum"]
PREDICT_CHUNK = 2_000_000


def _lgb():
    try:
        import lightgbm
    except OSError as e:  # missing OpenMP runtime on macOS
        raise SystemExit("LightGBM could not load its OpenMP runtime. On macOS run: brew install libomp") from e
    return lightgbm


def context(q: np.ndarray, p1: np.ndarray) -> np.ndarray:
    df = pl.DataFrame({"q": q, "p1": p1})
    top1 = pl.col("p1").max().over("q")
    top2 = pl.col("p1").top_k(2).min().over("q")
    n = pl.len().over("q")
    other = pl.when(n == 1).then(0.0).when(pl.col("p1") >= top1).then(top2).otherwise(top1)
    out = df.select(
        pl.col("p1"),
        pl.col("p1").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("p1_rank"),
        other.alias("p1_other"),
        (pl.col("p1") - other).alias("p1_margin"),
        (pl.col("p1") > 0.5).sum().over("q").cast(pl.Float32).alias("p1_n_hi"),
        pl.col("p1").sum().over("q").alias("p1_sum"),
    )
    return out.to_numpy().astype(np.float32)


def _params(cfg: Config) -> dict:
    return dict(cfg.lgb_params, num_threads=cfg.n_jobs + 2, seed=cfg.seed)


def _fit(train_set, valid_set, cfg: Config):
    lgb = _lgb()
    booster = lgb.train(_params(cfg), train_set, num_boost_round=cfg.lgb_rounds, valid_sets=[valid_set],
                        callbacks=[lgb.early_stopping(cfg.lgb_early_stop, verbose=False),
                                   lgb.log_evaluation(200)])
    log(f"   best iteration {booster.best_iteration}, valid logloss {booster.best_score['valid_0']['binary_logloss']:.5f}")
    return booster


def build_dataset(X, y, names, cfg: Config):
    """Bin the (memory-mapped) training matrix once; folds use subsets of it."""
    lgb = _lgb()
    D = lgb.Dataset(X, label=y, feature_name=names, free_raw_data=True,
                    params={"max_bin": cfg.lgb_params["max_bin"], "verbose": -1})
    D.construct()
    return D


def take(X, rows: np.ndarray, cols=None) -> np.ndarray:
    """Rows (and optionally columns) of a memory-mapped matrix, read in chunks."""
    parts = []
    for st in range(0, len(rows), PREDICT_CHUNK):
        r = rows[st:st + PREDICT_CHUNK]
        parts.append(np.asarray(X[r] if cols is None else X[np.ix_(r, cols)], dtype=np.float32))
    return np.concatenate(parts) if parts else np.empty((0, X.shape[1] if cols is None else len(cols)), np.float32)


def fold_of(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    uniq, inv = np.unique(groups, return_inverse=True)
    return np.random.default_rng(seed).integers(0, n_folds, len(uniq))[inv]


def train_stage1(D, X, groups: np.ndarray, rows: np.ndarray, cfg: Config):
    """Out-of-fold stage-1 scores for `rows`, the fold models and their mean gain importance."""
    folds = fold_of(groups[rows], cfg.n_folds, cfg.seed)
    oof = np.zeros(len(rows), np.float32)
    models, imps = [], []
    for f in range(cfg.n_folds):
        tr, va = rows[folds != f], rows[folds == f]
        log(f"   stage 1 fold {f + 1}/{cfg.n_folds}: train {len(tr):,} rows")
        m = _fit(D.subset(tr), D.subset(va), cfg)
        pos = np.flatnonzero(folds == f)
        for st in range(0, len(va), PREDICT_CHUNK):
            oof[pos[st:st + PREDICT_CHUNK]] = m.predict(take(X, va[st:st + PREDICT_CHUNK]),
                                                        num_iteration=m.best_iteration)
        imp = m.feature_importance("gain")
        imps.append(imp / max(imp.sum(), 1e-12))
        models.append(m)
    return oof, models, np.mean(imps, axis=0)


def train_stage2(X2, y, groups, cfg: Config, names):
    """Single stage-2 model; early stopping on a 10% group holdout of the training rows."""
    lgb = _lgb()
    va = fold_of(groups, 10, cfg.seed + 1) == 0
    log(f"   stage 2: train {(~va).sum():,} rows, early-stop on {va.sum():,}")
    dtr = lgb.Dataset(X2[~va], y[~va], feature_name=names, free_raw_data=True)
    dva = lgb.Dataset(X2[va], y[va], reference=dtr, free_raw_data=True)
    return _fit(dtr, dva, cfg)


def fit_matcher(D, X, y, tq, groups, rows, names, cfg: Config):
    """Train stage 1 (out-of-fold) and stage 2 on `rows`; returns (Matcher, stage-1 OOF scores)."""
    oof, stage1, imp = train_stage1(D, X, groups, rows, cfg)
    cols = np.sort(np.argsort(-imp)[:cfg.stage2_top_features])
    X2 = np.hstack([take(X, rows, cols), context(tq[rows], oof)])
    stage2 = train_stage2(X2, y[rows], groups[rows], cfg, [names[i] for i in cols] + CTX)
    return Matcher(stage1, stage2, cols=cols, importance=imp), oof


class Matcher:
    """Loaded models + calibrator, applied chunk by chunk at inference."""

    def __init__(self, stage1, stage2, calib_x=None, calib_y=None, cols=None, importance=None):
        self.stage1, self.stage2 = stage1, stage2
        self.calib_x, self.calib_y = calib_x, calib_y
        self.cols = cols            # base-feature columns fed to stage 2 (None: all, older models)
        self.importance = importance

    def predict_p1(self, X):
        return np.mean([m.predict(X, num_iteration=m.best_iteration) for m in self.stage1], axis=0).astype(np.float32)

    def predict(self, X, q):
        p1 = self.predict_p1(X)
        base = X if self.cols is None else X[:, self.cols]
        X2 = np.hstack([base, context(q, p1)])
        p2 = self.stage2.predict(X2, num_iteration=self.stage2.best_iteration).astype(np.float32)
        return p1, p2

    def calibrate(self, p):
        if self.calib_x is None:
            return p
        return np.interp(p, self.calib_x, self.calib_y).astype(np.float32)

    def save(self, folder: Path):
        folder.mkdir(parents=True, exist_ok=True)
        for i, m in enumerate(self.stage1):
            m.save_model(str(folder / f"stage1_fold{i}.txt"), num_iteration=m.best_iteration)
        self.stage2.save_model(str(folder / "stage2.txt"), num_iteration=self.stage2.best_iteration)
        np.savez(folder / "calibration.npz", x=self.calib_x, y=self.calib_y)
        if self.cols is not None:
            (folder / "stage2_cols.json").write_text(json.dumps([int(c) for c in self.cols]))

    @classmethod
    def load(cls, folder: Path, n_folds: int):
        lgb = _lgb()
        stage1 = [lgb.Booster(model_file=str(folder / f"stage1_fold{i}.txt")) for i in range(n_folds)]
        stage2 = lgb.Booster(model_file=str(folder / "stage2.txt"))
        cal = np.load(folder / "calibration.npz")
        cols_file = folder / "stage2_cols.json"
        cols = np.array(json.loads(cols_file.read_text())) if cols_file.exists() else None
        return cls(stage1, stage2, cal["x"], cal["y"], cols=cols)


def fit_calibration(p, y):
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(p, y)
    return iso.X_thresholds_.astype(np.float64), iso.y_thresholds_.astype(np.float64)


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
