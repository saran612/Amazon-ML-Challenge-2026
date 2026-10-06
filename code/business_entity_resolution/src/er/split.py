"""Validation split and smoke-test sampling."""
from __future__ import annotations

import numpy as np
import polars as pl

# S1 roles
OTHER, VAL_A, VAL_B = 0, 1, 2


def s1_roles(n_s1: int, val_frac: float, seed: int) -> np.ndarray:
    """VAL_A / VAL_B: two halves of the held-out S1 set. VAL_A tunes, VAL_B reports."""
    rng = np.random.default_rng(seed)
    u = rng.random(n_s1)
    role = np.full(n_s1, OTHER, np.int8)
    role[u < val_frac / 2] = VAL_A
    role[(u >= val_frac / 2) & (u < val_frac)] = VAL_B
    return role


def sample_universe(s1: pl.DataFrame, q: pl.DataFrame, owner: np.ndarray | None,
                    frac: float, seed: int):
    """Consistent subset for smoke tests.

    Labeled: a share of S1 entities with all their matched records, plus the same share of
    unmatched distractors. Unlabeled: the same share of every file.
    Row indices are renumbered; the new owner array is returned for labeled data.
    """
    rng = np.random.default_rng(seed)
    keep_s = rng.random(s1.height) < frac
    if owner is None:
        keep_q = rng.random(q.height) < frac
    else:
        keep_q = np.where(owner >= 0, keep_s[np.maximum(owner, 0)], rng.random(q.height) < frac)
    new_sidx = np.cumsum(keep_s) - 1
    s1 = s1.filter(pl.Series(keep_s)).with_columns(idx=pl.int_range(pl.len(), dtype=pl.Int64))
    q = q.filter(pl.Series(keep_q)).with_columns(idx=pl.int_range(pl.len(), dtype=pl.Int64))
    if owner is not None:
        o = owner[keep_q]
        owner = np.where(o >= 0, new_sidx[np.maximum(o, 0)], -1)
    return s1, q, owner
