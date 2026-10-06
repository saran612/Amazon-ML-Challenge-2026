"""Macro F0.5: F0.5 per Source 1 entity, averaged over all of them (singletons included)."""
from __future__ import annotations

import numpy as np


def per_entity_f05(pred_s: np.ndarray, pred_q: np.ndarray, owner: np.ndarray,
                   eval_s: np.ndarray) -> np.ndarray:
    """F0.5 for every S1 in eval_s, given predicted pairs (pred_s[i], pred_q[i]).

    With tp, n_pred, n_true per S1:  F0.5 = 1.25 tp / (0.25 n_true + n_pred);
    an S1 with no true matches and no predictions scores 1.0.
    """
    n_s1 = int(max(eval_s.max(initial=-1), pred_s.max(initial=-1), owner.max(initial=-1))) + 1
    true_s = owner[owner >= 0]
    n_true = np.bincount(true_s, minlength=n_s1)
    n_pred = np.bincount(pred_s, minlength=n_s1)
    tp = np.bincount(pred_s[owner[pred_q] == pred_s], minlength=n_s1)
    nt, npd, t = n_true[eval_s], n_pred[eval_s], tp[eval_s]
    denom = 0.25 * nt + npd
    f = np.where(denom > 0, 1.25 * t / np.maximum(denom, 1e-12), 1.0)
    return f


def macro_f05(pred_s, pred_q, owner, eval_s) -> float:
    return float(per_entity_f05(pred_s, pred_q, owner, eval_s).mean()) if len(eval_s) else float("nan")


def report(pred_s, pred_q, owner, eval_s, s_country: np.ndarray) -> dict:
    f = per_entity_f05(pred_s, pred_q, owner, eval_s)
    n_true = np.bincount(owner[owner >= 0], minlength=len(s_country))[eval_s]
    in_eval = np.zeros(len(s_country), bool)
    in_eval[eval_s] = True
    m = in_eval[pred_s]
    tp = int((owner[pred_q[m]] == pred_s[m]).sum())
    out = {
        "macro_f05": float(f.mean()),
        "singleton_f05": float(f[n_true == 0].mean()) if (n_true == 0).any() else None,
        "nonsingleton_f05": float(f[n_true > 0].mean()) if (n_true > 0).any() else None,
        "pair_precision": tp / max(1, int(m.sum())),
        "pair_recall": tp / max(1, int(n_true.sum())),
        "n_eval_s1": int(len(eval_s)),
    }
    for c in np.unique(s_country[eval_s]):
        sel = s_country[eval_s] == c
        out[f"f05_{c}"] = float(f[sel].mean())
    buckets = {"0": n_true == 0, "1-2": (n_true >= 1) & (n_true <= 2),
               "3-5": (n_true >= 3) & (n_true <= 5), "6+": n_true >= 6}
    out["by_true_matches"] = {k: {"share": float(m.mean()), "f05": float(f[m].mean()) if m.any() else None}
                              for k, m in buckets.items()}
    return out


def self_check() -> None:
    """Worked example: predict [a, b, c], truth [a, c] -> 0.714."""
    owner = np.array([0, 1, 0])          # q0 and q2 belong to S1 #0; q1 belongs to S1 #1
    pred_s = np.array([0, 0, 0])
    pred_q = np.array([0, 1, 2])
    f = per_entity_f05(pred_s, pred_q, owner, np.array([0]))[0]
    assert abs(f - 0.7143) < 1e-3, f
    # singleton handling: S1 #2 has no matches; empty prediction -> 1.0, any prediction -> 0.0
    owner2 = np.array([0, -1])
    assert per_entity_f05(np.array([], int), np.array([], int), owner2, np.array([2]))[0] == 1.0
    assert per_entity_f05(np.array([2]), np.array([1]), owner2, np.array([2]))[0] == 0.0
