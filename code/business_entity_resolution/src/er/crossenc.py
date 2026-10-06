"""Cross-encoder re-scoring of uncertain pairs (on by default; --no-cross-encoder to skip).

A small transformer reads both records together ("name | address" of each) and outputs a match
probability. It is fine-tuned on training pairs the LightGBM stage 1 found hard (out-of-fold
score inside ce_train_band), then applied only to validation / test pairs whose calibrated score
falls inside ce_band. Its score is blended with LightGBM's in logit space with a weight tuned on
validation half A together with "off" (weight 1.0 = LightGBM only).

torch / transformers run only inside spawned worker processes (OpenMP isolation, as in embed.py).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

from .embed import _device, run_isolated
from .log import log


def pair_texts(qi: np.ndarray, si: np.ndarray, q: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    def txt(frame, idx):
        f = frame.select("name_raw", "addr_norm")[idx]
        return (f["name_raw"] + " | " + f["addr_norm"]).to_list()
    return pl.DataFrame({"a": txt(q, qi), "b": txt(s1, si)})


def train_worker(pairs_path: str, base_model: str, out_dir: str, batch: int, lr: float, max_len: int,
                 seed: int) -> str:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed)
    device = _device()
    df = pl.read_parquet(pairs_path).sample(fraction=1.0, shuffle=True, seed=seed)
    a, b, y = df["a"].to_list(), df["b"].to_list(), df["y"].to_numpy().astype(np.float32)
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=1, dtype=torch.float32)
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(a) // batch
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / steps))
    loss_fn = torch.nn.BCEWithLogitsLoss()
    t0, run = time.time(), 0.0
    for s in range(steps):
        sl = slice(s * batch, (s + 1) * batch)
        enc = tok(a[sl], b[sl], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
        logits = model(**enc).logits.squeeze(-1)
        loss = loss_fn(logits, torch.from_numpy(y[sl]).to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
        if s % 200 == 0:
            rate = (s + 1) / (time.time() - t0)
            log(f"   cross-encoder step {s:,}/{steps:,} loss {run:.4f} ({rate:.1f} steps/s on {device}, "
                f"~{(steps - s) / rate / 60:.0f} min left)")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return device


def score_worker(pairs_path: str, model_dir: str, max_len: int, batch: int, out_path: str) -> str:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = _device()
    df = pl.read_parquet(pairs_path)
    a, b = df["a"].to_list(), df["b"].to_list()
    n = len(a)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float16 if device != "cpu" else torch.float32).to(device).eval()
    out = np.empty(n, np.float32)
    order = np.argsort(np.fromiter((len(x) + len(y) for x, y in zip(a, b)), np.int32, n), kind="stable")
    t0 = time.time()
    with torch.inference_mode():
        for k, st in enumerate(range(0, n, batch)):
            idx = order[st:st + batch]
            enc = tok([a[j] for j in idx], [b[j] for j in idx], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
            out[idx] = torch.sigmoid(model(**enc).logits.squeeze(-1).float()).cpu().numpy()
            if k % 500 == 0 and st:
                rate = st / (time.time() - t0)
                log(f"   cross-encoder scoring {st:,}/{n:,} pairs ({rate:,.0f}/s on {device}, "
                    f"~{(n - st) / rate / 60:.0f} min left)")
    np.save(out_path, out)
    return device


def train(pairs: pl.DataFrame, out_dir: Path, work: Path, cfg) -> None:
    path = work / "ce_train_pairs.parquet"
    pairs.write_parquet(path)
    log(f"   training cross-encoder on {pairs.height:,} pairs ({int(pairs['y'].sum()):,} matches)")
    t = time.time()
    device = run_isolated(train_worker, str(path), cfg.ce_base_model, str(out_dir), cfg.ce_batch, cfg.ce_lr,
                          cfg.ce_max_len, cfg.seed)
    log(f"   cross-encoder trained on {device} in {(time.time() - t) / 60:.1f} min")


def score(texts: pl.DataFrame, model_dir: Path, work: Path, cfg, tag: str) -> np.ndarray:
    if texts.height == 0:
        return np.empty(0, np.float32)
    path, out = work / f"ce_{tag}_pairs.parquet", work / f"ce_{tag}_scores.npy"
    texts.write_parquet(path)
    log(f"   cross-encoder scoring {texts.height:,} uncertain {tag} pairs")
    t = time.time()
    device = run_isolated(score_worker, str(path), str(model_dir), cfg.ce_max_len, 256, str(out))
    log(f"   cross-encoder scored on {device} in {(time.time() - t) / 60:.1f} min")
    return np.load(out)


def blend(p: np.ndarray, band: np.ndarray, ce: np.ndarray, w: float) -> np.ndarray:
    """Logit-space blend on the band rows; w = 1.0 keeps LightGBM's score."""
    if w >= 1.0 or not band.any():
        return p
    eps = 1e-6
    lp = np.log(np.clip(p[band], eps, 1 - eps) / np.clip(1 - p[band], eps, 1))
    lc = np.log(np.clip(ce, eps, 1 - eps) / np.clip(1 - ce, eps, 1))
    out = p.copy()
    out[band] = (1 / (1 + np.exp(-(w * lp + (1 - w) * lc)))).astype(np.float32)
    return out
