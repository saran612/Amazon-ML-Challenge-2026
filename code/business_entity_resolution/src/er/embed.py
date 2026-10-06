"""Multilingual name embeddings.

A small multilingual encoder (multilingual-e5-small, MIT, 118M parameters) embeds each
record's name in its ORIGINAL script, so "व्हाइट प्राइवेट" lands next to "White Private"
where character n-grams of the transliteration ("hoy ait praibhet") cannot.
Uses:
  * blocking pass D: approximate nearest-neighbour search (FAISS) per country;
  * feature: cosine similarity of the two names.
Optionally the encoder is first fine-tuned on training match pairs (contrastive, in-batch
negatives), using only pairs outside the validation set.

torch / transformers / faiss are imported only inside worker processes: each ships its own
OpenMP runtime, and keeping them out of the main process avoids clashes with LightGBM's.

Test hook: ER_EMBED_BACKEND=hashing replaces the neural encoder with hashed character
3-grams (no torch needed). It exists only to smoke-test the plumbing; real runs never set it.
"""
from __future__ import annotations

import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import polars as pl

from .log import log

PREFIX = "query: "  # e5 convention for symmetric similarity


def _hashing() -> bool:
    return os.environ.get("ER_EMBED_BACKEND") == "hashing"


def run_isolated(fn, *args):
    """Run fn(*args) in a fresh spawned process and return its result."""
    with ProcessPoolExecutor(1, mp_context=get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def _device():
    import torch
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _mean_pool(out, mask):
    m = mask.unsqueeze(-1).to(out.dtype)
    return (out * m).sum(1) / m.sum(1).clamp(min=1)


# ---------------------------------------------------------------------------- encoding
def _hash_encode(texts: list[str], dim: int) -> np.ndarray:
    out = np.zeros((len(texts), dim), np.float32)
    for i, t in enumerate(texts):
        t = f" {t.lower()} "
        for j in range(len(t) - 2):
            h = zlib.crc32(t[j:j + 3].encode())
            out[i, h % dim] += 1.0 if (h >> 31) else -1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


def encode_worker(texts_path: str, out_path: str, model_path: str, max_len: int, batch: int, dim: int) -> str:
    texts = pl.read_parquet(texts_path)["text"].to_list()
    n = len(texts)
    E = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(n, dim))
    if _hashing():
        for st in range(0, n, 100_000):
            E[st:st + 100_000] = _hash_encode(texts[st:st + 100_000], dim).astype(np.float16)
        E.flush()
        return "hashing"

    import torch
    from transformers import AutoModel, AutoTokenizer
    device = _device()
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, dtype=torch.float16 if device != "cpu" else torch.float32)
    model = model.to(device).eval()
    order = np.argsort(np.fromiter((len(t) for t in texts), np.int32, n), kind="stable")
    t0 = time.time()
    with torch.inference_mode():
        for b, st in enumerate(range(0, n, batch)):
            idx = order[st:st + batch]
            enc = tok([PREFIX + texts[j] for j in idx], padding=True, truncation=True,
                      max_length=max_len, return_tensors="pt").to(device)
            v = _mean_pool(model(**enc).last_hidden_state, enc["attention_mask"])
            v = torch.nn.functional.normalize(v.float(), dim=-1)
            E[idx] = v.cpu().numpy().astype(np.float16)
            if b % 200 == 0 and st:
                rate = st / (time.time() - t0)
                log(f"   embedding {st:,}/{n:,} names ({rate:,.0f}/s on {device}, ~{(n - st) / rate / 60:.0f} min left)")
    E.flush()
    return device


def encode_texts(texts_path: Path, out_path: Path, model_path: str, cfg) -> None:
    n = pl.scan_parquet(texts_path).select(pl.len()).collect().item()
    log(f"   embedding {n:,} unique names with {model_path}")
    t = time.time()
    device = run_isolated(encode_worker, str(texts_path), str(out_path), model_path,
                          cfg.embed_max_len, cfg.embed_batch, cfg.embed_dim)
    log(f"   embeddings done on {device} in {(time.time() - t) / 60:.1f} min")


# ---------------------------------------------------------------------------- fine-tuning
def finetune_worker(pairs_path: str, base_model: str, out_dir: str, batch: int, lr: float, temp: float,
                    max_len: int, seed: int) -> str:
    """Contrastive fine-tuning: each (S1 name, matching S2/S3 name) is a positive pair; the other
    names in the batch are negatives. Symmetric InfoNCE loss."""
    if _hashing():
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "HASHING_BACKEND").write_text("test hook: no fine-tuning")
        return "hashing"
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(seed)
    device = _device()
    df = pl.read_parquet(pairs_path).sample(fraction=1.0, shuffle=True, seed=seed)
    a, b = df["a"].to_list(), df["b"].to_list()
    tok = AutoTokenizer.from_pretrained(base_model)
    model = AutoModel.from_pretrained(base_model, dtype=torch.float32).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(a) // batch
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (steps - s) / steps))

    def emb(texts):
        enc = tok([PREFIX + t for t in texts], padding=True, truncation=True, max_length=max_len,
                  return_tensors="pt").to(device)
        return torch.nn.functional.normalize(_mean_pool(model(**enc).last_hidden_state, enc["attention_mask"]), dim=-1)

    labels = torch.arange(batch, device=device)
    t0, run = time.time(), 0.0
    for s in range(steps):
        ea = emb(a[s * batch:(s + 1) * batch])
        eb = emb(b[s * batch:(s + 1) * batch])
        logits = ea @ eb.T / temp
        loss = (torch.nn.functional.cross_entropy(logits, labels)
                + torch.nn.functional.cross_entropy(logits.T, labels)) / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if s else loss.item()
        if s % 200 == 0:
            rate = (s + 1) / (time.time() - t0)
            log(f"   fine-tune step {s:,}/{steps:,} loss {run:.3f} ({rate:.1f} steps/s on {device})")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return device


def finetune(pairs_path: Path, out_dir: Path, cfg) -> None:
    t = time.time()
    device = run_isolated(finetune_worker, str(pairs_path), cfg.embed_model, str(out_dir), cfg.finetune_batch,
                          cfg.finetune_lr, cfg.finetune_temperature, cfg.embed_max_len, cfg.seed)
    log(f"   fine-tuning done on {device} in {(time.time() - t) / 60:.1f} min")


# ---------------------------------------------------------------------------- nearest neighbours
def ann_worker(emb_path: str, index_eids: np.ndarray, query_eids: np.ndarray, k: int, nprobe: int,
               n_threads: int, seed: int):
    """Top-k index rows for every query row by cosine (inner product of unit vectors)."""
    import faiss
    faiss.omp_set_num_threads(n_threads)
    E = np.load(emb_path, mmap_mode="r")
    X = np.ascontiguousarray(E[index_eids], dtype=np.float32)
    n, d = X.shape
    if n < 20_000:
        index = faiss.IndexFlatIP(d)
    else:
        nlist = int(min(16384, max(64, 4 * np.sqrt(n))))
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(seed)
        index.train(X[rng.choice(n, min(n, 50 * nlist), replace=False)])
        index.nprobe = nprobe
    index.add(X)
    rows, cols, vals = [], [], []
    for st in range(0, len(query_eids), 200_000):
        Q = np.ascontiguousarray(E[query_eids[st:st + 200_000]], dtype=np.float32)
        sims, nbrs = index.search(Q, min(k, n))
        ok = nbrs >= 0
        r = np.repeat(np.arange(st, st + len(Q), dtype=np.int32)[:, None], nbrs.shape[1], axis=1)
        rows.append(r[ok])
        cols.append(nbrs[ok].astype(np.int32))
        vals.append(sims[ok].astype(np.float32))
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def ann_topk(emb_path: Path, index_eids: np.ndarray, query_eids: np.ndarray, cfg):
    return run_isolated(ann_worker, str(emb_path), index_eids, query_eids, cfg.topk_embed,
                        cfg.ann_nprobe, cfg.n_jobs + 2, cfg.seed)


def rowwise_cos(E: np.ndarray, a_eids: np.ndarray, b_eids: np.ndarray, chunk: int = 250_000) -> np.ndarray:
    out = np.empty(len(a_eids), np.float32)
    for st in range(0, len(a_eids), chunk):
        x = np.asarray(E[a_eids[st:st + chunk]], dtype=np.float32)
        y = np.asarray(E[b_eids[st:st + chunk]], dtype=np.float32)
        out[st:st + chunk] = np.einsum("ij,ij->i", x, y)
    return out
