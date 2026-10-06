"""Timestamped logging with stage timing and peak memory."""
from __future__ import annotations

import resource
import sys
import time
from contextlib import contextmanager

_T0 = time.time()


def peak_mem_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:8.1f}s] {msg}", flush=True)


@contextmanager
def stage(name: str):
    t = time.time()
    log(f"── {name} …")
    yield
    log(f"── {name} done in {time.time() - t:.1f}s (peak mem {peak_mem_gb():.1f} GB)")
