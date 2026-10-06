"""Reading the TSV inputs and writing the two output files."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .log import log

SOURCE_FILES = {"S1": "source1", "S2": "source2", "S3": "source3"}


def _count_rows(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            n += block.count(b"\n")
    return n - 1  # header


def read_source(path: Path) -> pl.DataFrame:
    """Read one source TSV with quoting disabled (names contain quote characters)."""
    df = pl.read_csv(
        path,
        separator="\t",
        quote_char=None,
        infer_schema=False,
    )
    df = df.select(
        pl.col("entity_id").str.strip_chars(),
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null(""),
        pl.col("country").fill_null("").str.strip_chars(),
    )
    expected = _count_rows(path)
    if df.height != expected:
        raise ValueError(f"{path}: read {df.height} rows, file has {expected}")
    return df


def read_split(data_dir: Path, split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (s1, q): S1 records, and S2+S3 records stacked with a `source` column."""
    base = Path(data_dir) / split
    s1 = read_source(base / f"{split}_source1.tsv")
    s2 = read_source(base / f"{split}_source2.tsv").with_columns(source=pl.lit(2, pl.Int8))
    s3 = read_source(base / f"{split}_source3.tsv").with_columns(source=pl.lit(3, pl.Int8))
    q = pl.concat([s2, s3])
    log(f"read {split}: S1={s1.height:,} S2={s2.height:,} S3={s3.height:,}")
    return s1, q


def read_ground_truth(data_dir: Path, s1: pl.DataFrame, q: pl.DataFrame) -> np.ndarray:
    """Owner S1 row index for every q row (-1 when the record matches no S1)."""
    gt = pl.read_csv(
        Path(data_dir) / "train" / "train_ground_truth.tsv",
        separator="\t", quote_char=None, infer_schema=False,
    )
    long = (
        gt.select(
            pl.col("source1_entity_id").alias("s1_id"),
            pl.col("matched_entity_ids").fill_null("").str.split(",").alias("q_id"),
        )
        .explode("q_id", empty_as_null=True)
        .filter(pl.col("q_id").str.len_chars() > 0)
    )
    s1_map = s1.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("sidx"))
    q_map = q.select(pl.col("entity_id").alias("q_id"), pl.col("idx").alias("qidx"))
    long = long.join(s1_map, on="s1_id", how="inner").join(q_map, on="q_id", how="inner")
    owner = np.full(q.height, -1, dtype=np.int64)
    owner[long["qidx"].to_numpy()] = long["sidx"].to_numpy()
    return owner


def write_id_lists(path: Path, s1_ids: pl.Series, s_idx: np.ndarray, q_ids: np.ndarray,
                   list_col: str) -> None:
    """One row per S1 (input order); comma-joined S2/S3 IDs, empty when none."""
    pairs = pl.DataFrame({"sidx": s_idx.astype(np.int64), "q_id": q_ids})
    pairs = pairs.unique(maintain_order=True)
    grouped = pairs.group_by("sidx").agg(pl.col("q_id").str.join(","))
    out = (
        pl.DataFrame({"source1_entity_id": s1_ids, "sidx": np.arange(len(s1_ids), dtype=np.int64)})
        .join(grouped, on="sidx", how="left")
        .sort("sidx")
        .select("source1_entity_id", pl.col("q_id").fill_null("").alias(list_col))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
