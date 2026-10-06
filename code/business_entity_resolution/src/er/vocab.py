"""Vocabulary learned from the provided data.

* Token map: abbreviation / spelling variants mined from training match pairs
  (e.g. st->street, pvt->private, hriyana->haryana). Supervised, so it is learned in
  train.py only, on the training part of the split, and reused unchanged by test.py.
* Low-information name tokens and place segments: frequency statistics computed on the
  source files of whichever dataset is being processed.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl
from rapidfuzz.distance import JaroWinkler

from .config import Config

VOWELS = r"[aeiou]"


def is_abbrev(short: str, long: str) -> bool:
    """Language-agnostic abbreviation test: same first letter, letters appear in order."""
    if len(short) < 2 or len(short) >= len(long) or short[0] != long[0]:
        return False
    it = iter(long)
    return all(c in it for c in short)


def skeleton(tok: str) -> str:
    return "".join(c for c in tok if c not in "aeiou")


def _related(a: str, b: str) -> bool:
    if a.isdigit() or b.isdigit():
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if is_abbrev(short, long):
        return True
    if len(short) >= 3 and skeleton(a) == skeleton(b):
        return True
    return len(short) >= 4 and JaroWinkler.normalized_similarity(a, b) >= 0.88


def mine_token_map(a_texts: list[str], b_texts: list[str], token_df: dict[str, int],
                   cfg: Config) -> dict[str, str]:
    """Mine token variants from aligned text pairs of true matches.

    For each pair, tokens left unmatched on both sides are candidate variants of each other.
    A variant is kept when it recurs (support), explains a real share of the rarer token's
    unmatched occurrences (confidence), and passes a language-agnostic similarity test.
    Each token maps to its single strongest partner, always from the rarer to the more
    frequent spelling (both sides are mapped, so only consistency matters).
    """
    pair_cnt: Counter = Counter()
    tok_cnt: Counter = Counter()
    for a, b in zip(a_texts, b_texts):
        ta, tb = set(a.split()), set(b.split())
        ua, ub = ta - tb, tb - ta
        if not ua or not ub or len(ua) > 4 or len(ub) > 4:
            continue
        tok_cnt.update(ua)
        tok_cnt.update(ub)
        for x in ua:
            for y in ub:
                pair_cnt[(x, y) if x < y else (y, x)] += 1

    best: dict[str, tuple[int, str]] = {}
    for (x, y), c in pair_cnt.items():
        if c < cfg.abbrev_min_support or not _related(x, y):
            continue
        src, dst = (x, y) if (token_df.get(x, 0), x) < (token_df.get(y, 0), y) else (y, x)
        if c / tok_cnt[src] < cfg.abbrev_min_confidence:
            continue
        if c > best.get(src, (0, ""))[0]:
            best[src] = (c, dst)

    raw = {src: dst for src, (_, dst) in best.items()}
    resolved = {}
    for src in raw:  # follow chains to a fixed point, guarding against cycles
        seen, cur = {src}, raw[src]
        while cur in raw and raw[cur] not in seen:
            seen.add(cur)
            cur = raw[cur]
        if cur != src:
            resolved[src] = cur
    return resolved


def token_doc_freq(texts: pl.Series) -> dict[str, int]:
    t = texts.str.split(" ").list.unique().explode(empty_as_null=True)
    vc = t.filter(t.str.len_chars() > 0).value_counts()
    return dict(zip(vc[:, 0].to_list(), vc[:, 1].to_list()))


def apply_token_map(s: pl.Series, mapping: dict[str, str]) -> pl.Series:
    if not mapping:
        return s
    return s.str.split(" ").list.eval(pl.element().replace(mapping)).list.join(" ")


def map_segments(segs: pl.Series, mapping: dict[str, str]) -> pl.Series:
    if not mapping:
        return segs
    joined = segs.list.join(" | ")
    return apply_token_map(joined, mapping).str.split(" | ").list.eval(
        pl.element().filter(pl.element().str.len_chars() > 0))


def learn_stopwords(frame: pl.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Name tokens that occur in more than stop_df_ratio of a country's names."""
    out = {}
    for country, part in frame.group_by("country"):
        n = part.height
        t = part["name_norm"].str.split(" ").list.unique().explode(empty_as_null=True)
        vc = t.filter(t.str.len_chars() > 0).value_counts()
        out[country[0]] = vc.filter(pl.col("count") > cfg.stop_df_ratio * n)[:, 0].to_list()
    return out


def learn_places(frame: pl.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Digit-free address segments that recur across many records of a country."""
    out = {}
    for country, part in frame.group_by("country"):
        n = part.height
        min_count = max(cfg.place_min_count, int(cfg.place_min_ratio * n))
        seg = part["addr_segs"].list.unique().explode(empty_as_null=True)
        seg = seg.filter(seg.is_not_null() & ~seg.str.contains(r"\d"))
        vc = seg.value_counts()
        out[country[0]] = vc.filter(pl.col("count") >= min_count)[:, 0].to_list()
    return out


SHARED = "*"  # variants found in every training country; used for countries unseen in training


def shared_map(per_country: dict[str, dict]) -> dict:
    maps = list(per_country.values())
    if not maps:
        return {}
    return {k: v for k, v in maps[0].items() if all(m.get(k) == v for m in maps[1:])}


def apply_maps(frame: pl.DataFrame, maps: dict) -> pl.DataFrame:
    """maps = {"name": {country: map, "*": map}, "addr": {...}}; unseen countries get "*"."""
    parts = []
    for country, part in frame.group_by("country", maintain_order=True):
        c = country[0] if country[0] in maps["name"] else SHARED
        parts.append(part.with_columns(
            name_norm=apply_token_map(part["name_norm"], maps["name"].get(c, {})),
            addr_segs=map_segments(part["addr_segs"], maps["addr"].get(c, {})),
        ))
    return pl.concat(parts).sort("idx")


def derive(frame: pl.DataFrame, stop: dict[str, list[str]], places: dict[str, list[str]]) -> pl.DataFrame:
    """Add the columns used by blocking and features (after apply_maps)."""
    parts = []
    for country, part in frame.group_by("country", maintain_order=True):
        sw = stop.get(country[0], [])
        pl_set = places.get(country[0], [])
        part = part.with_columns(
            core=pl.col("name_norm").str.split(" ")
            .list.eval(pl.element().filter(~pl.element().is_in(sw))).list.join(" "),
            place_segs=pl.col("addr_segs").list.eval(pl.element().filter(pl.element().is_in(pl_set))),
            street_segs=pl.col("addr_segs").list.eval(pl.element().filter(~pl.element().is_in(pl_set))),
        )
        parts.append(part)
    frame = pl.concat(parts).sort("idx")
    frame = frame.with_columns(
        core=pl.when(pl.col("core").str.len_chars() > 0).then("core").otherwise("name_norm"),
        addr_norm=pl.col("addr_segs").list.join(" "),
        place_str=pl.col("place_segs").list.join(" "),
        place_list=pl.col("place_segs").list.unique(maintain_order=True).list.join("|"),
        street_str=pl.col("street_segs").list.join(" ").str.replace_all(r"\b\d+\b", "")
        .str.replace_all(r"\s+", " ").str.strip_chars(),
    ).with_columns(
        core_ns=pl.col("core").str.replace_all(" ", ""),
        nums_str=pl.col("addr_norm").str.extract_all(r"\b\d+\b").list.join(" "),
    ).with_columns(
        skel=pl.col("core_ns").str.replace_all(VOWELS, ""),
    )
    return frame.drop("place_segs", "street_segs", "addr_segs")


def learn_place_links(frame: pl.DataFrame, cfg: Config) -> list[str]:
    """Pairs of place segments that co-occur in the same addresses ("a|b", a < b).

    Links a city with its region and its département (bordeaux | gironde,
    bordeaux | nouvelle aquitaine) from the data itself, so two addresses that name the same
    area at different levels still agree. Learned on the dataset being processed.
    """
    out = []
    for country, part in frame.select("country", "place_list").group_by("country"):
        if part.height > cfg.place_link_sample:
            part = part.sample(cfg.place_link_sample, seed=cfg.seed)
        min_count = max(cfg.place_min_count, int(cfg.place_link_min_ratio * part.height))
        segs = part.select(pl.int_range(pl.len()).alias("r"), pl.col("place_list").str.split("|").alias("a"))
        segs = segs.explode("a", empty_as_null=True).filter(pl.col("a").str.len_chars() > 0)
        pairs = (segs.join(segs.rename({"a": "b"}), on="r").filter(pl.col("a") < pl.col("b"))
                 .group_by("a", "b").len().filter(pl.col("len") >= min_count))
        out += (pairs["a"] + "|" + pairs["b"]).to_list()
    return out


def add_local_counts(s1: pl.DataFrame, q: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Local name rarity: how many S1 records share the core name *in the same place*.

    A common name is only ambiguous if another business with that name is nearby. Each record's
    place key is its most specific place segment (the rarest one in its country); `local_cnt` is
    the number of S1 records with the same (country, core name, place key), or -1 when the record
    has no place. Computed on the dataset being processed; nothing is hand-written.
    """
    def segs(frame):
        return (frame.select("idx", "country", pl.col("place_list").str.split("|").alias("seg"))
                .explode("seg", empty_as_null=True).filter(pl.col("seg").str.len_chars() > 0))

    freq = pl.concat([segs(s1), segs(q)]).group_by("country", "seg").len("f")

    def with_key(frame):
        key = (segs(frame).join(freq, on=["country", "seg"])
               .sort(["idx", "f", "seg"]).unique("idx", keep="first", maintain_order=True)
               .select("idx", pl.col("seg").alias("place_key")))
        return (frame.join(key, on="idx", how="left").sort("idx")
                .with_columns(pl.col("place_key").fill_null("")))

    s1, q = with_key(s1), with_key(q)
    cnt = s1.filter(pl.col("place_key") != "").group_by("country", "core", "place_key").len("local_cnt")
    out = []
    for frame in (s1, q):
        frame = (frame.drop("local_cnt", strict=False).join(cnt, on=["country", "core", "place_key"], how="left")
                 .sort("idx")
                 .with_columns(local_cnt=pl.when(pl.col("place_key") == "").then(-1)
                               .otherwise(pl.col("local_cnt").fill_null(0)).cast(pl.Int32)))
        out.append(frame)
    return out[0], out[1]


def add_numbers(frame: pl.DataFrame) -> pl.DataFrame:
    """First house number of the address, as text (num1_str) and integer (num1; -1 when none)."""
    first = pl.col("nums_str").str.split(" ").list.first().fill_null("").str.slice(0, 15)
    return frame.with_columns(num1_str=first).with_columns(
        num1=pl.when(pl.col("num1_str") != "")
        .then(pl.col("num1_str").cast(pl.Int64, strict=False)).otherwise(-1).fill_null(-1))


def core_counts(s1: pl.DataFrame, q: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Name rarity: how many S1 records of the country share each core name."""
    cnt = s1.group_by("country", "core").len("core_cnt")
    s1 = s1.join(cnt, on=["country", "core"], how="left").sort("idx")
    q = (q.join(cnt, on=["country", "core"], how="left").sort("idx")
         .with_columns(pl.col("core_cnt").fill_null(0)))
    return s1, q


def sample_pairs(owner: np.ndarray, keep_s: np.ndarray, max_pairs: int, seed: int):
    """(q_idx, s_idx) of true pairs whose S1 is in keep_s, subsampled to max_pairs."""
    qi = np.flatnonzero(owner >= 0)
    qi = qi[keep_s[owner[qi]]]
    if len(qi) > max_pairs:
        qi = np.random.default_rng(seed).choice(qi, max_pairs, replace=False)
    return qi, owner[qi]
