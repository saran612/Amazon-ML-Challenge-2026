#!/usr/bin/env python3
"""Independent scorer for entity-resolution predictions (standard library only).

Metric:
  * F_0.5 = (1.25 * P * R) / (0.25 * P + R), computed per Source 1 entity;
  * macro-average over ALL Source 1 entities in the evaluation set (singletons included);
  * singleton (no true matches): empty prediction -> 1.0, any prediction -> 0.0;
  * an entity with true matches but an empty prediction -> 0.0 (recall 0);
  * a non-empty prediction with no correct IDs -> 0.0.

Scores predictions against any labelled set, e.g. the validation files scripts/train.py exports
(work/model/val_matching_results.tsv + work/model/val_ground_truth.tsv).

Usage:
    python scripts/score.py --pred PRED.tsv --truth TRUTH.tsv [--source1 SOURCE1.tsv] [--per-entity OUT.tsv]
    python scripts/score.py --self-test          # reproduces the worked example (0.714)

Format problems (missing or duplicate S1 rows, duplicate IDs in a list) are reported as warnings
and scored anyway: a missing S1 counts as an empty prediction; duplicate IDs count once.
"""
import argparse
import csv
import sys
from collections import defaultdict

BETA2 = 0.25  # beta = 0.5


def f05(pred: set, truth: set) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def read_lists(path: str, id_col: str, list_col: str, warnings: list) -> dict:
    csv.field_size_limit(sys.maxsize)
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        if header[:2] != [id_col, list_col]:
            warnings.append(f"{path}: header is {header[:2]}, expected {[id_col, list_col]}")
        for n, row in enumerate(reader, start=2):
            if not row:
                continue
            s1 = row[0].strip()
            ids = [x.strip() for x in (row[1] if len(row) > 1 else "").split(",") if x.strip()]
            if len(ids) != len(set(ids)):
                warnings.append(f"{path}:{n}: duplicate IDs in the list for {s1}")
            if s1 in out:
                warnings.append(f"{path}:{n}: duplicate row for {s1}")
            out[s1] = set(ids)
    return out


def read_countries(path: str) -> dict:
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        i_id, i_c = header.index("entity_id"), header.index("country")
        return {row[i_id]: row[i_c] for row in reader if row}


def score(pred: dict, truth: dict, countries: dict | None = None, warnings: list | None = None):
    warnings = warnings if warnings is not None else []
    missing = [s for s in truth if s not in pred]
    extra = [s for s in pred if s not in truth]
    if missing:
        warnings.append(f"{len(missing):,} S1 entities have no prediction row (scored as empty)")
    if extra:
        warnings.append(f"{len(extra):,} predicted S1 entities are not in the ground truth (ignored)")

    per = {}
    for s1, t in truth.items():
        per[s1] = f05(pred.get(s1, set()), t)
    n = len(per)
    macro = sum(per.values()) / n if n else float("nan")

    groups = defaultdict(list)
    for s1, f in per.items():
        k = len(truth[s1])
        groups["singletons (0 true matches)" if k == 0 else "non-singletons"].append(f)
        groups[f"true matches {'0' if k == 0 else '1-2' if k <= 2 else '3-5' if k <= 5 else '6+'}"].append(f)
        if countries is not None:
            groups[f"country {countries.get(s1, '?')}"].append(f)
    tp = sum(len(pred.get(s, set()) & t) for s, t in truth.items())
    n_pred = sum(len(pred.get(s, set())) for s in truth)
    n_true = sum(len(t) for t in truth.values())
    summary = {
        "macro_f05": macro,
        "n_entities": n,
        "pair_precision": tp / n_pred if n_pred else float("nan"),
        "pair_recall": tp / n_true if n_true else float("nan"),
        "breakdown": {k: (len(v), sum(v) / len(v)) for k, v in sorted(groups.items())},
    }
    return summary, per


def self_test() -> None:
    # worked example: predict [S2-00047, S2-00193, S3-00812], truth [S2-00047, S3-00812] -> 0.714
    f = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(f - 0.7143) < 1e-3, f
    assert f05(set(), set()) == 1.0            # singleton, correctly empty
    assert f05({"S2-1"}, set()) == 0.0         # singleton, false merge
    assert f05(set(), {"S2-1"}) == 0.0         # missed everything
    assert f05({"S2-9"}, {"S2-1"}) == 0.0      # wrong match only
    s, _ = score({"a": {"x", "y", "z"}, "b": set()}, {"a": {"x", "z"}, "b": set()})
    assert abs(s["macro_f05"] - (0.7143 + 1.0) / 2) < 1e-3
    print(f"self-test passed: worked example F0.5 = {f:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", help="predictions: source1_entity_id <TAB> matched_entity_ids")
    ap.add_argument("--truth", help="ground truth: source1_entity_id <TAB> matched_entity_ids")
    ap.add_argument("--source1", help="optional Source 1 file, adds a per-country breakdown")
    ap.add_argument("--per-entity", help="optional output TSV with the F0.5 of every S1 entity")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not (args.pred and args.truth):
        ap.error("--pred and --truth are required (or use --self-test)")

    warnings: list = []
    pred = read_lists(args.pred, "source1_entity_id", "matched_entity_ids", warnings)
    truth = read_lists(args.truth, "source1_entity_id", "matched_entity_ids", warnings)
    countries = read_countries(args.source1) if args.source1 else None
    summary, per = score(pred, truth, countries, warnings)

    for w in warnings:
        print(f"WARNING: {w}")
    print(f"Macro F0.5: {summary['macro_f05']:.4f}   ({summary['n_entities']:,} Source 1 entities)")
    print(f"Pair precision: {summary['pair_precision']:.4f}   pair recall: {summary['pair_recall']:.4f}")
    for k, (n, f) in summary["breakdown"].items():
        print(f"  {k:32s} n={n:>9,}  F0.5={f:.4f}")
    if args.per_entity:
        with open(args.per_entity, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["source1_entity_id", "f05", "n_pred", "n_true"])
            for s1, v in per.items():
                w.writerow([s1, f"{v:.6f}", len(pred.get(s1, ())), len(truth[s1])])


if __name__ == "__main__":
    main()
