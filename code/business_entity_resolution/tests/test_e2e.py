"""End-to-end: train + test on a small synthetic dataset (no embeddings, 2 workers, ~1 minute)."""
import csv
import json

import pytest

from er.config import Config
from er.pipeline import run_test, run_train
from er.synthetic import write_dataset


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        header = next(r)
        return header, {row[0]: set(filter(None, row[1].split(","))) for row in r}


@pytest.mark.e2e
def test_train_and_test_on_synthetic_data(tmp_path):
    data, work, out = tmp_path / "data", tmp_path / "work", tmp_path / "out"
    write_dataset(data, n_s1=300, seed=1)
    cfg = Config(n_jobs=2, use_embeddings=False, use_cross_encoder=False, finetune_embed=False,
                 feature_chunk=20_000, soft_task_size=2_000)
    run_train(data, work, cfg, sample=None, force=False)

    report = json.loads((work / "model" / "train_report.json").read_text())
    assert report["blocking"]["val"]["pair_recall"] > 0.9
    assert report["validation_valB"]["stage2_global_threshold"]["macro_f05"] > 0.7

    run_test(data, work, out, sample=None, force=False, n_jobs=None)
    header_m, match = _rows(out / "matching_results.tsv")
    header_c, cand = _rows(out / "candidate_pairs.tsv")
    assert header_m == ["source1_entity_id", "matched_entity_ids"]
    assert header_c == ["source1_entity_id", "candidate_entity_ids"]
    with open(data / "test" / "test_source1.tsv", encoding="utf-8") as f:
        n_s1 = sum(1 for _ in f) - 1
    assert len(match) == len(cand) == n_s1                          # one row per S1
    assert all(match[s] <= cand[s] for s in match)                  # matches are candidates
    assert all(x.startswith(("S2-", "S3-")) for ids in match.values() for x in ids)
