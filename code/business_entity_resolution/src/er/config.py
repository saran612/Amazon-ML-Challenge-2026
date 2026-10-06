"""All paths and tunable parameters in one place."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Dataset location: $ER_DATA_DIR if set, else <repo>/dataset (see dataset/README.md)."""
    return Path(os.environ.get("ER_DATA_DIR", REPO_ROOT / "dataset"))


def default_work_dir() -> Path:
    return REPO_ROOT / "work"


def default_output_dir() -> Path:
    return REPO_ROOT / "output"


@dataclass
class Config:
    seed: int = 2026
    n_jobs: int = 10

    # --- learned vocabulary -------------------------------------------------
    # A name token present in more than this share of a country's names is low-information.
    stop_df_ratio: float = 0.01
    # A digit-free address segment is a "place" if it recurs in this share of a country's records.
    place_min_ratio: float = 2e-5
    place_min_count: int = 5
    abbrev_min_support: int = 25
    abbrev_min_confidence: float = 0.3
    abbrev_max_pairs: int = 2_000_000

    # --- blocking -------------------------------------------------------------
    char_ngram: int = 3
    # Columns whose S1 document frequency exceeds this share are skipped during retrieval only.
    retrieval_max_df: float = 0.01
    retrieval_min_df_cap: int = 200
    topk_char: int = 10
    topk_addr: int = 8
    topk_joint: int = 10
    max_candidates: int = 16
    keep_top_per_pass: int = 3
    query_chunk: int = 4000

    # --- validation split -----------------------------------------------------
    val_frac: float = 0.05
    # Features for the training rows are written to a memory-mapped file and LightGBM bins them
    # once for all folds, so 30M rows fit in 24 GB of RAM.
    max_train_pairs: int = 30_000_000

    # --- features -------------------------------------------------------------
    feature_chunk: int = 1_500_000
    soft_task_size: int = 100_000

    # --- model ----------------------------------------------------------------
    n_folds: int = 3
    lgb_rounds: int = 2000
    lgb_early_stop: int = 100
    lgb_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "verbose": -1,
    })
    # stage 2 uses the stage-1 score, its query context and the top-N base features by importance
    stage2_top_features: int = 24

    # --- multilingual name embeddings (blocking pass D + cosine feature) -----------
    use_embeddings: bool = True
    embed_model: str = "intfloat/multilingual-e5-small"  # MIT license, 118M parameters
    embed_dim: int = 384
    embed_max_len: int = 32
    embed_batch: int = 256
    topk_embed: int = 15
    ann_nprobe: int = 16
    finetune_embed: bool = True
    finetune_pairs: int = 300_000
    finetune_batch: int = 128
    finetune_lr: float = 2e-5
    finetune_temperature: float = 0.05

    # --- cross-encoder re-scoring of uncertain pairs --------------------------------
    use_cross_encoder: bool = True
    ce_base_model: str = "intfloat/multilingual-e5-small"   # MIT; a classification head is added
    ce_train_pairs: int = 300_000
    ce_train_band: tuple = (0.02, 0.98)   # stage-1 out-of-fold score band for training pairs
    ce_band: tuple = (0.02, 0.98)         # calibrated score band re-scored at validation / test
    ce_max_len: int = 96
    ce_batch: int = 64
    ce_lr: float = 3e-5
    ce_weight_grid: tuple = (1.0, 0.75, 0.5, 0.25)   # weight of the LightGBM score; 1.0 = off

    # --- sibling records ----------------------------------------------------------
    # "Strong siblings" of an S1: records whose best blocking match it is, with a combined blocking
    # score of at least sib_min_comb (at most sib_max_per_s per S1, best first).
    sib_min_comb: float = 0.5
    sib_max_per_s: int = 10

    # --- consistency between records of the same business -----------------------
    use_consistency: bool = True
    consistency_anchor_min: float = 0.5
    consistency_target_min: float = 0.02
    consistency_lambda_grid: tuple = (0.0, 0.5, 0.75, 1.0)
    consistency_sim_grid: tuple = (0.7, 0.85)
    # place segments that co-occur in the same addresses (links a city with its region)
    place_link_min_ratio: float = 2e-5
    place_link_sample: int = 3_000_000

    # --- transfer to countries unseen in training (optional, --proxy) -----------
    # Leave-one-country-out: train on one training country, evaluate on the other. Also searches
    # a stricter minimum match probability for countries absent from training, kept only if it
    # improves both directions.
    use_proxy: bool = False
    unseen_min_p_grid: tuple = (0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98)

    # --- denser-crowd simulation (optional, training / validation only) ----------
    # Removes this share of the non-validation S1 records from the index, so their S2/S3 records
    # become distractors. Useful when the data to be matched has more distractors per S1 than
    # the training data. Validation S1 records are never removed.
    sim_drop_s1_frac: float = 0.0

    # --- decision layer grids -------------------------------------------------
    t_assign_grid: tuple = (0.02, 0.05, 0.1, 0.15, 0.2, 0.3)
    miss_mass_grid: tuple = (0.0, 0.1, 0.25, 0.5)
    global_t_grid: tuple = tuple(round(0.2 + 0.025 * i, 3) for i in range(29))

    def to_dict(self) -> dict:
        return asdict(self)
