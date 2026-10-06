"""Train the full entity-resolution pipeline end to end.

    python scripts/train.py                  # full training run (default settings)
    python scripts/train.py --sample 0.05    # quick run on a consistent 5% subset
    python scripts/train.py --no-embed       # without the multilingual embeddings (no torch needed)

Writes every artifact scripts/test.py needs to work/model/ (work/model_sample<f>/ for samples).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import Config, default_data_dir, default_work_dir  # noqa: E402
from er.pipeline import run_train  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--work-dir", type=Path, default=default_work_dir())
    ap.add_argument("--sample", type=float, default=None, help="fraction of the data, for quick runs")
    ap.add_argument("--force", action="store_true", help="ignore cached stage outputs")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default 10)")
    ap.add_argument("--no-embed", action="store_true",
                    help="skip the multilingual name embeddings (blocking pass D and the cosine feature)")
    ap.add_argument("--no-finetune", action="store_true", help="use the pretrained encoder without fine-tuning")
    ap.add_argument("--embed-model", default=None, help="encoder name or local path (default multilingual-e5-small)")
    ap.add_argument("--no-cross-encoder", action="store_true", help="skip the cross-encoder re-scoring")
    ap.add_argument("--no-consistency", action="store_true", help="skip the record-consistency step")
    ap.add_argument("--proxy", action="store_true",
                    help="also train and report leave-one-country-out models (transfer to unseen countries)")
    ap.add_argument("--sim-drop", type=float, default=None,
                    help="share of non-validation S1 records removed to simulate a denser crowd (default 0)")
    args = ap.parse_args()

    cfg = Config()
    if args.jobs:
        cfg.n_jobs = args.jobs
    cfg.use_embeddings = not args.no_embed
    cfg.finetune_embed = not args.no_finetune
    cfg.use_cross_encoder = not args.no_cross_encoder and not args.no_embed
    cfg.use_consistency = not args.no_consistency
    cfg.use_proxy = args.proxy
    if args.embed_model:
        cfg.embed_model = args.embed_model
    if args.sim_drop is not None:
        cfg.sim_drop_s1_frac = args.sim_drop
    run_train(args.data_dir, args.work_dir, cfg, args.sample, args.force)


if __name__ == "__main__":
    main()
