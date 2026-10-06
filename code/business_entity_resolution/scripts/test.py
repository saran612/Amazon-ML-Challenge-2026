"""Run the trained pipeline on the test set and write the two output files.

    python scripts/test.py                  # writes output/matching_results.tsv and output/candidate_pairs.tsv
    python scripts/test.py --sample 0.05    # uses the sample model and a data subset (output/sample/)

Requires a finished scripts/train.py run (same --sample value and --work-dir).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.config import default_data_dir, default_output_dir, default_work_dir  # noqa: E402
from er.pipeline import run_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--work-dir", type=Path, default=default_work_dir())
    ap.add_argument("--output-dir", type=Path, default=default_output_dir())
    ap.add_argument("--sample", type=float, default=None, help="use the sample model and a data subset")
    ap.add_argument("--force", action="store_true", help="ignore cached stage outputs")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes (default: as trained)")
    args = ap.parse_args()
    run_test(args.data_dir, args.work_dir, args.output_dir, args.sample, args.force, args.jobs)


if __name__ == "__main__":
    main()
