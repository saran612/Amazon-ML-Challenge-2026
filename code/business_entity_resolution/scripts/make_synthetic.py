"""Write a small synthetic dataset to try the pipeline without the real data.

    python scripts/make_synthetic.py --out data_synthetic --n-s1 300
    python scripts/train.py --data-dir data_synthetic --work-dir work_synthetic --no-embed --jobs 2
    python scripts/test.py  --data-dir data_synthetic --work-dir work_synthetic --output-dir output_synthetic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from er.synthetic import write_dataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-s1", type=int, default=300, help="S1 businesses per country and split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    write_dataset(args.out, args.n_s1, args.seed)
    print(f"wrote synthetic dataset to {args.out}")


if __name__ == "__main__":
    main()
