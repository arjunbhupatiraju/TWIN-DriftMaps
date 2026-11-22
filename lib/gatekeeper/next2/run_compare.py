"""Head-to-head with Findling 2025 and Zhang 2025-style decoding on the IBL cache.

    python next/run_compare.py --cache <your IBL cache folder> --split confirm --out results/compare

Quick test first:  add  --limit 3 --nulls 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_methods  # noqa: E402
import report_compare  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", required=True)
ap.add_argument("--split", default="confirm", choices=["explore", "confirm", "all"])
ap.add_argument("--out", default="results/compare")
ap.add_argument("--nulls", type=int, default=200)
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--folds", default="random", choices=["random", "contiguous"])
a = ap.parse_args()

csv = Path(a.out) / f"compare_rows_{a.split}.csv"
compare_methods.run(a.cache, csv, split=a.split, n_null=a.nulls, n_jobs=a.jobs, limit=a.limit, folds=a.folds)
report_compare.main(csv, a.out)
