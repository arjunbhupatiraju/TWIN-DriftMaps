"""Second dataset: download Steinmetz 2019, run all three procedures, write the summary.

    python next/run_steinmetz.py --data ~/Downloads/MULTIVERSE/steinmetz_data --out results/steinmetz
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import steinmetz  # noqa: E402
import report_steinmetz  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default="results/steinmetz")
ap.add_argument("--jobs", type=int, default=4)
ap.add_argument("--limit", type=int, default=0, help="only the first N sessions (quick test)")
ap.add_argument("--no-download", action="store_true")
a = ap.parse_args()

data = Path(a.data).expanduser()
if not a.no_download:
    steinmetz.download(data)
csv = Path(a.out) / "steinmetz_rows.csv"
steinmetz.run(data, csv, n_jobs=a.jobs, limit=a.limit)
report_steinmetz.main(csv, a.out)
