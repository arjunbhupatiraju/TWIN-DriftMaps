"""The new model on real IBL data. From the gatekeeper folder:
    python3 scripts/run_model_real.py --limit 3          # quick test
    python3 scripts/run_model_real.py                    # explore mice
    python3 scripts/run_model_real.py --split confirm    # confirm mice (replication)
    python3 scripts/model_report.py                      # summary + figure
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper.model_real import run
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/mv_cache")
ap.add_argument("--split", default="explore", choices=["explore", "confirm", "all"])
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--units", default="good", choices=["good", "all"])
a = ap.parse_args()
tag = ('test' if a.limit else a.split) + ('' if a.units == 'good' else '_allunits')
run(a.cache, f"results/model_real/rows_{tag}.csv.gz", a.split, a.jobs, a.limit, (a.units,))
