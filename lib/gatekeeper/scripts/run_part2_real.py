"""Part 2 real-brain validation. Usage (from the gatekeeper folder):
    python3 scripts/run_part2_real.py --limit 3       # 3-session test (a few minutes)
    python3 scripts/run_part2_real.py                 # explore mice (default)
    python3 scripts/run_part2_real.py --split all     # all 459 sessions
Then:  python3 scripts/part2_report.py results/part2_real/rows_explore.csv.gz
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper.part2_real import run
ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="data/mv_cache")
ap.add_argument("--split", default="explore", choices=["explore", "confirm", "all"])
ap.add_argument("--jobs", type=int, default=-1)
ap.add_argument("--limit", type=int, default=0, help="test run on the first N sessions")
a = ap.parse_args()
name = "test" if a.limit else a.split
run(a.cache, f"results/part2_real/rows_{name}.csv.gz", a.split, a.jobs, a.limit)
