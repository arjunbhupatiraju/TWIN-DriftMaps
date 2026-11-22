"""REPMAP amendment 1: fixed-strength evaluation on the confirm mice. Runs ONCE, only after
REPMAP_AMEND1_LOCK.txt holds the OSF link of the amendment.

    python3 scripts/run_repmap_followup.py        (~1-2 min)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper import repmap_followup as F  # noqa: E402

LOCK = Path("REPMAP_AMEND1_LOCK.txt")
STATS = Path("results/multiverse_confirm/session_stats.csv.gz")
OUT = Path("results/repmap_confirm/repmap_amend1.json")

if __name__ == "__main__":
    if not LOCK.exists() or "osf.io" not in LOCK.read_text():
        raise SystemExit("Refusing: REPMAP_AMEND1_LOCK.txt with the OSF link of the amendment is missing.")
    if OUT.exists() and "--force" not in sys.argv:
        raise SystemExit("Amendment 1 has already been run; it is run once.")
    F.run(STATS, OUT, n_rep=300)
