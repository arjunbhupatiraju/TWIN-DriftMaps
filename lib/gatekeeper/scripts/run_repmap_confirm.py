"""REPMAP confirmatory test: runs ONCE on the confirm mice, only after REPMAP_PREREG_LOCK.txt holds the OSF link.

    python3 scripts/run_repmap_confirm.py        (~5-15 min)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper import repmap_confirm as R  # noqa: E402

LOCK = Path("REPMAP_PREREG_LOCK.txt")
STATS = Path("results/multiverse_confirm/session_stats.csv.gz")
OUT = Path("results/repmap_confirm")

if __name__ == "__main__":
    if not LOCK.exists() or "osf.io" not in LOCK.read_text():
        raise SystemExit("Refusing: REPMAP_PREREG_LOCK.txt with the OSF link of the REPMAP registration is missing.")
    if (OUT / "repmap_confirm.json").exists() and "--force" not in sys.argv:
        raise SystemExit("The REPMAP confirm test already exists; it is run once.")
    if not STATS.exists():
        raise SystemExit(f"missing {STATS}")
    R.run(STATS, OUT, n_rep=300)
