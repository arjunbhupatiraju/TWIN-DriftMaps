"""REPMAP on the Mac.

    python3 scripts/run_repmap.py sens       # exploration round 3 (explore mice only; done)
    python3 scripts/run_repmap.py confirm    # THE registered REPMAP test on the confirm mice: once, only after
                                             # REPMAP_PREREG_LOCK.txt holds the OSF link (~2-5 min)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOCK = Path("REPMAP_PREREG_LOCK.txt")
OUT = Path("results/repmap_confirm")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "sens":
        from gatekeeper import sensitivity as S
        out = Path("results/repmap_explore/sens_stats.csv.gz")
        if out.exists():
            raise SystemExit(f"{out} already exists.")
        S.run(Path("data/mv_cache"), out, n_null=1000)
    elif cmd == "confirm":
        from gatekeeper import repmap_final as F
        if not LOCK.exists() or "osf.io" not in LOCK.read_text():
            raise SystemExit("Refusing: REPMAP_PREREG_LOCK.txt with the OSF link of the REPMAP registration is missing.")
        if (OUT / "repmap_result.json").exists() and "--force" not in sys.argv:
            raise SystemExit("The REPMAP confirm test already exists; it is run once.")
        stats = Path("results/multiverse_confirm/session_stats.csv.gz")
        if not stats.exists():
            raise SystemExit(f"missing {stats}")
        F.run(stats, OUT)
    else:
        raise SystemExit(__doc__)
