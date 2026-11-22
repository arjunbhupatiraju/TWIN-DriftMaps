"""MULTIVERSE literature test (run from the repo root).

    python3 scripts/run_literature.py frame      # 1. pull the sampling frame from OpenAlex, once (~5-15 min)
                                                 #    -> manifests/lit_frame.csv + lit_frame.meta.json (sha256 goes in the registration)
    -- register the literature pre-registration on OSF; put its link in LIT_PREREG_LOCK.txt --
    python3 scripts/run_literature.py sheet      # 2. make literature/screening.csv and prefilter_audit.csv (needs the lock)
    -- screen in screen_order until 100 eligible papers; code literature/claims.csv (CODEBOOK.md) --
    python3 scripts/run_literature.py coder2     #    list the 20% of eligible papers for the second coder
    python3 scripts/run_literature.py kappa      #    after both have coded: Cohen's kappa per field
    python3 scripts/run_literature.py freeze     # 3. freeze the coding: writes literature/CODING_LOCK.txt (sha256 of claims.csv)
    python3 scripts/run_literature.py analyse    # 4. score every claim against IBL, once (~5-15 min)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatekeeper import literature as L  # noqa: E402

LOCK = Path("LIT_PREREG_LOCK.txt")
LIT = Path("literature")
CLAIMS = LIT / "claims.csv"
CODING_LOCK = LIT / "CODING_LOCK.txt"
STATS = [Path("results/multiverse_explore/session_stats.csv.gz"), Path("results/multiverse_confirm/session_stats.csv.gz")]
OUT = Path("results/literature")


def _need_lock():
    if not LOCK.exists() or "osf.io" not in LOCK.read_text():
        raise SystemExit("Refusing: LIT_PREREG_LOCK.txt with the OSF link of the literature registration is missing.")


def frame(a):
    L.frame()


def sheet(a):
    _need_lock()
    if (LIT / "screening.csv").exists():
        raise SystemExit("literature/screening.csv already exists; keep working in it.")
    n, m = L.screening_sheet()
    print(f"literature/screening.csv: {n} works to screen in screen_order (stop at {L.N_TARGET} eligible)")
    print(f"literature/prefilter_audit.csv: {m} pre-filter rejects to screen as an audit")


def coder2(a):
    _need_lock()
    import csv
    rows = L.coder2_papers()
    out = LIT / "claims_coder2.csv"
    if out.exists():
        raise SystemExit(f"{out} exists; the second coder fills it in.")
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, L.CLAIM_COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: (r["doi"] if c == "paper_doi" else "") for c in L.CLAIM_COLS})
    print(f"{len(rows)} papers for the second coder -> {out} (one row per paper; add rows for more claims)")
    for r in rows:
        print(f"  {r['doi']}  {r['title'][:90]}")


def kappa(a):
    import csv
    import json
    k = L.kappa(list(csv.DictReader(open(CLAIMS))), list(csv.DictReader(open(LIT / "claims_coder2.csv"))))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kappa.json").write_text(json.dumps(k, indent=1))
    print(json.dumps(k, indent=1))


def freeze(a):
    _need_lock()
    if CODING_LOCK.exists():
        raise SystemExit("Coding is already frozen.")
    sha = hashlib.sha256(CLAIMS.read_bytes()).hexdigest()
    CODING_LOCK.write_text(sha + "\n")
    print(f"coding frozen: claims.csv sha256 {sha}")


def analyse(a):
    _need_lock()
    if not CODING_LOCK.exists():
        raise SystemExit("Freeze the coding first: python3 scripts/run_literature.py freeze")
    if hashlib.sha256(CLAIMS.read_bytes()).hexdigest() != CODING_LOCK.read_text().strip():
        raise SystemExit("claims.csv changed after it was frozen. Refusing.")
    if (OUT / "literature_summary.json").exists() and not a.force:
        raise SystemExit("The literature analysis already exists; it is run once.")
    missing = [s for s in STATS if not s.exists()]
    if missing:
        raise SystemExit(f"missing {missing}")
    L.analyse(CLAIMS, STATS, OUT, n_rep=a.n_rep)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("frame")
    sub.add_parser("sheet")
    sub.add_parser("coder2")
    sub.add_parser("kappa")
    sub.add_parser("freeze")
    an = sub.add_parser("analyse")
    an.add_argument("--n-rep", type=int, default=300)
    an.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dict(frame=frame, sheet=sheet, coder2=coder2, kappa=kappa, freeze=freeze, analyse=analyse)[a.cmd](a)


if __name__ == "__main__":
    main()
