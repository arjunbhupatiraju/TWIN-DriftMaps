"""INFOFLOW: where does the brain lose what it sees?

    python scripts/run_infoflow.py download --split explore   # ~40% of mice first (~5-6 h)
    python scripts/run_infoflow.py download --split confirm   # the rest, later
    python scripts/run_infoflow.py download          # internet, long, resumable; reuses the 301 selected sessions
    python scripts/run_infoflow.py status            # how many sessions per split, regions per split
    python scripts/run_infoflow.py explore           # EXPLORE split only (~40% of mice); confirm split untouched

The confirm split is never loaded by `explore` (enforced in code). See docs/INFOFLOW_PLAN.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = Path("data/info_cache")
SEL = Path("results/brainwide_coverage/selected_sessions.json")


def download(args):
    from check_ibl_coverage import connect
    from gatekeeper.infoflow_data import load_session_info
    eids = json.loads(SEL.read_text())["sessions"]
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eids if not (CACHE / f"{e}.npz").exists()]
    print(f"{len(eids)} sessions; {len(eids) - len(todo)} cached; {len(todo)} to download", flush=True)
    one = connect()
    if args.split:
        from gatekeeper.infoflow_data import split_of
        keep = []
        print(f"checking which mouse each session belongs to (fast; only downloading the {args.split} split)...", flush=True)
        for eid in todo:
            try:
                if split_of(str(one.get_details(eid)["subject"])) == args.split:
                    keep.append(eid)
            except Exception:
                keep.append(eid)          # unknown subject: download, split decided after
        print(f"{len(keep)} of {len(todo)} remaining sessions are in the {args.split} split", flush=True)
        todo = keep
    fails, t0 = {}, time.time()
    for i, eid in enumerate(todo):
        try:
            p = load_session_info(one, eid, CACHE, delete_raw=not args.keep_raw)
            z = np.load(p)
            meta = json.loads(str(z["meta"]))
            nreg = len([k for k in z.files if k.startswith("early__")])
            el = (time.time() - t0) / 60
            print(f"[{i+1}/{len(todo)}] {eid}: {len(z['contrast'])} trials, {nreg} regions, split={meta['split']} "
                  f"({el:.0f} min elapsed, ~{el / (i + 1) * (len(todo) - i - 1):.0f} min left)", flush=True)
        except Exception as e:
            fails[eid] = repr(e)[:300]
            print(f"[{i+1}/{len(todo)}] {eid}: FAILED {e!r:.200}", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/infoflow_download_failures.json").write_text(json.dumps(fails, indent=1))
    print(f"done; {len(fails)} failures (re-run the same command to retry)")


def status(args):
    from gatekeeper.infoflow_data import load_info_cache
    S = load_info_cache(CACHE)
    for sp in ("explore", "confirm", "unknown"):
        ss = [s for s in S if s.split == sp]
        if not ss:
            continue
        mice = {s.meta.get("subject") for s in ss}
        print(f"{sp}: {len(ss)} sessions, {len(mice)} mice")


def explore(args):
    from gatekeeper import infoflow
    tag = f"K{args.k}" + ("" if not args.no_block else "_noblock")
    infoflow.run_explore(CACHE, Path(f"results/infoflow_explore_{tag}"), n_boot=args.n_boot, K=args.k, use_block=not args.no_block)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--keep-raw", action="store_true")
    d.add_argument("--split", choices=["explore", "confirm"], default=None,
                   help="only download sessions from this split (explore first, confirm later)")
    sub.add_parser("status")
    e = sub.add_parser("explore")
    e.add_argument("--n-boot", type=int, default=500)
    e.add_argument("--no-block", action="store_true", help="do NOT give the block prior to the GLM-HMM (old behavior)")
    e.add_argument("--k", type=int, default=2, help="number of GLM-HMM states (with the block prior as input, 2 = engaged vs disengaged; try 3 too. Largest stimulus weight = engaged, others = disengaged)")
    args = ap.parse_args()
    dict(download=download, status=status, explore=explore)[args.cmd](args)


if __name__ == "__main__":
    main()
