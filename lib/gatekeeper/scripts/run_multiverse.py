"""MULTIVERSE: how reliable is a brain map?

    python3 scripts/run_multiverse.py download     # all 459 BWM sessions; overnight (~8 h), resumable
    python3 scripts/run_multiverse.py status       # sessions / mice / labs per split
    python3 scripts/run_multiverse.py explore      # full exploration, EXPLORE mice only (~30-60 min)
    python3 scripts/run_multiverse.py k4           # kill test K4 on simulated brains (~20-40 min)
    python3 scripts/run_multiverse.py k1stim       # K1 retry with IBL's own stimulus recipe (~30-90 min)
    python3 scripts/run_multiverse.py confirm      # THE confirm run: once, only after PREREG_LOCK.txt holds the OSF link

Downloading BOTH splits is fine: no analysis touches the confirm mice until the pre-registration
is locked (enforced in the analysis code, as in the pilot).
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

CACHE = Path("data/mv_cache")


def download(args):
    from check_ibl_coverage import connect
    from gatekeeper.mv_data import load_session_mv, manifest
    man = manifest()
    eids = list(man)
    if args.split:
        from gatekeeper.infoflow_data import split_of
        eids = [e for e in eids if split_of(man[e]["subject"]) == args.split]
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [e for e in eids if not (CACHE / f"{e}.npz").exists()]
    print(f"{len(eids)} sessions; {len(eids) - len(todo)} already cached; {len(todo)} to download", flush=True)
    one = connect()
    fails, t0 = {}, time.time()
    for i, eid in enumerate(todo):
        try:
            p = load_session_mv(one, eid, man[eid], CACHE, delete_raw=not args.keep_raw)
            with np.load(p) as z:
                meta = json.loads(str(z["meta"]))
                nreg = len([k for k in z.files if k.startswith("stim_a__")])
                ntr = len(z["contrast"])
            el = (time.time() - t0) / 60
            print(f"[{i+1}/{len(todo)}] {eid}: {ntr} trials, {nreg} regions, lab={meta['lab']}, "
                  f"split={meta['split']} ({el:.0f} min elapsed, ~{el / (i + 1) * (len(todo) - i - 1):.0f} min left)", flush=True)
        except Exception as e:
            fails[eid] = repr(e)[:300]
            print(f"[{i+1}/{len(todo)}] {eid}: FAILED {e!r:.200}", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/multiverse_download_failures.json").write_text(json.dumps(fails, indent=1))
    print(f"done; {len(fails)} failures (run the same command again to retry them)")


def status(args):
    from gatekeeper.mv_data import load_mv_cache
    S = load_mv_cache(CACHE)
    for sp in ("explore", "confirm"):
        ss = [s for s in S if s.split == sp]
        print(f"{sp}: {len(ss)} sessions, {len({s.subject for s in ss})} mice, {len({s.lab for s in ss})} labs")
    print(f"total: {len(S)} of 459 sessions cached")


def explore(args):
    from gatekeeper.multiverse import run_explore
    run_explore(CACHE, Path("results/multiverse_explore"), n_perm=args.n_perm, n_rep=args.n_rep,
                manifest_dir=Path(__file__).resolve().parents[1] / "manifests" / "ibl_bwm_region_results")


def k4(args):
    from gatekeeper.multiverse import run_k4
    run_k4(CACHE, Path("results/multiverse_k4"), n_perm=args.n_perm, n_rep=args.n_rep)


def k1stim(args):
    from gatekeeper.multiverse import run_k1_stim
    run_k1_stim(CACHE, Path("results/multiverse_explore"),
                Path(__file__).resolve().parents[1] / "manifests" / "ibl_bwm_region_results", n_pseudo=args.n_pseudo)


LOCK = Path("PREREG_LOCK.txt")


def confirm(args):
    """Runs ONCE, only after the pre-registration is public. PREREG_LOCK.txt must hold the OSF link."""
    from gatekeeper.multiverse import run_explore, run_k1_stim
    if not LOCK.exists() or "osf.io" not in LOCK.read_text():
        raise SystemExit("Refusing: PREREG_LOCK.txt with the OSF registration link is missing. Lock the pre-registration first.")
    out = Path("results/multiverse_confirm")
    if (out / "confirm.json").exists() and not args.force:
        raise SystemExit("The confirm run already exists; it is run once. (Re-print the table with: run_multiverse.py evaluate)")
    man = Path(__file__).resolve().parents[1] / "manifests" / "ibl_bwm_region_results"
    run_explore(CACHE, out, n_perm=200, n_rep=300, split="confirm", manifest_dir=man)
    run_k1_stim(CACHE, out, man, n_pseudo=100, split="confirm", min_sessions=5)
    evaluate(args)


def evaluate(args):
    from gatekeeper.multiverse import evaluate_prereg
    out = Path("results/multiverse_confirm")
    conf = json.loads((out / "confirm.json").read_text())
    ef = Path("results/multiverse_explore/explore.json")
    expl = json.loads(ef.read_text()) if ef.exists() else None
    kf = out / "k1_stim.json"
    k1s = json.loads(kf.read_text()) if kf.exists() else None
    res = evaluate_prereg(conf, expl, k1s)
    print("\nPRE-REGISTERED CLAIMS (confirm mice, run once):")
    for r in res:
        v = r["value"]
        vs = "n/a" if v is None else f"{v:.2f}"
        print(f"  {r['id']:18s} {'PASS' if r['passed'] else 'FAIL'}  value={vs:>6s}  {r['claim']}")
    print(f"{sum(r['passed'] for r in res)}/{len(res)} pre-registered checks passed")
    (out / "prereg_evaluation.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--keep-raw", action="store_true")
    d.add_argument("--split", choices=["explore", "confirm"], default=None)
    sub.add_parser("status")
    e = sub.add_parser("explore", help="full exploration on the EXPLORE mice only (confirm mice never opened)")
    e.add_argument("--n-perm", type=int, default=200)
    e.add_argument("--n-rep", type=int, default=300)
    k = sub.add_parser("k4", help="kill test K4: estimator check on simulated brains")
    k.add_argument("--n-perm", type=int, default=100)
    k.add_argument("--n-rep", type=int, default=200)
    k1 = sub.add_parser("k1stim", help="K1 retry: IBL-matched stimulus decoder (explore mice only)")
    k1.add_argument("--n-pseudo", type=int, default=100)
    c = sub.add_parser("confirm", help="THE confirm run (once, after the pre-registration is public)")
    c.add_argument("--force", action="store_true")
    sub.add_parser("evaluate", help="re-print the pre-registered PASS/FAIL table from the confirm results")
    a = ap.parse_args()
    dict(download=download, status=status, explore=explore, k4=k4, k1stim=k1stim, confirm=confirm, evaluate=evaluate)[a.cmd](a)


if __name__ == "__main__":
    main()
