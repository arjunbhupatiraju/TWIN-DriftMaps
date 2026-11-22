"""Modules 1-3 on real IBL data. Run ONLY after the coverage check says GO/BORDERLINE and the
pre-registration (docs/PREREGISTRATION.md) is committed.

    # step 1: download + cache the NI sessions found by the coverage check (needs internet)
    python scripts/run_ibl_analysis.py download --coverage results/coverage/coverage_units.csv
    # step 2: analyze from cache (no internet)
    python scripts/run_ibl_analysis.py analyze

Analysis set: sessions with >= MIN_NI good NI units and >= 300 trials with a valid choice.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper import characterize, lapse, leadlag  # noqa: E402
from gatekeeper.decision import HORIZON, lapse_verdict, lead_verdict  # noqa: E402
from gatekeeper.glmhmm import GLMHMM  # noqa: E402

MIN_NI = 1          # amended 2026-10-10 from 3 using unit COUNTS only (24 good units / 5 sessions)
MIN_TRIALS = 300
MIN_LC_SESSIONS = 5
CACHE = Path("data/ibl_cache")


def download(args):
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from check_ibl_coverage import connect
    from gatekeeper.ibl import load_session
    CACHE.mkdir(parents=True, exist_ok=True)
    u = pd.read_csv(args.coverage)
    eids = sorted(set(u[(u.target == "NI") & (u.units > 0)].eid))
    if args.include_lc:
        eids = sorted(set(eids) | set(u[(u.target == "LC") & (u.units > 0)].eid))
    print(f"{len(eids)} sessions to cache")
    one = connect()
    fails = {}
    for i, eid in enumerate(eids):
        try:
            s = load_session(one, eid, CACHE)
            print(f"[{i+1}/{len(eids)}] {eid}: {s.T} trials, units {s.meta.get('units')}, pupil {s.meta.get('pupil_source')}", flush=True)
        except Exception as e:
            fails[eid] = repr(e)[:300]
            print(f"[{i+1}/{len(eids)}] {eid}: FAILED {e!r}", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/download_failures.json").write_text(json.dumps(fails, indent=1))


def analyze(args):
    from gatekeeper.ibl import load_cached
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    allS = load_cached(CACHE, all_units=args.all_units)
    if args.all_units:
        print("SENSITIVITY RUN: all sorted units (secondary analysis; cannot upgrade a primary NOT SUPPORTED)")
    S = [s for s in allS if s.signals.get("NI", np.zeros((0, 0))).shape[1] >= MIN_NI and (s.choice >= 0).sum() >= MIN_TRIALS]
    print(f"{len(S)} of {len(allS)} cached sessions meet inclusion (>= {MIN_NI} NI units, >= {MIN_TRIALS} trials)")
    if len(S) < 3:
        print("Too few sessions for modules 2-3. Report module 1 descriptives only / switch to fallback.")
    R = dict(n_sessions=len(S), sessions=[s.sid for s in S])
    # GLM-HMM on ALL cached sessions (more behavior = better state model; neural data unused here)
    behS = [s for s in allS if (s.choice >= 0).sum() >= MIN_TRIALS]
    m = GLMHMM(K=2, n_restarts=10, seed=0).fit(behS)
    m3 = GLMHMM(K=3, n_restarts=10, seed=0).fit(behS)
    R["glmhmm"] = dict(W=m.W.tolist(), A=m.A.tolist(), n_behavior_sessions=len(behS),
                       K3_W=m3.W.tolist(), loglik_K2=m.ll, loglik_K3=m3.ll)
    post = [m.p_engaged(s) for s in S]
    # module 1
    rows = characterize.unit_modulation(S, post, "NI")
    R["module1_NI_units"] = characterize.summarize(rows)
    (out / "module1_units.json").write_text(json.dumps(rows, indent=1))
    # module 2
    have_lc = sum(1 for s in S if s.signals.get("LC", np.zeros((0, 0))).shape[1] >= MIN_NI) >= MIN_LC_SESSIONS
    regions = ["NI", "pupil"] + (["LC"] if have_lc else [])
    traces = [leadlag.session_traces(s, p, regions) for s, p in zip(S, post)]
    ll = leadlag.group_stats(traces, regions, n_boot=5000, seed=0)
    if not have_lc:
        ll["onset_NI-LC"] = dict(est=np.nan, lo=np.nan, hi=np.nan)
    R["module2_leadlag"] = ll
    R["module2_verdict"] = lead_verdict(ll, have_lc=have_lc)
    # module 3
    lp_regions = [r for r in regions if all(r in s.signals for s in S)]
    if "NI" in lp_regions and len(lp_regions) > 1:
        for h in (HORIZON, 0):
            lp = lapse.evaluate(lapse.build_table(S, m, lp_regions, horizon=h), lp_regions)
            delta = {"NI_vs_behavior": lapse.session_bootstrap_delta(lp, "behavior+NI", "behavior", n_boot=5000),
                     "NI_beyond_LC_pupil": lapse.session_bootstrap_delta(lp, "behavior+all", "behavior+all-NI", n_boot=5000)}
            tag = "" if h == HORIZON else "_same_trial_secondary"
            R[f"module3_lapse{tag}"] = {k: dict(auc=v["auc"], log_loss=v["log_loss"], n=v["n"]) for k, v in lp.items()}
            R[f"module3_delta{tag}"] = delta
            R[f"module3_verdict{tag}"] = lapse_verdict(delta)
    R["runtime_min"] = (time.time() - t0) / 60
    (out / "results.json").write_text(json.dumps(R, indent=1, default=float))
    print(json.dumps({k: R[k] for k in R if "verdict" in k or k.startswith("module1")}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--coverage", default="results/coverage/coverage_units.csv")
    d.add_argument("--include-lc", action="store_true", help="also cache LC-only sessions")
    a = sub.add_parser("analyze")
    a.add_argument("--out", default=None)
    a.add_argument("--all-units", action="store_true", help="pre-registered sensitivity analysis")
    args = ap.parse_args()
    if args.cmd == "analyze" and args.out is None:
        args.out = "results/ibl_all_units_sensitivity" if args.all_units else "results/ibl"
    download(args) if args.cmd == "download" else analyze(args)


if __name__ == "__main__":
    main()
