"""Validate the INFOFLOW lapse-locus rule on simulated pathways with a known locus.

    python scripts/validate_infoflow.py [--reps 3] [--only scenario:rep] [--merge]
Truth: source -> stage 1; downstream -> stage 4 (stages 1-3 intact); decision -> no neural locus.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper import infoflow  # noqa: E402
from gatekeeper.glmhmm import GLMHMM  # noqa: E402
from gatekeeper.infoflow_sim import SCENARIOS, PathwayConfig, simulate_pathway  # noqa: E402

TRUTH = {"source": "1 visual thalamus", "downstream": "4 midbrain", "decision": None}


def run(sc, seed, n_sessions=40):
    S = simulate_pathway(PathwayConfig(scenario=sc, n_sessions=n_sessions, p_present=0.6, seed=seed))
    beh = [s.as_behavior_session() for s in S]
    m = GLMHMM(K=2, n_restarts=2, seed=seed).fit(beh)
    rows = []
    for i, (s, b) in enumerate(zip(S, beh)):
        post = m.posteriors(b)["loo"][:, 0]
        for r in s.regions:
            x = infoflow.session_measures(s, r, post=post, seed=i, n_rep=1)
            if x:
                rows.append(x)
    L = infoflow.lapse_locus(rows, n_boot=2000, seed=seed)
    t = TRUTH[sc]
    V = {st: v["verdict"] for st, v in L["stages"].items()}
    # SAFETY (must hold): no truly intact stage is called LOSSY, and no truly lossy stage is called INTACT
    false_loss = [st for st, v in V.items() if v == "LOSSY" and (t is None or st < t)]
    false_intact = [st for st, v in V.items() if v == "INTACT" and t is not None and st >= t]
    safe = not false_loss and not false_intact
    # POWER (reported): was the true locus located exactly?
    exact = (L["locus"] == t) if t else (L["locus"] is None)
    return dict(scenario=sc, seed=seed, locus=L["locus"], locus_kind=L["locus_kind"], truth=t, ok=bool(safe),
                exact=bool(exact), false_loss=false_loss, false_intact=false_intact,
                rho_behavior=L["rho_behavior"], stages=L["stages"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default=None)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--out", default="results/infoflow_validation")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.merge:
        jobs = [(a.only.split(":")[0], int(a.only.split(":")[1]))] if a.only else [(s, r) for s in SCENARIOS for r in range(a.reps)]
        for sc, rep in jobs:
            r = run(sc, 500 + rep)
            r["rep"] = rep
            (out / f"run_{sc}_{rep}.json").write_text(json.dumps(r, default=float))
            print(f"{sc:10s} rep {rep}: locus={r['locus']} ({r['locus_kind']})  truth={r['truth']}  safe={r['ok']}  exact={r['exact']}  "
                  f"false_loss={r['false_loss']} false_intact={r['false_intact']}  behavior ratio={r['rho_behavior']:.2f}", flush=True)
        if a.only:
            return
    R = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
    lines = ["# INFOFLOW lapse-locus validation", "",
             "Rule: rho = mean d'(disengaged) / mean d'(engaged) per stage, held-out scores of one decoder per session,",
             "leave-one-out GLM-HMM states, session bootstrap, Bonferroni across stages; locus = first stage whose CI is below 1.", "",
             "Verdicts per stage: LOSSY (CI below 1), INTACT (CI contains 1 and excludes a >40% loss), UNRESOLVED.",
             "SAFE = no false LOSSY upstream of the truth and no false INTACT at/after it (required). EXACT = locus found exactly (power).", "",
             "| scenario | rep (seed) | locus found | truth | per-stage verdicts | behavior lapse ratio | safe | exact |",
             "|---|---|---|---|---|---|---|---|"]
    for r in R:
        v = " ".join(f"{st[0]}:{x['verdict'][0]}" for st, x in r["stages"].items())
        lines.append(f"| {r['scenario']} | {r['rep']} ({r['seed']}) | {r['locus'] or 'none'} | {r['truth'] or 'none'} | {v} | "
                     f"{r['rho_behavior']:.2f} | {r['ok']} | {r['exact']} |")
    lines += ["", f"**Safe in all runs: {all(r['ok'] for r in R)}** ({sum(r['ok'] for r in R)}/{len(R)}); "
              f"exact locus in {sum(r['exact'] for r in R)}/{len(R)}."]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
