"""Validate the brain-wide decision rules on simulated brains with known leaders.

    python scripts/validate_brainwide.py [--reps 3] [--quick]

Scenarios
  leaders      3 true leaders (lead 6, 4, 2 trials), 6 concurrent, 2 followers, 8 noise regions
  global_null  no leaders: 8 concurrent, 2 followers, 8 noise regions
Checks (all must hold in every run)
  leaders:     every true leader flagged; no non-leader flagged
  global_null: no region flagged (H1 NOT SUPPORTED)
Reported, not required: ordering of leaders, lapse-forecast results.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper import brainwide  # noqa: E402
from gatekeeper.glmhmm import GLMHMM  # noqa: E402
from gatekeeper.simulate import BrainwideConfig, default_brainwide_regions, global_null_regions, simulate_brainwide  # noqa: E402

SEED_BASE = 300
SCEN = {"leaders": default_brainwide_regions, "global_null": global_null_regions}


def run(scen, seed, n_sessions, n_boot):
    regions = SCEN[scen]()
    cfg = BrainwideConfig(n_sessions=n_sessions, regions=regions, seed=seed)
    S = simulate_brainwide(cfg)
    m = GLMHMM(K=2, n_restarts=2, seed=seed).fit(S)
    res = brainwide.analyze(S, m, sorted(regions), n_boot=n_boot, seed=seed, log=lambda *a: None)
    v = brainwide.verdict(res)
    truth = {r for r, sp in regions.items() if sp["kind"] == "lead"}
    flagged = set(v["H1_some_region_leads"]["leaders"])
    analyzed = {x["region"] for x in res["regions"]}
    ok = (flagged == (truth & analyzed)) if scen == "leaders" else (not flagged)
    lead_rows = [x for x in res["regions"] if x["region"] in truth]
    rho = spearmanr([-regions[x["region"]]["lead"] for x in lead_rows], [x["onset"] for x in lead_rows])[0] if len(lead_rows) >= 3 else np.nan
    return dict(scenario=scen, seed=seed, ok=bool(ok), true_leaders=sorted(truth & analyzed), flagged=sorted(flagged),
                false_positives=sorted(flagged - truth), missed=sorted((truth & analyzed) - flagged),
                leader_order_spearman=float(rho), verdict=v, regions=res["regions"])


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rs = [next(r for r in results if r["scenario"] == s) for s in SCEN]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    col = dict(LEAD="C3", STATE="0.4", FOLLOW="C0", NOISE="0.8")
    for ax, r in zip(axes, rs):
        rows = [x for x in r["regions"] if np.isfinite(x["onset"]) and not x["region"].startswith("NOISE")]
        for i, x in enumerate(rows):
            c = next(v for k, v in col.items() if x["region"].startswith(k))
            ax.errorbar(x["onset"], i, xerr=[[x["onset"] - x["onset_lo"]], [x["onset_hi"] - x["onset"]]], fmt="o",
                        color=c, mfc=c if x.get("leader") else "white")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([x["region"] for x in rows], fontsize=8)
        ax.axvline(0, color="k", ls=":")
        ax.set_title(f"{r['scenario']}: flagged = {', '.join(r['flagged']) or 'none'}", fontsize=9)
        ax.set_xlabel("onset relative to behavioral switch (trials)")
    fig.suptitle("Brain-wide lead map on simulated brains (filled = flagged leader; noise regions omitted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default="results/brainwide_validation")
    ap.add_argument("--only", default=None, help="run a single scenario:rep (for parallel runs), e.g. leaders:1")
    ap.add_argument("--merge", action="store_true", help="merge per-run JSONs in --out into the summary")
    ap.add_argument("--seed-base", type=int, default=400, help="300 = runs used to set LEAD_MARGIN; 400 = fresh")
    a = ap.parse_args()
    global SEED_BASE
    SEED_BASE = a.seed_base
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_sess, n_boot = (16, 200) if a.quick else (30, 1000)
    t0 = time.time()
    results = []
    if a.merge:
        results = [json.loads(p.read_text()) for p in sorted(out.glob("run_*.json"))]
        write_summary(results, out, f"{len(results)} simulated brains, {n_sess} sessions each (parallel runs).")
        return
    jobs = [(a.only.split(":")[0], int(a.only.split(":")[1]))] if a.only else [(s, r) for s in SCEN for r in range(a.reps)]
    for scen, rep in jobs:
        if True:
            r = run(scen, SEED_BASE + rep, n_sess, n_boot)
            r["rep"] = rep
            results.append(r)
            (out / f"run_{scen}_{rep}.json").write_text(json.dumps(r, default=float))
            print(f"{scen:11s} rep {rep}: ok={r['ok']}  flagged={r['flagged']}  missed={r['missed']}  "
                  f"false+={r['false_positives']}  leader-order rho={r['leader_order_spearman']:+.2f}", flush=True)
    if a.only:
        return
    write_summary(results, out, f"{len(results)} simulated brains, {n_sess} sessions each, {time.time()-t0:.0f} s.")


def write_summary(results, out, header):
    lines = ["# Brain-wide validation", "", header, "",
             f"Leader rule: BH q < 0.05 on p_lead, onset < -{brainwide.LEAD_MARGIN:g} trial, asym > 0.", "",
             "| scenario | rep | flagged leaders | missed | false positives | leader order (Spearman vs truth) | H3 lapse regions | pass |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['scenario']} | {r.get('rep', r['seed'] - 300)} (seed {r['seed']}) | {', '.join(r['flagged']) or '-'} | {', '.join(r['missed']) or '-'} | "
                     f"{', '.join(r['false_positives']) or '-'} | {r['leader_order_spearman']:+.2f} | "
                     f"{', '.join(r['verdict']['H3_leaders_forecast_lapses']['regions']) or '-'} | {r['ok']} |")
    lines += ["", f"**All runs pass: {all(r['ok'] for r in results)}**"]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    figure(results, out / "figure.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
