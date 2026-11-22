"""Validate the module 2 + 3 pipeline on simulated data where the answer is known.

    python scripts/validate_synthetic.py [--quick] [--reps 3]

For each scenario (ni_leads, lc_leads, null, noise) and replicate it simulates sessions,
fits the GLM-HMM, runs the PRE-REGISTERED lead-lag decision rule and the lapse forecast,
and checks the verdict matches the truth:
    ni_leads -> NI lead claim SUPPORTED
    lc_leads -> NOT supported (LC leads, NI does not)
    null     -> NOT supported (nobody leads)
    noise    -> NOT supported (NI carries nothing)
Writes results/synthetic_validation/{results.json, SUMMARY.md, figure.png}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper import lapse, leadlag  # noqa: E402
from gatekeeper.decision import EXPECTED, HORIZON, lead_verdict, lapse_verdict  # noqa: E402
from gatekeeper.glmhmm import GLMHMM  # noqa: E402
from gatekeeper.simulate import SimConfig, simulate  # noqa: E402

REGIONS = ["NI", "LC", "pupil"]


def run_one(scenario, seed, n_sessions, n_trials, n_boot):
    S = simulate(SimConfig(scenario=scenario, n_sessions=n_sessions, n_trials=n_trials, seed=seed))
    m = GLMHMM(K=2, n_restarts=3, seed=seed).fit(S)
    z = np.concatenate([s.truth["state"] for s in S])
    p_all = [m.p_engaged(s) for s in S]
    acc = float(((np.concatenate(p_all) > 0.5) == z).mean())
    traces = [leadlag.session_traces(s, p, REGIONS) for s, p in zip(S, p_all)]
    ll = leadlag.group_stats(traces, REGIONS, n_boot=n_boot, seed=seed)
    lp_all, lp_delta = {}, {}
    for h in (HORIZON, 0):
        lp = lapse.evaluate(lapse.build_table(S, m, REGIONS, horizon=h), REGIONS)
        d = {"NI_vs_behavior": lapse.session_bootstrap_delta(lp, "behavior+NI", "behavior", n_boot=n_boot, seed=seed),
             "NI_beyond_LC_pupil": lapse.session_bootstrap_delta(lp, "behavior+all", "behavior+all-NI", n_boot=n_boot, seed=seed)}
        lp_all[h], lp_delta[h] = lp, d
    lp, lp_delta_same, lp_delta = lp_all[HORIZON], lp_delta[0], lp_delta[HORIZON]
    traces_avg = {r: leadlag._pooled(traces, r, range(len(traces))) for r in ["p"] + REGIONS}
    return dict(scenario=scenario, seed=seed, state_accuracy=acc, glm_W=m.W.tolist(),
                leadlag=ll, lead_verdict=lead_verdict(ll),
                lapse={k: dict(auc=v["auc"], log_loss=v["log_loss"], n=v["n"]) for k, v in lp.items()},
                lapse_delta=lp_delta, lapse_verdict=lapse_verdict(lp_delta),
                lapse_same_trial_delta=lp_delta_same, lapse_same_trial_verdict=lapse_verdict(lp_delta_same),
                traces={k: (v.tolist() if v is not None else None) for k, v in traces_avg.items()})


def figure(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scen = list(EXPECTED)
    fig, ax = plt.subplots(1, len(scen), figsize=(4 * len(scen), 3.2), sharey=True)
    k = np.arange(-leadlag.WIN, leadlag.WIN + 1)
    for a, sc in zip(ax, scen):
        r = next(x for x in results if x["scenario"] == sc)
        for name, col in [("p", "k"), ("NI", "C3"), ("LC", "C0"), ("pupil", "C2")]:
            tr = r["traces"].get(name)
            if tr is None:
                continue
            tr = np.array(tr)
            tr = (tr - tr[:5].mean()) / (abs(tr[-5:].mean() - tr[:5].mean()) + 1e-9)
            a.plot(k, tr, col, lw=2 if name in ("p", "NI") else 1.2, label="behavior P(engaged)" if name == "p" else name)
        a.axvline(0, color="grey", ls=":")
        a.set_title(f"{sc}\nverdict: {r['lead_verdict']['verdict']}", fontsize=9)
        a.set_xlabel("trials from behavioral switch")
    ax[0].set_ylabel("normalized change")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default="results/synthetic_validation")
    ap.add_argument("--summarize-only", action="store_true", help="rebuild SUMMARY.md/figure from results.json")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        results = json.loads((out / "results.json").read_text())
        hdr = (out / "SUMMARY.md").read_text().splitlines()[2] if (out / "SUMMARY.md").exists() else ""
        write_summary(results, out, hdr)
        figure(results, out / "figure.png")
        return
    n_sess, n_trials, n_boot = (6, 400, 200) if args.quick else (12, 600, 1000)
    t0 = time.time()
    results = []
    for sc in EXPECTED:
        for rep in range(args.reps):
            r = run_one(sc, 100 + rep, n_sess, n_trials, n_boot)
            results.append(r)
            print(f"{sc:9s} rep {rep}: state acc {r['state_accuracy']:.2f}  lead: {r['lead_verdict']['verdict']:13s}"
                  f" NI onset {r['leadlag']['onset_NI']['est']:+.2f} [{r['leadlag']['onset_NI']['lo']:+.2f},{r['leadlag']['onset_NI']['hi']:+.2f}]"
                  f"  lapse(h={HORIZON}): {r['lapse_verdict']['verdict']}  same-trial: {r['lapse_same_trial_verdict']['verdict']}", flush=True)
    write_summary(results, out, f"{len(results)} runs, {n_sess} sessions x {n_trials} trials each, {time.time() - t0:.0f} s.")


def write_summary(results, out, header):
    lines = ["# Synthetic validation of the module 2 (lead-lag) and module 3 (lapse) decision rules", "",
             header, "",
             "| scenario | rep | state acc | NI onset vs behavior (trials) | NI - LC | NI - pupil | lead verdict | expected | lapse forecast AUC gain, NI beyond LC+pupil, h=5 | lapse verdict | expected |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    lines.insert(3, "Lapse columns are the pre-registered forecast (lapse 5 trials ahead); same-trial results are in results.json.")
    ok_all = True
    for r in results:
        L = r["leadlag"]
        e = EXPECTED[r["scenario"]]
        lv, lpv = r["lead_verdict"]["verdict"], r["lapse_verdict"]["verdict"]
        ok = (lv == e["lead"]) and (e["lapse"] is None or lpv == e["lapse"])
        ok_all &= ok
        f = lambda d: f"{d['est']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"
        lines.append(f"| {r['scenario']} | {r['seed'] - 100} | {r['state_accuracy']:.2f} | {f(L['onset_NI'])} | {f(L['onset_NI-LC'])} | "
                     f"{f(L['onset_NI-pupil'])} | {lv} | {e['lead']} | {f(r['lapse_delta']['NI_beyond_LC_pupil'])} | {lpv} | {e['lapse'] or 'either (not scored)'} |")
    lines += ["", f"**All verdicts match the truth: {ok_all}**"]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    figure(results, out / "figure.png")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
