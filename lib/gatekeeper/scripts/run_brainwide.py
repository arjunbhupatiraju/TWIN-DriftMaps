"""Brain-wide engagement-switch analysis on IBL data.

    python scripts/run_brainwide.py select     # choose sessions from the coverage results (offline, seconds)
    python scripts/run_brainwide.py download   # cache them (internet; long; resumable; deletes raw spikes)
    python scripts/run_brainwide.py analyze    # offline; writes results/brainwide/

Requires results/brainwide_coverage/ from scripts/check_brainwide_coverage.py.
The region list is fixed by the coverage step (regions.csv, analyzable == True) and the
pre-registration (docs/PREREGISTRATION_BRAINWIDE.md); it is not chosen after seeing activity.
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
from gatekeeper import brainwide  # noqa: E402

COV = Path("results/brainwide_coverage")
CACHE = Path("data/bwm_cache")
SEL = COV / "selected_sessions.json"
MIN_UNITS, MIN_SESSIONS, MIN_TRIALS = 5, 5, 300


def analyzable_regions():
    import pandas as pd
    reg = pd.read_csv(COV / "regions.csv")
    return sorted(reg[reg.analyzable].region.astype(str))


def select(args):
    """Greedy: add sessions until every analyzable region has >= target sessions with >= 5 good units."""
    import pandas as pd
    df = pd.read_csv(COV / "units_by_insertion.csv")
    regs = set(analyzable_regions())
    ps = df[df.region.isin(regs)].groupby(["eid", "region"]).good_units.sum().reset_index()
    ps = ps[ps.good_units >= MIN_UNITS]
    by_sess = ps.groupby("eid").region.apply(set).to_dict()
    need = {r: args.per_region for r in regs}
    avail = ps.groupby("region").eid.nunique().to_dict()
    for r in regs:
        need[r] = min(need[r], avail.get(r, 0))
    chosen = []
    while any(v > 0 for v in need.values()) and by_sess:
        best = max(by_sess, key=lambda e: sum(1 for r in by_sess[e] if need.get(r, 0) > 0))
        gain = sum(1 for r in by_sess[best] if need.get(r, 0) > 0)
        if gain == 0:
            break
        chosen.append(best)
        for r in by_sess.pop(best):
            if need.get(r, 0) > 0:
                need[r] -= 1
        if args.max_sessions and len(chosen) >= args.max_sessions:
            break
    SEL.write_text(json.dumps(dict(sessions=chosen, per_region=args.per_region, regions=sorted(regs)), indent=1))
    print(f"{len(regs)} analyzable regions; selected {len(chosen)} sessions "
          f"(target {args.per_region} sessions per region). Saved {SEL}")
    print(f"Rough download size: {len(chosen)} sessions x ~0.5-1 GB of spikes each, deleted after caching.")


def download(args):
    from check_ibl_coverage import connect
    from gatekeeper.ibl import load_session_brainwide
    sel = json.loads(SEL.read_text())["sessions"]
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [e for e in sel if not (CACHE / f"{e}.npz").exists()]
    print(f"{len(sel)} selected; {len(sel) - len(todo)} already cached; {len(todo)} to download", flush=True)
    one = connect()
    fails = {}
    t0 = time.time()
    for i, eid in enumerate(todo):
        try:
            s = load_session_brainwide(one, eid, CACHE, delete_raw=not args.keep_raw)
            nreg = len([k for k in s.signals if k != "pupil"])
            el = (time.time() - t0) / 60
            print(f"[{i+1}/{len(todo)}] {eid}: {s.T} trials, {nreg} regions, pupil={'pupil' in s.signals} "
                  f"({el:.0f} min elapsed, ~{el / (i + 1) * (len(todo) - i - 1):.0f} min left)", flush=True)
        except Exception as e:
            fails[eid] = repr(e)[:300]
            print(f"[{i+1}/{len(todo)}] {eid}: FAILED {e!r:.200}", flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/brainwide_download_failures.json").write_text(json.dumps(fails, indent=1))
    print(f"done; {len(fails)} failures (re-run the same command to retry them)")


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [x for x in res["regions"] if np.isfinite(x["onset"]) and np.isfinite(x["onset_lo"])]
    rows = sorted(rows, key=lambda x: x["onset"])
    fig, ax = plt.subplots(figsize=(7, max(4, 0.22 * len(rows))))
    for i, x in enumerate(rows):
        c = "C3" if x.get("leader") else "0.5"
        ax.errorbar(x["onset"], i, xerr=[[max(0.0, x["onset"] - x["onset_lo"])], [max(0.0, x["onset_hi"] - x["onset"])]], fmt="o",
                    color=c, mfc=c if x.get("leader") else "white", ms=4, lw=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([x["region"] for x in rows], fontsize=7)
    ax.axvline(0, color="k", ls=":")
    ax.set_xlabel("change time relative to behavioral engagement switch (trials; < 0 = before)")
    ax.set_title("Brain-wide engagement-switch timing (red = pre-registered leader, FDR 5%)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def analyze(args):
    from gatekeeper.glmhmm import GLMHMM
    from gatekeeper.ibl import load_cached
    out = Path(args.out or ("results/brainwide_all_units_sensitivity" if args.all_units else "results/brainwide"))
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = [s for s in load_cached(CACHE, all_units=args.all_units) if (s.choice >= 0).sum() >= MIN_TRIALS]
    regions = analyzable_regions()
    print(f"{len(S)} sessions, {len(regions)} pre-registered regions. Fitting GLM-HMM...", flush=True)
    m = GLMHMM(K=2, n_restarts=args.restarts, seed=0).fit(S)
    print(f"GLM-HMM done ({(time.time()-t0)/60:.1f} min). Engaged-state stimulus weight {m.W[0,0]:.2f}, "
          f"disengaged {m.W[1,0]:.2f}. Analyzing regions...", flush=True)
    res = brainwide.analyze(S, m, regions, min_units=MIN_UNITS, min_sessions=MIN_SESSIONS, n_boot=args.n_boot, seed=0)
    v = brainwide.verdict(res)
    import pandas as pd
    pd.DataFrame(res["regions"]).to_csv(out / "regions.csv", index=False)
    (out / "results.json").write_text(json.dumps(dict(verdict=v, glmhmm=dict(W=m.W.tolist(), A=m.A.tolist()),
                                                      n_sessions=len(S), **res), indent=1, default=float))
    figure(res, out / "lead_map.png")
    print(json.dumps(v, indent=1, default=float))
    print(f"\nwrote {out}/results.json, regions.csv, lead_map.png ({(time.time()-t0)/60:.1f} min)")


def redraw(args):
    """Rebuild lead_map.png from results.json (no re-analysis)."""
    out = Path(args.out or "results/brainwide")
    res = json.loads((out / "results.json").read_text())
    figure(res, out / "lead_map.png")
    v = res.get("verdict") or brainwide.verdict(res)
    print(json.dumps(v, indent=1, default=float))
    print(f"wrote {out}/lead_map.png")


def diagnose(args):
    """EXPLORATORY (not pre-registered). Is there any engagement signal around switches at all?
    Uses the GLM-HMM saved in results.json (no refit). Per region:
      held-out separation d': state axis fit on the first half of each session's confident trials,
      tested on the second half (engaged vs disengaged), averaged over sessions;
      pooled switch-triggered averages (saved) for the regions with the largest d'.
    Plus behavior: fraction engaged, switches per session, run lengths."""
    from gatekeeper import leadlag
    from gatekeeper.glmhmm import GLMHMM
    from gatekeeper.ibl import load_cached
    out = Path("results/brainwide_diagnostics")
    out.mkdir(parents=True, exist_ok=True)
    res = json.loads(Path("results/brainwide/results.json").read_text())
    m = GLMHMM(K=2)
    m.W, m.A = np.array(res["glmhmm"]["W"]), np.array(res["glmhmm"]["A"])
    m.pi = np.array([0.5, 0.5])
    S = [s for s in load_cached(CACHE) if (s.choice >= 0).sum() >= MIN_TRIALS]
    regions = analyzable_regions()
    post = [m.p_engaged(s) for s in S]
    beh = []
    for s, p in zip(S, post):
        z = (p > 0.5).astype(int)
        sw = leadlag.find_switches(p)
        runs = np.diff(np.flatnonzero(np.r_[1, np.diff(z) != 0, 1]))
        beh.append(dict(frac_engaged=float(z.mean()), raw_switches=int((np.diff(z) != 0).sum()),
                        clean_switches=len(sw), median_run=float(np.median(runs)),
                        confident_frac=float(((p > .9) | (p < .1)).mean())))
    B = {k: float(np.median([b[k] for b in beh])) for k in beh[0]}
    B["sessions_with_0_clean_switches"] = int(sum(b["clean_switches"] == 0 for b in beh))
    print("behavior (median over sessions):", json.dumps(B))
    rows, avgs = [], {}
    for r in regions:
        ds, tr = [], []
        for s, p in zip(S, post):
            if r not in s.signals or s.signals[r].shape[1] < MIN_UNITS:
                continue
            X = s.signals[r]
            Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
            half = np.arange(len(p)) >= len(p) // 2
            e, d = p > 0.9, p < 0.1
            a, b = ~half, half
            if min((e & a).sum(), (d & a).sum(), (e & b).sum(), (d & b).sum()) < 15:
                continue
            w = Z[e & a].mean(0) - Z[d & a].mean(0)
            w /= np.linalg.norm(w) + 1e-9
            x = Z @ w
            ds.append((x[e & b].mean() - x[d & b].mean()) / (np.sqrt(0.5 * (x[e & b].var() + x[d & b].var())) + 1e-9))
            sw = leadlag.find_switches(p)
            ax_ = leadlag.region_axis_signal(X, p, sw)
            if ax_ is not None and sw:
                tr.append(leadlag.triggered(ax_, sw))
        if not ds:
            continue
        rows.append(dict(region=r, n_sessions=len(ds), dprime_heldout=float(np.mean(ds)),
                         dprime_se=float(np.std(ds) / np.sqrt(len(ds))), n_switches=int(sum(len(t) for t in tr))))
        if tr:
            avgs[r] = np.concatenate(tr).mean(0).tolist()
    rows.sort(key=lambda x: -x["dprime_heldout"])
    pavg = [leadlag.triggered(p, leadlag.find_switches(p)) for p in post]
    pavg = np.concatenate([a for a in pavg if len(a)]).mean(0).tolist()
    (out / "diagnostics.json").write_text(json.dumps(dict(behavior=B, per_session_behavior=beh, regions=rows,
                                                          triggered=avgs, behavior_triggered=pavg), indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = np.arange(-leadlag.WIN, leadlag.WIN + 1)
    top = [x["region"] for x in rows if x["region"] in avgs][:12]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    axs[0].bar(range(len(rows)), [x["dprime_heldout"] for x in rows], color="0.5")
    axs[0].set_xticks(range(len(rows)))
    axs[0].set_xticklabels([x["region"] for x in rows], rotation=90, fontsize=5)
    axs[0].set_ylabel("held-out engaged vs disengaged separation (d')")
    axs[0].set_title("Does each region carry engagement state at all? (exploratory)", fontsize=9)
    axs[1].plot(k, (np.array(pavg) - np.mean(pavg[:5])) / (np.ptp(pavg) + 1e-9), "k", lw=2.5, label="behavior P(engaged)")
    for r in top:
        a = np.array(avgs[r])
        axs[1].plot(k, (a - a[:5].mean()) / (abs(a[-5:].mean() - a[:5].mean()) + 1e-9), lw=1, label=r)
    axs[1].axvline(0, color="grey", ls=":")
    axs[1].set_ylim(-1.5, 2.5)
    axs[1].set_xlabel("trials from behavioral switch")
    axs[1].set_title("Switch-triggered activity, 12 most state-modulated regions", fontsize=9)
    axs[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png", dpi=140)
    print("top regions by held-out d':", [(x["region"], round(x["dprime_heldout"], 2)) for x in rows[:15]])
    print(f"wrote {out}/diagnostics.json and diagnostics.png")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--per-region", type=int, default=10)
    s.add_argument("--max-sessions", type=int, default=0)
    d = sub.add_parser("download")
    d.add_argument("--keep-raw", action="store_true")
    a = sub.add_parser("analyze")
    a.add_argument("--out", default=None)
    a.add_argument("--all-units", action="store_true")
    a.add_argument("--n-boot", type=int, default=2000)
    a.add_argument("--restarts", type=int, default=5)
    f = sub.add_parser("figure")
    f.add_argument("--out", default=None)
    sub.add_parser("diagnose")
    args = ap.parse_args()
    dict(select=select, download=download, analyze=analyze, figure=redraw, diagnose=diagnose)[args.cmd](args)


if __name__ == "__main__":
    main()
