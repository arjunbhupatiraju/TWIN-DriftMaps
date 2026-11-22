"""PRIOR MAP follow-ups.

  python3 part6/prior_followup.py behavior   # B1, PRE-REGISTERED (claude/preregistration-prior-behavior.md):
                                             #   do sessions whose brains carry more prior information show more
                                             #   prior-driven choices?
  python3 part6/prior_followup.py figure     # EXPLORATORY: which regions and brain systems carry the hidden prior,
                                             #   and how many sessions would settle each region

Both read the REGISTERED prior-map run (results_prior/prior_rows.csv, prior_map_regions.csv); --tag amend1 reads the
amended run instead.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_prior"
sys.path.insert(0, str(HERE.parent / "twin"))


def _files(tag):
    s = f"_{tag}" if tag else ""
    return OUT / f"prior_rows{s}.csv", OUT / f"prior_map_regions{s}.csv", s


def _gk(root):
    root = Path(root or os.environ.get("GATEKEEPER_ROOT", "~/Downloads/GATEKEEPER/gatekeeper")).expanduser()
    for p in (root, root / "next"):
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------- B1 behaviour (pre-registered)
def behaviour_weight(s, burn=90):
    """How strongly the mouse's choices follow the Bayes-optimal prior: the prior weight of a logistic choice model
    (stimulus, prior, 4 lags of choice and of rewarded choice), signed so that following the prior is positive."""
    from twin.task import BlockTask, ChoiceModel
    c = np.asarray(s.contrast, float)
    side = np.where(c != 0, (c > 0).astype(float), np.nan)
    prior = BlockTask().bayes_prior(side)
    ch = np.asarray(s.choice, float).copy()
    ch[:burn] = 0                                            # unbiased opening block is not used
    m = ChoiceModel.fit(c, ch, np.asarray(s.feedback, float), prior)
    w_stim, w_prior = m.w[1], m.w[2]
    return float(w_prior * np.sign(w_stim)), int((ch != 0).sum())


def cmd_behavior(a):
    import pandas as pd
    from scipy.stats import rankdata
    _gk(a.gatekeeper)
    from gatekeeper.mv_data import MVSession
    rows_f, _, sfx = _files(a.tag)
    D = pd.read_csv(rows_f)
    D["excess"] = D.score - D.null_mean
    S = D.groupby("session").agg(neural=("excess", "mean"), n_regions=("region", "nunique"), mean_N=("N", "mean"),
                                 T=("T", "first"), lab=("lab", "first"), subject=("subject", "first")).reset_index()
    cache = Path(a.cache).expanduser()
    beh = []
    for sid in S.session:
        try:
            w, n = behaviour_weight(MVSession(cache / f"{sid}.npz"))
        except Exception as e:
            print(f"{sid}: behaviour fit failed ({e})"); w, n = np.nan, 0
        beh.append((w, n))
    S["behaviour"] = [b[0] for b in beh]; S["n_choices"] = [b[1] for b in beh]
    S = S[np.isfinite(S.behaviour)].reset_index(drop=True)

    def resid(v, Z):
        rv = rankdata(v); Zr = np.column_stack([np.ones(len(v))] + [rankdata(z) for z in Z.T])
        return rv - Zr @ np.linalg.lstsq(Zr, rv, rcond=None)[0]

    Z = S[["n_regions", "mean_N", "T"]].values.astype(float)
    rn = resid(S.neural.values, Z)

    def rho(b):
        rb = resid(b, Z)
        return float(np.corrcoef(rn, rb)[0, 1])

    obs = rho(S.behaviour.values)
    rng = np.random.default_rng(0)
    labs = S.lab.values; null = []
    for _ in range(a.perm):
        b = S.behaviour.values.copy()
        for lab in np.unique(labs):                        # shuffle behaviour across sessions WITHIN each lab
            i = np.nonzero(labs == lab)[0]; b[i] = b[rng.permutation(i)]
        null.append(rho(b))
    p = float((1 + np.sum(np.array(null) >= obs)) / (1 + a.perm))
    # secondary (descriptive): within mouse, sessions of the same mouse
    within = []
    for _, g in S.groupby("subject"):
        if len(g) >= 3 and g.behaviour.std() > 0 and g.neural.std() > 0:
            within.append(np.corrcoef(rankdata(g.neural), rankdata(g.behaviour))[0, 1])
    B1 = obs > 0 and p < 0.05
    L = [f"# B1: neural prior information vs behavioural prior use ({len(S)} sessions, {S.subject.nunique()} mice, "
         f"{S.lab.nunique()} labs){' [' + a.tag + ']' if a.tag else ''}",
         f"  partial Spearman rho (controlling regions, units, trials) = {obs:+.3f}; within-lab permutation p = {p:.4f}",
         f"  B1 {'PASS' if B1 else 'FAIL'} (criterion: rho > 0 and one-sided p < 0.05)",
         f"  descriptive: within-mouse rank correlation, {len(within)} mice with >= 3 sessions: median "
         f"{np.median(within) if within else float('nan'):+.3f}, positive in {np.mean(np.array(within) > 0) if within else float('nan'):.0%}",
         f"  behavioural prior weight: median {S.behaviour.median():.2f}, positive in {np.mean(S.behaviour > 0):.0%} of sessions"]
    print("\n".join(L))
    S.to_csv(OUT / f"prior_behaviour_sessions{sfx}.csv", index=False)
    (OUT / f"PRIOR_BEHAVIOUR{sfx}.md").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- figure (exploratory)
def _fisher(ps):
    from scipy.stats import chi2
    ps = np.clip(np.asarray(ps, float), 1e-12, 1)
    return chi2.sf(-2 * np.log(ps).sum(axis=-1), 2 * ps.shape[-1])


def sessions_needed(pvals, target=0.8, s_max=200, B=400, rng=None):
    rng = rng or np.random.default_rng(0)
    pvals = np.asarray(pvals, float)
    for S in list(range(1, 21)) + list(range(25, s_max + 1, 5)):
        if np.mean(_fisher(pvals[rng.integers(0, len(pvals), (B, S))]) < 0.05) >= target:
            return S
    return np.inf


def _systems(acr):
    try:
        from iblatlas.regions import BrainRegions
        return list(BrainRegions().acronym2acronym(np.asarray(acr), mapping="Cosmos"))
    except Exception as e:
        print(f"(iblatlas not available for brain-system labels: {e}; install with: pip3 install iblatlas)")
        return ["all"] * len(acr)


def cmd_figure(a):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_f, reg_f, sfx = _files(a.tag)
    D = pd.read_csv(rows_f); R = pd.read_csv(reg_f, index_col=0)
    rng = np.random.default_rng(0)
    R["category"] = np.select([R.fdr_valid.astype(bool), R.p_carries > 0.5, R.p_carries >= 0.2],
                              ["detected", "hidden (likely carries it)", "unanswerable"], "likely absent")
    need = {}
    for reg, g in D.groupby("region"):
        pp = np.concatenate([json.loads(c)["0.06"] for c in g.power_curve])
        need[reg] = sessions_needed(pp, rng=rng)
    R["sessions_needed_r0.06"] = pd.Series(need)
    R["more_sessions_needed"] = (R["sessions_needed_r0.06"] - R.n_sess).clip(lower=0)
    R["system"] = _systems(R.index.values)
    R.to_csv(OUT / f"prior_map_figure_table{sfx}.csv")

    cats = ["detected", "hidden (likely carries it)", "unanswerable", "likely absent"]
    col = {"detected": "#0d6b73", "hidden (likely carries it)": "#7fc4c9", "unanswerable": "#c9c9c9", "likely absent": "#e7d6cc"}
    tab = R.groupby(["system", "category"]).size().unstack(fill_value=0).reindex(columns=cats, fill_value=0)
    tab = tab.loc[tab.sum(1).sort_values().index]
    fig = plt.figure(figsize=(13, 4.6))
    ax1 = fig.add_subplot(1, 3, 1)
    left = np.zeros(len(tab))
    for c in cats:
        ax1.barh(tab.index, tab[c], left=left, color=col[c], label=c); left += tab[c].values
    ax1.set_xlabel("regions"); ax1.set_title("Where the prior is, by brain system", fontsize=10)
    ax1.legend(fontsize=7, frameon=False, loc="lower right")
    ax2 = fig.add_subplot(1, 3, 2)
    Rs = R.sort_values("p_carries")
    ax2.scatter(np.arange(len(Rs)), Rs.p_carries, s=9, c=[col[c] for c in Rs.category])
    ax2.axhline(0.5, ls="--", lw=0.8, c="k"); ax2.set_xlabel("regions (sorted)"); ax2.set_ylabel("P(region carries the prior)")
    ax2.set_title(f"Detected {int((R.category == 'detected').sum())}, hidden {int((R.category == cats[1]).sum())}, "
                  f"unanswerable {int((R.category == cats[2]).sum())}", fontsize=10)
    ax3 = fig.add_subplot(1, 3, 3)
    fin = np.isfinite(R["sessions_needed_r0.06"])
    ax3.scatter(R.n_sess[fin], R["sessions_needed_r0.06"][fin], s=9, c=[col[c] for c in R.category[fin]])
    mx = float(np.nanmax(R["sessions_needed_r0.06"][fin])) if fin.any() else 10
    ax3.plot([0, mx], [0, mx], "k--", lw=0.8)
    ax3.set_xlabel("sessions recorded"); ax3.set_ylabel("sessions needed (80% power, r = 0.06)")
    ax3.set_title("What would settle each region", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"prior_map_figure{sfx}.{ext}", dpi=200)
    try:                                                   # optional Swanson flatmap of P(carries the prior)
        from iblatlas.plots import plot_swanson_vector
        f2, ax = plt.subplots(figsize=(8, 5))
        plot_swanson_vector(R.index.values, R.p_carries.values, ax=ax, cmap="viridis", vmin=0, vmax=1, br=None)
        ax.set_axis_off(); ax.set_title("P(region carries the block prior)")
        f2.savefig(OUT / f"prior_map_swanson{sfx}.png", dpi=200)
    except Exception as e:
        print(f"(Swanson flatmap skipped: {e})")
    print(R.category.value_counts().to_string())
    print("\nBy brain system:\n" + tab.to_string())
    H = R[R.category == cats[1]].sort_values("p_carries", ascending=False)
    print(f"\nTop hidden regions: {', '.join(H.index[:25])}")
    print(f"Median extra sessions needed to settle an unanswerable region (r = 0.06): "
          f"{np.nanmedian(R.loc[R.category == cats[2], 'more_sessions_needed'].replace(np.inf, np.nan)):.0f}")
    print(f"Wrote {OUT}/prior_map_figure{sfx}.png and prior_map_figure_table{sfx}.csv")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("behavior"); b.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
    b.add_argument("--gatekeeper"); b.add_argument("--perm", type=int, default=5000)
    f = sub.add_parser("figure")
    for q in (b, f):
        q.add_argument("--tag", default="")
    a = ap.parse_args()
    {"behavior": cmd_behavior, "figure": cmd_figure}[a.cmd](a)


if __name__ == "__main__":
    main()
