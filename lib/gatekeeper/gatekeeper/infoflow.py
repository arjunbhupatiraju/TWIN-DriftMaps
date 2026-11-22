"""INFOFLOW analysis: how much stimulus information each region carries, how it scales with the
number of simultaneously recorded neurons, how it compares with behavior on the SAME trials,
and what disengagement does to it, stage by stage from thalamus to motor output.

Core measures (per session x region, all cross-validated):
  d'_neural(N)   stimulus-side discriminability from N simultaneously recorded neurons
                 (shrinkage LDA, 5-fold stratified CV, averaged over random neuron subsets).
                 Trials: low contrasts (6.25-25%), valid choice, no movement before the window ends.
  d'_within      the same, computed within left-choice and within right-choice trials and
                 averaged: stimulus information that is NOT just the upcoming choice.
  d'_behavior    z(P(right | stim right)) - z(P(right | stim left)) on the same trials.
  ceiling        fit I(N) = Imax * N / (N + K) to d'^2 averaged across sessions (pooled curve per region).
  lapse ratio    rho_region = d'_disengaged / d'_engaged of held-out scores from ONE decoder per
                 session, versus the same ratio for behavior. States come from the LEAVE-ONE-OUT
                 GLM-HMM posterior (each trial's own choice is never used to label it; using it
                 sorts errors into 'disengaged' and fakes downstream loss -- found in simulation).
                 rho_region ~ 1 while rho_behavior << 1 => information intact in that region.
                 Primary: all-trial d'. Within-choice d' is secondary only: conditioning on choice
                 is biased when choice-stimulus coupling differs between states (also found in simulation).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

LOW_CONTRASTS = (0.0625, 0.125, 0.25)

STAGES = {
    "1 visual thalamus": ["LGd", "LP"],
    "2 primary visual cortex": ["VISp"],
    "3 higher visual cortex": ["VISl", "VISal", "VISrl", "VISa", "VISam", "VISpm", "VISli", "VISpor"],
    "4 midbrain": ["SCs", "SCm", "APN", "MRN"],
    "5 basal ganglia": ["CP", "ACB", "GPe", "SNr"],
    "6 frontal cortex": ["MOs", "ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl", "FRP"],
    "7 motor output": ["MOp", "GRN", "IRN", "PRNr", "PARN", "RN"],
}
REGION_STAGE = {r: s for s, rs in STAGES.items() for r in rs}


# ----------------------------------------------------------------------------- core estimators
def cv_dprime(X, y, n_splits=5, seed=0):
    """Cross-validated d' of a shrinkage-LDA readout. NaN if too few trials per class."""
    y = np.asarray(y).astype(int)
    if X.ndim == 1:
        X = X[:, None]
    n0, n1 = (y == 0).sum(), (y == 1).sum()
    k = min(n_splits, n0, n1)
    if k < 2 or X.shape[1] == 0:
        return np.nan
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    scores = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xs, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xs[tr], y[tr])
        scores[te] = m.decision_function(Xs[te])
    a, b = scores[y == 1], scores[y == 0]
    sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan


def cv_scores(X, y, n_splits=5, seed=0):
    """Held-out decision scores of a shrinkage-LDA stimulus readout (one decoder per session)."""
    y = np.asarray(y).astype(int)
    k = min(n_splits, (y == 0).sum(), (y == 1).sum())
    if k < 2:
        return None
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    sc = np.zeros(len(y))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Xs, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(Xs[tr], y[tr])
        sc[te] = m.decision_function(Xs[te])
    return sc


def score_dprime(sc, y):
    a, b = sc[y == 1], sc[y == 0]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    sd = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    return float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan


def within_choice_score_dprime(sc, y, choice, min_per_cell=4):
    """d' of held-out scores computed inside each choice group, averaged (choice cannot fake it)."""
    ds, ws = [], []
    for ch in (0, 1):
        m = choice == ch
        if min((y[m] == 0).sum(), (y[m] == 1).sum()) < min_per_cell:
            continue
        d = score_dprime(sc[m], y[m])
        if np.isfinite(d):
            ds.append(d)
            ws.append(m.sum())
    return float(np.average(ds, weights=ws)) if ds else np.nan


def behavior_dprime(side, choice):
    pr = lambda m: np.clip(np.mean(choice[m] == 1), 0.5 / m.sum(), 1 - 0.5 / m.sum()) if m.sum() else np.nan
    return float(norm.ppf(pr(side == 1)) - norm.ppf(pr(side == 0)))


def trial_mask(contrast, choice, rt, window_end=0.13, contrasts=LOW_CONTRASTS):
    c = np.abs(contrast)
    m = np.isin(np.round(c, 4), np.round(contrasts, 4)) & (choice >= 0)
    if rt is not None:
        m &= np.isfinite(rt) & (rt > window_end)
    return m


def within_choice_dprime(X, side, choice, seed=0):
    ds, ws = [], []
    for ch in (0, 1):
        m = choice == ch
        if m.sum() < 10:
            continue
        d = cv_dprime(X[m], side[m], seed=seed)
        if np.isfinite(d):
            ds.append(d)
            ws.append(m.sum())
    return float(np.average(ds, weights=ws)) if ds else np.nan


def scaling_curve(X, side, n_grid=None, n_rep=10, seed=0):
    """Mean cross-validated d'^2 vs number of neurons (random subsets)."""
    rng = np.random.default_rng(seed)
    N = X.shape[1]
    if n_grid is None:
        n_grid = [n for n in (1, 2, 4, 8, 16, 32, 64, 128) if n <= N] + ([N] if N not in (1, 2, 4, 8, 16, 32, 64, 128) else [])
    out = {}
    for n in n_grid:
        vals = []
        for r in range(n_rep if n < N else 1):
            idx = rng.choice(N, n, replace=False)
            d = cv_dprime(X[:, idx], side, seed=seed + r)
            if np.isfinite(d):
                vals.append(np.sign(d) * d ** 2)
        if vals:
            out[int(n)] = float(np.mean(vals))
    return out


def _sat(n, imax, k):
    return imax * n / (n + k)


def fit_ceiling(curve):
    """Fit Imax*N/(N+K) to {N: mean d'^2}. Returns dict(imax, k, saturating: bool)."""
    if len(curve) < 3:
        return dict(imax=np.nan, k=np.nan, ok=False)
    n = np.array(sorted(curve))
    y = np.array([curve[i] for i in n])
    try:
        p, _ = curve_fit(_sat, n, y, p0=[max(y.max(), 1e-3) * 2, n.max() / 2],
                         bounds=([0, 0.1], [np.inf, 1e5]), maxfev=10000)
    except Exception:
        return dict(imax=np.nan, k=np.nan, ok=False)
    # "saturating within range" if half-saturation K lies inside the sampled N
    return dict(imax=float(p[0]), k=float(p[1]), ok=True, saturating=bool(p[1] < n.max()))


# ----------------------------------------------------------------------------- per session
def session_measures(sess, region, window="early", post=None, seed=0, n_rep=10):
    """All measures for one session x region. `post` = smoothed P(engaged) from the GLM-HMM (optional)."""
    X = sess.X(region, window)
    if X is None:
        return None
    m = trial_mask(sess.contrast, sess.choice, sess.rt, window_end=0.13 if window == "early" else 0.23)
    if m.sum() < 40:
        return None
    Xm, side, ch = X[m], (sess.contrast[m] > 0).astype(int), sess.choice[m]
    out = dict(sid=sess.sid, region=region, stage=REGION_STAGE.get(region), n_neurons=int(X.shape[1]),
               n_trials=int(m.sum()), d_neural=cv_dprime(Xm, side, seed=seed),
               d_within=within_choice_dprime(Xm, side, ch, seed=seed),
               d_behavior=behavior_dprime(side, ch),
               curve=scaling_curve(Xm, side, n_rep=n_rep, seed=seed))
    if post is not None:
        # LAPSE ANALYSIS: all non-zero contrasts, one decoder per session, held-out scores compared
        # between engaged (P>.8) and disengaged (P<.2) trials, within choice.
        mm = trial_mask(sess.contrast, sess.choice, sess.rt, window_end=0.13 if window == "early" else 0.23,
                        contrasts=(0.0625, 0.125, 0.25, 1.0))
        Xa, ya, ca, pa = X[mm], (sess.contrast[mm] > 0).astype(int), sess.choice[mm], post[mm]
        sc = cv_scores(Xa, ya, seed=seed)
        if sc is not None:
            e, d = pa > 0.8, pa < 0.2
            if e.sum() >= 20 and d.sum() >= 20:
                out["_lapse"] = dict(sc=sc, y=ya, c=ca, p=pa)
                out.update(d_engaged=within_choice_score_dprime(sc[e], ya[e], ca[e]),
                           d_disengaged=within_choice_score_dprime(sc[d], ya[d], ca[d]),
                           d_engaged_all=score_dprime(sc[e], ya[e]), d_disengaged_all=score_dprime(sc[d], ya[d]),
                           db_engaged=behavior_dprime(ya[e], ca[e]), db_disengaged=behavior_dprime(ya[d], ca[d]),
                           n_engaged=int(e.sum()), n_disengaged=int(d.sum()))
    return out


# ----------------------------------------------------------------------------- aggregation
def region_summary(rows, min_sessions=3):
    by = {}
    for r in rows:
        by.setdefault(r["region"], []).append(r)
    out = []
    for reg, rs in by.items():
        if len(rs) < min_sessions:
            continue
        pooled = {}
        for r in rs:
            for n, v in r["curve"].items():
                pooled.setdefault(int(n), []).append(v)
        curve = {n: float(np.mean(v)) for n, v in pooled.items() if len(v) >= min_sessions}
        fit = fit_ceiling(curve)
        eng = [r for r in rs if np.isfinite(r.get("d_engaged", np.nan)) and np.isfinite(r.get("d_disengaged", np.nan))]
        rho = rho_b = np.nan
        if len(eng) >= min_sessions:
            rho = float(np.mean([r["d_disengaged"] for r in eng]) / np.mean([r["d_engaged"] for r in eng]))
            rho_b = float(np.mean([r["db_disengaged"] for r in eng]) / np.mean([r["db_engaged"] for r in eng]))
        f = lambda k: float(np.nanmean([r[k] for r in rs]))
        out.append(dict(region=reg, stage=REGION_STAGE.get(reg), n_sessions=len(rs),
                        median_neurons=float(np.median([r["n_neurons"] for r in rs])),
                        d_neural=f("d_neural"), d_within=f("d_within"), d_behavior=f("d_behavior"),
                        efficiency=f("d_behavior") / f("d_neural") if f("d_neural") > 0 else np.nan,
                        curve=curve, **{f"ceiling_{k}": v for k, v in fit.items()},
                        rho_neural=rho, rho_behavior=rho_b, n_sessions_engagement=len(eng)))
    return sorted(out, key=lambda x: (x["stage"] or "9", x["region"]))


def stage_profile(summary):
    prof = {}
    for s in STAGES:
        rs = [x for x in summary if x["stage"] == s]
        if not rs:
            continue
        w = np.array([x["n_sessions"] for x in rs], float)
        g = lambda k: float(np.average([x[k] for x in rs if np.isfinite(x[k])], weights=[wi for x, wi in zip(rs, w) if np.isfinite(x[k])])) \
            if any(np.isfinite(x[k]) for x in rs) else np.nan
        prof[s] = dict(regions=[x["region"] for x in rs], d_neural=g("d_neural"), d_within=g("d_within"),
                       rho_neural=g("rho_neural"), rho_behavior=g("rho_behavior"))
    return prof


def _rho_from_lapse(rs, shifts=None):
    """Stage rho recomputed from stored held-out scores; `shifts` circularly shifts each session's
    state posterior relative to its trials (null: state unrelated to the neural signal)."""
    de, dd = [], []
    for r in rs:
        L = r["_lapse"]
        p = L["p"] if shifts is None else np.roll(L["p"], shifts[r["sid"]])
        e, d = p > 0.8, p < 0.2
        if e.sum() < 20 or d.sum() < 20:
            continue
        a, b = score_dprime(L["sc"][e], L["y"][e]), score_dprime(L["sc"][d], L["y"][d])
        if np.isfinite(a) and np.isfinite(b):
            de.append(a)
            dd.append(b)
    if not de or np.mean(de) <= 0:
        return np.nan
    return np.mean(dd) / np.mean(de)


def lapse_locus(rows, n_boot=5000, seed=0, alpha=0.05, key="all", min_sessions=5, intact_floor=0.6, n_perm=2000):
    """Stage-level lapse ratio rho = mean d'_disengaged / mean d'_engaged (within-choice),
    session-bootstrap CI, Bonferroni-corrected across the stages tested (family-wise alpha).
    Each stage gets one of three verdicts:
        LOSSY       corrected CI upper bound < 1
        INTACT      CI contains 1 AND its lower bound >= intact_floor (a large loss is excluded)
        UNRESOLVED  otherwise (too noisy to say: typical where stimulus d' is small)
    locus = first LOSSY stage, claimed only if every earlier tested stage is INTACT.
    If an UNRESOLVED stage precedes the first LOSSY stage the locus is reported as a range.
    If no stage is LOSSY: 'none detected' (with the list of INTACT and UNRESOLVED stages)."""
    rng = np.random.default_rng(seed)
    ke, kd = ("d_engaged_all", "d_disengaged_all") if key == "all" else ("d_engaged", "d_disengaged")
    by = {}
    for r in rows:
        if r.get("stage") and np.isfinite(r.get(ke, np.nan)) and np.isfinite(r.get(kd, np.nan)):
            by.setdefault(r["stage"], []).append(r)
    beh = {}
    for r in rows:
        if np.isfinite(r.get("db_engaged", np.nan)):
            beh[r["sid"]] = (r["db_engaged"], r["db_disengaged"])

    def rho(rs):
        de = np.mean([r[ke] for r in rs])
        return np.mean([r[kd] for r in rs]) / de if de > 0 else np.nan

    res = {}
    n_tested = sum(1 for st in STAGES if len({r["sid"] for r in by.get(st, [])}) >= min_sessions)
    a = alpha / max(n_tested, 1)
    for st in STAGES:
        rs = by.get(st, [])
        if len({r["sid"] for r in rs}) < min_sessions:
            continue
        sid_rows = {}
        for r in rs:
            sid_rows.setdefault(r["sid"], []).append(r)
        keys = list(sid_rows)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(len(keys), len(keys))
            boots.append(rho([x for i in pick for x in sid_rows[keys[i]]]))
        boots = np.array(boots)
        boots = boots[np.isfinite(boots)]
        lo, hi = np.percentile(boots, [100 * a / 2, 100 * (1 - a / 2)])
        # LOSSY requires a circular-shift permutation test (exact false-alarm control even with few
        # sessions, where bootstrap CIs are too narrow -- found in simulation); CI decides INTACT.
        p_perm = np.nan
        if key == "all" and all("_lapse" in r for r in rs):
            obs = _rho_from_lapse(rs)
            T = {r["sid"]: len(r["_lapse"]["p"]) for r in rs}
            null = []
            for _ in range(n_perm):
                sh = {sid: int(rng.integers(min(50, t // 4), max(min(50, t // 4) + 1, t - min(50, t // 4)))) for sid, t in T.items()}
                null.append(_rho_from_lapse(rs, sh))
            null = np.array(null)
            null = null[np.isfinite(null)]
            p_perm = float((1 + np.sum(null <= obs)) / (1 + len(null))) if np.isfinite(obs) and len(null) else np.nan
            lossy = bool(np.isfinite(p_perm) and p_perm < a and obs < 1)
        else:
            lossy = bool(hi < 1)
        verdict = "LOSSY" if lossy else ("INTACT" if (lo >= intact_floor and hi >= 1) else "UNRESOLVED")
        res[st] = dict(rho=float(rho(rs)), lo=float(lo), hi=float(hi), p_perm=p_perm, n_sessions=len(keys), verdict=verdict,
                       lossy=verdict == "LOSSY", d_engaged=float(np.mean([r[ke] for r in rs])))
    bv = np.array(list(beh.values()))
    rb = float(bv[:, 1].mean() / bv[:, 0].mean()) if len(bv) else np.nan
    tested = [st for st in STAGES if st in res]
    first = next((i for i, st in enumerate(tested) if res[st]["verdict"] == "LOSSY"), None)
    if first is None:
        locus, kind = None, "none detected"
    else:
        before = [res[st]["verdict"] for st in tested[:first]]
        if all(v == "INTACT" for v in before):
            locus, kind = tested[first], "located"
        else:
            last_intact = max([i for i in range(first) if res[tested[i]]["verdict"] == "INTACT"], default=None)
            locus = f"between {tested[last_intact] if last_intact is not None else 'start'} and {tested[first]}"
            kind = "range"
    return dict(stages=res, rho_behavior=rb, locus=locus, locus_kind=kind)


# ----------------------------------------------------------------------------- explore runner
def state_check(model, beh_sessions, post_kind="loo"):
    """Does the 'engaged' state really differ from the rest? Accuracy on 100% contrast trials
    by state (engaged = P>.8, disengaged = P<.2). A real engagement split shows a clear gap."""
    acc_e, acc_d, n_e, n_d = [], [], 0, 0
    for b in beh_sessions:
        p = model.posteriors(b)[post_kind][:, 0]
        hi = (np.abs(b.contrast) >= 0.99) & (b.choice >= 0)
        corr = (b.choice == (b.contrast > 0).astype(int))
        e, d = hi & (p > 0.8), hi & (p < 0.2)
        acc_e += list(corr[e])
        acc_d += list(corr[d])
    return dict(acc_engaged_100pct=float(np.mean(acc_e)) if acc_e else np.nan,
                acc_disengaged_100pct=float(np.mean(acc_d)) if acc_d else np.nan,
                n_engaged=len(acc_e), n_disengaged=len(acc_d))


def run_explore(cache_dir, out_dir, n_boot=500, regions=None, log=print, K=3, use_block=True):
    from .glmhmm import GLMHMM
    from .infoflow_data import load_info_cache
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    S = load_info_cache(cache_dir, split="explore")
    assert all(s.split == "explore" for s in S), "confirm data must not be loaded during exploration"
    S = [s for s in S if (s.choice >= 0).sum() >= 300]
    log(f"EXPLORE split: {len(S)} sessions, {len({s.meta.get('subject') for s in S})} mice")
    beh = [s.as_behavior_session(use_block=use_block) for s in S]
    log(f"block prior as GLM-HMM input: {use_block and beh[0].block is not None}")
    m = GLMHMM(K=K, n_restarts=3, seed=0).fit(beh)
    log(f"GLM-HMM K={K} weights [stimulus, bias, prev choice, win-stay, block] per state (state 1 = 'engaged'):")
    for k in range(K):
        log(f"   state {k+1}: " + "  ".join(f"{w:+.2f}" for w in m.W[k]))
    chk = state_check(m, beh)
    log(f"STATE CHECK (100% contrast accuracy): engaged {chk['acc_engaged_100pct']:.2f} (n={chk['n_engaged']}) "
        f"vs disengaged {chk['acc_disengaged_100pct']:.2f} (n={chk['n_disengaged']})")
    (out_dir / "glmhmm.json").write_text(json.dumps(dict(K=K, W=m.W.tolist(), A=m.A.tolist(), check=chk), indent=1))
    rows = []
    for i, (s, b) in enumerate(zip(S, beh)):
        post = m.posteriors(b)["loo"][:, 0]
        for r in (regions or s.regions):
            res = session_measures(s, r, "early", post=post, seed=i)
            if res:
                rows.append(res)
        if (i + 1) % 10 == 0:
            log(f"  {i+1}/{len(S)} sessions")
    summ = region_summary(rows)
    prof = stage_profile(summ)
    locus = lapse_locus(rows, n_boot=n_boot)
    locus_wc = lapse_locus(rows, n_boot=n_boot, key="within")
    (out_dir / "lapse_locus.json").write_text(json.dumps(dict(primary=locus, within_choice_secondary=locus_wc), indent=1, default=float))
    log(f"LAPSE LOCUS (exploratory): {locus['locus']}  behavior lapse ratio {locus['rho_behavior']:.2f}")
    for st, v in locus["stages"].items():
        log(f"  {st:26s} rho={v['rho']:.2f} [{v['lo']:.2f},{v['hi']:.2f}] n={v['n_sessions']} {v['verdict']}")
    (out_dir / "session_rows.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "_lapse"} for r in rows], default=float))
    (out_dir / "regions.json").write_text(json.dumps(summ, indent=1, default=float))
    (out_dir / "stages.json").write_text(json.dumps(prof, indent=1, default=float))
    plot_explore(summ, prof, out_dir / "explore.png")
    for s, v in prof.items():
        log(f"{s:26s} d'={v['d_neural']:.2f} within-choice d'={v['d_within']:.2f} "
            f"lapse ratio neural={v['rho_neural']:.2f} behavior={v['rho_behavior']:.2f}  ({', '.join(v['regions'])})")
    return summ, prof


def plot_explore(summ, prof, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    stages = list(prof)
    x = np.arange(len(stages))
    ax[0].bar(x - 0.2, [prof[s]["d_neural"] for s in stages], 0.4, label="stimulus d' (all)")
    ax[0].bar(x + 0.2, [prof[s]["d_within"] for s in stages], 0.4, label="within-choice d'")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([s.split(" ", 1)[1] for s in stages], rotation=40, ha="right", fontsize=8)
    ax[0].set_title("Stimulus information along the pathway (recorded neurons)", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].plot(x, [prof[s]["rho_neural"] for s in stages], "o-", label="neural: disengaged / engaged")
    ax[1].plot(x, [prof[s]["rho_behavior"] for s in stages], "k--", label="behavior: disengaged / engaged")
    ax[1].axhline(1, color="grey", ls=":")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([s.split(" ", 1)[1] for s in stages], rotation=40, ha="right", fontsize=8)
    ax[1].set_title("What a lapse does to information, stage by stage", fontsize=9)
    ax[1].legend(fontsize=7)
    top = sorted([r for r in summ if r["curve"]], key=lambda r: -r["d_neural"])[:8]
    for r in top:
        n = sorted(r["curve"])
        ax[2].plot(n, [r["curve"][i] for i in n], "o-", ms=3, label=r["region"])
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("simultaneously recorded neurons")
    ax[2].set_ylabel("d'^2 (cross-validated)")
    ax[2].set_title("Information scaling, 8 most informative regions", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
