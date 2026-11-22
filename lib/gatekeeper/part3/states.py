"""PART 3, step 3: what does the brain-wide slow mode DO?

Q4  ENGAGEMENT. Mice switch between an engaged strategy and disengaged (biased) strategies (GLM-HMM; Ashwood et al.
    2022, Nature Neuroscience). We fit a 3-state GLM-HMM per mouse on choices alone, then ask whether the global slow
    mode tracks the slow probability of being engaged, beyond other sessions' slow modes, with the task design AND
    slow pupil partialled out of both signals (same null and symmetric partialling as slowmode.py / controls.py).
Q3  LEADING REGIONS. For each region, does its slow signal change BEFORE or AFTER the rest of the brain's slow mode
    (global mode recomputed without that region)? Lead index = mean over lags 1..LAG of corr(region_t, rest_t+L) minus
    corr(region_t+L, rest_t). Under no lead it has mean 0 in every session (time-reversal symmetry), so a region's
    sessions are tested against 0 (Wilcoxon), with BH across regions.

    python3 part3/states.py sim               (checks on synthetic data, no IBL data needed)
    python3 part3/states.py run --limit 10    (quick test)
    python3 part3/states.py run               (explore split)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import slowmode as sm  # noqa: E402

LAG = 20          # trials
FAST = 10         # lighter smoothing for lead-lag, so timing is not blurred by the 50-trial window
NEUROMOD = {"LC", "SLC", "DR", "MRN", "VTA", "SNc", "NI", "PPN", "LDT", "RN", "CS", "SI", "MA", "NDB", "PRNr", "PRNc"}


# ------------------------------------------------------------------------------- 3-state GLM-HMM (choices only)
def glm_inputs(contrast, choice, feedback):
    """Ashwood-style inputs: stimulus, previous choice, win-stay-lose-switch, bias."""
    stim = np.tanh(5 * contrast) / np.tanh(5)
    ch = np.where(choice >= 0, 2.0 * choice - 1, 0.0)
    prev = np.r_[0.0, ch[:-1]]
    ws = np.r_[0.0, (ch * feedback)[:-1]]
    return np.column_stack([stim, prev, ws, np.ones(len(stim))])


def _sig(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def _fit_weighted_logistic(X, y, w, b0, l2=1.0, iters=25):
    b = b0.copy()
    for _ in range(iters):
        p = _sig(X @ b)
        g = X.T @ (w * (y - p)) - l2 * b
        H = (X * (w * p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        b = b + step
        if np.abs(step).max() < 1e-6:
            break
    return b


def fit_glmhmm(sessions, K=3, iters=60, seed=0):
    """sessions: list of (X (T,4), y (T,) in {0,1} or nan for misses). Returns weights, transitions, posteriors.
    State 0 is initialized engaged (stimulus-driven), states 1 and 2 as biased left and right."""
    W = np.array([[4.0, 0.2, 0.2, 0.0], [1.0, 0.2, 0.2, -2.0], [1.0, 0.2, 0.2, 2.0]])[:K]
    A = np.full((K, K), 0.02 / (K - 1)); np.fill_diagonal(A, 0.98)
    pi0 = np.full(K, 1.0 / K)
    for it in range(iters):
        post_all, xi_sum, g0 = [], np.zeros((K, K)), np.zeros(K)
        for X, y in sessions:
            T = len(y)
            p1 = _sig(X @ W.T)                                            # (T, K) prob of choosing right
            lik = np.where(np.isfinite(y)[:, None], np.where(y[:, None] == 1, p1, 1 - p1), 1.0)
            al = np.zeros((T, K)); c = np.zeros(T)
            al[0] = pi0 * lik[0]; c[0] = al[0].sum(); al[0] /= c[0]
            for t in range(1, T):
                al[t] = (al[t - 1] @ A) * lik[t]; c[t] = al[t].sum(); al[t] /= c[t]
            be = np.ones((T, K))
            for t in range(T - 2, -1, -1):
                be[t] = (A @ (lik[t + 1] * be[t + 1])) / c[t + 1]
            gm = al * be
            gm /= gm.sum(1, keepdims=True)
            xi_sum += (al[:-1, :, None] * A[None] * (lik[1:] * be[1:])[:, None, :] / c[1:, None, None]).sum(0)
            g0 += gm[0]
            post_all.append(gm)
        A = xi_sum + 1.0                                                      # weak prior keeps rows valid
        A /= A.sum(1, keepdims=True)
        pi0 = (g0 + 1) / (g0 + 1).sum()
        Xc = np.vstack([X for X, _ in sessions]); yc = np.concatenate([y for _, y in sessions])
        Gc = np.vstack(post_all); ok = np.isfinite(yc)
        for k in range(K):
            W[k] = _fit_weighted_logistic(Xc[ok], yc[ok], Gc[ok, k], W[k])
    engaged = int(np.argmax(np.abs(W[:, 0])))                                 # largest stimulus weight
    return W, A, post_all, engaged


# ------------------------------------------------------------------------------- per-session signals
def session_signals(path, aux_dir):
    """Region slow signals (for lead-lag), global mode, slow design, slow pupil, behaviour inputs."""
    from gatekeeper.mv_data import MVSession
    import controls
    s = MVSession(path)
    v = s.valid["stimOn"].copy()
    if v.sum() < sm.MIN_TRIALS:
        return None
    D = sm.task_design(s)[v]
    regs = {}
    for r in s.regions:
        X = s.X(r, sm.WINDOW)
        if X is None:
            continue
        X = X[v]
        X = X[:, X.std(0) > 0]
        if X.shape[1] < sm.MIN_UNITS:
            continue
        regs[r] = sm.residualize(X.astype(float), D)
    if len(regs) < 2:
        return None
    Z = {r: (X - X.mean(0)) / (X.std(0) + 1e-12) for r, X in regs.items()}
    allZ = np.hstack(list(Z.values()))
    g, _ = sm.slow_pc(allZ)
    Ds = np.column_stack([sm.movavg(D[:, k]) for k in range(D.shape[1])])
    aux = controls.load_aux(aux_dir, s.sid, v, len(s.contrast)) if aux_dir else None
    pupil = aux["arousal"] if aux is not None and aux["arousal"] is not None else None
    # lead-lag signals: region slow signal vs the slow mode of everything else, lightly smoothed
    lead = {}
    names = list(Z)
    for r in names:
        rest = np.hstack([Z[q] for q in names if q != r])
        Sr = np.column_stack([sm.movavg(Z[r][:, i], FAST) for i in range(Z[r].shape[1])])
        Sq = np.column_stack([sm.movavg(rest[:, i], FAST) for i in range(rest.shape[1])])
        ur = _pc1(Sr); uq = _pc1(Sq)
        if np.corrcoef(ur, uq)[0, 1] < 0:
            ur = -ur
        lead[r] = lead_index(ur, uq)
    return dict(sid=s.sid, subject=s.subject, T=int(v.sum()), global_mode=g, slow_design=Ds, slow_exog=Ds[:, :5], behaviour=sm.behaviour(s, v), pupil=pupil,
                glm=(glm_inputs(s.contrast[v], s.choice[v], s.feedback[v]),
                     np.where(s.choice[v] >= 0, s.choice[v].astype(float), np.nan)),
                lead=lead)


def _pc1(S):
    S = S - S.mean(0)
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    return U[:, 0] * sv[0]


def lead_index(a, b, lag=LAG):
    """Positive when a changes before b."""
    def c(x, y):
        return np.corrcoef(x, y)[0, 1]
    vals = [c(a[:-L], b[L:]) - c(a[L:], b[:-L]) for L in range(1, lag + 1)]
    return float(np.nanmean(vals))


# ------------------------------------------------------------------------------- analysis
def engagement_test(S, n_null=200, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    by_mouse = {}
    for i, x in enumerate(S):
        by_mouse.setdefault(x["subject"], []).append(i)
    rows, weights = [], []
    for m, idx in by_mouse.items():
        W, A, post, k = fit_glmhmm([S[i]["glm"] for i in idx])
        weights.append(dict(subject=m, engaged_state=k, **{f"w{s}_{n}": W[s, j] for s in range(W.shape[0])
                                                            for j, n in enumerate(("stim", "prev", "wsls", "bias"))},
                            stay_engaged=A[k, k]))
        for i, gm in zip(idx, post):
            S[i]["p_engaged"] = sm.movavg(gm[:, k])
    for i, x in enumerate(S):
        others = [y["global_mode"] for j, y in enumerate(S) if j != i]
        idx = rng.integers(0, len(others), n_null)
        e = x["p_engaged"]
        row = dict(sid=x["sid"], subject=x["subject"], engaged_frac=float(np.mean(e > 0.5)), engaged_sd=float(np.std(e)))
        for name, D in (("task", x["slow_design"]),
                        ("task+pupil", np.column_stack([x["slow_design"], x["pupil"]]) if x["pupil"] is not None else None)):
            if D is None:
                continue
            obs = abs(sm._corr(x["global_mode"], e, D))
            null = np.array([abs(sm._corr(others[k], e, D)) for k in idx])
            null = null[np.isfinite(null)]
            if np.isfinite(obs) and len(null) > 20:
                row[f"r_{name}"] = obs
                row[f"rnull_{name}"] = float(np.median(null))
                row[f"p_{name}"] = float((1 + (null >= obs).sum()) / (1 + len(null)))
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(weights)


def design_check(S, n_null=200, seed=0):
    """Same tests with three choices of what is partialled out of both signals:
    'full'           all slow task regressors (incl. slow choice and slow feedback = reward rate). Over-controls:
                     slow reward rate is close to the definition of engagement and accuracy.
    'stimulus/block' only what the experimenter controls (block, opening block, Bayes prior, contrast, abs contrast).
    'stimulus/block+pupil'  the same plus slow pupil."""
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for i, x in enumerate(S):
        others = [y["global_mode"] for j, y in enumerate(S) if j != i]
        idx = rng.integers(0, len(others), n_null)
        targets = dict(x["behaviour"]); targets["p_engaged"] = x["p_engaged"]
        # negative control: ANOTHER session's engagement trace (cannot be linked to this brain); must give about 0.05
        targets["ctl_p_engaged_other_session"] = S[(i + 1) % len(S)]["p_engaged"][: len(x["p_engaged"])]
        designs = {"full": x["slow_design"], "stimulus/block": x["slow_exog"],
                   "stimulus/block+pupil": np.column_stack([x["slow_exog"], x["pupil"]]) if x["pupil"] is not None else None}
        for tname, e in targets.items():
            for dname, D in designs.items():
                if D is None:
                    continue
                obs = abs(sm._corr(x["global_mode"], e, D))
                null = np.array([abs(sm._corr(others[k], e, D)) for k in idx])
                null = null[np.isfinite(null)]
                if np.isfinite(obs) and len(null) > 20:
                    rows.append(dict(sid=x["sid"], subject=x["subject"], target=tname, design=dname, r=obs,
                                     r_null=float(np.median(null)), p=float((1 + (null >= obs).sum()) / (1 + len(null)))))
    return pd.DataFrame(rows)


def report_check(d, out_dir):
    L = ["# Part 3: is the behaviour link hidden by over-control?\n",
         "Global slow mode vs slow behaviour, null = other sessions' global modes. Fraction p < 0.05 should be about 0.05 "
         "when there is nothing.\n",
         "| Behaviour | Partialled out | Sessions | Median abs r | Median abs r (null) | Fraction p < 0.05 | Fisher p |",
         "|---|---|---|---|---|---|---|"]
    for (t, dn), h in d.groupby(["target", "design"], sort=True):
        L.append(f"| {t} | {dn} | {len(h)} | {h.r.median():.2f} | {h.r_null.median():.2f} | {(h.p < .05).mean():.3f} | {_fisher(h.p):.1e} |")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "PART3_DESIGN_CHECK.md").write_text(text + "\n")
    d.to_csv(Path(out_dir) / "design_check.csv", index=False)
    print(text)


def lead_table(S):
    import pandas as pd
    from scipy.stats import wilcoxon
    recs = [dict(sid=x["sid"], subject=x["subject"], region=r, lead=v) for x in S for r, v in x["lead"].items() if np.isfinite(v)]
    d = pd.DataFrame(recs)
    if d.empty:
        return d, d
    div = sm.division_of(sorted(d.region.unique()))
    out = []
    for r, h in d.groupby("region"):
        if h.sid.nunique() < 3:
            continue
        p = float(wilcoxon(h.lead.values).pvalue) if np.any(h.lead.values != 0) else 1.0
        out.append(dict(region=r, division=div.get(r, r), neuromodulatory=r in NEUROMOD, sessions=h.sid.nunique(),
                        mice=h.subject.nunique(), mean_lead=h.lead.mean(), frac_leading=(h.lead > 0).mean(), p=p))
    t = pd.DataFrame(out)
    if len(t):
        t["significant_fdr5"] = bh(t.p.values, 0.05)
        t = t.sort_values("mean_lead", ascending=False)
    return d, t


def bh(p, q=0.05):
    p = np.asarray(p, float); n = len(p)
    o = np.argsort(p); ok = p[o] <= q * np.arange(1, n + 1) / n
    out = np.zeros(n, bool)
    if ok.any():
        out[o[: np.flatnonzero(ok).max() + 1]] = True
    return out


def _fisher(p):
    from scipy.stats import chi2
    p = np.asarray(p, float); p = p[np.isfinite(p)]
    return float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan


def report(eng, w, raw, lt, out_dir):
    L = ["# Part 3 step 3: engagement states and leading regions\n",
         f"## Q4: does the global slow mode track the engaged state? ({len(eng)} sessions, {eng.subject.nunique()} mice)\n",
         f"GLM-HMM (3 states per mouse, choices only). Median stimulus weight engaged vs other states: "
         f"{np.median([r[f'w{int(r.engaged_state)}_stim'] for _, r in w.iterrows()]):.2f}; median P(stay engaged) "
         f"{w.stay_engaged.median():.3f}. Median fraction of trials engaged {eng.engaged_frac.median():.2f}.\n",
         "| Partialled out | Sessions | Median abs r | Median abs r (null) | Fraction p < 0.05 | Fisher p |", "|---|---|---|---|---|---|"]
    for name in ("task", "task+pupil"):
        c = f"p_{name}"
        if c in eng:
            d = eng.dropna(subset=[c])
            L.append(f"| {name} | {len(d)} | {d[f'r_{name}'].median():.2f} | {d[f'rnull_{name}'].median():.2f} | "
                     f"{(d[c] < .05).mean():.3f} | {_fisher(d[c]):.1e} |")
    L += ["\n## Q3: which regions lead the brain-wide slow mode?\n",
          f"Lead index over lags 1 to {LAG} trials (positive = region changes first). Regions with >= 3 sessions; "
          "Wilcoxon against 0; BH across regions.\n"]
    if len(lt):
        L.append(f"Regions tested: {len(lt)}; significant at FDR 5%: {int(lt.significant_fdr5.sum())} "
                 f"(leading {int((lt.significant_fdr5 & (lt.mean_lead > 0)).sum())}, lagging "
                 f"{int((lt.significant_fdr5 & (lt.mean_lead < 0)).sum())}).\n")
        L += ["| Region | Division | Neuromodulatory | Sessions | Mice | Mean lead | Fraction of sessions leading | p | FDR 5% |",
              "|---|---|---|---|---|---|---|---|---|"]
        show = lt[lt.significant_fdr5 | lt.neuromodulatory].head(40)
        for _, r in show.iterrows():
            L.append(f"| {r.region} | {r.division} | {'yes' if r.neuromodulatory else ''} | {r.sessions} | {r.mice} | "
                     f"{r.mean_lead:+.3f} | {r.frac_leading:.2f} | {r.p:.1e} | {'yes' if r.significant_fdr5 else ''} |")
        byd = lt.groupby("division").mean_lead.agg(["mean", "count"]).sort_values("mean", ascending=False)
        L += ["\nBy division (mean lead, regions):", "", "| Division | Mean lead | Regions |", "|---|---|---|"]
        L += [f"| {k} | {v['mean']:+.3f} | {int(v['count'])} |" for k, v in byd.iterrows()]
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "PART3_STATES.md").write_text(text + "\n")
    eng.to_csv(Path(out_dir) / "engagement.csv", index=False)
    w.to_csv(Path(out_dir) / "glmhmm_weights.csv", index=False)
    raw.to_csv(Path(out_dir) / "lead_sessions.csv", index=False)
    lt.to_csv(Path(out_dir) / "lead_regions.csv", index=False)
    print(text)


def run(cache, aux_dir, out_dir, split, limit, n_jobs):
    from joblib import Parallel, delayed
    from gatekeeper.mv_data import MVSession
    paths = [p for p in sorted(Path(cache).expanduser().glob("*.npz")) if MVSession(p).split == split][: limit or None]
    print(f"{len(paths)} sessions ({split})", flush=True)
    t0 = time.time()
    S = [x for x in Parallel(n_jobs=n_jobs, verbose=5)(delayed(session_signals)(p, aux_dir) for p in paths) if x]
    print(f"signals done in {(time.time() - t0) / 60:.1f} min; fitting GLM-HMMs", flush=True)
    eng, w = engagement_test(S)
    raw, lt = lead_table(S)
    report(eng, w, raw, lt, out_dir)


# ------------------------------------------------------------------------------- simulation checks
def _ar(T, rng, rho=0.99):
    z = np.zeros(T); e = rng.standard_normal(T)
    for t in range(1, T):
        z[t] = rho * z[t - 1] + np.sqrt(1 - rho ** 2) * e[t]
    return z


def sim(n_sessions=40, T=700, link=0.0, lead_shift=0, seed=0):
    """Each session: a hidden engaged/disengaged chain; choices from a 3-state GLM-HMM; 5 regions sharing one slow
    state. 'link' makes the slow state follow engagement; 'lead_shift' makes region R1 run that many trials AHEAD."""
    rng = np.random.default_rng(seed)
    S = []
    for i in range(n_sessions):
        # engagement chain
        z = np.zeros(T, int)
        for t in range(1, T):
            z[t] = z[t - 1] if rng.random() < 0.98 else rng.integers(0, 3)
        c = rng.choice([-1, -.25, -.125, -.0625, 0, .0625, .125, .25, 1], T)
        ch = np.zeros(T); fb = np.zeros(T)
        Wt = np.array([[5, .2, .2, 0], [.5, .2, .2, -2], [.5, .2, .2, 2]])
        for t in range(T):
            x = glm_inputs(c[max(0, t - 1):t + 1], ch[max(0, t - 1):t + 1], fb[max(0, t - 1):t + 1])[-1]
            ch[t] = rng.random() < _sig(x @ Wt[z[t]])
            fb[t] = 1.0 if (ch[t] == (c[t] > 0)) or c[t] == 0 else -1.0
        eng = sm.movavg((z == 0).astype(float))
        state = link * (eng - eng.mean()) / (eng.std() + 1e-9) + np.sqrt(max(1 - link ** 2, 0.2)) * _ar(T + 50, rng)[:T]
        full = np.r_[state, np.zeros(50)]
        Z = {}
        for j, r in enumerate(("R1", "R2", "R3", "R4", "R5")):
            sig_r = full[lead_shift:lead_shift + T] if (r == "R1" and lead_shift) else state
            X = rng.poisson(np.exp(0.5 + 0.4 * sig_r[:, None] * rng.uniform(.5, 1.5, 15))).astype(float)
            Z[r] = (X - X.mean(0)) / (X.std(0) + 1e-12)
        g, _ = sm.slow_pc(np.hstack(list(Z.values())))
        lead = {}
        for r in Z:
            rest = np.hstack([Z[q] for q in Z if q != r])
            ur = _pc1(np.column_stack([sm.movavg(Z[r][:, k], FAST) for k in range(Z[r].shape[1])]))
            uq = _pc1(np.column_stack([sm.movavg(rest[:, k], FAST) for k in range(rest.shape[1])]))
            if np.corrcoef(ur, uq)[0, 1] < 0:
                ur = -ur
            lead[r] = lead_index(ur, uq)
        S.append(dict(sid=f"s{i}", subject=f"m{i // 4}", T=T, global_mode=g, slow_design=np.zeros((T, 0)), pupil=None,
                      glm=(glm_inputs(c, ch, fb), ch.astype(float)), lead=lead, z=z))
    eng, w = engagement_test(S, n_null=100, seed=seed)
    recov = np.mean([np.mean((S[i]["p_engaged"] > .5) == (sm.movavg((S[i]["z"] == 0).astype(float)) > .5)) for i in range(len(S))])
    raw, lt = lead_table(S)
    return eng, w, lt, recov


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "run", "check", "simcheck"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--aux", default="data/part3_aux")
    ap.add_argument("--out", default="results/part3_states")
    ap.add_argument("--split", default="explore")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.step == "sim":
        for kw in (dict(), dict(link=0.8), dict(lead_shift=8)):
            eng, w, lt, recov = sim(**kw)
            r1 = lt[lt.region == "R1"].iloc[0] if len(lt) else None
            print(f"{kw}: engaged-state recovery {recov:.2f} | Q4 frac p<.05 = {(eng.p_task < .05).mean():.3f} | "
                  f"R1 mean lead {r1.mean_lead:+.3f} p={r1.p:.1e} | other regions significant: "
                  f"{int(lt[lt.region != 'R1'].significant_fdr5.sum())}/4")
    elif a.step == "simcheck":
        for kw in (dict(), dict(link=0.8)):
            eng, w, lt, recov = sim(**kw)
        print("see check")
    elif a.step == "check":
        from joblib import Parallel, delayed
        from gatekeeper.mv_data import MVSession
        paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz")) if MVSession(p).split == a.split][: a.limit or None]
        print(f"{len(paths)} sessions ({a.split})", flush=True)
        S = [x for x in Parallel(n_jobs=a.jobs)(delayed(session_signals)(p, a.aux) for p in paths) if x]
        engagement_test(S)
        report_check(design_check(S), a.out)
    else:
        run(a.cache, a.aux, a.out, a.split, a.limit, a.jobs)
