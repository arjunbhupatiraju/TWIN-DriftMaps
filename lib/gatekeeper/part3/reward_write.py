"""PART 3 mechanism: how is the reward memory WRITTEN?

Everything so far says the memory of past rewarded choices drives behaviour for 4-6 trials but is not held in
ongoing activity (lags 2-6 absent with power, no reactivation). The standard account of such an activity-silent
memory is that it is written by the outcome signal (a reward prediction error, RPE) into synaptic weights and read
out later. This file tests the two links of that account that the recordings can test.

Step W1 (behaviour, write rule). Does the size of the memory update depend on how surprising the reward was?
    logit P(right_t) = stimulus + block prior + choice lags 1-4 + rewarded-choice lags 1-4 + outcome lags 1-4
                       + g_k * rewarded_choice(t-k) * surprise(t-k)        k = 1, 2
    surprise = 1 - confidence, confidence = cross-fitted P(chosen side | stimulus, block prior) for that trial.
    RPE-written memory predicts g_1 > 0 (Lak et al. 2020 showed this in a related task; this is the replication on
    the IBL / Steinmetz mice and gives the write rule used in step W2).

Step W2 (neural write signal, new). On each trial, the population response after the outcome (feedback window
0-200 ms) is decoded into an RPE estimate. Its trial-to-trial deviation from what the model expects (epsilon:
decoded RPE minus everything the model and the covariates explain) should predict how strongly the NEXT choice
repeats this choice:
    logit P(right_{t+1}) = [everything above at t+1] + c_t x [movement / arousal / RT / contrast at t] + b * c_t * eps_t
    prediction: b > 0 (a larger-than-expected outcome response writes a stronger memory)
    Null: the same regression with eps circularly shifted within the session (200 shifts, at least 30 trials).
    Control: the same pipeline on the STIMULUS window of the same trial (before the outcome). A write signal predicts
    the effect after the outcome, not before.
    Slow drift is removed from eps (cosine basis, periods > 30 trials) before testing.

    python3 part3/reward_write.py sim
    python3 part3/reward_write.py behaviour
    python3 part3/reward_write.py run --limit 4          then   run --every 1
    python3 part3/reward_write.py run --regions --every 2   (where in the brain is the write signal)
    python3 part3/reward_write.py steinmetz --data <folder>
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "next", ROOT / "driftaware" / "src", HERE):
    sys.path.insert(0, str(p))
from history_map import lags  # noqa: E402

K = 4
N_SHIFT = 200
MIN_SHIFT = 30
MIN_UNITS = 10
SLOW_PERIOD = 30


# ------------------------------------------------------------------------------------------- small tools
def logit_fit(X, y, lam=1e-2, iters=50):
    """Ridge-penalised logistic regression (Newton). Intercept is column 0 and is not penalised."""
    X = np.column_stack([np.ones(len(y)), X])
    w = np.zeros(X.shape[1])
    P = lam * np.eye(X.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + P @ w
        Hs = (X * (p * (1 - p))[:, None]).T @ X + P
        step = np.linalg.solve(Hs + 1e-9 * np.eye(len(w)), g)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def logit_prob(X, w):
    return 1 / (1 + np.exp(-np.clip(np.column_stack([np.ones(len(X)), X]) @ w, -30, 30)))


def folds(n, k=5):
    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(edges[i], edges[i + 1]) for i in range(k)]


def stim_design(contrast):
    c = np.nan_to_num(np.asarray(contrast, float))
    f = lambda x: np.tanh(5 * x) / np.tanh(5)
    return np.column_stack([f(np.clip(c, 0, None)), f(np.clip(-c, 0, None))])


def cosine_basis(T, period=SLOW_PERIOD):
    nb = max(1, int(2 * T / period))
    t = (np.arange(T) + 0.5) / T
    return np.column_stack([np.cos(np.pi * j * t) for j in range(1, nb + 1)])


def resid(y, Z):
    Z = np.column_stack([np.ones(len(y)), Z])
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def task_vars(contrast, choice, feedback, prior):
    """Signed choice c (+1/-1, 0 = no choice), outcome r (1/0), confidence of the chosen side, surprise, design."""
    choice = np.asarray(choice); f = np.asarray(feedback, float)
    c = np.where(choice >= 0, 2.0 * choice - 1, 0.0)
    r = (f > 0).astype(float)
    y = (c > 0).astype(float)
    ok = c != 0
    S = np.column_stack([stim_design(contrast), np.asarray(prior, float)])
    p_right = np.full(len(c), 0.5)
    idx = np.flatnonzero(ok)
    for te in folds(len(idx)):                                   # cross-fitted P(right | stimulus, prior)
        tr = np.setdiff1d(np.arange(len(idx)), te)
        w = logit_fit(S[idx[tr]], y[idx[tr]])
        p_right[idx[te]] = logit_prob(S[idx[te]], w)
    conf = np.where(c > 0, p_right, 1 - p_right)
    rc = c * r
    sur = np.where(r > 0, 1 - conf, 0.0)
    sur = np.where(r > 0, sur - sur[r > 0].mean(), 0.0) if (r > 0).any() else sur
    base = np.column_stack([stim_design(contrast), prior, lags(c, K), lags(rc, K), lags(f, K)])
    return dict(c=c, r=r, y=y, ok=ok, conf=conf, rc=rc, sur=sur, base=base)


# ------------------------------------------------------------------------------------------- W1 behaviour
def write_rule(contrast, choice, feedback, prior):
    tv = task_vars(contrast, choice, feedback, prior)
    G = lags(tv["rc"] * tv["sur"], 2)
    X = np.column_stack([tv["base"], G])
    m = tv["ok"].copy(); m[:K + 1] = False
    w = logit_fit(X[m], tv["y"][m], lam=1.0)
    nb = tv["base"].shape[1]
    # surprise coefficient and the plain rewarded-choice lag-1 coefficient for scale
    return dict(g1=w[1 + nb], g2=w[2 + nb], rc1=w[1 + 3 + K], n=int(m.sum()))


# ------------------------------------------------------------------------------------------- W2 write test
def decode_rpe(X, delta, v):
    """Cross-validated ridge decoding of delta from activity, contiguous folds, on trials v."""
    idx = np.flatnonzero(v)
    Xv = X[idx].astype(float)
    out = np.full(len(delta), np.nan)
    for te in folds(len(idx)):
        tr = np.setdiff1d(np.arange(len(idx)), te)
        mu, sd = Xv[tr].mean(0), Xv[tr].std(0) + 1e-9
        A = (Xv[tr] - mu) / sd; B = (Xv[te] - mu) / sd
        yt = delta[idx[tr]] - delta[idx[tr]].mean()
        lam = A.shape[1] * 10.0
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yt)
        out[idx[te]] = B @ w + delta[idx[tr]].mean()
    return out


def write_test(X, tv, contrast, lrt, Hc, v, rng, n_shift=N_SHIFT, eps_fn=None):
    """Returns real coefficient b, z vs circular-shift null, one-sided p, and decoding r of the RPE."""
    T = len(tv["c"])
    delta = tv["r"] - tv["conf"]
    dh = decode_rpe(X, delta, v)
    idx = np.flatnonzero(v)
    dec_r = float(np.corrcoef(dh[idx], delta[idx])[0, 1])
    Kc = [delta, tv["r"], tv["conf"], np.abs(np.nan_to_num(contrast)), lrt]
    if Hc is not None:
        Kc.append(Hc)
    Kc = np.column_stack(Kc)[idx]
    Kc = np.column_stack([Kc, cosine_basis(T)[idx]])
    eps = np.zeros(T)
    e = resid(dh[idx], Kc)
    eps[idx] = (e - e.mean()) / (e.std() + 1e-12)
    if eps_fn is not None:
        eps = eps_fn(eps)
    # next-choice regression on trials t+1 where t is in v and t+1 has a choice
    t1 = idx[idx + 1 < T]
    t1 = t1[tv["ok"][t1 + 1]]
    t1 = t1[t1 >= K]
    c = tv["c"]
    side_cov = [c * np.abs(np.nan_to_num(contrast)), c * lrt, tv["rc"] * tv["sur"]]
    if Hc is not None:
        side_cov += [c * Hc[:, j] for j in range(Hc.shape[1])]
    side_cov = np.column_stack(side_cov)
    B = np.column_stack([tv["base"][t1 + 1], side_cov[t1]])
    B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    y = tv["y"][t1 + 1]

    r1 = tv["r"][t1]

    def coef(ev):
        z = c[t1] * ev[t1]
        w = logit_fit(np.column_stack([B, z * r1, z * (1 - r1)]), y, lam=1.0)
        return w[-2], w[-1]

    real = np.array(coef(eps))
    pos = np.flatnonzero(v)
    null = []
    for _ in range(n_shift):
        k = int(rng.integers(MIN_SHIFT, max(MIN_SHIFT + 1, len(pos) - MIN_SHIFT)))
        e2 = np.zeros(T); e2[pos] = np.roll(eps[pos], k)
        null.append(coef(e2))
    null = np.asarray(null)
    zz = (real - null.mean(0)) / (null.std(0) + 1e-12)
    # main test: rewarded trials (the memory is reward-gated); error trials reported as a diagnostic, two-sided
    return dict(b=float(real[0]), z=float(zz[0]), p=float((1 + (null[:, 0] >= real[0]).sum()) / (1 + len(null))),
                b_err=float(real[1]), z_err=float(zz[1]),
                p_err=float((1 + (np.abs(null[:, 1] - null[:, 1].mean()) >= abs(real[1] - null[:, 1].mean())).sum())
                            / (1 + len(null))),
                dec_r=dec_r, n=int(len(t1)))


# ------------------------------------------------------------------------------------------- simulation
def sim(reps=60, seed=0):
    """Calibration (no coupling) and power (coupling) of the write test, with drifting neurons and realistic sizes."""
    from joblib import Parallel, delayed
    from history_map import sim_session

    def one(i, eta, confound):
        rng = np.random.default_rng(seed + 7919 * i)
        T, N = 600, 60
        X0, side, _, _, prior = sim_session(rng, T=T, N=N, mode="none")
        contrast = (2 * side - 1) * rng.choice([0, .0625, .125, .25, 1.0], T)
        u = rng.standard_normal(T)                                   # hidden trial-to-trial variation of the RPE
        a = rng.standard_normal(T)                                   # hidden arousal (confound mode)
        Q = 0.0; P = 0.0; choice = np.zeros(T, int); fb = np.zeros(T)
        sR = stim_design(contrast)
        for t in range(T):
            lg = 3 * (sR[t, 0] - sR[t, 1]) + 1.2 * (prior[t] - 0.5) * 2 + Q + P
            pr = 1 / (1 + np.exp(-lg))
            choice[t] = rng.random() < pr
            fb[t] = 1.0 if (choice[t] == side[t]) else -1.0
            if contrast[t] == 0:
                fb[t] = 1.0 if rng.random() < .5 else -1.0
            conf = pr if choice[t] == 1 else 1 - pr
            d = (fb[t] > 0) - conf
            Q = 0.7 * Q + 1.0 * (2 * choice[t] - 1) * (d + eta * u[t]) * (fb[t] > 0)
            # confound: arousal makes the mouse repeat its choice whatever the outcome (not a reward memory)
            P = confound * 0.8 * a[t] * (2 * choice[t] - 1)
        tv = task_vars(contrast, choice, fb, prior - 0.5)
        delta = tv["r"] - tv["conf"]
        wr, wd, wa = rng.standard_normal(N), rng.standard_normal(N), rng.standard_normal(N)
        Xf = X0 + 0.8 * np.outer(tv["r"], wr) + 0.4 * np.outer(delta + 0.6 * u, wd) + confound * 0.6 * np.outer(a, wd)
        Xs = X0 + 0.8 * np.outer(np.sign(contrast), wr) + confound * 0.6 * np.outer(a, wd)    # stimulus window
        lrt = rng.standard_normal(T) * 0.3
        v = np.ones(T, bool); v[:90] = False
        of = write_test(Xf, tv, contrast, lrt, None, v, rng, n_shift=100)
        os_ = write_test(Xs, tv, contrast, lrt, None, v, rng, n_shift=100)
        return of["p"] < .05, of["p_err"] < .05, os_["p"] < .05, of["dec_r"]

    for eta, confound in ((0.0, 0), (0.5, 0), (1.0, 0), (0.0, 1)):
        R = np.array(Parallel(n_jobs=-1)(delayed(one)(i, eta, confound) for i in range(reps)))
        print(f"coupling {eta} confound {confound}: feedback rewarded {R[:, 0].mean():.3f}   feedback errors "
              f"{R[:, 1].mean():.3f}   stimulus-window control {R[:, 2].mean():.3f}   (RPE decoding r {R[:, 3].mean():.2f})",
              flush=True)


# ------------------------------------------------------------------------------------------- IBL
def _ibl_session(path, aux_dir, regions=False, seed=0, only=None):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from gatekeeper.multiverse import _sides
    from history_confounds import confounds
    s = MVSession(path)
    v0 = cm.mask_of(s)
    if v0 is None or v0.sum() < 200:
        return []
    T = len(s.contrast)
    prior = cm.bayes_prior(_sides(s)) - 0.5
    tv = task_vars(s.contrast, s.choice, s.feedback, prior)
    lrt = np.where(np.isfinite(s.rt) & (s.rt > 0), np.log(np.clip(s.rt, 0.05, 10)), np.nan)
    lrt = np.where(np.isfinite(lrt), lrt, np.nanmedian(lrt))
    Hc, _ = confounds(aux_dir, s.sid, T)
    if Hc is not None:
        Hc = (Hc - Hc.mean(0)) / (Hc.std(0) + 1e-9)
    rng = np.random.default_rng(seed + zlib.crc32(s.sid.encode()) % 100000)
    vw = {"feedback_a": v0 & s.valid["feedback"], "stim_a": v0 & s.valid["feedback"]}
    groups = [("ALL", list(s.regions))] if not regions else [(r, [r]) for r in s.regions]
    if only is not None:
        groups = [(r, [r]) for r in s.regions if r in only]
    rows = []
    for name, regs in groups:
        row = dict(sid=s.sid, subject=s.subject, lab=s.lab, region=name)
        for w in ("feedback_a", "stim_a"):
            parts = [s.X(q, w) for q in regs]
            parts = [x for x in parts if x is not None]
            if not parts:
                break
            X = np.hstack(parts)
            X = X[:, X[vw[w]].std(0) > 0]
            if X.shape[1] < MIN_UNITS:
                break
            o = write_test(X, tv, s.contrast, lrt, Hc, vw[w], rng)
            row.update({f"{w}_{k}": val for k, val in o.items()})
            row["units"] = X.shape[1]
        else:
            rows.append(row)
    return rows


def behaviour(cache, out_dir):
    import pandas as pd
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from gatekeeper.multiverse import _sides
    rows = []
    for p in sorted(Path(cache).expanduser().glob("*.npz")):
        s = MVSession(p)
        if cm.mask_of(s) is None:
            continue
        prior = cm.bayes_prior(_sides(s)) - 0.5
        rows.append(dict(sid=s.sid, subject=s.subject, **write_rule(s.contrast, s.choice, s.feedback, prior)))
    df = pd.DataFrame(rows)
    _behaviour_report(df, out_dir, "IBL")


def _behaviour_report(df, out_dir, title):
    from scipy.stats import ttest_1samp, wilcoxon
    m = df.groupby("subject")[["g1", "g2", "rc1"]].mean()
    L = [f"# W1 write rule ({title}): does a more surprising reward write a stronger memory?\n",
         f"{len(df)} sessions, {len(m)} mice. Coefficients per mouse (mean of its sessions); test across mice.\n",
         "| Term | Mean across mice | SE | t-test p (one-sided, > 0) | Wilcoxon p | Mice > 0 |", "|---|---|---|---|---|---|"]
    for k, lab in (("rc1", "rewarded choice, lag 1 (scale)"), ("g1", "x surprise, lag 1 (write rule)"),
                   ("g2", "x surprise, lag 2")):
        x = m[k].values
        L.append(f"| {lab} | {x.mean():.3f} | {x.std(ddof=1) / np.sqrt(len(x)):.3f} | "
                 f"{ttest_1samp(x, 0, alternative='greater').pvalue:.1e} | "
                 f"{wilcoxon(x, alternative='greater').pvalue:.1e} | {(x > 0).sum()}/{len(x)} |")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "W1_WRITE_RULE.md").write_text(text + "\n")
    df.to_csv(Path(out_dir) / "w1_sessions.csv", index=False)
    print(text)


def run(cache, aux_dir, out_dir, limit, jobs, every, regions=False):
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / ("sessions_regions" if regions else "sessions"); out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[::every][: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(todo)} to run", flush=True)
    t0 = time.time()

    def work(p):
        try:
            rows = _ibl_session(p, aux_dir, regions)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True); rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    for i, n in enumerate(Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)):
        print(f"[{i + 1}/{len(todo)}] {n} rows  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fs = [f for f in out.glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    write_report(df, out_dir, "IBL", regions)


def write_report(df, out_dir, title, regions=False):
    from scipy.stats import chi2, norm, binomtest
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    L = [f"# W2 neural write signal ({title}): does the outcome response predict the next memory update?\n",
         f"{df.sid.nunique()} sessions, {df.subject.nunique()} mice. Prediction: feedback window b > 0; "
         "stimulus window (before the outcome) is the control. Null = circular shifts of the neural deviation.\n"]
    ses = df[df.region == "ALL"] if not regions else df
    if not regions:
        L += ["Decision rule (fixed in advance, from the simulation): a write signal requires (1) feedback window, "
              "rewarded trials: Stouffer z > 0 and Fisher p < 0.05; (2) stimulus-window control not significant; "
              "(3) error trials not showing the same positive effect (an arousal / repeat-bias confound shows all three).\n",
              "| Window | Sessions | RPE decoding r (median) | Fraction p < 0.05 | Binomial p | Fisher p | "
              "Stouffer z (signed) | Mice with mean z > 0 | Error trials: Stouffer z |", "|---|---|---|---|---|---|---|---|---|"]
        for w in ("feedback_a", "stim_a"):
            d = ses.dropna(subset=[f"{w}_p"])
            if not len(d):
                continue
            n = int((d[f"{w}_p"] < .05).sum())
            zs = d[f"{w}_z"].values
            mz = d.groupby("subject")[f"{w}_z"].mean()
            L.append(f"| {w} | {len(d)} | {d[f'{w}_dec_r'].median():.2f} | {n / len(d):.3f} | "
                     f"{binomtest(n, len(d), .05, 'greater').pvalue:.1e} | {fisher(d[f'{w}_p'].values):.1e} | "
                     f"{zs.sum() / np.sqrt(len(zs)):.2f} (p {norm.sf(zs.sum() / np.sqrt(len(zs))):.1e}) | "
                     f"{(mz > 0).sum()}/{len(mz)} | "
                     f"{d[f'{w}_z_err'].sum() / np.sqrt(len(d)) if f'{w}_z_err' in d else float('nan'):.2f} |")
        if "lab" in ses:
            L += ["\nBy lab (feedback window, Stouffer z):", "| Lab | Sessions | z |", "|---|---|---|"]
            for lab, g in ses.dropna(subset=["feedback_a_z"]).groupby("lab"):
                L.append(f"| {lab} | {len(g)} | {g.feedback_a_z.sum() / np.sqrt(len(g)):.2f} |")
    else:
        from verify_numbers import region_p, sig
        L += ["| Window | Regions significant (FDR 5%) / tested | Session fraction p < 0.05 |", "|---|---|---|"]
        for w in ("feedback_a", "stim_a"):
            pr = region_p(df.dropna(subset=[f"{w}_p"]), f"{w}_p")
            L.append(f"| {w} | {len(sig(pr))} / {len(pr)} | {(df[f'{w}_p'] < .05).mean():.3f} |")
            L.append(f"\nRegions ({w}): " + (", ".join(sorted(sig(pr))) or "none") + "\n")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / ("W2_REGIONS.md" if regions else "W2_WRITE_SIGNAL.md")).write_text(text + "\n")
    print(text)


def _next_design(tv, contrast, lrt, Hc, v):
    """Same trials and covariates as write_test's next-choice regression."""
    T = len(tv["c"]); idx = np.flatnonzero(v); c = tv["c"]
    t1 = idx[idx + 1 < T]; t1 = t1[tv["ok"][t1 + 1]]; t1 = t1[t1 >= K]
    side_cov = [c * np.abs(np.nan_to_num(contrast)), c * lrt, tv["rc"] * tv["sur"]]
    if Hc is not None:
        side_cov += [c * Hc[:, j] for j in range(Hc.shape[1])]
    B = np.column_stack([tv["base"][t1 + 1], np.column_stack(side_cov)[t1]])
    B = (B - B.mean(0)) / (B.std(0) + 1e-9)
    return t1, B, tv["y"][t1 + 1]


def planted(X, tv, contrast, lrt, Hc, v, g):
    """Real feedback-window activity plus a known write signal: on rewarded trials, each neuron's own response to
    reward surprise, scaled by how much the mouse actually updated on the next trial (beyond everything the
    regression already explains). g = size relative to the natural trial-to-trial spread of reward surprise."""
    T = len(tv["c"])
    t1, B, y = _next_design(tv, contrast, lrt, Hc, v)
    wb = logit_fit(B, y, lam=1.0)
    res = y - logit_prob(B, wb)
    r1 = tv["r"][t1]
    sig = np.zeros(T)
    sig[t1] = tv["c"][t1] * res * r1
    rew = t1[r1 > 0]
    if len(rew) < 20:
        return X
    sig[rew] = (sig[rew] - sig[rew].mean()) / (sig[rew].std() + 1e-12)
    delta = tv["r"] - tv["conf"]
    idx = np.flatnonzero(v)
    dv = delta[idx] - delta[idx].mean()
    beta = (dv @ (X[idx] - X[idx].mean(0))) / (dv @ dv + 1e-12)       # each neuron's response per unit of surprise
    scale = g * delta[rew].std()
    return X + np.outer(sig * scale, beta)


def check(cache, aux_dir, out_dir, every, jobs):
    """Real-data checks of the write test (feedback window, all neurons pooled): real, fake (neural deviation's sign
    flipped at random on each trial, which destroys any real link), planted (g = 0.05, 0.10, 0.25)."""
    import pandas as pd
    from joblib import Parallel, delayed
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from gatekeeper.multiverse import _sides
    from history_confounds import confounds

    def one(p):
        try:
            s = MVSession(p)
            v0 = cm.mask_of(s)
            if v0 is None or v0.sum() < 200:
                return None
            T = len(s.contrast)
            tv = task_vars(s.contrast, s.choice, s.feedback, cm.bayes_prior(_sides(s)) - 0.5)
            lrt = np.where(np.isfinite(s.rt) & (s.rt > 0), np.log(np.clip(s.rt, 0.05, 10)), np.nan)
            lrt = np.where(np.isfinite(lrt), lrt, np.nanmedian(lrt))
            Hc, _ = confounds(aux_dir, s.sid, T)
            if Hc is not None:
                Hc = (Hc - Hc.mean(0)) / (Hc.std(0) + 1e-9)
            v = v0 & s.valid["feedback"]
            parts = [s.X(q, "feedback_a") for q in s.regions]
            X = np.hstack([x for x in parts if x is not None]); X = X[:, X[v].std(0) > 0]
            rng = np.random.default_rng(zlib.crc32(s.sid.encode()) % 100000)
            row = dict(sid=s.sid, subject=s.subject)
            flip = rng.choice([-1.0, 1.0], T)
            for tag, Xc, fn in (("real", X, None), ("fake", X, lambda e: e * flip),
                                ("plant05", planted(X, tv, s.contrast, lrt, Hc, v, 0.05), None),
                                ("plant10", planted(X, tv, s.contrast, lrt, Hc, v, 0.10), None),
                                ("plant25", planted(X, tv, s.contrast, lrt, Hc, v, 0.25), None)):
                o = write_test(Xc, tv, s.contrast, lrt, Hc, v, rng, n_shift=100, eps_fn=fn)
                row.update({f"{tag}_z": o["z"], f"{tag}_p": o["p"], f"{tag}_z_err": o["z_err"]})
            return row
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True)
            return None

    paths = sorted(Path(cache).expanduser().glob("*.npz"))[::every]
    print(f"{len(paths)} sessions", flush=True)
    rows = [r for r in Parallel(n_jobs=jobs, verbose=5)(delayed(one)(p) for p in paths) if r]
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "w2_check.csv", index=False)
    L = [f"# W2 real-data checks ({len(df)} sessions, {df.subject.nunique()} mice)\n",
         "fake: neural deviation's sign flipped at random per trial (must be ~0.05, z ~0). planted: a write signal of "
         "known size added to the real activity, at 5%, 10% and 25% of the natural spread of reward surprise.\n",
         "| Condition | Fraction p < 0.05 (rewarded) | Stouffer z (rewarded) | Stouffer z (errors) |", "|---|---|---|---|"]
    for tag in ("real", "fake", "plant05", "plant10", "plant25"):
        L.append(f"| {tag} | {(df[f'{tag}_p'] < .05).mean():.3f} | {df[f'{tag}_z'].sum() / np.sqrt(len(df)):.2f} | "
                 f"{df[f'{tag}_z_err'].sum() / np.sqrt(len(df)):.2f} |")
    text = "\n".join(L)
    (Path(out_dir) / "W2_CHECK.md").write_text(text + "\n")
    print(text)


TARGETS = ("VTA", "SNc", "CP", "ACB")      # fixed in advance: claude/preregistration-w2-dopamine-striatum.md


def run_targets(cache, aux_dir, out_dir, jobs):
    """Pre-registered targeted test (Oct 26): the W2 write test in VTA, SNc, CP and ACB only."""
    import pandas as pd
    from joblib import Parallel, delayed
    from scipy.stats import chi2, norm
    from gatekeeper.infoflow_data import split_of
    out = Path(out_dir) / "sessions_targets"; out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(todo)} to run; target regions {TARGETS}", flush=True)
    t0 = time.time()

    def work(p):
        try:
            rows = _ibl_session(p, aux_dir, True, only=set(TARGETS))
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True); rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    try:
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(work)(p) for p in todo)
    except Exception:
        gen = Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)
    for i, n in enumerate(gen):
        if (i + 1) % 25 == 0 or i + 1 == len(todo):
            print(f"[{i + 1}/{len(todo)}] ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fs = [f for f in out.glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    stz = lambda z: float(z.sum() / np.sqrt(len(z))) if len(z) else np.nan
    L = ["# Pre-registered targeted write test: dopamine (VTA, SNc) and striatum (CP, ACB)\n",
         "Rule (fixed Oct 26 before results): rewarded Fisher p < 0.0125 and Stouffer z > 0; stimulus-window control "
         "Fisher p > 0.05; error trials not significantly positive (two-sided 0.05); same sign in both splits.\n",
         "| Region | Sessions | Mice | Rewarded z | Rewarded Fisher p | Control Fisher p | Error z (p two-sided) | "
         "z explore / confirm | Passes |", "|---|---|---|---|---|---|---|---|---|"]
    for r in TARGETS:
        d = df[df.region == r].dropna(subset=["feedback_a_p"]) if len(df) else df
        if not len(d):
            L.append(f"| {r} | 0 | 0 | - | - | - | - | - | no data |"); continue
        zr, pr = stz(d.feedback_a_z), fisher(d.feedback_a_p.values)
        pc = fisher(d.stim_a_p.dropna().values)
        ze = stz(d.feedback_a_z_err); pe = 2 * norm.sf(abs(ze))
        sp = d.subject.map(split_of)
        zs = {k: stz(d.feedback_a_z[sp == k]) for k in ("explore", "confirm")}
        ok = (pr < 0.0125) and (zr > 0) and (pc > 0.05) and not (ze > 0 and pe < 0.05) and all(
            np.isfinite(v) and v > 0 for v in zs.values())
        L.append(f"| {r} | {len(d)} | {d.subject.nunique()} | {zr:.2f} | {pr:.3g} | {pc:.3g} | {ze:.2f} ({pe:.3g}) | "
                 f"{zs['explore']:.2f} / {zs['confirm']:.2f} | {'YES' if ok else 'no'} |")
    text = "\n".join(L)
    (Path(out_dir) / "W2_TARGETS.md").write_text(text + "\n")
    print(text)


# ------------------------------------------------------------------------------------------- Steinmetz
def steinmetz(data_dir, out_dir, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    import steinmetz_replication as sr
    sessions = sr.load_sessions(data_dir)
    rows_b = []
    for i, d in enumerate(sessions):
        contrast, choice, fb, side = sr.task(d)
        rows_b.append(dict(sid=f"s{i}", subject=str(d["mouse_name"]),
                           **write_rule(contrast, choice, fb, np.zeros(len(choice)))))
    _behaviour_report(pd.DataFrame(rows_b), out_dir, "Steinmetz 2019")

    def work(i, d):
        contrast, choice, fb, side = sr.task(d)
        T = len(choice)
        tv = task_vars(contrast, choice, fb, np.zeros(T))
        rt = np.asarray(d["response_time"], float).ravel()
        lrt = np.log(np.clip(rt, 0.05, 10))
        Hc = sr.confounds(d, T)
        if Hc is not None:
            Hc = (Hc - Hc.mean(0)) / (Hc.std(0) + 1e-9)
        b = lambda t: 50 + np.round(np.asarray(t, float).ravel() / 0.01).astype(int)
        fbin = b(d["feedback_time"]); nb = d["spks"].shape[2]
        v = (choice >= 0) & (fbin + 20 <= nb); v[:10] = False
        S = d["spks"]
        Xf = np.stack([S[:, t, fbin[t]:fbin[t] + 20].sum(1) for t in range(T)]).astype(float)
        Xs = S[:, :, 50:60].sum(2).T.astype(float)
        rng = np.random.default_rng(91 + i)
        row = dict(sid=f"s{i}", subject=str(d["mouse_name"]), lab="cortexlab", region="ALL")
        for w, X in (("feedback_a", Xf), ("stim_a", Xs)):
            X = X[:, X[v].std(0) > 0]
            row.update({f"{w}_{k}": val for k, val in write_test(X, tv, contrast, lrt, Hc, v, rng).items()})
        return row

    res = Parallel(n_jobs=jobs, verbose=5)(delayed(work)(i, d) for i, d in enumerate(sessions))
    write_report(pd.DataFrame(res), out_dir, "Steinmetz 2019")


def steinmetz_acb(data_dir, out_dir, jobs, region="ACB"):
    """Pre-registered addendum (Oct 26, 17:30): independent confirmation of the IBL ACB lead in Steinmetz 2019.
    One region, one test. Planted checks (10%, 25%) are reported only to show whether the test had power."""
    import pandas as pd
    from joblib import Parallel, delayed
    from scipy.stats import chi2, norm
    import steinmetz_replication as sr
    sessions = sr.load_sessions(data_dir)

    def work(i, d):
        areas = np.asarray(d["brain_area"]).astype(str)
        sel = areas == region
        if sel.sum() < MIN_UNITS:
            return None
        contrast, choice, fb, side = sr.task(d)
        T = len(choice)
        tv = task_vars(contrast, choice, fb, np.zeros(T))
        rt = np.asarray(d["response_time"], float).ravel()
        lrt = np.log(np.clip(rt, 0.05, 10))
        Hc = sr.confounds(d, T)
        if Hc is not None:
            Hc = (Hc - Hc.mean(0)) / (Hc.std(0) + 1e-9)
        b = lambda t: 50 + np.round(np.asarray(t, float).ravel() / 0.01).astype(int)
        fbin = b(d["feedback_time"]); nb = d["spks"].shape[2]
        v = (choice >= 0) & (fbin + 20 <= nb); v[:10] = False
        S = d["spks"][sel]
        Xf = np.stack([S[:, t, fbin[t]:fbin[t] + 20].sum(1) for t in range(T)]).astype(float)
        Xs = S[:, :, 50:60].sum(2).T.astype(float)
        Xf = Xf[:, Xf[v].std(0) > 0]; Xs = Xs[:, Xs[v].std(0) > 0]
        if Xf.shape[1] < MIN_UNITS:
            return None
        rng = np.random.default_rng(191 + i)
        row = dict(sid=f"s{i}", subject=str(d["mouse_name"]), units=int(Xf.shape[1]))
        for tag, X, w in (("real", Xf, "fb"), ("control", Xs, "stim"),
                          ("plant10", planted(Xf, tv, contrast, lrt, Hc, v, 0.10), "fb"),
                          ("plant25", planted(Xf, tv, contrast, lrt, Hc, v, 0.25), "fb")):
            o = write_test(X, tv, contrast, lrt, Hc, v, rng)
            row.update({f"{tag}_z": o["z"], f"{tag}_p": o["p"], f"{tag}_z_err": o["z_err"]})
        return row

    res = [r for r in Parallel(n_jobs=jobs, verbose=0)(delayed(work)(i, d) for i, d in enumerate(sessions)) if r]
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    stz = lambda z: float(np.sum(z) / np.sqrt(len(z))) if len(z) else np.nan
    L = [f"# Pre-registered confirmation in Steinmetz 2019: write signal in {region}\n",
         "Rule (fixed Oct 26, 17:30, before running): rewarded Fisher p < 0.05 and Stouffer z > 0; control Fisher "
         "p > 0.05; error trials not significantly positive. Planted rows show whether the test had power.\n"]
    if not res:
        L.append(f"No session has {MIN_UNITS}+ {region} units. The lead cannot be tested in this dataset.")
    else:
        df = pd.DataFrame(res)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(out_dir) / f"steinmetz_{region}.csv", index=False)
        ze = stz(df.real_z_err); pe = 2 * norm.sf(abs(ze))
        pr, zr, pc = fisher(df.real_p.values), stz(df.real_z), fisher(df.control_p.values)
        ok = pr < 0.05 and zr > 0 and pc > 0.05 and not (ze > 0 and pe < 0.05)
        L += [f"{len(df)} sessions, {df.subject.nunique()} mice, median {int(df.units.median())} {region} units\n",
              "| Row | Stouffer z | Fisher p |", "|---|---|---|",
              f"| real (rewarded) | {zr:.2f} | {pr:.3g} |", f"| control (stimulus window) | {stz(df.control_z):.2f} | {pc:.3g} |",
              f"| errors | {ze:.2f} | two-sided {pe:.3g} |",
              f"| planted 10% | {stz(df.plant10_z):.2f} | {fisher(df.plant10_p.values):.3g} |",
              f"| planted 25% | {stz(df.plant25_z):.2f} | {fisher(df.plant25_p.values):.3g} |",
              f"\nPasses: {'YES' if ok else 'no'}"]
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / f"W2_STEINMETZ_{region}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "behaviour", "run", "steinmetz", "check", "targets", "steinmetz_acb"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--aux", default="data/part3_aux2")
    ap.add_argument("--data", default="~/Downloads/MULTIVERSE/steinmetz_data")
    ap.add_argument("--out", default="results/reward_write")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--regions", action="store_true")
    a = ap.parse_args()
    if a.step == "sim":
        sim()
    elif a.step == "behaviour":
        behaviour(a.cache, a.out)
    elif a.step == "steinmetz_acb":
        steinmetz_acb(a.data, a.out + "/steinmetz", a.jobs)
    elif a.step == "targets":
        run_targets(a.cache, a.aux, a.out + "/ibl", a.jobs)
    elif a.step == "check":
        check(a.cache, a.aux, a.out + "/check", a.every if a.every > 1 else 4, a.jobs)
    elif a.step == "run":
        run(a.cache, a.aux, a.out + "/ibl", a.limit, a.jobs, a.every, a.regions)
    else:
        steinmetz(a.data, a.out + "/steinmetz", a.jobs)
