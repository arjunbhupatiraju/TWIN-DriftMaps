"""Head-to-head on the same IBL sessions: the new test vs the two most recent prior-decoding pipelines.

Target: the Bayes-optimal prior (the main target of Findling et al. 2025, Nature) and the block prior, each
with a fake version (ctl_) drawn from the task's generative process, so it does not exist in the brain.
Window: -600 to -100 ms before stimulus onset. The first 90 (unbiased) trials are excluded.

Methods (every method uses the SAME real and null labels in each session x region):
    findling        Lasso regression, nested 5-fold cross-validation (outer and inner), cross-validated R2,
                    pseudo-session null. Our reimplementation of Findling et al. 2025 (Methods, Ephys).
    findling_shuf   the same decoder with a trial-shuffle null (the common decoding procedure)
    zhang           Zhang et al. 2025-style trial-correlation decoding: the cross-validated Lasso predictions are
                    smoothed across trials by an AR(1) Kalman smoother whose parameters come from the labels'
                    own autocorrelation (an empirical prior over trials), then scored by R2. Pseudo-session null.
    plain           standard pooled single-neuron statistic, pseudo-session null
    new             the drift-aware test (driftaware), pseudo-session null

Region maps: Fisher's method across sessions, then Benjamini-Hochberg (5% and 1%, as in Findling et al.).
Differences from the published pipelines are listed in COMPARISON_NOTES.md.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "driftaware" / "src"))
from driftaware import drift_test  # noqa: E402
from driftaware.core import plain_stat  # noqa: E402
from gatekeeper.multiverse import _sides, pseudo_pl  # noqa: E402
from gatekeeper.mv_data import MVSession  # noqa: E402

VARS = ("bprior", "prior", "ctl_bprior", "ctl_prior")
ALPHAS = np.logspace(-3, 0, 6)
MIN_UNITS = 5
UNBIASED = 90
HAZARD = 1 / 60


def bayes_prior(side, unbiased=UNBIASED, hazard=HAZARD, p_bias=0.8):
    """P(next stimulus on the right | past stimulus sides), for the IBL block structure (forward HMM filter).
    Blocks: right-biased (P(right) = 0.8) or left-biased (0.2), switching with probability `hazard` per trial;
    0.5 during the first `unbiased` trials."""
    T = len(side)
    out = np.full(T, 0.5)
    b = 0.5                                     # P(right-biased block)
    for t in range(T):
        if t >= unbiased:
            out[t] = b * p_bias + (1 - b) * (1 - p_bias)
        s = side[t]
        if t >= unbiased and np.isfinite(s):
            lr, ll = (p_bias, 1 - p_bias) if s == 1 else (1 - p_bias, p_bias)
            b = b * lr / (b * lr + (1 - b) * ll)
        if t >= unbiased - 1:
            b = b * (1 - hazard) + (1 - b) * hazard
    return out


def mask_of(s):
    pl = s.meta.get("probabilityLeft")
    if pl is None or len(pl) != len(s.contrast):
        return None
    v = s.valid["stimOn"] & (s.choice >= 0)
    v[:UNBIASED] = False
    return v


def label_matrix(s, var, v, n_null, rng):
    """Column 0 = the tested label (real, or one fake draw), columns 1.. = pseudo-session nulls."""
    Tf = len(s.contrast)
    pl_real = np.asarray(s.meta["probabilityLeft"], float)

    def draw(kind):
        if kind == "real":
            pl, side = pl_real, _sides(s)
        else:
            pl = pseudo_pl(Tf, rng)
            side = (rng.random(Tf) >= pl).astype(float)
        if var.endswith("bprior"):
            return bayes_prior(side)[v]
        return (pl[v] < 0.5).astype(float)
    first = draw("real") if not var.startswith("ctl_") else draw("fake")
    return np.column_stack([first] + [draw("fake") for _ in range(n_null)])


FOLDS = ["random"]          # "random" (interleaved trials, the usual sklearn KFold(shuffle=True)) or "contiguous"


def _kfold(T, k, rng):
    idx = rng.permutation(T) if FOLDS[0] == "random" else np.arange(T)
    parts = np.array_split(idx, k)
    return [(np.sort(np.concatenate(parts[:i] + parts[i + 1:])), np.sort(parts[i])) for i in range(k)]


def _lasso_multi(X, Y, alpha):
    from sklearn.linear_model import Lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Lasso(alpha=alpha, max_iter=2000, tol=1e-3, precompute=True).fit(X, Y)   # Gram matrix: same fit, ~5x faster
    return m


def cv_predictions(X, Y, outer=5, inner=5, seed=0):
    """Nested-CV Lasso predictions for every label column (alpha chosen per column in the inner loop)."""
    T, M = Y.shape
    pred = np.zeros((T, M))
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    rng = np.random.default_rng(seed)
    for tr, te in _kfold(T, outer, rng):
        err = np.zeros((len(ALPHAS), M))
        for itr, ite in _kfold(len(tr), inner, rng):
            a_tr, a_te = tr[itr], tr[ite]
            for j, a in enumerate(ALPHAS):
                m = _lasso_multi(Xz[a_tr], Y[a_tr], a)
                err[j] += ((m.predict(Xz[a_te]).reshape(len(a_te), M) - Y[a_te]) ** 2).sum(0)
        best = err.argmin(0)
        for j, a in enumerate(ALPHAS):
            cols = np.flatnonzero(best == j)
            if len(cols) == 0:
                continue
            m = _lasso_multi(Xz[tr], Y[tr][:, cols], a)
            pred[np.ix_(te, cols)] = m.predict(Xz[te]).reshape(len(te), len(cols))
    return pred


def r2(Y, P):
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ((Y - P) ** 2).sum(0) / (ss + 1e-12)


def ar1_smooth(pred, y):
    """RTS smoother of the decoder output, with an AR(1) latent whose coefficient and variance come from y."""
    yc = y - y.mean()
    phi = float(np.clip((yc[1:] @ yc[:-1]) / (yc @ yc + 1e-12), 0, 0.999))
    q = max(yc.var() * (1 - phi ** 2), 1e-8)
    r = max(float(np.var(pred - y)), 1e-8)
    z = pred - y.mean()
    T = len(z)
    m, P = np.zeros(T), np.zeros(T)
    mp, Pp = 0.0, yc.var() + 1e-8
    for t in range(T):
        K = Pp / (Pp + r)
        m[t] = mp + K * (z[t] - mp)
        P[t] = (1 - K) * Pp
        mp, Pp = phi * m[t], phi ** 2 * P[t] + q
    ms = m.copy()
    for t in range(T - 2, -1, -1):
        Pn = phi ** 2 * P[t] + q
        G = P[t] * phi / Pn
        ms[t] = m[t] + G * (ms[t + 1] - phi * m[t])
    return ms + y.mean()


def _p(stat):
    return float((1 + (stat[1:] >= stat[0]).sum()) / len(stat))


def session_rows(path, n_null, seed, split, folds="random"):
    FOLDS[0] = folds
    s = MVSession(path)
    if split != "all" and s.split != split:
        return []
    v = mask_of(s)
    if v is None or v.sum() < 100:
        return []
    rng = np.random.default_rng(seed)
    rows = []
    for var in VARS:
        Y = label_matrix(s, var, v, n_null, rng)
        if np.std(Y[:, 0]) == 0:
            continue
        Ysh = np.column_stack([Y[:, 0]] + [rng.permutation(Y[:, 0]) for _ in range(n_null)])
        for r in s.regions:
            X = s.X(r, "prior_a")
            if X is None:
                continue
            X = X[v]
            X = X[:, X.std(0) > 0]
            if X.shape[1] < MIN_UNITS:
                continue
            P = cv_predictions(X, Y)
            R = r2(Y, P)
            Psh = cv_predictions(X, Ysh)
            Rsh = r2(Ysh, Psh)
            Rz = np.array([r2(Y[:, [j]], ar1_smooth(P[:, j], Y[:, j])[:, None])[0] for j in range(Y.shape[1])])
            rows.append(dict(sid=s.sid, subject=s.subject, split=s.split, var=var, region=r, n_units=X.shape[1],
                             n_trials=int(v.sum()), r2_findling=float(R[0]), r2_zhang=float(Rz[0]),
                             p_findling=_p(R), p_findling_shuf=_p(Rsh), p_zhang=_p(Rz),
                             p_plain=_p(plain_stat(X, Y)), p_new=drift_test(X, Y[:, 0], Y[:, 1:]).p))
    return rows


def run(cache_dir, out_csv, split="confirm", n_null=200, n_jobs=-1, limit=0, folds="random", log=print):
    FOLDS[0] = folds
    import pandas as pd
    from joblib import Parallel, delayed
    paths = sorted(Path(cache_dir).expanduser().glob("*.npz"))
    paths = [p for p in paths if split == "all" or MVSession(p).split == split]
    todo = paths[:limit] if limit else paths
    per = Path(out_csv).parent / f"sessions_{split}"
    per.mkdir(parents=True, exist_ok=True)
    jobs = [(i, p) for i, p in enumerate(todo) if not (per / f"{p.stem}.csv").exists()]
    log(f"{len(todo)} sessions ({split}), {n_null} null labels each; {len(todo) - len(jobs)} already saved, {len(jobs)} to run")
    t0 = time.time()

    def one(i, p):
        t = time.time()
        rows = session_rows(p, n_null, 4242 + i, split, folds)
        pd.DataFrame(rows).to_csv(per / f"{p.stem}.csv", index=False)       # saved as soon as it finishes
        print(f"session {p.stem} done in {(time.time() - t) / 60:.1f} min", flush=True)
    Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(i, p) for i, p in jobs)
    parts = []
    for p in todo:
        f = per / f"{p.stem}.csv"
        if f.exists() and f.stat().st_size > 1:
            try:
                parts.append(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                pass
    df = pd.concat(parts, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"done in {(time.time() - t0) / 60:.1f} min: {len(df)} rows -> {out_csv}")
    return df
