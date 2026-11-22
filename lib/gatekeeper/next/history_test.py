"""Does a region encode the PRIOR, or only the last few trials?  (conditional test)

Idea: remove what the last k trials can explain, from BOTH sides, then test what is left.

  Neural side (label-free, so the null stays valid):
      regress each neuron, across trials, on the last k trials' stimulus side, choice, feedback and
      choice x feedback (4k columns + intercept); keep the residuals.
  Label side (each label uses ONLY its own stimulus sequence, so real and pseudo labels are treated alike):
      regress the prior on the last k stimulus sides OF THAT SAME SEQUENCE; keep the residual.
      The residual is the part of the prior that is NOT a linear function of the last k stimuli,
      i.e. evidence integrated over a longer history.
  Then run the usual tests (new test; Findling-style decoding) on residuals vs residuals,
  with pseudo-session nulls built exactly the same way.

Null hypothesis this tests: activity depends on task history only through the last k trials (linearly),
plus drift. Rejecting it means activity tracks the longer-horizon prior.
Validated in simulation by sim_check() before any real-data use.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "driftaware" / "src"))
from driftaware import drift_test  # noqa: E402

UNBIASED = 90
KS = (1, 3, 5)


def lags(v, k):
    """(T, k) matrix of v shifted by 1..k trials (zeros before the start)."""
    T = len(v)
    out = np.zeros((T, k))
    for j in range(1, k + 1):
        out[j:, j - 1] = v[:-j]
    return out


def history_design(side, choice, feedback, k):
    """Task history of the last k trials, centered codes.
    side: 1 right / 0 left / nan;  choice: 1 right / 0 left / -1 none;  feedback: +1 / -1 / 0."""
    s = np.nan_to_num(np.asarray(side, float) - 0.5)
    c = np.where(np.asarray(choice) >= 0, np.asarray(choice, float) - 0.5, 0.0)
    f = np.asarray(feedback, float) / 2.0
    return np.hstack([lags(s, k), lags(c, k), lags(f, k), lags(c * f * 2, k)])


def residualize(A, D):
    """Residuals of each column of A after least-squares regression on [1, D]."""
    D1 = np.column_stack([np.ones(len(D)), D])
    beta, *_ = np.linalg.lstsq(D1, A, rcond=None)
    return A - D1 @ beta


def label_residuals(priors, sides, v, k, Hv):
    """priors, sides: lists of full-length arrays (column 0 real, then nulls). EVERY label is residualized on
    the same real task history Hv (the regressors removed from the neurons) PLUS the last k stimulus sides of
    its OWN sequence. So all labels end up in the same subspace as the residual neurons, and each loses the
    part explained by its own recent stimuli."""
    cols = []
    for pi, sd in zip(priors, sides):
        D = np.hstack([Hv, lags(np.nan_to_num(np.asarray(sd, float) - 0.5), k)[v]])
        cols.append(residualize(pi[v][:, None], D)[:, 0])
    return np.column_stack(cols)


def cond_drift_test(X, H, priors, sides, v, k, pmax=10, k_shared=1, smooth=50):
    """Conditional version of the new test. Everything (neurons, labels, history regressors) goes through the
    same label-free steps: remove the shared slow component, then each neuron's own AR filter. In that
    whitened space, the neuron is residualized on the filtered history H, and each label on [filtered H,
    filtered last-k stimulus sides of its OWN sequence]. S = sum over neurons of squared correlations.
    Returns the permutation p-value (column 0 = real label)."""
    from scipy.signal import lfilter
    from driftaware.core import _slow_components, ar_bic
    Xv = X[v]; Xv = Xv[:, Xv.std(0) > 0]
    Y = np.column_stack([p[v] for p in priors])                                     # (T, M+1)
    L = np.stack([lags(np.nan_to_num(np.asarray(sd, float) - 0.5), k)[v] for sd in sides])  # (M+1, T, k)
    Z = (Xv - Xv.mean(0)) / Xv.std(0)
    U = _slow_components(Xv, k_shared, smooth)
    Zc = Z - U @ (U.T @ Z)
    Yc = Y - U @ (U.T @ Y)
    Hc = H - U @ (U.T @ H)
    Lc = L - np.einsum("tu,mus->mts", U, np.einsum("tu,mts->mus", U, L))
    T, N = Zc.shape
    S = np.zeros(Y.shape[1])
    for i in range(N):
        phi = ar_bic(Zc[:, i], pmax)
        a = np.r_[1.0, -phi]; p = len(phi)
        f = lambda A: lfilter(a, [1.0], A - A.mean(0), axis=0)[p:]
        e = f(Zc[:, i])
        G = f(Yc)
        Hf = np.column_stack([np.ones(T - p), f(Hc)])
        Lf = lfilter(a, [1.0], Lc - Lc.mean(1, keepdims=True), axis=1)[:, p:, :]
        # residualize neuron and labels on filtered H
        e = e - Hf @ np.linalg.lstsq(Hf, e, rcond=None)[0]
        G = G - Hf @ np.linalg.lstsq(Hf, G, rcond=None)[0]
        Lf = Lf - np.einsum("tq,mqs->mts", Hf, np.linalg.lstsq(Hf, Lf.transpose(1, 0, 2).reshape(T - p, -1), rcond=None)[0].reshape(Hf.shape[1], Lf.shape[0], -1).transpose(1, 0, 2))
        # residualize each label on its OWN filtered recent stimulus sides (batched k x k solves)
        QtQ = np.einsum("mts,mtu->msu", Lf, Lf) + 1e-9 * np.eye(Lf.shape[2])
        Qtg = np.einsum("mts,tm->ms", Lf, G)
        G = G - np.einsum("mts,ms->tm", Lf, np.linalg.solve(QtQ, Qtg[..., None])[..., 0])
        e = (e - e.mean()) / (e.std() + 1e-12)
        G = G - G.mean(0); G = G / (np.linalg.norm(G, axis=0) + 1e-12)
        S += (e @ G) ** 2 / len(e)
    return float((1 + (S[1:] >= S[0]).sum()) / len(S))


# ---------------------------------------------------------------- simulation check
def _sim_session(rng, T, N, mode, drift=0.6, beta=0.15):
    from gatekeeper.multiverse import pseudo_pl
    from compare_methods import bayes_prior
    pl = pseudo_pl(T, rng)
    side = (rng.random(T) >= pl).astype(float)
    prior = bayes_prior(side)
    choice = np.where(rng.random(T) < 0.8, side, 1 - side).astype(int)
    fb = np.where(choice == side, 1.0, -1.0)
    z = np.zeros((T, N)); e = rng.standard_normal((T, N)); z[0] = e[0]
    for t in range(1, T):
        z[t] = 0.98 * z[t - 1] + np.sqrt(1 - 0.98 ** 2) * e[t]
    common = z[:, :1] * 0.6 + z * 0.8
    X = np.sqrt(drift) * common / common.std(0) + np.sqrt(1 - drift) * rng.standard_normal((T, N))
    w = rng.standard_normal((N, 4))
    H1 = history_design(side, choice, fb, 1)
    if mode == "recent":      # strong encoding of the LAST trial only, no prior
        X = X + 3 * beta * H1 @ w.T
    elif mode == "prior":     # encoding of the prior itself
        X = X + 2 * beta * np.outer(prior - prior.mean(), w[:, 0]) / prior.std()
    return X, side, choice, fb, prior, pl


def sim_check(reps=150, T=600, N=30, n_null=100, k=3, seed=0):
    from joblib import Parallel, delayed
    from gatekeeper.multiverse import pseudo_pl
    from compare_methods import bayes_prior

    def one(i, mode):
        rng = np.random.default_rng(seed + i)
        X, side, choice, fb, prior, pl = _sim_session(rng, T, N, mode)
        v = np.ones(T, bool); v[:UNBIASED] = False
        sides = [side]; priors = [prior]
        for _ in range(n_null):
            p2 = pseudo_pl(T, rng); s2 = (rng.random(T) >= p2).astype(float)
            sides.append(s2); priors.append(bayes_prior(s2))
        Y = np.column_stack([p[v] for p in priors])
        uncond = drift_test(X[v], Y[:, 0], Y[:, 1:]).p
        cond = cond_drift_test(X, history_design(side, choice, fb, k)[v], priors, sides, v, k)
        return uncond < 0.05, cond < 0.05

    out = {}
    for mode in ("none", "recent", "prior"):
        r = np.array(Parallel(n_jobs=-1)(delayed(one)(i, mode) for i in range(reps)))
        out[mode] = r.mean(0)
        print(f"{mode:7s} unconditional reject {r[:, 0].mean():.3f}   conditional (k={k}) reject {r[:, 1].mean():.3f}", flush=True)
    return out


# ---------------------------------------------------------------- real data
def session_rows(path, n_null, seed, split, with_findling):
    import compare_methods as cm
    from gatekeeper.multiverse import _sides, pseudo_pl
    from gatekeeper.mv_data import MVSession
    s = MVSession(path)
    if split != "all" and s.split != split:
        return []
    v = cm.mask_of(s)
    if v is None or v.sum() < 100:
        return []
    rng = np.random.default_rng(seed)
    Tf = len(s.contrast)
    real_side = _sides(s)
    rows = []
    for var in ("bprior", "ctl_bprior"):
        sides, priors = [], []
        n_draw = n_null + (1 if var.startswith("ctl_") else 0)
        if not var.startswith("ctl_"):
            sides.append(real_side); priors.append(cm.bayes_prior(real_side))
        for _ in range(n_draw):
            pl = pseudo_pl(Tf, rng); sd = (rng.random(Tf) >= pl).astype(float)
            sides.append(sd); priors.append(cm.bayes_prior(sd))
        Y = np.column_stack([p[v] for p in priors])
        for r in s.regions:
            X = s.X(r, "prior_a")
            if X is None:
                continue
            X = X[v]
            X = X[:, X.std(0) > 0]
            if X.shape[1] < cm.MIN_UNITS:
                continue
            row = dict(sid=s.sid, subject=s.subject, split=s.split, var=var, region=r, n_units=X.shape[1],
                       n_trials=int(v.sum()), p_new_k0=drift_test(X, Y[:, 0], Y[:, 1:]).p)
            for k in KS:
                Hv = history_design(real_side, s.choice, s.feedback, k)[v]
                Xfull = np.zeros((len(v), X.shape[1])); Xfull[v] = X
                row[f"p_new_k{k}"] = cond_drift_test(Xfull, Hv, priors, sides, v, k)
                if with_findling and k == 3:
                    Xr = residualize(X, Hv)
                    Yr = label_residuals(priors, sides, v, k, Hv)
                    R = cm.r2(Yr, cm.cv_predictions(Xr, Yr))
                    row["p_findling_k3"] = cm._p(R)
            rows.append(row)
    return rows


def run(cache_dir, out_dir, split="confirm", n_null=200, n_jobs=-1, limit=0, with_findling=True, log=print):
    import pandas as pd
    from joblib import Parallel, delayed
    from gatekeeper.mv_data import MVSession
    paths = sorted(Path(cache_dir).expanduser().glob("*.npz"))
    paths = [p for p in paths if split == "all" or MVSession(p).split == split]
    todo = paths[:limit] if limit else paths
    per = Path(out_dir) / f"sessions_{split}"
    per.mkdir(parents=True, exist_ok=True)
    jobs = [(i, p) for i, p in enumerate(todo) if not (per / f"{p.stem}.csv").exists()]
    log(f"{len(todo)} sessions ({split}); {len(todo) - len(jobs)} already saved, {len(jobs)} to run; findling={with_findling}")
    t0 = time.time()

    def one(i, p):
        t = time.time()
        pd.DataFrame(session_rows(p, n_null, 5151 + i, split, with_findling)).to_csv(per / f"{p.stem}.csv", index=False)
        print(f"session {p.stem} done in {(time.time() - t) / 60:.1f} min", flush=True)
    Parallel(n_jobs=n_jobs, verbose=0)(delayed(one)(i, p) for i, p in jobs)
    parts = [pd.read_csv(f) for f in per.glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat(parts, ignore_index=True)
    out = Path(out_dir) / f"history_rows_{split}.csv"
    df.to_csv(out, index=False)
    log(f"done in {(time.time() - t0) / 60:.1f} min: {len(df)} rows -> {out}")
    return df


if __name__ == "__main__":
    import argparse
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true", help="run the simulation check only")
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--split", default="confirm", choices=["explore", "confirm", "all"])
    ap.add_argument("--out", default="results/history")
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-findling", action="store_true", help="skip the slow decoder (new test only)")
    a = ap.parse_args()
    if a.sim:
        sim_check()
    else:
        run(a.cache, a.out, a.split, a.nulls, a.jobs, a.limit, not a.no_findling)
