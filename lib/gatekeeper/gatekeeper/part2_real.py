"""Part 2, real-brain validation on the IBL cache (one row per session x region x variable x unit set).

For the pooled single-neuron statistic S = T * sum_i corr(x_i, y)^2 (the most common analysis in the literature:
61% of coded claims use single-neuron tests), compute p-values under every null:
    naive     chi2_N (independent trials and neurons)
    shuffle   trial permutation (200)
    circ      circular shift (200)
    pseudo    labels redrawn from IBL's task generative process / imposter sessions (200)  - the reference
    dct_hac   drift-corrected test, Bartlett-tapered lag window
    dct_long  drift-corrected test, untapered window K = T/3
    dct_pw    drift-corrected test, label prewhitened
plus the measured drift quantities that the theory uses:
    V_hat     tr(Sigma)/N = measured Bartlett inflation (Sigma with K = T/3, untapered)
    pred_fp   parameter-free predicted shuffle false-positive probability for this session
              (S ~ Satterthwaite(Sigma) vs the shuffle critical value from Satterthwaite(R0))
Fake variables (ctl_*) give the false-positive rate of each null; real variables give power.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .multiverse import (MIN_PER_CLASS, MIN_UNITS, WINDOW_OF, _ALPHA_CUR, _base, design, label_matrix,
                         null_draw, subjective_alpha)
from .mv_data import MVSession
from theory.corrected_test import dct, dct_pw, _acf

VARS = ("stim", "choice", "feedback", "prior", "sprior",
        "ctl_stim", "ctl_choice", "ctl_feedback", "ctl_prior", "ctl_sprior")
N_PERM = 200


def _S(Z, Y):
    T = Z.shape[0]
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    return T * ((Z.T @ Y / T) ** 2).sum(0)


def _p(S):
    return float((1 + (S[1:] >= S[0]).sum()) / len(S))


def _sigma(Z, yc, K):
    T = len(yc)
    ry = _acf(yc, K)
    Sig = Z.T @ Z / T
    R0 = Sig.copy()
    for k in range(1, K + 1):
        G = Z[:-k].T @ Z[k:] / T
        Sig += ry[k] * (G + G.T)
    return Sig, R0


def _satt(M):
    t1, t2 = np.trace(M), np.sum(M * M)
    return t2 / t1, t1 ** 2 / t2


def session_rows(path, pools, seed=0, split="explore", units=("good", "all")):
    views = {u: MVSession(path, all_units=(u == "all")) for u in units}
    s = views[units[0]]
    if split != "all" and s.split != split:
        return []
    rng = np.random.default_rng(seed)
    rows, Xc = [], {}
    for var in VARS:
        b = _base(var)
        pool = None
        if b in pools:
            pool = np.concatenate([y for sid, y in pools[b] if sid != s.sid] or [np.zeros(1)])
        d = design(s, var, "a")
        if d[0] is None:
            continue
        if b == "sprior":
            _ALPHA_CUR[0] = subjective_alpha(s)
        if var.startswith("ctl_"):
            v, _, crng = d
            y = null_draw(var, v, len(s.contrast), pool, crng)
        else:
            v, y = d
        y = np.asarray(y, float)
        if b == "sprior":
            if len(y) < 2 * MIN_PER_CLASS or np.std(y) == 0:
                continue
        elif min((y == 0).sum(), (y == 1).sum()) < MIN_PER_CLASS:
            continue
        T = len(y)
        Ysh = label_matrix(y, "shuffle", N_PERM, rng, var, v, len(s.contrast), pool)
        Yci = label_matrix(y, "circ", N_PERM, rng, var, v, len(s.contrast), pool)
        Yps = label_matrix(y, "pseudo", N_PERM, rng, var, v, len(s.contrast), pool)
        yc = (y - y.mean()) / (y.std() + 1e-12)
        win = f"{WINDOW_OF.get(b, b)}_a"
        for u in units:
            sv = views[u]
            for r in sv.regions:
                if (u, r, win) not in Xc:
                    Xc[(u, r, win)] = sv.X(r, win)
                X = Xc[(u, r, win)]
                if X is None:
                    continue
                X = X[v]
                X = X[:, X.std(0) > 0]
                N = X.shape[1]
                if N < MIN_UNITS:
                    continue
                Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
                S0 = _S(Z, yc[:, None])[0]
                row = dict(sid=s.sid, subject=s.subject, lab=s.lab, var=var, units=u, region=r, n_units=N, n_trials=T,
                           p_naive=float(chi2.sf(S0, N)), p_shuffle=_p(_S(Z, Ysh)), p_circ=_p(_S(Z, Yci)),
                           p_pseudo=_p(_S(Z, Yps)), p_dct_hac=dct(X, y)[1], p_dct_long=dct(X, y, K=T // 3, taper=False)[1],
                           p_dct_pw=dct_pw(X, y)[1])
                Sig, R0 = _sigma(Z, yc, T // 3)
                a0, n0 = _satt(R0)
                crit = a0 * chi2.isf(0.05, n0)
                tr = np.trace(Sig)
                if tr > 0:
                    a1, n1 = _satt(Sig)
                    row["pred_fp"] = float(chi2.sf(crit / a1, n1))
                else:
                    row["pred_fp"] = np.nan
                row["V_hat"] = float(tr / N)
                row["R0_offdiag"] = float((R0.sum() - np.trace(R0)) / (N * (N - 1)))
                rows.append(row)
    return rows


def run(cache_dir, out_csv, split="explore", n_jobs=-1, limit=0, log=print):
    import pandas as pd
    from joblib import Parallel, delayed
    allp = sorted(Path(cache_dir).glob("*.npz"))
    paths = [p for p in allp if split == "all" or MVSession(p).split == split]
    log(f"{len(paths)} sessions ({split}); building imposter pools ...")
    pools = {}
    for b in ("choice", "feedback"):
        seqs = []
        for p in paths:
            s = MVSession(p)
            vv, yy = design(s, b, "a")[:2]
            if vv is not None:
                seqs.append((s.sid, yy))
        pools[b] = seqs
    t0 = time.time()
    todo = paths[:limit] if limit else paths
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(session_rows)(p, pools, 9000 + i, split) for i, p in enumerate(todo))
    df = pd.DataFrame([r for rows in res for r in rows])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"done in {(time.time() - t0) / 60:.0f} min: {len(df)} rows -> {out_csv}")
    return df
