"""The new model on real IBL data.

Per session x region x variable (real and fake), the pooled single-neuron statistic under four schemes:
    S_pseudo      plain statistic, pseudo-session null (IBL generative model)          - current gold standard
    S_imposter    plain statistic, imposter null (labels from OTHER sessions)          - no task model needed
    G_pseudo      GLS statistic (each neuron whitened by its own drift model), pseudo null
    G_imposter    GLS statistic, imposter null
    NEW_imposter  shared slow drift removed + GLS whitening, imposter null              - THE NEW MODEL
    NEW_pseudo    same with pseudo-session null
Fake variables measure false positives; real variables measure power. Region maps follow in model_report.py.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from .multiverse import (MIN_PER_CLASS, MIN_UNITS, WINDOW_OF, _ALPHA_CUR, _base, _imposter, _prior_trace, _sides, design, label_matrix,
                         null_draw, subjective_alpha)
from .mv_data import MVSession
from .gls_test import gls_stat, ar_bic, remove_shared

VARS = ("stim_all", "choice", "feedback", "prior", "sprior",
        "ctl_stim_all", "ctl_choice", "ctl_feedback", "ctl_prior", "ctl_sprior")
N_PERM = 200


def _p(s):
    return float((1 + (s[1:] >= s[0]).sum()) / len(s))


def _plain(X, Y):
    T = X.shape[0]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Yc = Y - Y.mean(0); Yc /= (Yc.std(0) + 1e-12)
    return ((Z.T @ Yc / T) ** 2).sum(0)


def label_of(s, var, pools):
    """(mask, labels) for var in session s; fake variables drawn from the null process as in Part 1."""
    b = _base(var)
    d = design(s, var, "a")
    if d[0] is None:
        return None, None
    if b == "sprior":
        _ALPHA_CUR[0] = subjective_alpha(s)
    if var.startswith("ctl_"):
        v, _, crng = d
        pool = np.concatenate([y for sid, y in pools.get(b, []) if sid != s.sid] or [np.zeros(1)]) if b in ("choice", "feedback") else None
        y = null_draw(var, v, len(s.contrast), pool, crng)
    else:
        v, y = d
    return v, np.asarray(y, float)


def session_rows(path, pools, seed=0, split="explore", units=("good",)):
    views = {u: MVSession(path, all_units=(u == "all")) for u in units}
    s = views[units[0]]
    if split != "all" and s.split != split:
        return []
    rng = np.random.default_rng(seed)
    rows, Xc = [], {}
    for var in VARS:
        b = _base(var)
        v, y = label_of(s, var, pools)
        if v is None:
            continue
        if b == "sprior":
            if len(y) < 2 * MIN_PER_CLASS or np.std(y) == 0:
                continue
        elif min((y == 0).sum(), (y == 1).sum()) < MIN_PER_CLASS:
            continue
        T = len(y)
        cpool = np.concatenate([yy for sid, yy in pools[b] if sid != s.sid] or [np.zeros(1)]) if b in ("choice", "feedback") else None
        Yps = label_matrix(y, "pseudo", N_PERM, rng, var, v, len(s.contrast), cpool)
        if b == "sprior":
            # FIX (2026-10-17): imposter subjective prior = another session's stimulus SIDES, run through THIS
            # session's fitted alpha, then this session's trial mask. Old version reused other mice's traces
            # (their own alpha) -> not exchangeable -> liberal.
            spool = np.concatenate([ss for sid, ss in pools["sides"] if sid != s.sid])
            Tf = len(s.contrast); a_here = subjective_alpha(s)
            Yim = np.column_stack([y] + [_prior_trace(_imposter(spool, Tf, rng), a_here)[v] for _ in range(N_PERM)]).astype(float)
        else:
            ipool = np.concatenate([yy for sid, yy in pools[b] if sid != s.sid])
            Yim = np.column_stack([y] + [_imposter(ipool, T, rng) for _ in range(N_PERM)]).astype(float)
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
                phis = [ar_bic(X[:, i], 10) for i in range(N)]
                # NEW MODEL: remove the top shared slow component (label-free), then per-neuron drift whitening
                Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
                Zs = np.apply_along_axis(lambda c: np.convolve(c, np.ones(50) / 50, mode="same"), 0, Z)
                U = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)[0][:, :1]
                Xn = Z - U @ (U.T @ Z)
                phin = [ar_bic(Xn[:, i], 10) for i in range(N)]
                phip = ar_bic(Xn.mean(1), 10)                     # FINAL MODEL (Arjun's playground winner): pooled drift model
                phipool = [phip] * N
                proj = lambda Y: Y - U @ (U.T @ Y)
                rows.append(dict(sid=s.sid, subject=s.subject, lab=s.lab, var=var, units=u, region=r, n_units=N, n_trials=T,
                                 p_S_pseudo=_p(_plain(X, Yps)), p_S_imposter=_p(_plain(X, Yim)),
                                 p_G_pseudo=_p(gls_stat(X, Yps, phis=phis)), p_G_imposter=_p(gls_stat(X, Yim, phis=phis)),
                                 p_NEW_pseudo=_p(gls_stat(Xn, proj(Yps), phis=phin)), p_NEW_imposter=_p(gls_stat(Xn, proj(Yim), phis=phin)),
                                 p_FINAL_pseudo=_p(gls_stat(Xn, proj(Yps), phis=phipool)), p_FINAL_imposter=_p(gls_stat(Xn, proj(Yim), phis=phipool)),
                                 ar_order=float(np.mean([len(p) for p in phis]))))
    return rows


def build_pools(paths):
    """Label sequences from every session, per base variable (for imposter nulls)."""
    pools = {}
    for b in ("stim_all", "choice", "feedback", "prior", "sprior"):
        seqs = []
        for p in paths:
            s = MVSession(p)
            v, y = label_of(s, b, {})
            if v is not None and len(y) > 0:
                seqs.append((s.sid, np.asarray(y, float)))
        pools[b] = seqs
    pools["sides"] = [(MVSession(p).sid, _sides(MVSession(p))) for p in paths]
    return pools


def run(cache_dir, out_csv, split="explore", n_jobs=-1, limit=0, units=("good",), log=print):
    import pandas as pd
    from joblib import Parallel, delayed
    allp = sorted(Path(cache_dir).glob("*.npz"))
    paths = [p for p in allp if split == "all" or MVSession(p).split == split]
    log(f"{len(paths)} sessions ({split}); building label pools ...")
    pools = build_pools(paths)
    todo = paths[:limit] if limit else paths
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(session_rows)(p, pools, 7000 + i, split, units) for i, p in enumerate(todo))
    df = pd.DataFrame([r for rows in res for r in rows])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"done in {(time.time() - t0) / 60:.0f} min: {len(df)} rows -> {out_csv}")
    return df
