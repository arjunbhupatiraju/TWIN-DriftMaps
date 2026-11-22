"""REPMAP round 3 (explore mice only): more sensitive per-session evidence.

The standard per-session test (MULTIVERSE default: ridge decoder with lam = 1*N, 200 pseudo-sessions,
empirical p) has two sensitivity limits:
  1. p cannot go below 1/201, so a strong session and a merely good one look alike to Fisher's method;
  2. one fixed ridge penalty, which can be badly matched to a region's size or signal.
Round 3 computes, for every session x region, from the SAME null process:
  p200      empirical p from the first 200 pseudo labels, lam = 1   (reproduces the standard)
  p1000     empirical p from 1000 pseudo labels, lam = 1            (finer p, floor 1/1001)
  z1        (score - null mean) / null sd, lam = 1                  (continuous evidence, no floor)
  zmax      the same, with the score = max over lam in (0.1, 1, 10)*N, applied identically to real AND
            null labels (selection is mirrored in the null, so the test stays valid)
Fake variables (ctl_*) are run through the same code: they check that the extra sensitivity does not
come from false positives.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .multiverse import (MIN_PER_CLASS, MIN_UNITS, WINDOW_OF, _ALPHA_CUR, _base, design, label_matrix,
                         null_draw, subjective_alpha)
from .mv_data import MVSession

VARS = ("stim", "choice", "feedback", "sprior", "ctl_stim", "ctl_choice", "ctl_feedback", "ctl_sprior")
LAMS = (0.1, 1.0, 10.0)


def decode_scores(X, Y, lams=LAMS):
    """Leave-one-out ridge correlation for every label column, for each lam (scaled by N)."""
    T, N = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    Yc = Y - Y.mean(0)
    ny = np.linalg.norm(Yc, axis=0) + 1e-12
    G = Z.T @ Z
    out = []
    for lam in lams:
        A = G + lam * N * np.eye(N)
        H = Z @ np.linalg.solve(A, Z.T)
        h = np.clip(np.diag(H), None, 0.999)
        loo = (H @ Yc - h[:, None] * Yc) / (1 - h[:, None])
        out.append((loo * Yc).sum(0) / (np.linalg.norm(loo, axis=0) * ny))
    return np.array(out)                       # (len(lams), 1 + n_null)


def _p(v):
    return float((1 + (v[1:] >= v[0]).sum()) / len(v))


def _z(v):
    nl = v[1:]
    return float((v[0] - nl.mean()) / (nl.std(ddof=1) + 1e-12))


def pools_for(paths, split):
    pools = {}
    for b in ("choice", "feedback"):
        seqs = []
        for p in paths:
            s = MVSession(p)
            if s.split != split:
                continue
            v, y = design(s, b, "a")[:2]
            if v is not None:
                seqs.append((s.sid, y))
        pools[b] = seqs
    return pools


def session_rows(path, pools, n_null=1000, seed=0, split="explore"):
    s = MVSession(path)
    if s.split != split:
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
        if var.startswith("ctl_"):
            v, _, crng = d
            if b == "sprior":
                _ALPHA_CUR[0] = subjective_alpha(s)
            y = null_draw(var, v, len(s.contrast), pool, crng)
        else:
            v, y = d
        if b == "sprior":
            _ALPHA_CUR[0] = subjective_alpha(s)
            if len(y) < 2 * MIN_PER_CLASS or np.std(y) == 0:
                continue
        elif min((y == 0).sum(), (y == 1).sum()) < MIN_PER_CLASS:
            continue
        Y = label_matrix(y, "pseudo", n_null, rng, var, v, len(s.contrast), pool)
        win = f"{WINDOW_OF.get(b, b)}_a"
        for r in s.regions:
            if (r, win) not in Xc:
                Xc[(r, win)] = s.X(r, win)
            X = Xc[(r, win)]
            if X is None:
                continue
            X = X[v]
            X = X[:, X.std(0) > 0]
            if X.shape[1] < MIN_UNITS:
                continue
            D = decode_scores(X, Y)
            d1 = D[LAMS.index(1.0)]
            dmax = D.max(0)
            rows.append(dict(sid=s.sid, subject=s.subject, lab=s.lab, var=var, region=r,
                             n_units=int(X.shape[1]), n_trials=int(len(y)),
                             p200=_p(d1[:201]), p1000=_p(d1), z1=_z(d1), zmax=_z(dmax), pmax1000=_p(dmax),
                             eff=float(d1[0])))
    return rows


def run(cache_dir, out_csv, n_null=1000, n_jobs=-1, split="explore", log=print):
    import pandas as pd
    from joblib import Parallel, delayed
    paths = [p for p in sorted(Path(cache_dir).glob("*.npz")) if MVSession(p).split == split]
    if split != "explore":
        raise SystemExit("Round 3 is exploration: explore mice only.")
    log(f"{len(paths)} {split} sessions; building imposter pools ...")
    pools = pools_for(paths, split)
    log(f"per-session evidence with {n_null} pseudo-sessions and 3 ridge penalties (parallel) ...")
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(session_rows)(p, pools, n_null, 5000 + i, split)
                                               for i, p in enumerate(paths))
    df = pd.DataFrame([r for rows in res for r in rows])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"done in {(time.time() - t0) / 60:.0f} min: {len(df)} rows -> {out_csv}")
    return df
