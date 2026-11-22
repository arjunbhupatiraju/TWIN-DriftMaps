"""Module 1: how do single NI units relate to engagement state?

Per unit: mean pre-stimulus rate in engaged vs disengaged trials (GLM-HMM smoothed posterior
> .8 / < .2), modulation index (E - D) / (E + D), and a circular-shift permutation p-value
(shifting the state sequence relative to spikes preserves both autocorrelations, so slow
drift alone cannot produce significance). Benjamini-Hochberg across units.
"""
from __future__ import annotations

import numpy as np


def _bh(p):
    p = np.asarray(p)
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def unit_modulation(sessions, posteriors, region="NI", n_perm=1000, seed=0, min_shift=50):
    rng = np.random.default_rng(seed)
    rows = []
    for s, p in zip(sessions, posteriors):
        if region not in s.signals:
            continue
        X = s.signals[region]
        T = len(p)
        if T < 2 * min_shift + 10:
            continue
        eng, dis = p > 0.8, p < 0.2
        if eng.sum() < 20 or dis.sum() < 20:
            continue
        shifts = rng.integers(min_shift, T - min_shift, n_perm)
        for u in range(X.shape[1]):
            x = X[:, u]
            obs = x[eng].mean() - x[dis].mean()
            null = np.array([x[np.roll(eng, k)].mean() - x[np.roll(dis, k)].mean() for k in shifts])
            pval = (1 + (np.abs(null) >= abs(obs)).sum()) / (n_perm + 1)
            e, d = x[eng].mean(), x[dis].mean()
            rows.append(dict(sid=s.sid, unit=u, rate_engaged=float(e), rate_disengaged=float(d),
                             mod_index=float((e - d) / (e + d + 1e-9)), p=float(pval)))
    if rows:
        q = _bh([r["p"] for r in rows])
        for r, qq in zip(rows, q):
            r["q"] = float(qq)
    return rows


def summarize(rows):
    if not rows:
        return dict(n_units=0)
    mi = np.array([r["mod_index"] for r in rows])
    sig = np.array([r["q"] < 0.05 for r in rows])
    return dict(n_units=len(rows), n_sessions=len({r["sid"] for r in rows}),
                frac_significant=float(sig.mean()),
                frac_up_in_engaged=float((sig & (mi > 0)).mean()),
                frac_down_in_engaged=float((sig & (mi < 0)).mean()),
                median_mod_index=float(np.median(mi)))
