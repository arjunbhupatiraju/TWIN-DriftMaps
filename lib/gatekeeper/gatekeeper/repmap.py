"""REPMAP: a brain map built for replication (exploration stage: explore mice only).

Diagnosis (MULTIVERSE): the standard pipeline combines sessions with Fisher's method, which calls a region
significant if ANY session is strongly significant, so one lucky session can carry a claim.

Fix candidates compared here, all on the calibrated pseudo-session null (decode, good units, IBL window):
    fisher_session   standard: Fisher across sessions, BH          (IBL / MULTIVERSE default)
    fisher_mouse     Fisher within mouse, then across mice, BH
    pc_u2            partial conjunction across mice: effect in >= 2 mice (Benjamini & Heller 2008), BH
    pc_f30 / pc_f50  partial conjunction: effect in >= max(2, ceil(f * n_mice)) mice, BH
Partial-conjunction p for "at least u of n": Fisher combination of the n-u+1 LARGEST mouse-level p-values.

Evaluation: random study of k mice (method under test) vs held-out reference = standard calibrated verdict
on ALL other mice. precision = P(reference significant | claimed); claims per study; recall = share of
reference-significant (testable) regions claimed. Sweeping q gives each method a precision-claims curve,
so methods are compared at matched numbers of claims.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .multiverse import _bh, default_key

METHODS = ("fisher_session", "fisher_mouse", "pc_u2", "pc_f30", "pc_f50")
QS = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
VARS = ("stim", "choice", "sprior", "feedback")


def mouse_p(U, var):
    d = U.mats[default_key(var)[:5]]
    C, S = d["C"], d["S"]
    with np.errstate(invalid="ignore"):
        pm = np.where(C > 0, chi2.sf(S, 2 * np.maximum(C, 1)), np.nan)
    return d["regions"], pm, C


def pc_pvals(pm, a, u_rule):
    """pm (R, M) mouse-level p (nan = no sessions); a (M,) 0/1. Returns p (R,), n mice (R,)."""
    sel = pm[:, a > 0]
    n = np.sum(~np.isnan(sel), 1)
    s = np.sort(np.where(np.isnan(sel), -1.0, sel), 1)[:, ::-1]      # descending; missing (-1) go last
    p = np.ones(len(n))
    for i in range(len(n)):
        u = u_rule(n[i])
        if n[i] < max(u, 1):
            p[i] = np.nan
            continue
        top = s[i, : n[i] - u + 1]
        p[i] = chi2.sf(-2 * np.log(np.clip(top, 1e-300, 1)).sum(), 2 * len(top))
    return p, n


U_RULES = {"pc_u2": lambda n: 2, "pc_f30": lambda n: max(2, int(np.ceil(0.3 * n))),
           "pc_f50": lambda n: max(2, int(np.ceil(0.5 * n)))}


def claims(U, var, method, a, q, cache):
    """Return sig array (R,) with -1 = untested, over the regions of the default key."""
    key = default_key(var)
    if method in ("fisher_session", "fisher_mouse"):
        d = U.mats[key[:5]]
        n = d["C"] @ a
        tested = n >= 2
        if method == "fisher_session":
            p = chi2.sf(d["S"] @ a, 2 * np.maximum(n, 1))
        else:
            p = chi2.sf(d["Lm"] @ a, 2 * np.maximum(d["Cm"] @ a, 1))
    else:
        regs, pm, C = cache[var]
        p, nm = pc_pvals(pm, a, U_RULES[method])
        tested = ~np.isnan(p) & ((C @ a) >= 2)
    out = np.full(len(p), -1, np.int8)
    if tested.any():
        out[tested] = _bh(p[tested], q).astype(np.int8)
    return out


def reference(U, var, b):
    v, _ = U.verdict(default_key(var), b)
    return v


def evaluate(U, ks=(4, 8, 12, 16), n_rep=200, seed=0, log=print):
    rng = np.random.default_rng(seed)
    M = len(U.mice)
    cache = {v: mouse_p(U, v) for v in VARS}
    res = {}
    for var in VARS:
        for k in ks:
            acc = {(m, q): np.zeros(4) for m in METHODS for q in QS}   # hits, claims, ref_pos_testable, studies
            for _ in range(n_rep):
                p = rng.permutation(M)
                a = np.zeros(M)
                a[p[:k]] = 1
                ref = reference(U, var, 1 - a)
                for m in METHODS:
                    for q in QS:
                        c = claims(U, var, m, a, q, cache)
                        ok = (c >= 0) & (ref >= 0)
                        r = acc[(m, q)]
                        r[0] += ((c == 1) & (ref == 1) & ok).sum()
                        r[1] += ((c == 1) & ok).sum()
                        r[2] += ((ref == 1) & ok).sum()
                        r[3] += 1
            for (m, q), r in acc.items():
                res[f"{var}|{k}|{m}|{q}"] = dict(precision=r[0] / r[1] if r[1] else None,
                                                  claims_per_study=r[1] / r[3], recall=r[0] / r[2] if r[2] else None)
            log(f"  {var} k={k} done")
    return res


def matched(res, var, k, target_claims):
    """For each method: precision at the q whose claims/study is closest to target (and the curve)."""
    out = {}
    for m in METHODS:
        pts = [(q, res[f"{var}|{k}|{m}|{q}"]) for q in QS]
        q, r = min(pts, key=lambda t: abs(t[1]["claims_per_study"] - target_claims))
        out[m] = dict(q=q, **r)
    return out


def summary(res, ks=(4, 8, 12, 16)):
    lines = []
    for var in VARS:
        for k in ks:
            base = res[f"{var}|{k}|fisher_session|0.05"]
            mt = matched(res, var, k, base["claims_per_study"])
            lines.append(dict(var=var, k=k, standard=base, matched=mt))
    return lines


def run(stats_files, out_dir, n_rep=200, log=print):
    import pandas as pd
    from .multiverse import Universes
    df = pd.concat([pd.read_csv(f) for f in stats_files], ignore_index=True)
    df = df[df["var"].isin(VARS) & (df.window == "a") & (df.units == "good")]
    U = Universes(df)
    log(f"{len(U.mice)} mice; evaluating {len(METHODS)} methods x {len(QS)} q ...")
    res = evaluate(U, n_rep=n_rep, log=log)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "repmap_curves.json").write_text(json.dumps(res, indent=1))
    S = summary(res)
    (out / "repmap_summary.json").write_text(json.dumps(S, indent=1, default=float))
    return res, S
