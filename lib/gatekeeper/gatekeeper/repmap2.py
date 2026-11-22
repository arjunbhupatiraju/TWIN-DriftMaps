"""REPMAP exploration round 2 (explore mice only): principled variants that pool evidence across mice.

All on the calibrated pseudo-session null (decode, good units, IBL window). Per session: p -> z = isf(p).
    F_sess    standard: Fisher across sessions                          (IBL / MULTIVERSE default)
    PC2       partial conjunction: effect in >= 2 mice (round 1)
    STO_sess  Stouffer across sessions: sum z / sqrt(N)  (less driven by one extreme session than Fisher)
    STO_mouse Stouffer within mouse, then across mice (each mouse weighs the same)
    RE_mouse  random-effects test: one-sample t-test of mouse-level z across mice (population inference;
              needs >= 3 mice) -- asks "does the population of mice encode it", which is what replicates
    HYB       intersection-union: max(p_PC2, p_STO_mouse) -- consistent across mice AND strong in aggregate
BH across tested regions at level q; q is swept so methods are compared at matched claims per study.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2, norm, t as tdist

from .multiverse import _bh

PMIN = 1.0 / 201


class Mats:
    def __init__(self, df, var, col="p_decode_pseudo"):
        g = df[(df["var"] == var) & (df.window == "a") & (df.units == "good")]
        self.mice = sorted(df.subject.unique())
        self.regions = sorted(g.region.unique())
        ri = {r: i for i, r in enumerate(self.regions)}
        mi = {m: j for j, m in enumerate(self.mice)}
        R, M = len(self.regions), len(self.mice)
        rr, mm = g.region.map(ri).to_numpy(), g.subject.map(mi).to_numpy()
        p = np.clip(g[col].to_numpy(), PMIN, 1 - 1e-9)
        z = norm.isf(p)
        self.C = np.zeros((R, M)); np.add.at(self.C, (rr, mm), 1)
        self.S = np.zeros((R, M)); np.add.at(self.S, (rr, mm), -2 * np.log(p))
        self.Z = np.zeros((R, M)); np.add.at(self.Z, (rr, mm), z)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.pm = np.where(self.C > 0, chi2.sf(self.S, 2 * np.maximum(self.C, 1)), np.nan)
            self.zm = np.where(self.C > 0, self.Z / np.sqrt(np.maximum(self.C, 1)), np.nan)


def _pc2(pm_sel):
    n = np.sum(~np.isnan(pm_sel), 1)
    s = np.sort(np.where(np.isnan(pm_sel), -1.0, pm_sel), 1)[:, ::-1]
    p = np.full(len(n), np.nan)
    for i in np.nonzero(n >= 2)[0]:
        top = s[i, : n[i] - 1]
        p[i] = chi2.sf(-2 * np.log(np.clip(top, 1e-300, 1)).sum(), 2 * len(top))
    return p


def pvals(X, method, a):
    sel = a > 0
    C = X.C[:, sel]
    N = C.sum(1)
    nm = (C > 0).sum(1)
    ok = N >= 2
    with np.errstate(invalid="ignore", divide="ignore"):
        if method == "F_sess":
            p = chi2.sf(X.S[:, sel].sum(1), 2 * np.maximum(N, 1))
        elif method == "STO_sess":
            p = norm.sf(X.Z[:, sel].sum(1) / np.sqrt(np.maximum(N, 1)))
        elif method == "STO_mouse":
            zm = np.nan_to_num(X.zm[:, sel])
            p = norm.sf(zm.sum(1) / np.sqrt(np.maximum(nm, 1)))
            ok = ok & (nm >= 1)
        elif method == "RE_mouse":
            zm = X.zm[:, sel]
            mu = np.nanmean(zm, 1)
            sd = np.nanstd(zm, 1, ddof=1)
            tt = mu / (sd / np.sqrt(np.maximum(nm, 1)))
            p = tdist.sf(tt, np.maximum(nm - 1, 1))
            ok = ok & (nm >= 3) & (sd > 0)
        elif method == "PC2":
            p = _pc2(X.pm[:, sel])
            ok = ok & (nm >= 2)
        elif method == "HYB":
            p1 = _pc2(X.pm[:, sel])
            zm = np.nan_to_num(X.zm[:, sel])
            p2 = norm.sf(zm.sum(1) / np.sqrt(np.maximum(nm, 1)))
            p = np.fmax(p1, p2)
            ok = ok & (nm >= 2)
        else:
            raise ValueError(method)
    return np.where(ok & np.isfinite(p), p, np.nan)


def verdict(X, method, a, q):
    p = pvals(X, method, a)
    out = np.full(len(p), -1, np.int8)
    t = ~np.isnan(p)
    if t.any():
        out[t] = _bh(p[t], q).astype(np.int8)
    return out
