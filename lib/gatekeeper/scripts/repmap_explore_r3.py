"""REPMAP round 3 evaluation (explore mice only): evidence type x combination rule, matched claims."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import chi2, norm
sys.path.insert(0, '.')
from gatekeeper.multiverse import _bh
d = pd.read_csv('results/repmap_explore/sens_stats.csv.gz')
EVID = {"p200": lambda g: g.p200.to_numpy(), "p1000": lambda g: g.p1000.to_numpy(),
        "z1": lambda g: norm.sf(g.z1.to_numpy()), "zmax": lambda g: norm.sf(g.zmax.to_numpy())}
QS = (1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)

class M:
    def __init__(self, var, ev):
        g = d[d['var'] == var]
        self.mice = sorted(d.subject.unique()); self.regions = sorted(g.region.unique())
        ri = {r: i for i, r in enumerate(self.regions)}; mi = {m: j for j, m in enumerate(self.mice)}
        rr, mm = g.region.map(ri).to_numpy(), g.subject.map(mi).to_numpy()
        p = np.clip(EVID[ev](g), 1e-12, 1 - 1e-12); z = norm.isf(p)
        R, Mn = len(self.regions), len(self.mice)
        self.C = np.zeros((R, Mn)); np.add.at(self.C, (rr, mm), 1)
        self.S = np.zeros((R, Mn)); np.add.at(self.S, (rr, mm), -2 * np.log(p))
        self.Z = np.zeros((R, Mn)); np.add.at(self.Z, (rr, mm), z)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.pm = np.where(self.C > 0, chi2.sf(self.S, 2 * np.maximum(self.C, 1)), np.nan)

def pv(X, comb, a):
    sel = a > 0; C = X.C[:, sel]; N = C.sum(1); nm = (C > 0).sum(1)
    if comb == "Fisher":
        p = chi2.sf(X.S[:, sel].sum(1), 2 * np.maximum(N, 1)); ok = N >= 2
    elif comb == "Stouffer":
        p = norm.sf(X.Z[:, sel].sum(1) / np.sqrt(np.maximum(N, 1))); ok = N >= 2
    else:  # PC2
        pm = X.pm[:, sel]; s = np.sort(np.where(np.isnan(pm), -1, pm), 1)[:, ::-1]
        p = np.full(len(nm), np.nan)
        for i in np.nonzero(nm >= 2)[0]:
            top = s[i, :nm[i] - 1]; p[i] = chi2.sf(-2 * np.log(np.clip(top, 1e-300, 1)).sum(), 2 * len(top))
        ok = (nm >= 2) & (N >= 2)
    return np.where(ok, p, np.nan)

def verdict(p, q):
    out = np.full(len(p), -1, np.int8); t = ~np.isnan(p)
    if t.any(): out[t] = _bh(p[t], q).astype(np.int8)
    return out

COMBOS = [(e, c) for e in EVID for c in ("Fisher", "Stouffer", "PC2")]
REFS = [("p200", "Fisher"), ("p200", "PC2"), ("zmax", "Stouffer")]
n_rep = 150
out = {}
calib = {}
for var in ("stim", "choice", "sprior", "feedback"):
    Xs = {e: M(var, e) for e in EVID}
    Xc = {e: M("ctl_" + var, e) for e in EVID}
    Mn = len(Xs["p200"].mice)
    # calibration: fake variable, full explore data, q=0.05 -> share of tested regions flagged
    calib[var] = {f"{e}|{c}": float(np.nanmean(verdict(pv(Xc[e], c, np.ones(Mn)), 0.05)[verdict(pv(Xc[e], c, np.ones(Mn)), 0.05) >= 0] == 1)) for e, c in COMBOS}
    for k in (4, 8, 12, 16):
        rng = np.random.default_rng(11)
        acc = {(r, ec, q): np.zeros(2) for r in range(len(REFS)) for ec in COMBOS for q in QS}
        fake = {(ec, q): np.zeros(2) for ec in COMBOS for q in (0.05,)}
        for _ in range(n_rep):
            p = rng.permutation(Mn); a = np.zeros(Mn); a[p[:k]] = 1
            refs = [verdict(pv(Xs[e], c, 1 - a), 0.05) for e, c in REFS]
            for ec in COMBOS:
                pp = pv(Xs[ec[0]], ec[1], a)
                for q in QS:
                    cl = verdict(pp, q)
                    for ri, ref in enumerate(refs):
                        ok = (cl >= 0) & (ref >= 0)
                        acc[(ri, ec, q)] += [((cl == 1) & (ref == 1) & ok).sum(), ((cl == 1) & ok).sum()]
                fv = verdict(pv(Xc[ec[0]], ec[1], a), 0.05)
                fake[(ec, 0.05)] += [(fv == 1).sum(), (fv >= 0).sum()]
        for ri in range(len(REFS)):
            for ec in COMBOS:
                out[f"{var}|{k}|{ri}|{ec[0]}|{ec[1]}"] = [[acc[(ri, ec, q)][1] / n_rep, acc[(ri, ec, q)][0] / n_rep] for q in QS]
        for ec in COMBOS:
            out[f"fake|{var}|{k}|{ec[0]}|{ec[1]}"] = fake[(ec, 0.05)][0] / max(fake[(ec, 0.05)][1], 1)
        print(var, k, flush=True)
json.dump(dict(curves=out, calib_full=calib), open('results/repmap_explore/round3_curves.json', 'w'))
