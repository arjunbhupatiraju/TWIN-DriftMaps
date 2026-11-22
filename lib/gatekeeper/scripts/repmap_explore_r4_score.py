"""REPMAP round 4 (explore mice only): learned replication score, cross-validated across disjoint mouse halves.
Features per (study, region): -log10 Fisher p, -log10 PC2 p, n mice, n sessions, share of mice with p<.05,
largest single-mouse share of the Fisher statistic. Logistic regression trained on studies + held-out references
drawn inside one half of the mice; tested inside the other half; then swapped."""
import sys, hashlib, numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, '.')
from gatekeeper.multiverse import _bh
d = pd.read_csv('results/repmap_explore/sens_stats.csv.gz')
mice_all = sorted(d.subject.unique())
half = {m: int(hashlib.sha256(f"REPMAP-CV|{m}".encode()).hexdigest(), 16) % 2 for m in mice_all}

def mats(var, mice):
    g = d[(d['var'] == var) & d.subject.isin(mice)]
    regs = sorted(d[d['var'] == var].region.unique()); ri = {r: i for i, r in enumerate(regs)}; mi = {m: j for j, m in enumerate(mice)}
    rr, mm = g.region.map(ri).to_numpy(), g.subject.map(mi).to_numpy(); p = np.clip(g.p200.to_numpy(), 1e-12, 1)
    C = np.zeros((len(regs), len(mice))); np.add.at(C, (rr, mm), 1); S = np.zeros_like(C); np.add.at(S, (rr, mm), -2 * np.log(p))
    with np.errstate(all='ignore'): pm = np.where(C > 0, chi2.sf(S, 2 * np.maximum(C, 1)), np.nan)
    return C, S, pm

def feats(X, a):
    C, S, pm = X; sel = a > 0; c = C[:, sel]; s = S[:, sel]; p_m = pm[:, sel]
    N = c.sum(1); nm = (c > 0).sum(1); St = s.sum(1)
    fis = chi2.sf(St, 2 * np.maximum(N, 1))
    srt = np.sort(np.where(np.isnan(p_m), -1, p_m), 1)[:, ::-1]; pc = np.ones(len(N))
    for i in np.nonzero(nm >= 2)[0]:
        top = srt[i, :nm[i] - 1]; pc[i] = chi2.sf(-2 * np.log(np.clip(top, 1e-300, 1)).sum(), 2 * len(top))
    frac = np.where(nm > 0, np.nansum(p_m < 0.05, 1) / np.maximum(nm, 1), 0)
    Sm = np.where(np.isnan(p_m), 0, -2 * np.log(np.clip(p_m, 1e-300, 1)))
    dom = np.where(Sm.sum(1) > 0, Sm.max(1) / np.maximum(Sm.sum(1), 1e-9), 1)
    F = np.c_[-np.log10(np.clip(fis, 1e-300, 1)), -np.log10(np.clip(pc, 1e-300, 1)), np.log1p(nm), np.log1p(N), frac, dom]
    return F, N >= 2, fis, pc, nm

def ref_verdict(X, b):
    C, S, _ = X; N = C[:, b > 0].sum(1); p = chi2.sf(S[:, b > 0].sum(1), 2 * np.maximum(N, 1)); t = N >= 2
    out = np.full(len(N), -1, np.int8)
    if t.any(): out[t] = _bh(p[t], 0.05).astype(np.int8)
    return out

def studies(X, M, k, n, rng):
    for _ in range(n):
        p = rng.permutation(M); a = np.zeros(M); a[p[:k]] = 1
        yield a, ref_verdict(X, 1 - a)

QS = (1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
def bh_claims(p, ok, q):
    c = np.zeros(len(p), bool); t = ok & ~np.isnan(p)
    if t.any(): c[t] = _bh(p[t], q)
    return c

for var, G in (('stim', (0.5, 1, 2)), ('sprior', (0.5, 1, 2)), ('choice', (2, 4, 8)), ('feedback', (3, 6, 12))):
    res = {}
    for tr in (0, 1):
        Mtr = [m for m in mice_all if half[m] == tr]; Mte = [m for m in mice_all if half[m] != tr]
        Xtr, Xte = mats(var, Mtr), mats(var, Mte)
        rng = np.random.default_rng(21)
        Xs, ys = [], []
        for k in (4, 6, 8):
            for a, ref in studies(Xtr, len(Mtr), k, 200, rng):
                F, ok, fis, pc, nm = feats(Xtr, a); use = ok & (ref >= 0)
                Xs.append(F[use]); ys.append(ref[use] == 1)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack(Xs), np.concatenate(ys))
        rng = np.random.default_rng(99)
        for k in (4, 6, 8):
            acc = {}
            for a, ref in studies(Xte, len(Mte), k, 200, rng):
                F, ok, fis, pc, nm = feats(Xte, a); valid = ok & (ref >= 0); good = ref == 1
                sc = clf.predict_proba(F)[:, 1]
                rules = {'standard': (fis, ok), 'min3+PC': (pc, ok & (nm >= 3)), 'min2+PC': (pc, ok & (nm >= 2))}
                for name, (p, ok2) in rules.items():
                    for q in QS:
                        c = bh_claims(p, ok2, q) & valid
                        acc.setdefault((name, q), np.zeros(2))[:] += [(c & good).sum(), c.sum()]
                for th in (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
                    # learned score claims: must still pass the standard test at q=0.05 (keeps error control)
                    c = (sc >= th) & bh_claims(fis, ok, 0.05) & valid
                    acc.setdefault(('score', th), np.zeros(2))[:] += [(c & good).sum(), c.sum()]
            for name in ('standard', 'min2+PC', 'min3+PC', 'score'):
                pts = np.array([[v[1] / 200, v[0] / 200] for (n_, _), v in acc.items() if n_ == name])
                o = np.argsort(pts[:, 0]); cl, hi = pts[o, 0], pts[o, 1]
                for g in G:
                    res.setdefault(name, []).append(np.nan if g > cl.max() or g < cl.min() else np.interp(g, cl, hi) / g)
                res.setdefault(name + '_max', []).append(cl.max())
    print(f"{var:8s} " + "  ".join(f"{n}: {np.nanmean(res[n]):.3f} (max {np.mean(res[n+'_max']):.1f} claims)" for n in ('standard', 'min2+PC', 'min3+PC', 'score')), flush=True)
