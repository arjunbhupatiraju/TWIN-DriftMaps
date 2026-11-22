"""Independent re-implementation of the registered REPMAP evaluation (no code shared with repmap_confirm.py
except reading the same file). Written from the registration text, using pandas groupby instead of matrices,
statsmodels-free BH written separately, and an explicit loop over claims. Compares, for the same random studies:
  (1) every rule's p-values per region, (2) BH verdicts, (3) the final precision table, for both the registered
  'all other mice' reference and amendment 1's equal-size / fixed-36 references.
    python3 scripts/repmap_independent_check.py results/multiverse_confirm/session_stats.csv.gz
This RE-COMPUTES already-registered analyses to look for bugs; it does not create a new test."""
import sys, math
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, '.')
from gatekeeper import repmap_confirm as RC, repmap_followup as RF

def bh_indep(pdict, q):
    items = sorted(pdict.items(), key=lambda kv: kv[1]); m = len(items); cut = 0
    for i, (_, p) in enumerate(items, 1):
        if p <= q * i / m: cut = i
    return {r for r, _ in items[:cut]}

def region_pvals(g, mice, rule):
    """g: rows of one variable. mice: set of subjects in the study. Returns {region: p} for tested regions."""
    fam, m = rule.split('_'); m = int(m)
    sub = g[g.subject.isin(mice)]
    out = {}
    for reg, rr in sub.groupby('region'):
        ps = np.clip(rr.p_decode_pseudo.to_numpy(), 1e-12, 1)
        n_sess = len(ps); per_mouse = rr.groupby('subject').p_decode_pseudo.apply(lambda s: np.clip(s.to_numpy(), 1e-12, 1))
        n_mice = len(per_mouse)
        if n_sess < 2 or n_mice < (m if fam == 'F' else max(m, 2)): continue
        if fam == 'F':
            out[reg] = stats.chi2.sf(-2 * np.log(ps).sum(), 2 * n_sess)
        else:
            pm = sorted([stats.chi2.sf(-2 * np.log(v).sum(), 2 * len(v)) for v in per_mouse], reverse=True)
            top = pm[: n_mice - 1]
            out[reg] = stats.chi2.sf(-2 * np.log(np.clip(top, 1e-300, 1)).sum(), 2 * len(top))
    return out

def main(path, n_check=60):
    df = pd.read_csv(path)
    df = df[df['var'].isin(RC.VARS) & (df.window == 'a') & (df.units == 'good')]
    worst = 0.0; nver = 0; mism = 0
    for var in RC.VARS:
        g = df[df['var'] == var]
        X = RC.mats(df, var); mice_all = sorted(df.subject.unique()); regs = sorted(g.region.unique())
        rng = np.random.default_rng(7)
        for _ in range(n_check):
            k = int(rng.choice(RC.KS)); perm = rng.permutation(len(mice_all)); a = np.zeros(len(mice_all)); a[perm[:k]] = 1
            study = {mice_all[i] for i in perm[:k]}
            for rule in RC.RULES:
                p1 = RC.pvals(X, rule, a); p2 = region_pvals(g, study, rule)
                t1 = {regs[i] for i in np.nonzero(~np.isnan(p1))[0]}
                if t1 != set(p2): mism += 1; continue
                for r in t1:
                    x, y = p1[regs.index(r)], p2[r]
                    worst = max(worst, abs(math.log10(max(x, 1e-300)) - math.log10(max(y, 1e-300))))
                for q in (0.01, 0.05, 0.2):
                    v1 = RC.verdict(p1, q); s1 = {regs[i] for i in np.nonzero(v1 == 1)[0]}
                    nver += 1; mism += int(s1 != bh_indep(p2, q))
    print(f"p-values: worst |log10 difference| = {worst:.2e}; tested-set/verdict mismatches = {mism} of {nver} BH checks")
    # precision tables: independent recomputation of the amendment's equal-size reference for one var/k, few reps
    for var in ("stim", "sprior"):
        g = df[df['var'] == var]; X = RC.mats(df, var); mice_all = sorted(df.subject.unique()); M = len(mice_all)
        rng = np.random.default_rng(RF.REF_SIZE)  # any seed: compare implementations on identical draws
        hits = {r: 0 for r in ("F_1", "PC_3")}; cl = {r: 0 for r in ("F_1", "PC_3")}; h2 = dict(hits); c2 = dict(cl)
        for _ in range(40):
            p = rng.permutation(M); a = np.zeros(M); a[p[:8]] = 1; b = np.zeros(M); b[p[8:16]] = 1
            A = {mice_all[i] for i in p[:8]}; B = {mice_all[i] for i in p[8:16]}
            ref1 = RC.verdict(RC.pvals(X, "F_1", b), 0.05); refset2 = bh_indep(region_pvals(g, B, "F_1"), 0.05); tested2 = set(region_pvals(g, B, "F_1"))
            for r in ("F_1", "PC_3"):
                c1v = RC.verdict(RC.pvals(X, r, a), 0.05); ok = (c1v >= 0) & (ref1 >= 0)
                hits[r] += int(((c1v == 1) & (ref1 == 1) & ok).sum()); cl[r] += int(((c1v == 1) & ok).sum())
                pa = region_pvals(g, A, r); claims = bh_indep(pa, 0.05) & tested2
                h2[r] += len(claims & refset2); c2[r] += len(claims)
        print(f"{var}: equal-size precision at q=.05, k=8 - registered code F_1 {hits['F_1']}/{cl['F_1']}, PC_3 {hits['PC_3']}/{cl['PC_3']} | "
              f"independent F_1 {h2['F_1']}/{c2['F_1']}, PC_3 {h2['PC_3']}/{c2['PC_3']}")

if __name__ == "__main__":
    main(sys.argv[1])
