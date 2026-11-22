"""REPMAP round 2 exploration (explore mice only). Precision at matched claims, three held-out references."""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from gatekeeper import repmap2 as R
QS = (1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
METH = ("F_sess", "PC2", "STO_sess", "STO_mouse", "RE_mouse", "HYB")
REFS = ("F_sess", "PC2", "STO_sess")
VARS = ("stim", "choice", "sprior", "prior", "feedback")
df = pd.read_csv('results/multiverse_explore/session_stats.csv.gz')
n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 150
out = {}
for var in VARS:
    X = R.Mats(df, var)
    M = len(X.mice)
    for k in (4, 8, 12, 16):
        rng = np.random.default_rng(7)
        acc = {(rf, m, q): np.zeros(3) for rf in REFS for m in METH for q in QS}
        for _ in range(n_rep):
            p = rng.permutation(M); a = np.zeros(M); a[p[:k]] = 1
            refs = {rf: R.verdict(X, rf, 1 - a, 0.05) for rf in REFS}
            for m in METH:
                pv = R.pvals(X, m, a)
                t = ~np.isnan(pv)
                for q in QS:
                    c = np.full(len(pv), -1, np.int8)
                    if t.any():
                        c[t] = R._bh(pv[t], q).astype(np.int8)
                    for rf in REFS:
                        ref = refs[rf]; ok = (c >= 0) & (ref >= 0)
                        r = acc[(rf, m, q)]
                        r[0] += ((c == 1) & (ref == 1) & ok).sum(); r[1] += ((c == 1) & ok).sum(); r[2] += 1
        for rf in REFS:
            for m in METH:
                out[f"{var}|{k}|{rf}|{m}"] = [[acc[(rf, m, q)][1] / n_rep, acc[(rf, m, q)][0] / n_rep] for q in QS]
        print(var, k, flush=True)
json.dump(out, open('results/repmap_explore/round2_curves.json', 'w'))
