"""Fit the theory to real IBL false positives (explore mice, fake variables, shuffle null).
One free parameter per fake variable (its drift factor V) must explain the false-positive rate in every
neuron-count bin. Compares against a 'flat' model (a single FP rate, no N dependence)."""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
sys.path.insert(0, '.')
from theory.drift_fp import fp_theory
d = pd.read_csv('results/multiverse_explore/session_stats.csv.gz')
d = d[(d.window == 'a') & (d.units == 'good') & d['var'].str.startswith('ctl_')]
edges = [5, 8, 12, 18, 27, 40, 60, 90, 2000]
d['nb'] = pd.cut(d.n_units, edges, right=False)
out = []
for v, g in d.groupby('var'):
    t = g.groupby('nb', observed=True).agg(N=('n_units', 'median'), n=('n_units', 'size'), fp=('p_decode_shuffle', lambda x: (x < .05).mean()))
    N, n, fp = t.N.to_numpy(float), t.n.to_numpy(float), t.fp.to_numpy()
    def nll(V):
        p = np.clip(fp_theory(N, V), 1e-4, 1 - 1e-4); k = fp * n
        return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
    r = minimize_scalar(nll, bounds=(1.0, 20), method='bounded'); V = r.x
    pred = fp_theory(N, V)
    flat = np.sum(fp * n) / n.sum()
    ll_flat = -np.sum(fp * n * np.log(flat) + (n - fp * n) * np.log(1 - flat))
    out.append((v, V, r.fun, ll_flat))
    print(f"\n{v}: fitted V = {V:.3f}   theory NLL {r.fun:.1f} vs flat-rate NLL {ll_flat:.1f}  (lower is better)")
    for i in range(len(N)):
        print(f"   N~{N[i]:4.0f} (n={int(n[i]):3d})  observed {fp[i]:.2f}   theory {pred[i]:.2f}")
