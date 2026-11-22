"""Fit v2 to real IBL fake-variable false positives: ONE shared neural correlation structure (rho0, q) for all
variables, plus one drift factor delta per variable. 8 parameters for 48 data points."""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, '.')
from theory.drift_fp import fp_general
d = pd.read_csv('results/multiverse_explore/session_stats.csv.gz')
d = d[(d.window == 'a') & (d.units == 'good') & d['var'].str.startswith('ctl_')]
edges = [5, 8, 12, 18, 27, 40, 60, 90, 2000]
d['nb'] = pd.cut(d.n_units, edges, right=False)
V = sorted(d['var'].unique()); data = {}
for v, g in d.groupby('var'):
    t = g.groupby('nb', observed=True).agg(N=('n_units', 'median'), n=('n_units', 'size'), k=('p_decode_shuffle', lambda x: (x < .05).sum()))
    data[v] = (t.N.to_numpy(float), t.n.to_numpy(float), t.k.to_numpy(float))
def unpack(th): return 1 / (1 + np.exp(-th[0])) * 0.3, 1 / (1 + np.exp(-th[1])), np.exp(th[2:])
def nll(th):
    rho0, q, deltas = unpack(th); s = 0
    for v, dl in zip(V, deltas):
        N, n, k = data[v]; p = np.clip(fp_general(N, dl, rho0, q), 1e-6, 1 - 1e-6)
        s -= np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
    return s
best = min((minimize(nll, np.r_[a, b, np.log(np.full(len(V), 0.3))], method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-6, fatol=1e-6))
            for a in (-3, -1, 0) for b in (-2, 0, 2)), key=lambda r: r.fun)
rho0, q, deltas = unpack(best.x)
k_par = 2 + len(V); npts = sum(len(data[v][0]) for v in V)
print(f"shared: rho0 (instantaneous neuron corr) = {rho0:.3f}, q (shared fraction of drift) = {q:.3f};  NLL {best.fun:.1f}, {k_par} params, {npts} points")
tot_obs, tot_pred, flat_nll = [], [], 0
for v, dl in zip(V, deltas):
    N, n, k = data[v]; pred = fp_general(N, dl, rho0, q); obs = k / n
    fr = k.sum() / n.sum(); flat_nll -= np.sum(k * np.log(fr) + (n - k) * np.log(1 - fr))
    tot_obs += list(obs); tot_pred += list(pred)
    print(f"{v:13s} V = {1+dl:.3f}  saturation FP_inf = {fp_general(1e6, dl, rho0, q):.2f}   " + " ".join(f"{o:.2f}/{p:.2f}" for o, p in zip(obs, pred)))
o, p = np.array(tot_obs), np.array(tot_pred)
print(f"\nobserved vs predicted across all 48 bins: r = {np.corrcoef(o, p)[0,1]:.3f}, mean abs error = {np.mean(np.abs(o-p)):.3f}")
print(f"model NLL {best.fun:.1f} vs flat-rate-per-variable NLL {flat_nll:.1f} (6 params)")
