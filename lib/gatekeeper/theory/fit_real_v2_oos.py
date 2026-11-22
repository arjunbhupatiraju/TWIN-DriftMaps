"""Out-of-sample checks for theory v2.
(1) fit on window a / good units, predict window b / good units with NO refitting;
(2) fit each unit set separately to see whether the model form holds over a wider neuron-count range (all units)."""
import sys, numpy as np, pandas as pd
from scipy.optimize import minimize
sys.path.insert(0, '.')
from theory.drift_fp import fp_general
D = pd.read_csv('results/multiverse_explore/session_stats.csv.gz')
D = D[D['var'].str.startswith('ctl_')]
edges = [5, 8, 12, 18, 27, 40, 60, 90, 140, 220, 5000]
def bins(window, units):
    d = D[(D.window == window) & (D.units == units)].copy()
    d['nb'] = pd.cut(d.n_units, edges, right=False); out = {}
    for v, g in d.groupby('var'):
        t = g.groupby('nb', observed=True).agg(N=('n_units', 'median'), n=('n_units', 'size'), k=('p_decode_shuffle', lambda x: (x < .05).sum()))
        t = t[t.n >= 20]; out[v] = (t.N.to_numpy(float), t.n.to_numpy(float), t.k.to_numpy(float))
    return out
def unpack(th): return 1 / (1 + np.exp(-th[0])) * 0.3, 1 / (1 + np.exp(-th[1])), np.exp(th[2:])
def nll(th, data):
    rho0, q, dl = unpack(th); s = 0
    for v, de in zip(sorted(data), dl):
        N, n, k = data[v]; p = np.clip(fp_general(N, de, rho0, q), 1e-6, 1 - 1e-6); s -= np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
    return s
def fit(data):
    return min((minimize(nll, np.r_[a, b, np.log(np.full(len(data), .3))], args=(data,), method='Nelder-Mead', options=dict(maxiter=20000, xatol=1e-7, fatol=1e-7))
                for a in (-3, -1, 0) for b in (-2, 0, 2)), key=lambda r: r.fun)
def flat(data): return -sum(np.sum(k * np.log(k.sum() / n.sum()) + (n - k) * np.log(1 - k.sum() / n.sum())) for N, n, k in data.values())
def report(tag, th, data):
    rho0, q, dl = unpack(th); o, p = [], []
    for v, de in zip(sorted(data), dl):
        N, n, k = data[v]; o += list(k / n); p += list(fp_general(N, de, rho0, q))
    o, p = np.array(o), np.array(p)
    print(f"{tag}: rho0={rho0:.3f} q={q:.3f} V=" + ",".join(f"{1+x:.2f}" for x in dl) +
          f" | r={np.corrcoef(o,p)[0,1]:.3f} MAE={np.abs(o-p).mean():.3f} NLL={nll(th, data):.1f} flat={flat(data):.1f} bins={len(o)}")
import warnings; warnings.filterwarnings("ignore")
ag = bins('a', 'good'); bg = bins('b', 'good'); aa = bins('a', 'all'); ba = bins('b', 'all')
if __name__ == '__main__':
    fa = fit(ag); report("fit  window a good", fa.x, ag)
    report("PRED window b good (no refit)", fa.x, bg)
    fb = fit(bg); report("fit  window b good", fb.x, bg)
    fA = fit(aa); report("fit  window a all ", fA.x, aa)
    report("PRED window b all  (no refit from a all)", fA.x, ba)
    # shared structure only: keep rho0,q from good-units fit, refit deltas on all-units
    th = fA.x.copy(); th[:2] = fa.x[:2]
    r = minimize(lambda d: nll(np.r_[fa.x[:2], d], aa), fA.x[2:], method='Nelder-Mead', options=dict(maxiter=20000))
    report("all units, rho0/q fixed from good units", np.r_[fa.x[:2], r.x], aa)
