"""v3: drift factor grows with session length, delta = delta_var * (T/400)^g (long-memory drift: the Bartlett sum
2*sum_k rho_x(k) rho_y(k) keeps growing with T when drift has slow components). Session-level likelihood.
Out-of-sample: fit on window a / good units, predict window b / good units."""
import sys, numpy as np, pandas as pd, warnings
from scipy.optimize import minimize
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
from theory.drift_fp import fp_general
D = pd.read_csv('results/multiverse_explore/session_stats.csv.gz'); D = D[D['var'].str.startswith('ctl_')]
VARS = sorted(D['var'].unique())
def get(w, u):
    d = D[(D.window == w) & (D.units == u)]
    return [(d[d['var'] == v].n_units.to_numpy(float), d[d['var'] == v].n_trials.to_numpy(float),
             (d[d['var'] == v].p_decode_shuffle < .05).to_numpy(float)) for v in VARS]
def unpack(th, use_g):
    return 1/(1+np.exp(-th[0]))*.3, 1/(1+np.exp(-th[1])), (th[2] if use_g else 0.0), np.exp(th[3:])
def nll(th, data, use_g):
    rho0, q, g, dl = unpack(th, use_g); s = 0
    for (N, T, y), de in zip(data, dl):
        p = np.clip(fp_general(N, de*(T/400)**g, rho0, q), 1e-6, 1-1e-6); s -= np.sum(y*np.log(p)+(1-y)*np.log(1-p))
    return s
def fit(data, use_g):
    return min((minimize(nll, np.r_[a, b, .5, np.log(np.full(6, .4))], args=(data, use_g), method='Nelder-Mead',
                         options=dict(maxiter=40000, xatol=1e-7, fatol=1e-7)) for a in (-2, 0) for b in (-2, 0)), key=lambda r: r.fun)
def binned(th, data, use_g):
    rho0, q, g, dl = unpack(th, use_g); o, p = [], []
    for (N, T, y), de in zip(data, dl):
        pr = fp_general(N, de*(T/400)**g, rho0, q)
        for lo, hi in [(5,8),(8,12),(12,18),(18,27),(27,40),(40,60),(60,90),(90,9999)]:
            for tq in (T < np.median(T), T >= np.median(T)):
                m = (N >= lo) & (N < hi) & tq
                if m.sum() >= 15: o.append(y[m].mean()); p.append(pr[m].mean())
    o, p = np.array(o), np.array(p); return np.corrcoef(o, p)[0,1], np.abs(o-p).mean(), len(o)
a, b = get('a', 'good'), get('b', 'good')
for use_g in (False, True):
    f = fit(a, use_g); rho0, q, g, dl = unpack(f.x, use_g)
    ra = binned(f.x, a, use_g); rb = binned(f.x, b, use_g)
    print(f"{'v3 (V grows with T)' if use_g else 'v2 (no T)':20s} rho0={rho0:.3f} q={q:.3f} g={g:.2f}  fit NLL a={f.fun:.1f}  "
          f"PRED NLL b={nll(f.x, b, use_g):.1f}  | N-by-T bins: fit r={ra[0]:.3f} MAE={ra[1]:.3f}; predicted b r={rb[0]:.3f} MAE={rb[1]:.3f} ({rb[2]} bins)")
