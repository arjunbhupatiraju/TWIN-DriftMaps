"""Calibration and power of the drift-corrected test (DCT) vs shuffle, naive parametric, and the
oracle pseudo-session test (labels redrawn from the TRUE generative process - available in simulation only,
and in real data only when the task's generative model is known, as in IBL)."""
import sys, numpy as np, json
sys.path.insert(0, '.')
from theory.corrected_test import dct, dct_pw, shuffle_p, naive_p
from scipy.stats import chi2

from scipy.signal import lfilter
def ar1(T, a, n, rng):
    e = rng.standard_normal((T, n)); e[0] /= np.sqrt(1 - a*a)
    return lfilter([np.sqrt(1 - a*a)], [1, -a], e, axis=0)

def label(T, kind, rng):
    if kind == 'block':            # IBL-like block prior: blocks of 20-100 trials, binary
        y = np.empty(T); t = 0; s = rng.choice([-1., 1.])
        while t < T:
            L = rng.integers(20, 101); y[t:t+L] = s; t += L; s = -s
        return y
    if kind == 'ar':               # slow continuous (e.g. engagement, running speed, estimated prior)
        return ar1(T, 0.98, 1, rng)[:, 0]
    if kind == 'iid':              # stimulus side, no temporal structure
        return rng.choice([-1., 1.], T)

def neurons(T, N, w, q, rho0, rng, a=0.98):
    common = ar1(T, a, 1, rng); priv = ar1(T, a, N, rng)
    drift = np.sqrt(q) * common + np.sqrt(1 - q) * priv
    g = rng.standard_normal((T, 1)); e = rng.standard_normal((T, N))
    inst = np.sqrt(rho0) * g + np.sqrt(1 - rho0) * e
    return np.sqrt(w) * drift + np.sqrt(1 - w) * inst

def pseudo_p(X, y, kind, n, rng):
    T = len(y); Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Y = np.column_stack([y] + [label(T, kind, rng) for _ in range(n)])
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    S = ((Z.T @ Y / T) ** 2).sum(0)
    return float((1 + (S[1:] >= S[0]).sum()) / (n + 1))

def circ_p(X, y, n, rng):
    T = len(y); Z = (X - X.mean(0)) / (X.std(0) + 1e-12); lo = max(1, T // 10)
    Y = np.column_stack([y] + [np.roll(y, int(rng.integers(lo, T - lo + 1))) for _ in range(n)])
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    S = ((Z.T @ Y / T) ** 2).sum(0)
    return float((1 + (S[1:] >= S[0]).sum()) / (n + 1))

METHODS = ('naive', 'shuffle', 'circ', 'pseudo', 'dct_hac', 'dct_hac_long', 'dct_pw')

def cell(kind, T, N, w, q, beta, nrep, seed):
                        rng = np.random.default_rng(seed)
                        P = {m: [] for m in METHODS}
                        for _ in range(nrep):
                            y = label(T, kind, rng); X = neurons(T, N, w, q, 0.1, rng)
                            if beta:                                   # signal in 20% of neurons
                                m = max(1, N // 5); yc = (y - y.mean()) / y.std()
                                X[:, :m] += beta * yc[:, None]
                            P['naive'].append(naive_p(X, y)); P['shuffle'].append(shuffle_p(X, y, 200, rng))
                            P['pseudo'].append(pseudo_p(X, y, kind, 200, rng)); P['circ'].append(circ_p(X, y, 200, rng)); P['dct_hac'].append(dct(X, y)[1]); P['dct_hac_long'].append(dct(X, y, K=len(y)//3, taper=False)[1]); P['dct_pw'].append(dct_pw(X, y)[1])
                        row = dict(kind=kind, T=T, N=N, w=w, q=q, beta=beta, **{m: float(np.mean(np.array(v) < .05)) for m, v in P.items()})
                        print(row, flush=True); return row

def run(nrep=300, seed=1):
    from joblib import Parallel, delayed
    cells = [(k, T, N, w, q, b) for k in ('block', 'ar', 'iid') for T in (400, 800) for N in (10, 40, 120)
             for w, q in ((0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.6, 0.3)) for b in (0.0, 0.1)]
    return Parallel(n_jobs=-1)(delayed(cell)(*c, nrep, seed + i) for i, c in enumerate(cells))

if __name__ == '__main__':
    nrep = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    res = run(nrep)
    json.dump(res, open('theory/sim_corrected.json', 'w'), indent=1)
