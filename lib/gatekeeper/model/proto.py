"""Prototype: drift-robust imposter test (DRIFT-X).
Statistic: remove the K slowest cosine components (DCT basis) from neurons AND label, then S = T*sum_i r_i^2.
Null: the same statistic with labels from OTHER sessions (imposters; in simulation = fresh draws of the label process).
Valid for any K chosen without looking at y (exchangeability of sessions). Power depends on K."""
import sys, numpy as np
sys.path.insert(0, '.')
from theory.sim_corrected import label, ar1
from scipy.signal import lfilter

def drift(T, N, rng, kind):
    if kind == 'ar98':   return ar1(T, 0.98, N, rng)
    if kind == 'slow':   return ar1(T, 0.998, N, rng)
    if kind == 'mix':    # long-memory-like: sum of timescales
        return (ar1(T, 0.9, N, rng) + ar1(T, 0.98, N, rng) + ar1(T, 0.995, N, rng) + ar1(T, 0.999, N, rng)) / 2

def neurons(T, N, w, q, kind, rng):
    d = np.sqrt(q) * drift(T, 1, rng, kind) + np.sqrt(1 - q) * drift(T, N, rng, kind)
    return np.sqrt(w) * d + np.sqrt(1 - w) * rng.standard_normal((T, N))

def basis(T, K):
    t = (np.arange(T) + 0.5) / T
    B = np.column_stack([np.ones(T)] + [np.cos(np.pi * k * t) for k in range(1, K + 1)])
    Q, _ = np.linalg.qr(B); return Q

def stat(Xr, Y):   # Xr already residualised + standardised; Y columns residualised
    Y = Y / (np.linalg.norm(Y, axis=0) + 1e-12)
    Z = Xr / (np.linalg.norm(Xr, axis=0) + 1e-12)
    return ((Z.T @ Y) ** 2).sum(0)

def p_test(X, y, nulls, K):
    T = len(y); Q = basis(T, K)
    R = lambda A: A - Q @ (Q.T @ A)
    Xr = R(X); Y = R(np.column_stack([y] + nulls))
    s = stat(Xr, Y); return (1 + (s[1:] >= s[0]).sum()) / len(s)

if __name__ == '__main__':
    rng = np.random.default_rng(3); T = 600; nrep = 200
    Ks = (0, 2, 4, 8, 16)
    for lab in ('block', 'ar'):
        for dk in ('ar98', 'mix'):
            for w in (0.3, 0.6):
                for beta in (0.0, 0.1):
                    hits = {K: 0 for K in Ks}
                    for _ in range(nrep):
                        N = 40; y = label(T, lab, rng); X = neurons(T, N, w, 0.2, dk, rng)
                        if beta: X[:, :8] += beta * (y - y.mean())[:, None] / y.std()
                        nulls = [label(T, lab, rng) for _ in range(199)]
                        for K in Ks: hits[K] += p_test(X, y, nulls, K) < .05
                    print(f"label={lab:5s} drift={dk:4s} w={w} beta={beta}: " + "  ".join(f"K={K}:{hits[K]/nrep:.3f}" for K in Ks), flush=True)
