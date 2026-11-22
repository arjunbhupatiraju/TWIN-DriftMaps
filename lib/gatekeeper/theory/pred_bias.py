"""Fix 2 test: does a parametric (AR-model) estimate of drift remove the theory's underprediction?
Simulate sessions, run the shuffle test (200 perms), and predict each session's shuffle FP from
  OLD: empirical lagged covariances (K = T/3)
  NEW: diagonal from AR(p) models of each neuron and the label summed over all lags (less biased),
       off-diagonal still empirical. Compare binned predicted vs observed."""
import sys, numpy as np
sys.path.insert(0, '.')
from scipy.stats import chi2
from scipy.signal import lfilter
from theory.sim_corrected import label
from model.proto import neurons
from model.gls import ar_bic
from theory.corrected_test import shuffle_p, _acf

def ar_acf(phi, K):
    """theoretical ACF of AR(phi) up to lag K via impulse response."""
    if len(phi) == 0: return np.r_[1.0, np.zeros(K)]
    imp = np.zeros(4000); imp[0] = 1
    h = lfilter([1.0], np.r_[1.0, -phi], imp)
    g = np.array([h[:len(h) - k] @ h[k:] for k in range(K + 1)])
    return g / g[0]

def predict(X, y, param, K=None):
    T, N = X.shape; K = K or T // 3
    Z = (X - X.mean(0)) / X.std(0); yc = (y - y.mean()) / y.std()
    ry = _acf(yc, K)
    Sig = Z.T @ Z / T; R0 = Sig.copy()
    for k in range(1, K + 1):
        G = Z[:-k].T @ Z[k:] / T; Sig += ry[k] * (G + G.T)
    if param == 'unbiased':
        Sig = Z.T @ Z / T; R0 = Sig.copy()
        ry2 = np.array([1.0] + [yc[:-k] @ yc[k:] / (T - k) for k in range(1, K + 1)])
        for k in range(1, K + 1):
            G = Z[:-k].T @ Z[k:] / (T - k); Sig += ry2[k] * (G + G.T)
    elif param:
        ryp = ar_acf(ar_bic(yc, 10), 3000)
        for i in range(N):
            rx = ar_acf(ar_bic(Z[:, i], 10), 3000)
            Sig[i, i] = 1 + 2 * np.sum(rx[1:] * ryp[1:])
    if param == 'exact':
        l0 = np.clip(np.linalg.eigvalsh(R0), 0, None); l1 = np.clip(np.linalg.eigvalsh(Sig), 0, None)
        g = np.random.default_rng(0); Zs = g.standard_normal((20000, N)) ** 2
        c = np.quantile(Zs @ l0, 0.95); return float(np.mean(Zs @ l1 > c))
    sat = lambda M: (np.sum(M * M) / np.trace(M), np.trace(M) ** 2 / np.sum(M * M))
    a0, n0 = sat(R0); a1, n1 = sat(Sig)
    return chi2.sf(a0 * chi2.isf(.05, n0) / a1, n1)

rng = np.random.default_rng(4); rows = []
for _ in range(int(sys.argv[1]) if len(sys.argv) > 1 else 300):
    N = int(rng.choice([10, 20, 40])); w = rng.uniform(0.01, 0.25); T = int(rng.choice([400, 700]))
    y = label(T, 'block', rng); X = neurons(T, N, w, 0.15, rng.choice(['ar98', 'mix']), rng)
    rows.append((shuffle_p(X, y, 200, rng) < .05, predict(X, y, False), predict(X, y, 'exact'), predict(X, y, 'exact', K=T // 2)))
r = np.array(rows, float)
for j, name in ((1, 'OLD (empirical, T/3)'), (2, 'EXACT, T/3'), (3, 'EXACT, T/2')):
    q = np.quantile(r[:, j], np.linspace(0, 1, 7)); idx = np.clip(np.searchsorted(q, r[:, j]) - 1, 0, 5)
    pred = [r[idx == b, j].mean() for b in range(6)]; obs = [r[idx == b, 0].mean() for b in range(6)]
    print(f"{name:24s} pred: " + " ".join(f"{p:.2f}" for p in pred) + "\n" + " " * 25 + "obs:  " + " ".join(f"{o:.2f}" for o in obs) +
          f"   MAE {np.mean(np.abs(np.array(pred) - np.array(obs))):.3f}")
