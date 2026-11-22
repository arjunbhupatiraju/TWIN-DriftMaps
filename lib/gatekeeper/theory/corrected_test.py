"""Drift-corrected population test (DCT).

Question it answers: do N simultaneously recorded neurons carry information about a trial variable y,
when both the neurons and y drift slowly over the session?

Statistic:  S = T * sum_i r_i^2,   r_i = corr(x_i, y)                        (pooled single-neuron test)
Null law under drift (Bartlett 1935, multivariate):
    cov(r) ~= Sigma / T,  Sigma_ij = sum_{|k|<=K} w_k rho_y(k) R_ij(k)
    R_ij(k) = lagged cross-correlation of neurons i, j, estimated from residuals of x on y
    w_k = Bartlett (Newey-West) taper; K = lag window
S is a weighted sum of chi^2_1 with weights eig(Sigma). p from Satterthwaite: S ~ a chi^2_nu,
a = tr(Sigma^2)/tr(Sigma), nu = tr(Sigma)^2/tr(Sigma^2).
No model of how y was generated is needed (the pseudo-session test needs one), and no permutations.
"""
import numpy as np
from scipy.stats import chi2


def _acf(y, K):
    y = (y - y.mean()) / (y.std() + 1e-12); T = len(y)
    return np.array([1.0] + [np.dot(y[:-k], y[k:]) / T for k in range(1, K + 1)])


def dct(X, y, K=None, taper=True, resid=True):
    """X: (T, N) neural responses, y: (T,) label. Returns (S, p, details)."""
    T, N = X.shape
    if K is None:
        K = int(min(T // 4, 4 * T ** (1 / 2)))
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    yc = (y - y.mean()) / (y.std() + 1e-12)
    r = Z.T @ yc / T
    S = T * np.sum(r ** 2)
    E = Z - np.outer(yc, r) if resid else Z           # remove the y-related part before estimating drift
    E = (E - E.mean(0)) / (E.std(0) + 1e-12)
    ry = _acf(yc, K)
    Sig = E.T @ E / T
    for k in range(1, K + 1):
        w = (1 - k / (K + 1)) if taper else 1.0
        G = E[:-k].T @ E[k:] / T
        Sig += w * ry[k] * (G + G.T)
    tr1, tr2 = np.trace(Sig), np.sum(Sig * Sig)
    if tr1 <= 0:
        return S, 1.0, dict(a=np.nan, nu=np.nan, K=K)
    a, nu = tr2 / tr1, tr1 ** 2 / tr2
    return S, float(chi2.sf(S / a, nu)), dict(a=a, nu=nu, K=K, V_mean=tr1 / N)


def shuffle_p(X, y, n=200, rng=None):
    rng = np.random.default_rng(rng); T = len(y)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Y = np.column_stack([y] + [rng.permutation(y) for _ in range(n)])
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    S = ((Z.T @ Y / T) ** 2).sum(0)
    return float((1 + (S[1:] >= S[0]).sum()) / (n + 1))


def naive_p(X, y):
    """Textbook parametric: S ~ chi^2_N (independent trials, independent neurons)."""
    T, N = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12); yc = (y - y.mean()) / (y.std() + 1e-12)
    return float(chi2.sf(T * np.sum((Z.T @ yc / T) ** 2), N))


def _ar_fit(y, pmax=10):
    """Least-squares AR(p) for y, p by BIC. Returns coefficients phi (len p)."""
    y = y - y.mean(); T = len(y); best = (np.inf, np.zeros(0))
    for p in range(0, pmax + 1):
        if p == 0:
            res = y; phi = np.zeros(0)
        else:
            L = np.column_stack([y[p - j - 1:T - j - 1] for j in range(p)])
            phi, *_ = np.linalg.lstsq(L, y[p:], rcond=None); res = y[p:] - L @ phi
        bic = len(res) * np.log(np.mean(res ** 2) + 1e-300) + p * np.log(len(res))
        if bic < best[0]:
            best = (bic, phi)
    return best[1]


def _filt(x, phi):
    p = len(phi)
    if p == 0:
        return x
    out = x[p:].copy()
    for j in range(p):
        out = out - phi[j] * x[p - j - 1:len(x) - j - 1]
    return out


def dct_pw(X, y, pmax=10, phi=None):
    """Prewhitened drift-corrected test (Box-Jenkins / Haugh 1976 prewhitening, extended to a population of neurons).
    1. fit AR(p) to the label y (p by BIC), or pass phi if the label's temporal model is known;
    2. apply the SAME filter to y and to every neuron -> filtered label is white;
    3. with a white label, cov(r) = R0/T exactly (R0 = instantaneous neuron correlation, from residuals),
       whatever the neurons' own drift; S = T sum r_i^2 is tested against Satterthwaite a*chi2_nu from eig(R0)."""
    if phi is None:
        phi = _ar_fit(y, pmax)
    yf = _filt(y - y.mean(), phi)
    Xf = np.column_stack([_filt(X[:, i] - X[:, i].mean(), phi) for i in range(X.shape[1])]) if X.ndim == 2 else None
    T, N = Xf.shape
    Z = (Xf - Xf.mean(0)) / (Xf.std(0) + 1e-12); yc = (yf - yf.mean()) / (yf.std() + 1e-12)
    r = Z.T @ yc / T; S = T * np.sum(r ** 2)
    E = Z - np.outer(yc, r); E = (E - E.mean(0)) / (E.std(0) + 1e-12)
    R0 = E.T @ E / T
    tr1, tr2 = np.trace(R0), np.sum(R0 * R0)
    a, nu = tr2 / tr1, tr1 ** 2 / tr2
    return S, float(chi2.sf(S / a, nu)), dict(p=len(phi), a=a, nu=nu)
