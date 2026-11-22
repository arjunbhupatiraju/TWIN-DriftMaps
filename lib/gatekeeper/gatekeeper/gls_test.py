"""GLS-imposter test: whiten each neuron by ITS OWN estimated temporal noise model (AR(p) fit to x_i),
apply the same filter to the label, then S = sum_i corr(filtered x_i, filtered y)^2.
Null: identical computation with imposter labels. Valid for any label-free preprocessing (session exchangeability)."""
import numpy as np
from scipy.signal import lfilter

def ar_coef(x, p):
    x = x - x.mean(); T = len(x)
    L = np.column_stack([x[p - j - 1:T - j - 1] for j in range(p)])
    phi, *_ = np.linalg.lstsq(L, x[p:], rcond=None)
    return phi

def ar_bic(x, pmax):
    x = x - x.mean(); T = len(x); best = (np.inf, np.zeros(0))
    for p in range(0, pmax + 1):
        if p == 0: res = x; phi = np.zeros(0)
        else:
            L = np.column_stack([x[p - j - 1:T - j - 1] for j in range(p)]); phi, *_ = np.linalg.lstsq(L, x[p:], rcond=None); res = x[p:] - L @ phi
        b = len(res) * np.log(res.var() + 1e-300) + p * np.log(len(res))
        if b < best[0]: best = (b, phi)
    return best[1]

def gls_stat(X, Ys, pmax=10, shrink=1.0, phis=None):
    """X (T,N); Ys (T,M) label columns (col 0 real). Returns S for each column."""
    T, N = X.shape
    Yc = Ys - Ys.mean(0)
    S = np.zeros(Ys.shape[1])
    for i in range(N):
        phi = phis[i] if phis is not None else ar_bic(X[:, i], pmax) * shrink
        a = np.r_[1.0, -phi]
        xf = lfilter(a, [1.0], X[:, i] - X[:, i].mean())[len(phi):]
        Yf = lfilter(a, [1.0], Yc, axis=0)[len(phi):]
        xf = (xf - xf.mean()) / (xf.std() + 1e-12)
        Yf = Yf - Yf.mean(0); Yf /= (np.linalg.norm(Yf, axis=0) + 1e-12)
        S += (xf @ Yf) ** 2 / len(xf)
    return S

def p_gls(X, y, nulls, **kw):
    s = gls_stat(X, np.column_stack([y] + nulls), **kw)
    return (1 + (s[1:] >= s[0]).sum()) / len(s)


def remove_shared(X, k=1, smooth=25):
    """Remove the top-k shared SLOW components (PCs of the moving-averaged population activity) from every neuron.
    Label-free, so the imposter null stays valid."""
    if k == 0:
        return X
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ker = np.ones(smooth) / smooth
    Zs = np.apply_along_axis(lambda c: np.convolve(c, ker, mode='same'), 0, Z)
    U, s, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    C = U[:, :k]                                   # slow shared time courses
    return Z - C @ (C.T @ Z)


def p_gls_shared(X, y, nulls, k=1, **kw):
    Xr = remove_shared(X, k)
    Yr = np.column_stack([y] + nulls).astype(float)
    if k:
        Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
        ker = np.ones(25) / 25
        Zs = np.apply_along_axis(lambda c: np.convolve(c, ker, mode='same'), 0, Z)
        U = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)[0][:, :k]
        Yr = Yr - U @ (U.T @ Yr)                   # project the same components out of the labels
    s = gls_stat(Xr, Yr, **kw)
    return (1 + (s[1:] >= s[0]).sum()) / len(s)
