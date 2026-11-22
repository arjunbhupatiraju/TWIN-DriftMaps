"""Theory of shuffle-null false positives in population decoding under slow drift.

Setup (null hypothesis true): T trials; task label y_t with autocorrelation rho_y(k); N neurons x_it, independent of y,
each with autocorrelation rho_x(k). A label-shuffle null destroys temporal order.

1. One neuron (Bartlett 1946): the sample correlation r between x and y has variance
       Var_true(r) = V / T,   V = 1 + 2 * sum_{k>=1} rho_x(k) * rho_y(k)        (AR(1): V = (1 + a*b) / (1 - a*b))
   while the shuffle null assumes Var(r) = 1 / T.
2. Population (new): a linear decoder's evidence grows with S = sum_i r_i^2. Under the shuffle null
       E[S] = N / T,  sd[S] = sqrt(2N) / T;   under drift  E[S] = N V / T,  sd[S] = sqrt(2N) V / T.
   So the spurious signal, in units of the shuffle null, is
       z = (V - 1) * sqrt(N / 2)
   and the false-positive rate of a one-sided alpha test is
       FP(N, V) = Phi( ((V - 1) * sqrt(N / 2) - z_alpha) / V ).
   Consequence: for any drift (V > 1), FP -> 1 as the number of neurons grows. Larger recordings make shuffle
   nulls worse, not better. Cross-validation does not remove it, because train and test trials share the drift.
3. Pseudo-session (or session-permutation) nulls regenerate y with its own temporal structure, so the null
   distribution carries the same V and the inflation cancels.
"""
import numpy as np
from scipy.stats import norm


def V_from_acf(rx, ry, kmax=None):
    k = min(len(rx), len(ry)) if kmax is None else kmax
    return 1 + 2 * np.sum(rx[1:k] * ry[1:k])


def fp_theory(N, V, alpha=0.05):
    za = norm.isf(alpha)
    return norm.cdf(((V - 1) * np.sqrt(N / 2) - za) / V)


def acf(x, kmax):
    x = (x - x.mean()) / (x.std() + 1e-12)
    return np.array([1.0] + [np.mean(x[:-k] * x[k:]) for k in range(1, kmax)])


def decode(X, Y, lam=1.0):
    """Exact leave-one-out ridge correlation (the MULTIVERSE decoder) for every column of Y."""
    T, N = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    H = Z @ np.linalg.solve(Z.T @ Z + lam * N * np.eye(N), Z.T)
    h = np.clip(np.diag(H), None, 0.999)
    Yc = Y - Y.mean(0)
    loo = (H @ Yc - h[:, None] * Yc) / (1 - h[:, None])
    return (loo * Yc).sum(0) / (np.linalg.norm(loo, axis=0) * np.linalg.norm(Yc, axis=0) + 1e-12)


def ar1(T, N, a, rng):
    x = np.zeros((T, N))
    e = rng.standard_normal((T, N))
    x[0] = e[0]
    for t in range(1, T):
        x[t] = a * x[t - 1] + np.sqrt(1 - a * a) * e[t]
    return x


# ---------------------------------------------------------------- v2: correlated neurons (shared drift)
def fp_general(N, delta, rho0=0.0, q=0.0, alpha=0.05):
    """Population statistic S = sum r_i^2 with neuron correlations.
    Shuffle null: cov(r) = C/T with C = equicorrelated (1 on diagonal, rho0 off) - instantaneous neuron correlation.
    True null (drift): cov(r) = (C + delta * Q)/T, Q = equicorrelated (1, q) - drift shared across neurons,
    delta = V - 1 = 2 * sum_k rho_x(k) rho_y(k).
    Gaussian approximation:  FP = Phi( (N*delta - z_a * sqrt(2 tr C^2)) / sqrt(2 tr Sigma^2) ).
    With rho0 = q = 0 this reduces to Phi(((V-1) sqrt(N/2) - z_a) / V).
    For q > 0, FP saturates as N grows:  FP_inf = Phi( delta / (sqrt(2) * (rho0 + delta * q)) )."""
    za = norm.isf(alpha)
    N = np.asarray(N, float)
    trC2 = N + N * (N - 1) * rho0 ** 2
    trS2 = N * (1 + delta) ** 2 + N * (N - 1) * (rho0 + delta * q) ** 2
    return norm.cdf((N * delta - za * np.sqrt(2 * trC2)) / np.sqrt(2 * trS2))
