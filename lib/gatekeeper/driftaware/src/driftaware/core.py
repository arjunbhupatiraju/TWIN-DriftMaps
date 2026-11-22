"""Drift-aware test for whether a population of neurons encodes a trial-by-trial variable.

The problem
-----------
Neural firing rates drift slowly within a session, and many task variables (block priors, reward history,
engagement) also change slowly. A test that treats trials as independent reads the chance overlap between
the two as encoding, so it flags variables that do not exist.

The test
--------
1. Remove the top shared slow component of the population (a principal component of the 50-trial moving
   average). This step never looks at the label, so the null stays valid.
2. Whiten each neuron by its own autoregressive model (order chosen by BIC, up to 10) and apply the same
   filter to the label (generalised least squares).
3. Statistic S = sum over neurons of the squared correlation between whitened activity and whitened label.
4. Compare S with the same statistic computed on null labels: pseudo-sessions drawn from the task's
   generative process, or "imposter" label sequences taken from other sessions.

Every step that touches the data is label-free, so any null whose labels are exchangeable with the real one
gives an exact (finite-sample) p-value.

Reference: Bhupatiraju, A. MULTIVERSE (in preparation). Archive: https://osf.io/z3pmq
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.signal import lfilter

__all__ = ["ar_bic", "remove_shared_drift", "drift_stat", "plain_stat", "drift_test", "DriftTestResult",
           "imposter_labels", "shuffle_labels"]


def ar_bic(x: np.ndarray, pmax: int = 10) -> np.ndarray:
    """AR coefficients for a 1-D series, order 0..pmax chosen by BIC (least squares)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    T = len(x)
    pmax = int(min(pmax, max(0, T // 5)))
    best_bic, best_phi = np.inf, np.zeros(0)
    for p in range(pmax + 1):
        if p == 0:
            res, phi = x, np.zeros(0)
        else:
            L = np.column_stack([x[p - j - 1:T - j - 1] for j in range(p)])
            phi, *_ = np.linalg.lstsq(L, x[p:], rcond=None)
            res = x[p:] - L @ phi
        bic = len(res) * np.log(res.var() + 1e-300) + p * np.log(len(res))
        if bic < best_bic:
            best_bic, best_phi = bic, phi
    return best_phi


def _slow_components(X: np.ndarray, k: int, smooth: int) -> np.ndarray:
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    ker = np.ones(smooth) / smooth
    Zs = np.apply_along_axis(lambda c: np.convolve(c, ker, mode="same"), 0, Z)
    U = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)[0]
    return U[:, :k]


def remove_shared_drift(X: np.ndarray, Y: np.ndarray | None = None, k: int = 1, smooth: int = 50):
    """Project the top-k shared slow components out of the z-scored activity (and out of the labels, if given).

    X: (trials, neurons). Y: (trials,) or (trials, M). Returns X_clean (and Y_clean).
    """
    X = np.asarray(X, float)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    if k == 0:
        return (Z, None if Y is None else np.asarray(Y, float))
    U = _slow_components(X, k, smooth)
    Zc = Z - U @ (U.T @ Z)
    if Y is None:
        return Zc, None
    Y = np.asarray(Y, float)
    Y2 = Y[:, None] if Y.ndim == 1 else Y
    Yc = Y2 - U @ (U.T @ Y2)
    return Zc, (Yc[:, 0] if Y.ndim == 1 else Yc)


def drift_stat(X: np.ndarray, Y: np.ndarray, pmax: int = 10, phis: Sequence[np.ndarray] | None = None) -> np.ndarray:
    """GLS statistic for each label column. X (T, N), Y (T, M). Returns S of shape (M,)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T, N = X.shape
    Yc = Y - Y.mean(0)
    S = np.zeros(Y.shape[1])
    for i in range(N):
        phi = phis[i] if phis is not None else ar_bic(X[:, i], pmax)
        a = np.r_[1.0, -phi]
        xf = lfilter(a, [1.0], X[:, i] - X[:, i].mean())[len(phi):]
        Yf = lfilter(a, [1.0], Yc, axis=0)[len(phi):]
        sd = xf.std()
        if sd == 0:
            continue
        xf = (xf - xf.mean()) / sd
        Yf = Yf - Yf.mean(0)
        Yf = Yf / (np.linalg.norm(Yf, axis=0) + 1e-12)
        S += (xf @ Yf) ** 2 / len(xf)
    return S


def plain_stat(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Standard pooled single-neuron statistic: sum of squared correlations, no drift correction."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T = X.shape[0]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Yc = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
    return ((Z.T @ Yc / T) ** 2).sum(0)


@dataclass
class DriftTestResult:
    p: float
    stat: float
    null: np.ndarray
    n_neurons: int
    n_trials: int

    def __repr__(self):
        return f"DriftTestResult(p={self.p:.4g}, stat={self.stat:.4g}, n_null={len(self.null)}, N={self.n_neurons}, T={self.n_trials})"


def _null_matrix(y, nulls, n_null, rng):
    if callable(nulls):
        cols = [np.asarray(nulls(rng), float) for _ in range(n_null)]
    else:
        nulls = np.asarray(nulls, float)
        if nulls.ndim == 1:
            nulls = nulls[:, None]
        cols = [nulls[:, j] for j in range(nulls.shape[1])]
    for c in cols:
        if len(c) != len(y):
            raise ValueError(f"null label has length {len(c)}, expected {len(y)}")
    return np.column_stack([y] + cols)


def drift_test(X: np.ndarray, y: np.ndarray, nulls: np.ndarray | Callable[[np.random.Generator], np.ndarray],
               n_null: int = 200, k_shared: int = 1, smooth: int = 50, pmax: int = 10, seed: int | None = 0,
               method: str = "drift") -> DriftTestResult:
    """Test whether the population X encodes the label y.

    Parameters
    ----------
    X : (trials, neurons) spike counts or rates in the analysis window.
    y : (trials,) the variable (binary or continuous).
    nulls : null labels, either an array (trials, M) or a function rng -> (trials,) that draws one null label
        (a pseudo-session from the task's generative process, or an imposter sequence from another session).
        Do NOT use trial shuffles: they ignore the label's autocorrelation and the test is then invalid.
    n_null : number of null draws when `nulls` is a function.
    k_shared, smooth : shared slow components removed (default 1) and the moving-average width in trials.
    pmax : maximum AR order for the per-neuron whitening.
    method : "drift" (the drift-aware test) or "plain" (the standard statistic with the same null, for comparison).

    Returns
    -------
    DriftTestResult with the permutation p-value (1 + #null >= real) / (1 + #null).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    if X.ndim != 2 or X.shape[0] != len(y):
        raise ValueError("X must be (trials, neurons) with the same number of trials as y")
    keep = X.std(0) > 0
    X = X[:, keep]
    rng = np.random.default_rng(seed)
    Y = _null_matrix(y, nulls, n_null, rng)
    if method == "plain":
        s = plain_stat(X, Y)
    elif method == "drift":
        Xc, Yc = remove_shared_drift(X, Y, k=k_shared, smooth=smooth)
        s = drift_stat(Xc, Yc, pmax=pmax)
    else:
        raise ValueError("method must be 'drift' or 'plain'")
    p = float((1 + (s[1:] >= s[0]).sum()) / len(s))
    return DriftTestResult(p=p, stat=float(s[0]), null=s[1:], n_neurons=X.shape[1], n_trials=X.shape[0])


def imposter_labels(pool: Sequence[np.ndarray], T: int) -> Callable[[np.random.Generator], np.ndarray]:
    """Null-label generator from OTHER sessions' label sequences (never include the tested session).

    Each draw takes a random contiguous stretch of length T from the concatenated pool (wrapping at the end),
    so the null labels keep the real temporal structure of the variable.
    """
    cat = np.concatenate([np.asarray(p, float) for p in pool])
    if len(cat) < T:
        reps = int(np.ceil(T / len(cat))) + 1
        cat = np.tile(cat, reps)

    def draw(rng):
        start = int(rng.integers(0, len(cat)))
        return cat[(start + np.arange(T)) % len(cat)]
    return draw


def shuffle_labels(y: np.ndarray) -> Callable[[np.random.Generator], np.ndarray]:
    """Trial-shuffle null. Provided ONLY to reproduce the common (invalid under drift) procedure."""
    y = np.asarray(y, float)
    return lambda rng: rng.permutation(y)
