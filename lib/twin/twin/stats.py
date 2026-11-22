"""Test statistics. Each takes neurons X (T x N) and a label matrix Y (T x M) and returns one value per column,
so the real label and all null labels are scored in one call."""
from __future__ import annotations

import numpy as np


def _z(A):
    A = A - A.mean(0)
    return A / (A.std(0) + 1e-12)


def pooled_corr(X, Y):
    """S = T * sum_i r_i^2: the pooled single-neuron statistic most encoding claims use (Part 2 theory)."""
    Xz, Yz = _z(np.asarray(X, float)), _z(np.asarray(Y, float))
    T = len(Xz)
    R = Xz.T @ Yz / T                      # N x M correlations
    return T * (R ** 2).sum(0)


def decode_r2(X, Y, alphas=(1.0, 10.0, 100.0, 1000.0), folds=5, seed=0, contiguous=False):
    """Cross-validated ridge R^2 (alpha chosen per label by inner CV). contiguous=True uses blocked folds."""
    X = _z(np.asarray(X, float)); Y = np.asarray(Y, float)
    T, M = Y.shape
    idx = np.arange(T) if contiguous else np.random.default_rng(seed).permutation(T)
    parts = np.array_split(idx, folds)
    pred = np.zeros_like(Y)
    for te in parts:
        tr = np.setdiff1d(idx, te)
        Xt, Yt = X[tr], Y[tr] - Y[tr].mean(0)
        U, s, Vt = np.linalg.svd(Xt, full_matrices=False)
        UtY = U.T @ Yt
        # alpha per label by generalized cross-validation (closed form on the training fold)
        gcvs, Bs = [], []
        for a in alphas:
            H = s ** 2 / (s ** 2 + a)
            resid = Yt - U @ (H[:, None] * UtY)
            gcvs.append((resid ** 2).sum(0) / (1 - H.sum() / len(tr)) ** 2)
            Bs.append(Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY))
        pick = np.argmin(np.array(gcvs), axis=0)
        B = np.stack(Bs)[pick, :, np.arange(M)].T
        pred[te] = X[te] @ B + Y[tr].mean(0)
    ss = ((Y - Y.mean(0)) ** 2).sum(0)
    return 1 - ((Y - pred) ** 2).sum(0) / (ss + 1e-12)


STATS = {"pooled_corr": pooled_corr, "decode_r2": decode_r2}
