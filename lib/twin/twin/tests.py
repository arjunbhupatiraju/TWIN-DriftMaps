"""Every test of "does this population encode label y?" in one place, so they can be compared on equal footing.

Single session (X: trials x neurons, y: label per trial), statistic f(X, Y) scoring each column of Y:
  shuffle       permute trials of y (the common test; invalid for slow labels)
  pseudo        labels redrawn from the task's generative process (IBL / Findling; needs a task model)
  linear_shift  Harris 2020: compare the unshifted alignment with alignments of y shifted by -S..S trials, all on
                the same truncated window (no task model; minimum p = 1 / (2S + 1))
  twin          TWIN: neural twins (multivariate phase randomization of X, same phases for all neurons) keep the
                recording's own autocorrelation and correlations; y is untouched (no task model)
Several sessions:
  session_perm  Harris 2020: neural data of session i scored against labels of session j != i (truncated to the
                shorter length); the matched pairing is compared with all mismatched pairings
"""
from __future__ import annotations

import numpy as np

from .drift import phase_twin
from .stats import pooled_corr


def _p(real, null):
    null = np.asarray(null)
    return float((1 + np.sum(null >= real)) / (len(null) + 1))


def p_shuffle(X, y, rng, n=200, stat=pooled_corr):
    v = stat(X, np.column_stack([y] + [rng.permutation(y) for _ in range(n)]))
    return _p(v[0], v[1:])


def p_pseudo(X, y, rng, null_labels, n=200, stat=pooled_corr):
    v = stat(X, np.column_stack([y, null_labels.matrix(len(y), n, rng)]))
    return _p(v[0], v[1:])


def p_linear_shift(X, y, rng=None, S=None, stat=pooled_corr):
    T = len(y)
    S = S or max(10, T // 10)
    L = T - 2 * S
    Xw = X[S:S + L]
    Y = np.column_stack([y[S + s:S + s + L] for s in range(-S, S + 1)])
    v = stat(Xw, Y)
    real = v[S]
    return float(np.mean(v >= real))                           # rank of the unshifted alignment among 2S+1


def p_twin(X, y, rng, n=200, stat=pooled_corr):
    y = np.asarray(y, float)
    real = stat(X, y[:, None])[0]
    null = [stat(phase_twin(X, rng), y[:, None])[0] for _ in range(n)]
    return _p(real, null)


def p_session_perm(sessions, rng, n=500, stat=pooled_corr):
    """sessions: list of (X, y). Statistic summed over matched pairs vs over random derangements (every session's
    neurons scored against another session's labels, truncated to the shorter length)."""
    k = len(sessions)
    M = np.zeros((k, k))
    for i, (X, _) in enumerate(sessions):
        for j, (_, y) in enumerate(sessions):
            L = min(len(X), len(y))
            M[i, j] = stat(X[:L], np.asarray(y[:L], float)[:, None])[0] / L
    real = np.trace(M)
    null = []
    while len(null) < n:
        perm = rng.permutation(k)
        if np.any(perm == np.arange(k)):
            continue
        null.append(M[np.arange(k), perm].sum())
    return _p(real, null)
