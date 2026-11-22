"""Synthetic sessions with slow neural drift and block-structured labels, for tests and the tutorial."""
from __future__ import annotations

import numpy as np

__all__ = ["block_labels", "simulate_session"]


def block_labels(T: int, rng: np.random.Generator, mean_len: int = 60, lo: int = 20, hi: int = 100,
                 unbiased: int = 90) -> np.ndarray:
    """IBL-style block prior: 0.5 for the first `unbiased` trials, then alternating 0.2/0.8 blocks.
    Returns the binary block identity (1 = right-biased block); unbiased trials are labelled at random."""
    lab = list(rng.integers(0, 2, min(unbiased, T)))
    cur = int(rng.integers(0, 2))
    while len(lab) < T:
        L = 0
        while not lo <= L <= hi:
            L = int(rng.exponential(mean_len))
        lab += [cur] * L
        cur = 1 - cur
    return np.asarray(lab[:T], float)


def simulate_session(T: int = 600, N: int = 30, drift: float = 0.6, shared: float = 0.4, beta: float = 0.0,
                     ar: float = 0.98, seed: int = 0):
    """Return (X, y): X (T, N) activity with slow AR(1) drift (part shared across neurons), y block labels.

    drift: variance fraction of the slow component; shared: fraction of the drift common to all neurons;
    beta: true encoding strength of y (0 = no encoding, so any detection is a false positive).
    """
    rng = np.random.default_rng(seed)
    y = block_labels(T, rng)

    def ar1(n):
        z = np.zeros((T, n))
        e = rng.standard_normal((T, n)) * np.sqrt(1 - ar ** 2)
        z[0] = rng.standard_normal(n)
        for t in range(1, T):
            z[t] = ar * z[t - 1] + e[t]
        return z
    common = ar1(1)
    private = ar1(N)
    slow = np.sqrt(shared) * common + np.sqrt(1 - shared) * private
    X = np.sqrt(drift) * slow + np.sqrt(1 - drift) * rng.standard_normal((T, N))
    w = rng.standard_normal(N)
    X = X + beta * np.outer(y - y.mean(), w)
    return X, y
