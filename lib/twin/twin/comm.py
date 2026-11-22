"""Drift-safe tests of coupling ("communication") between two brain regions.

The common analysis predicts region B's trial-by-trial activity from region A (cross-validated regression) and
compares the result with a trial-shuffle null. Shuffling destroys each region's autocorrelation, so two regions that
merely drift slowly, even ones recorded on different days, look coupled (the spurious-regression problem).

comm_check(A, B) reports:
  p_common     the common pipeline: random-fold cross-validation against a trial shuffle (slow drift leaks between
               folds; on independent drifting regions this fires about 100% of the time)
  r2           cross-validated prediction of B from A (ridge, blocked folds, mean over B neurons)
  p_shuffle    blocked folds against a trial shuffle
  p_twin       null = twins of B (multivariate phase randomization: B keeps its own autocorrelation and internal
               correlations, its alignment with A is destroyed). A valid test of "any coupling" under drift.
  r2_slow / r2_fast, p_slow / p_fast
               the same analysis on the slow (periods > slow_trials) and fast parts of both regions, each against its
               own twin null: where in time the coupling lives. Slow-only coupling fits a shared state (arousal,
               engagement); fast coupling is what trial-by-trial communication would need. Note: statistics alone cannot
               tell communication from common fast input; that needs perturbations.
"""
from __future__ import annotations

import numpy as np

from .stats import decode_r2
from .drift import phase_twin


def comm_r2(A, B, folds=5, contiguous=True, seed=0):
    """Mean cross-validated R^2 of B's neurons predicted from A (ridge). contiguous=True: blocked folds (drift-safe);
    False: random folds, as most pipelines do (slow drift leaks between training and test trials)."""
    return float(np.mean(decode_r2(A, B, folds=folds, contiguous=contiguous, seed=seed)))


def split(X, slow_trials=5.0):
    """Slow (periods longer than slow_trials) and fast parts of a recording."""
    X = np.asarray(X, float)
    T = len(X); mu = X.mean(0)
    F = np.fft.rfft(X - mu, axis=0)
    lo = np.fft.rfftfreq(T) < 1.0 / slow_trials
    slow = np.fft.irfft(F * lo[:, None], n=T, axis=0) + mu
    return slow, X - slow + mu


def _p(real, null):
    return float((1 + (np.asarray(null) >= real).sum()) / (len(null) + 1))


def comm_check(A, B, n_null=100, slow_trials=5.0, seed=0, folds=5, parts=True):
    rng = np.random.default_rng(seed)
    A = np.asarray(A, float); B = np.asarray(B, float)
    # the common pipeline: random-fold CV against a trial shuffle
    rc = comm_r2(A, B, folds, contiguous=False)
    p_common = _p(rc, [comm_r2(A, B[rng.permutation(len(B))], folds, contiguous=False) for _ in range(n_null)])
    real = comm_r2(A, B, folds)
    shuf = [comm_r2(A, B[rng.permutation(len(B))], folds) for _ in range(n_null)]
    tw = [comm_r2(A, phase_twin(B, rng), folds) for _ in range(n_null)]
    out = dict(r2_common=rc, p_common=p_common, r2=real, p_shuffle=_p(real, shuf), p_twin=_p(real, tw), null_shuffle_mean=float(np.mean(shuf)),
               null_twin_mean=float(np.mean(tw)), n_null=n_null, slow_trials=slow_trials)
    if parts:
        (As, Af), (Bs, Bf) = split(A, slow_trials), split(B, slow_trials)
        for name, a, b in (("slow", As, Bs), ("fast", Af, Bf)):
            r = comm_r2(a, b, folds)
            nl = [comm_r2(a, phase_twin(b, rng), folds) for _ in range(max(20, n_null // 2))]
            out[f"r2_{name}"] = r; out[f"p_{name}"] = _p(r, nl); out[f"excess_{name}"] = float(max(r - np.mean(nl), 0))
        tot = out["excess_slow"] + out["excess_fast"]
        out["slow_share"] = float(out["excess_slow"] / tot) if tot > 0 else float("nan")
    out["verdict"] = _verdict(out)
    return out


def _verdict(o):
    if o["p_twin"] >= 0.05:
        if o["p_common"] < 0.05 or o["p_shuffle"] < 0.05:
            return ("NOT SUPPORTED: significant only against the trial shuffle; twins of B with the same drift but no alignment "
                    "with A predict as well (spurious coupling from autocorrelation).")
        return "No evidence of coupling."
    if "p_fast" not in o:
        return "Coupled (valid against twins)."
    if o["p_fast"] < 0.05 and o["p_slow"] < 0.05:
        return f"Coupled on slow AND fast timescales ({o['slow_share']:.0%} of the excess predictability is slow)."
    if o["p_fast"] < 0.05:
        return "Coupled on fast (trial-to-trial) timescales."
    return ("Coupled only on slow timescales: consistent with a shared slow state (arousal, engagement), not trial-by-trial "
            "communication.")


def simulate_pair(T, NA, NB, rng, drift=None, shared_between=True, coupling=0.0, frac=0.3):
    """Two regions with drift. shared_between=True: one drift process over both regions (shared slow state);
    False: independent drifts (e.g. different sessions). coupling > 0 adds fast communication: a fraction of B's
    neurons receive a random projection of A's fast activity with per-neuron correlation about `coupling`."""
    from .drift import DriftModel
    dm = drift or DriftModel(slow_frac=0.4, phi=0.97, shared=0.5, inst_corr=0.0)
    if shared_between:
        X = dm.simulate(T, NA + NB, rng); A, B = X[:, :NA], X[:, NA:].copy()
    else:
        A, B = dm.simulate(T, NA, rng), dm.simulate(T, NB, rng)
    if coupling > 0:
        drive = split(A)[1] @ rng.standard_normal((NA, NB))
        drive = (drive - drive.mean(0)) / (drive.std(0) + 1e-12)
        idx = rng.choice(NB, max(1, int(round(frac * NB))), replace=False)
        b = coupling / np.sqrt(max(1e-6, 1 - coupling ** 2))
        B[:, idx] += b * B[:, idx].std(0) * drive[:, idx]
    return A, B
