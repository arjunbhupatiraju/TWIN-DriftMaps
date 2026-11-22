"""PART 3 step 4c: a more powerful version of the brain-wide slow-signal test, run IDENTICALLY on IBL and Steinmetz.

Why: on short sessions (Steinmetz, ~300 trials) the v1 test (ridge from all neurons to the fitted bias path) has little
power, because two slow signals over a short session look alike by chance (null decoders already reach r ~ 0.85).
v3 changes only the statistic:
  1. history, current choice, block prior and movement/arousal are removed from the activity (as before)
  2. the population is summarized by its top 5 SLOW components (10-trial smoothing), which is where the hypothesis says
     the memory lives, instead of hundreds of noisy neurons
  3. very slow trends (periods over 100 trials) are removed from both the components and every target, so the test uses
     the 10-100 trial timescale, where a short session still has many independent samples
  4. same targets, same null (other mice's paths), same fake and planted checks
Simulation (300-trial sessions): calibration 0.017 (v1 0.083), power 0.48 (v1 0.25).

    python3 part3/mechanism3.py ibl
    python3 part3/mechanism3.py steinmetz --data <folder with steinmetz_part*.npz>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mechanism as m1  # noqa: E402
import mechanism2 as m2  # noqa: E402

K_PC, SMOOTH, PERIOD, LAM = 5, 10, 100, 1.0


def slow_pcs(X, k=K_PC, w=SMOOTH):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ker = np.ones(w) / w
    S = np.apply_along_axis(lambda x: np.convolve(x, ker, "same"), 0, Z)
    U, s, _ = np.linalg.svd(S - S.mean(0), full_matrices=False)
    return U[:, :k] * s[:k]


def detrend(Y, period=PERIOD):
    T = len(Y)
    nb = max(1, int(2 * T / period))
    t = (np.arange(T) + 0.5) / T
    B = np.column_stack([np.ones(T)] + [np.cos(np.pi * j * t) for j in range(1, nb + 1)])
    return Y - B @ np.linalg.lstsq(B, Y, rcond=None)[0]


def test_v3(X, targets, D, owns, lam=None):
    """Drop-in replacement for mechanism._test_lam (same inputs and outputs)."""
    Y = np.column_stack([m1.resid(targets[:, [j]], np.column_stack([D, owns[j]]))[:, 0] for j in range(targets.shape[1])])
    Xr = m1.resid(X.astype(float), D)
    P = detrend(slow_pcs(Xr))
    r = m1.ridge_scores(P, detrend(Y), lam=LAM)
    return float((1 + (r[1:] >= r[0]).sum()) / len(r)), float(r[0]), float(np.median(r[1:]))


import os
os.environ["MV_TEST"] = "v3"    # every pooled test in mechanism2 (including in parallel workers) now uses v3


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["ibl", "steinmetz"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--data", default="~/Downloads/MULTIVERSE/steinmetz_data")
    ap.add_argument("--out", default="results/mechanism3")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.step == "ibl":
        m2.run_ibl(a.cache, a.out + "/ibl", a.jobs)
    else:
        m2.run_steinmetz(a.data, a.out + "/steinmetz", a.jobs)
