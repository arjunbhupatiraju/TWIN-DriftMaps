"""Calcium-indicator support. Indicators (GCaMP etc.) decay over seconds, so trial-level fluorescence mixes the
current trial with previous ones. This module models that smearing so planted controls, planner simulations and
identifiability checks see what a calcium recording would actually show.

tau_trials = indicator decay time / trial duration (e.g. GCaMP6s ~1.5 s decay, 6 s trials -> 0.25).
"""
from __future__ import annotations

import numpy as np


def smear(x, tau_trials):
    """Exponential carry-over across trials: f_t = s_t + exp(-1/tau) f_{t-1} (per column)."""
    x = np.asarray(x, float)
    if tau_trials is None or tau_trials <= 0:
        return x
    k = np.exp(-1.0 / tau_trials)
    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + k * out[t - 1]
    return out


def carryover(tau_trials, lags=4):
    """Share of the signal at lags 1..lags relative to lag 0 (how much a trial leaks into later ones)."""
    k = np.exp(-1.0 / tau_trials) if tau_trials and tau_trials > 0 else 0.0
    return [float(k ** j) for j in range(1, lags + 1)]
