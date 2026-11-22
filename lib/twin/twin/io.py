"""Loaders: plain arrays, NWB files, and the GATEKEEPER IBL cache.

Every loader returns trial-by-neuron spike counts X (T x N) plus a dict of trial variables."""
from __future__ import annotations

import numpy as np


def counts_from_spikes(spike_times_per_unit, event_times, window):
    lo, hi = window
    ev = np.asarray(event_times, float)
    X = np.zeros((len(ev), len(spike_times_per_unit)))
    for j, st in enumerate(spike_times_per_unit):
        st = np.sort(np.asarray(st, float))
        X[:, j] = np.searchsorted(st, ev + hi) - np.searchsorted(st, ev + lo)
    return X


def from_nwb(path, event_column="start_time", window=(-0.6, -0.1), trial_columns=None):
    """Read units' spike times and the trials table from an NWB file (requires pynwb).
    window is relative to event_column, in seconds. Returns X and {column: array} for the trials table."""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), "r") as io:
        nwb = io.read()
        units = nwb.units
        spikes = [units["spike_times"][i] for i in range(len(units.id))]
        trials = nwb.trials.to_dataframe()
        ev = trials[event_column].values
        ok = np.isfinite(ev)
        X = counts_from_spikes(spikes, ev[ok], window)
        cols = trial_columns or [c for c in trials.columns if c not in ("start_time", "stop_time")]
        return X, {c: trials[c].values[ok] for c in cols}



def from_gatekeeper(path, region, window="prior_a", gatekeeper_root=None):
    """One region of one cached IBL session from the GATEKEEPER repo (MVSession .npz)."""
    import sys
    from pathlib import Path
    root = Path(gatekeeper_root or "~/Downloads/GATEKEEPER/gatekeeper").expanduser()
    for p in (root, root / "next"):
        sys.path.insert(0, str(p))
    from gatekeeper.mv_data import MVSession
    s = MVSession(path)
    X = s.X(region, window)
    trials = dict(contrast=np.asarray(s.contrast, float), choice=np.asarray(s.choice), feedback=np.asarray(s.feedback, float),
                  rt=np.asarray(s.rt, float), probability_left=np.asarray(s.meta.get("probabilityLeft", []), float))
    return X, trials, s
