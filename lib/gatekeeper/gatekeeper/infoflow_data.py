"""Data layer for INFOFLOW ("Where does the brain lose what it sees?").

Per session it caches, for every Beryl region, spike counts of every sorted unit in three
windows aligned to stimulus onset, plus a good-unit mask:
    pre    [-0.40, -0.05] s   (baseline / state)
    early  [ 0.03,  0.13] s   (primary: sensory, mostly before movement and choice signals)
    late   [ 0.13,  0.23] s   (secondary: where choice signals start to mix in)
and per trial: signed contrast, choice, feedback, reaction time (first movement - stimOn),
pre-stimulus pupil. Raw spike files are deleted after caching (disk stays small).

EXPLORE / CONFIRM SPLIT (fixed 2026-10-11, before any post-stimulus data was downloaded):
a session belongs to the EXPLORE set iff int(sha256(subject nickname), 16) % 10 < 4,
i.e. ~40% of mice. Splitting by mouse (not session) keeps the confirm set independent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .data import Session
from .ibl import EXCLUDE_REGIONS, GOOD_LABEL, _delete_raw_spikes, _pupil, _trials, region_labels

WINDOWS = {"pre": (-0.40, -0.05), "early": (0.03, 0.13), "late": (0.13, 0.23)}
EXPLORE_FRACTION_TENTHS = 4


def split_of(subject: str) -> str:
    """Deterministic, outcome-blind assignment of a mouse to 'explore' or 'confirm'."""
    h = int(hashlib.sha256(subject.encode()).hexdigest(), 16)
    return "explore" if h % 10 < EXPLORE_FRACTION_TENTHS else "confirm"


def counts_window(spike_times, spike_clusters, cluster_ids, onsets, lo, hi):
    order = np.argsort(spike_clusters, kind="stable")
    sc, st = spike_clusters[order], spike_times[order]
    a_b = np.searchsorted(sc, cluster_ids, "left")
    b_b = np.searchsorted(sc, cluster_ids, "right")
    out = np.zeros((len(onsets), len(cluster_ids)), np.int16)
    for j, (a, b) in enumerate(zip(a_b, b_b)):
        t = st[a:b]
        out[:, j] = np.searchsorted(t, onsets + hi) - np.searchsorted(t, onsets + lo)
    return out


def counts_all_windows(spike_times, spike_clusters, cluster_ids, onsets, windows=WINDOWS):
    """{window: (T, N) int16} for all clusters at once. Sorts the probe's spikes ONCE
    (the per-region/per-window re-sorting made downloads ~2.5x slower)."""
    order = np.argsort(spike_clusters, kind="stable")
    sc, st = spike_clusters[order], spike_times[order]
    a_b = np.searchsorted(sc, cluster_ids, "left")
    b_b = np.searchsorted(sc, cluster_ids, "right")
    out = {w: np.zeros((len(onsets), len(cluster_ids)), np.int16) for w in windows}
    edges = {w: (onsets + lo, onsets + hi) for w, (lo, hi) in windows.items()}
    for j, (a, b) in enumerate(zip(a_b, b_b)):
        t = st[a:b]
        for w, (lo, hi) in edges.items():
            out[w][:, j] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    return out


def _extra_trial_fields(one, eid, n):
    tr = one.load_object(eid, "trials")
    on = np.asarray(tr["stimOn_times"], float)
    keep = np.isfinite(on)
    fm = np.asarray(tr.get("firstMovement_times", np.full(len(on), np.nan)), float)
    fb = np.asarray(tr["feedbackType"], float)
    rt = (fm - on)[keep]
    fb = fb[keep]
    assert len(rt) == n, "trial count mismatch"
    return rt, fb


def load_session_info(one, eid, cache_dir, delete_raw=True):
    cache = Path(cache_dir) / f"{eid}.npz"
    if cache.exists():
        return cache
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    contrast, choice, onsets, meta = _trials(one, eid)
    rt, fb = _extra_trial_fields(one, eid, len(onsets))
    try:
        subject = str(one.get_details(eid)["subject"])
    except Exception:
        subject = "unknown"
    arrays = {}
    for ins in one.alyx.rest("insertions", "list", session=eid):
        pname = ins["name"]
        colls = sorted([c for c in one.list_collections(eid) if c.startswith(f"alf/{pname}")], key=len, reverse=True)
        for coll in colls:
            rl = region_labels(one, eid, coll)
            if rl is None:
                continue
            try:
                spikes = one.load_object(eid, "spikes", collection=coll, attribute=["times", "clusters"])
            except Exception:
                continue
            acr, label = rl
            beryl = np.asarray(br.acronym2acronym(acr, mapping="Beryl")).astype(str)
            st = np.asarray(spikes["times"], float)
            scl = np.asarray(spikes["clusters"], int)
            keep_ids = np.flatnonzero(~np.isin(beryl, list(EXCLUDE_REGIONS)))
            allw = counts_all_windows(st, scl, keep_ids, onsets)
            col = {cid: j for j, cid in enumerate(keep_ids)}
            for r in np.unique(beryl[keep_ids]):
                ids = np.flatnonzero(beryl == r)
                cols = [col[i] for i in ids]
                g = (label[ids] == GOOD_LABEL) | np.isnan(label[ids])
                for w in WINDOWS:
                    X = allw[w][:, cols]
                    k = f"{w}__{r}"
                    arrays[k] = np.hstack([arrays[k], X]) if k in arrays else X
                arrays[f"good__{r}"] = np.r_[arrays.get(f"good__{r}", np.zeros(0, bool)), g]
            break
    pupil, src = _pupil(one, eid, onsets)
    if pupil is not None:
        arrays["pupil"] = pupil.astype(np.float32)
    meta.update(eid=eid, subject=subject, split=split_of(subject), pupil_source=src,
                good_units={k[6:]: int(v.sum()) for k, v in arrays.items() if k.startswith("good__")})
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, contrast=contrast, choice=choice, rt=rt, feedback=fb, meta=json.dumps(meta), **arrays)
    if delete_raw:
        _delete_raw_spikes(one, eid)
    return cache


class InfoSession:
    """Lazy view of one cached session."""

    def __init__(self, path, all_units=False):
        self.path = Path(path)
        z = np.load(path, allow_pickle=False)
        self._z, self.all_units = z, all_units
        self.meta = json.loads(str(z["meta"]))
        self.sid = self.path.stem
        self.contrast = z["contrast"]
        self.choice = z["choice"].astype(int)
        self.rt = z["rt"]
        self.feedback = z["feedback"]
        self.pupil = z["pupil"] if "pupil" in z.files else None
        self.regions = sorted({k.split("__", 1)[1] for k in z.files if k.startswith("early__")})
        self.split = self.meta.get("split", "unknown")

    def X(self, region, window="early"):
        z = self._z
        k = f"{window}__{region}"
        if k not in z.files:
            return None
        X = z[k].astype(float)
        if not self.all_units:
            X = X[:, z[f"good__{region}"].astype(bool)]
        return X if X.shape[1] else None

    def as_behavior_session(self, use_block=True):
        """Session object for the GLM-HMM (behavior only), with the block prior as an input."""
        blk = None
        pl = self.meta.get("probabilityLeft")
        if use_block and pl is not None and len(pl) == len(self.contrast):
            pl = np.nan_to_num(np.asarray(pl, float), nan=0.5)
            blk = np.sign(0.5 - pl)          # +1 right-likely block, -1 left-likely, 0 unbiased
        return Session(self.sid, self.contrast, self.choice, {}, block=blk)


def load_info_cache(cache_dir, split=None, all_units=False):
    out = [InfoSession(p, all_units) for p in sorted(Path(cache_dir).glob("*.npz"))]
    return [s for s in out if split is None or s.split == split]
