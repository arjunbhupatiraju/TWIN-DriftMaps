"""Data layer for MULTIVERSE ("How reliable is a brain map?").

Session list: the IBL Brain-Wide Map 2023-12 release (459 sessions, 139 mice, 12 labs), copied
from int-brain-lab/paper-brain-wide-map into manifests/bwm_2023_12_release.csv. Only the probes
listed there are used (they passed IBL's quality control).

Per session we cache, for every Beryl region, spike counts of every sorted unit in each analysis
window below, a good-unit mask, the trial variables, and lab / subject / split. Raw spike files
are deleted after caching.

Windows (seconds relative to the event). "a" = the window used by the IBL Brain-Wide Map paper
(as far as we know; kill test K1 checks it); "b" = a shifted alternative (multiverse axis).
    stim_a      stimOn         [ 0.00, 0.10]     stim_b      stimOn         [ 0.05, 0.15]
    choice_a    firstMovement  [-0.10, 0.00]     choice_b    firstMovement  [-0.15,-0.05]
    feedback_a  feedback       [ 0.00, 0.20]     feedback_b  feedback       [ 0.05, 0.25]
    prior_a     stimOn         [-0.60,-0.10]     prior_b     stimOn         [-0.40,-0.10]
Trials whose event time is missing get zero counts and valid__<event> = False.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .ibl import EXCLUDE_REGIONS, GOOD_LABEL, _delete_raw_spikes, _trials, region_labels
from .infoflow_data import split_of

MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "bwm_2023_12_release.csv"

WINDOWS = {
    "stim_a": ("stimOn", 0.00, 0.10), "stim_b": ("stimOn", 0.05, 0.15),
    "choice_a": ("firstMovement", -0.10, 0.00), "choice_b": ("firstMovement", -0.15, -0.05),
    "feedback_a": ("feedback", 0.00, 0.20), "feedback_b": ("feedback", 0.05, 0.25),
    "prior_a": ("stimOn", -0.60, -0.10), "prior_b": ("stimOn", -0.40, -0.10),
}
EVENTS = ("stimOn", "firstMovement", "feedback")


def manifest():
    """{eid: dict(subject, lab, probes=[...])} from the frozen BWM release list."""
    import csv
    out = {}
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            d = out.setdefault(row["eid"], dict(subject=row["subject"], lab=row["lab"], probes=[]))
            d["probes"].append(row["probe_name"])
    return out


def counts_events(spike_times, spike_clusters, cluster_ids, edges):
    """edges: {window: (lo_times, hi_times)} -> {window: (T, N) int16}. Sorts the probe's spikes once."""
    order = np.argsort(spike_clusters, kind="stable")
    sc, st = spike_clusters[order], spike_times[order]
    a_b = np.searchsorted(sc, cluster_ids, "left")
    b_b = np.searchsorted(sc, cluster_ids, "right")
    T = len(next(iter(edges.values()))[0])
    out = {w: np.zeros((T, len(cluster_ids)), np.int16) for w in edges}
    for j, (a, b) in enumerate(zip(a_b, b_b)):
        t = st[a:b]
        for w, (lo, hi) in edges.items():
            out[w][:, j] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    return out


def _event_times(one, eid, n):
    tr = one.load_object(eid, "trials")
    on = np.asarray(tr["stimOn_times"], float)
    keep = np.isfinite(on)
    ev = {"stimOn": on[keep]}
    fm = np.asarray(tr.get("firstMovement_times", np.full(len(on), np.nan)), float)[keep]
    fb = np.asarray(tr.get("feedback_times", np.full(len(on), np.nan)), float)[keep]
    ev["firstMovement"], ev["feedback"] = fm, fb
    fbt = np.asarray(tr["feedbackType"], float)[keep]
    assert len(fm) == n, "trial count mismatch"
    return ev, fbt


def load_session_mv(one, eid, info, cache_dir, delete_raw=True):
    cache = Path(cache_dir) / f"{eid}.npz"
    if cache.exists():
        return cache
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    contrast, choice, onsets, meta = _trials(one, eid)
    ev, fbt = _event_times(one, eid, len(onsets))
    valid = {e: np.isfinite(t) for e, t in ev.items()}
    edges = {}
    for w, (e, lo, hi) in WINDOWS.items():
        t = np.where(valid[e], ev[e], 0.0)
        edges[w] = (np.where(valid[e], t + lo, 0.0), np.where(valid[e], t + hi, 0.0))
    arrays, probes_done = {}, []
    for pname in info["probes"]:
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
            allw = counts_events(st, scl, keep_ids, edges)
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
            probes_done.append(pname)
            break
    meta.update(eid=eid, subject=info["subject"], lab=info["lab"], split=split_of(info["subject"]),
                probes=probes_done, windows={w: list(v) for w, v in WINDOWS.items()},
                good_units={k[6:]: int(v.sum()) for k, v in arrays.items() if k.startswith("good__")})
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, contrast=contrast, choice=choice, feedback=fbt,
                        rt=ev["firstMovement"] - ev["stimOn"],
                        **{f"valid__{e}": v for e, v in valid.items()},
                        meta=json.dumps(meta), **arrays)
    if delete_raw:
        _delete_raw_spikes(one, eid)
    return cache


class MVSession:
    """Lazy view of one cached MULTIVERSE session."""

    def __init__(self, path, all_units=False):
        # The file is opened only while reading and closed again: holding 459 sessions open at once
        # exceeds the operating system's open-file limit (macOS: 256).
        self.path = Path(path)
        self.all_units = all_units
        with np.load(path, allow_pickle=False) as z:
            self.meta = json.loads(str(z["meta"]))
            self.contrast = z["contrast"]
            self.choice = z["choice"].astype(int)
            self.feedback = z["feedback"]
            self.rt = z["rt"]
            self.valid = {e: z[f"valid__{e}"] for e in EVENTS}
            self._files = set(z.files)
        self.sid = self.path.stem
        self.regions = sorted({k.split("__", 1)[1] for k in self._files if k.startswith("stim_a__")})
        self.split = self.meta.get("split", "unknown")
        self.subject = self.meta.get("subject", "unknown")
        self.lab = self.meta.get("lab", "unknown")

    def X(self, region, window):
        k = f"{window}__{region}"
        if k not in self._files:
            return None
        with np.load(self.path, allow_pickle=False) as z:
            X = z[k].astype(float)
            if not self.all_units:
                X = X[:, z[f"good__{region}"].astype(bool)]
        return X if X.shape[1] else None


def load_mv_cache(cache_dir, split=None, all_units=False):
    out = [MVSession(p, all_units) for p in sorted(Path(cache_dir).glob("*.npz"))]
    return [s for s in out if split is None or s.split == split]
