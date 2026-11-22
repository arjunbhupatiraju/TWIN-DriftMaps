"""Load IBL Brain-Wide-Map sessions into trial-level Sessions (module 1-3 input).

UNTESTED against the live server (the build workspace cannot reach IBL); written against the
public ONE / ALF conventions. Every step is wrapped so one bad session cannot stop a run,
and each loaded session is cached to an .npz so analyses never need the network again.

Per trial t (stimulus onset s_t):
  activity of unit u = spike count in [s_t - 0.40 s, s_t - 0.05 s]  (pre-stimulus, quiescent period)
  pupil              = mean pupil diameter in the same window (lightning pose or DLC features)
Choice convention: IBL `choice` is -1/+1 for the two wheel directions and 0 for no-go. The
mapping to "right" is chosen per session as the one that agrees with feedbackType on
non-zero-contrast trials (it should be choice == -1 -> right; the check guards against
convention changes).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import Session

WINDOW = (-0.40, -0.05)
REGION_SETS = {
    "NI": ("NI",),
    "LC": ("LC",),
    "LC+SLC": ("LC", "SLC"),
    "HPF": ("CA1", "CA2", "CA3", "DG", "DG-mo", "DG-po", "DG-sg"),
}
GOOD_LABEL = 1.0


def _try(one, eid, name, coll):
    try:
        return one.load_dataset(eid, name, collection=coll)
    except Exception:
        return None


def ids_to_acronyms(ids):
    from iblatlas.regions import BrainRegions
    global _BR
    try:
        _BR
    except NameError:
        _BR = BrainRegions()
    return np.asarray(_BR.id2acronym(np.asarray(ids, int))).astype(str)


def region_labels(one, eid, coll):
    """(acronyms, quality labels) per cluster, or None. IBL stores regions differently across
    releases, so try: cluster acronyms -> cluster Allen ids -> cluster channels + channel ids."""
    acr = _try(one, eid, "clusters.brainLocationAcronyms_ccf_2017.npy", coll)
    if acr is None:
        ids = _try(one, eid, "clusters.brainLocationIds_ccf_2017.npy", coll)
        if ids is not None:
            acr = ids_to_acronyms(ids)
    if acr is None:
        cch = _try(one, eid, "clusters.channels.npy", coll)
        chid = _try(one, eid, "channels.brainLocationIds_ccf_2017.npy", coll)
        if cch is not None and chid is not None:
            acr = ids_to_acronyms(np.asarray(chid)[np.asarray(cch, int)])
    if acr is None:
        return None
    acr = np.asarray(acr).astype(str)
    label = np.full(len(acr), np.nan)
    m = _try(one, eid, "clusters.metrics.pqt", coll)
    if m is not None and "label" in m:
        label = np.asarray(m["label"], float)
    return acr, label


def _counts(spike_times, spike_clusters, cluster_ids, onsets):
    """(T, N) spike counts per trial window for the given clusters."""
    out = np.zeros((len(onsets), len(cluster_ids)))
    lo, hi = onsets + WINDOW[0], onsets + WINDOW[1]
    for j, c in enumerate(cluster_ids):
        st = spike_times[spike_clusters == c]
        out[:, j] = np.searchsorted(st, hi) - np.searchsorted(st, lo)
    return out


def _trials(one, eid):
    tr = one.load_object(eid, "trials")
    cl = np.nan_to_num(np.asarray(tr["contrastLeft"], float))
    cr = np.nan_to_num(np.asarray(tr["contrastRight"], float))
    contrast = cr - cl
    ch = np.asarray(tr["choice"], float)
    fb = np.asarray(tr["feedbackType"], float)
    onsets = np.asarray(tr["stimOn_times"], float)
    nz = (contrast != 0) & (ch != 0)
    best = None
    for right_val in (-1, 1):
        right = (ch == right_val).astype(int)
        agree = np.mean((right[nz] == (contrast[nz] > 0)) == (fb[nz] == 1))
        if best is None or agree > best[0]:
            best = (agree, right_val)
    choice = np.where(ch == 0, -1, (ch == best[1]).astype(int))
    keep = np.isfinite(onsets)
    meta = dict(choice_mapping_agreement=float(best[0]), right_value=int(best[1]),
                probabilityLeft=np.asarray(tr.get("probabilityLeft", np.full(len(ch), np.nan)), float)[keep].tolist())
    return contrast[keep], choice[keep], onsets[keep], meta


def _pupil(one, eid, onsets):
    """Pre-stimulus pupil diameter per trial, or None."""
    for attempt in ("features", "lightningPose"):
        try:
            if attempt == "features":
                cam = one.load_object(eid, "leftCamera", attribute=["features", "times"])
                feat = cam["features"]
                col = "pupilDiameter_smooth" if "pupilDiameter_smooth" in feat else "pupilDiameter_raw"
                d = np.asarray(feat[col], float)
            else:
                cam = one.load_object(eid, "leftCamera", attribute=["lightningPose", "times"])
                lp = cam["lightningPose"]
                d = np.nanmean([np.abs(np.asarray(lp["pupil_top_r_y"]) - np.asarray(lp["pupil_bottom_r_y"])),
                                np.abs(np.asarray(lp["pupil_right_r_x"]) - np.asarray(lp["pupil_left_r_x"]))], axis=0)
            t = np.asarray(cam["times"], float)
            n = min(len(t), len(d))
            t, d = t[:n], d[:n]
            out = np.full(len(onsets), np.nan)
            lo = np.searchsorted(t, onsets + WINDOW[0])
            hi = np.searchsorted(t, onsets + WINDOW[1])
            for i, (a, b) in enumerate(zip(lo, hi)):
                if b > a:
                    out[i] = np.nanmean(d[a:b])
            if np.isfinite(out).mean() > 0.8:
                out[~np.isfinite(out)] = np.nanmedian(out)
                return out, attempt
        except Exception:
            continue
    return None, None


def load_session(one, eid, cache_dir, min_units=1):
    cache = Path(cache_dir) / f"{eid}.npz"
    if cache.exists():
        return from_cache(cache)
    contrast, choice, onsets, meta = _trials(one, eid)
    signals, unit_info, good = {}, {}, {}
    for ins in one.alyx.rest("insertions", "list", session=eid):
        pname = ins["name"]
        colls = sorted([c for c in one.list_collections(eid) if c.startswith(f"alf/{pname}")], key=len, reverse=True)
        spikes = None
        for coll in colls:
            rl = region_labels(one, eid, coll)
            if rl is None:
                continue
            try:
                spikes = one.load_object(eid, "spikes", collection=coll, attribute=["times", "clusters"])
                acr, label = rl
                break
            except Exception:
                spikes = None
        if spikes is None:
            continue
        for rs, acrs in REGION_SETS.items():
            ids = np.flatnonzero(np.isin(acr, acrs))          # ALL units; quality kept as a mask
            if len(ids) == 0:
                continue
            X = _counts(np.asarray(spikes["times"]), np.asarray(spikes["clusters"]), ids, onsets)
            signals[rs] = np.hstack([signals[rs], X]) if rs in signals else X
            unit_info.setdefault(rs, []).extend([f"{pname}:{i}" for i in ids])
            g = (label[ids] == GOOD_LABEL) | np.isnan(label[ids])
            good[rs] = np.r_[good.get(rs, np.zeros(0, bool)), g]
    pupil, pupil_src = _pupil(one, eid, onsets)
    if pupil is not None:
        signals["pupil"] = pupil[:, None]
    meta.update(units={k: len(v) for k, v in unit_info.items()},
                good_units={k: int(v.sum()) for k, v in good.items()}, pupil_source=pupil_src, eid=eid)
    np.savez_compressed(cache, contrast=contrast, choice=choice, meta=json.dumps(meta),
                        **{f"sig__{k}": v for k, v in signals.items()},
                        **{f"good__{k}": v for k, v in good.items()})
    return from_cache(cache)


def from_cache(path, all_units=False):
    """Default: good units only (primary). all_units=True: every sorted unit (sensitivity)."""
    z = np.load(path, allow_pickle=False)
    signals = {}
    for k in z.files:
        if not k.startswith("sig__"):
            continue
        r = k[5:]
        X = z[k]
        if not all_units and f"good__{r}" in z.files and r != "pupil":
            X = X[:, z[f"good__{r}"].astype(bool)]
        if X.shape[1] > 0:
            signals[r] = X
    meta = json.loads(str(z["meta"]))
    return Session(Path(path).stem, z["contrast"], z["choice"].astype(int), signals, meta=meta)


def load_cached(cache_dir, all_units=False):
    return [from_cache(p, all_units) for p in sorted(Path(cache_dir).glob("*.npz"))]


# ---------------------------------------------------------------------------
# Brain-wide loader: every region (IBL 'Beryl' parcellation) in one session.
# ---------------------------------------------------------------------------
EXCLUDE_REGIONS = {"root", "void"}


def counts_fast(spike_times, spike_clusters, cluster_ids, onsets):
    """(T, N) spike counts per pre-stimulus window, O(n_spikes log n_spikes) for all clusters."""
    order = np.argsort(spike_clusters, kind="stable")      # stable: times stay sorted within cluster
    sc, st = spike_clusters[order], spike_times[order]
    lo_b = np.searchsorted(sc, cluster_ids, "left")
    hi_b = np.searchsorted(sc, cluster_ids, "right")
    lo, hi = onsets + WINDOW[0], onsets + WINDOW[1]
    out = np.zeros((len(onsets), len(cluster_ids)))
    for j, (a, b) in enumerate(zip(lo_b, hi_b)):
        t = st[a:b]
        out[:, j] = np.searchsorted(t, hi) - np.searchsorted(t, lo)
    return out


def _delete_raw_spikes(one, eid):
    """Remove downloaded spike files (large) once a session is cached; keeps disk use small."""
    try:
        sp = Path(one.eid2path(eid))
        for f in sp.rglob("spikes.*"):
            f.unlink()
    except Exception:
        pass


def load_session_brainwide(one, eid, cache_dir, delete_raw=True):
    cache = Path(cache_dir) / f"{eid}.npz"
    if cache.exists():
        return from_cache(cache)
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    contrast, choice, onsets, meta = _trials(one, eid)
    signals, good = {}, {}
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
            for r in np.unique(beryl):
                if r in EXCLUDE_REGIONS:
                    continue
                ids = np.flatnonzero(beryl == r)
                X = counts_fast(st, scl, ids, onsets)
                g = (label[ids] == GOOD_LABEL) | np.isnan(label[ids])
                signals[r] = np.hstack([signals[r], X]) if r in signals else X
                good[r] = np.r_[good.get(r, np.zeros(0, bool)), g]
            break
    pupil, pupil_src = _pupil(one, eid, onsets)
    if pupil is not None:
        signals["pupil"] = pupil[:, None]
    meta.update(units={k: int(v.shape[1]) for k, v in signals.items() if k != "pupil"},
                good_units={k: int(v.sum()) for k, v in good.items()}, pupil_source=pupil_src, eid=eid)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, contrast=contrast, choice=choice, meta=json.dumps(meta),
                        **{f"sig__{k}": v for k, v in signals.items()},
                        **{f"good__{k}": v for k, v in good.items()})
    if delete_raw:
        _delete_raw_spikes(one, eid)
    return from_cache(cache)
