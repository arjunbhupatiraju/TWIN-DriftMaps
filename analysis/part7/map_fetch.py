"""STEP 3, data: stream the Druckmann/Svoboda MAP dataset (DANDI:000363) and keep only what the reanalysis needs.

For every session it streams (never downloads whole files; most are 100-700 GB of raw voltage):
  units     spike times, quality, depth on the probe, brain area, Kilosort drift metrics, firing rate
  trials    instruction (lick left/right), outcome, early lick, photostimulation, free/auto water
  events    sample, delay and go cue times, left and right lick times
  tracking  jaw, nose and tongue (DeepLabCut, side camera, about 300 Hz)
and saves, per trial and per epoch, every unit's spike count and 7 movement features, in one small file per session
(results_map/sessions/<name>.npz). Resumable: finished sessions are skipped.

Epochs (fixed): pre = 0.5 s before sample onset; sample = sample onset to offset; delay = last 1.0 s before the go cue;
response = go cue to +0.6 s.

    python3 part7/map_fetch.py --limit 2
    python3 part7/map_fetch.py
    python3 part7/map_fetch.py --part 0 --of 4      (and --part 1/2/3 in three more windows: 4x faster)
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_map" / "sessions"
DANDISET, VERSION = "000363", "0.231012.2129"
EPOCHS = ("pre", "sample", "delay", "response")
MOVE = ("jaw_speed", "jaw_y", "nose_speed", "tongue_visible", "tongue_speed", "licks_left", "licks_right")
LIK = 0.9


def _s(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def _col(h, name):
    return h[f"units/{name}"][:] if f"units/{name}" in h else None


def _first_in(ev, starts, stops):
    """For each trial, the first event time inside [start, stop] (NaN if none)."""
    ev = np.sort(np.asarray(ev, float)); out = np.full(len(starts), np.nan)
    i = np.searchsorted(ev, starts)
    ok = i < len(ev)
    cand = np.where(ok, ev[np.minimum(i, len(ev) - 1)], np.nan)
    good = ok & (cand <= stops)
    out[good] = cand[good]
    return out


def _wmean(t, x, lo, hi):
    """Mean of x (NaN ignored) over frames with lo <= t < hi, per window; NaN if no valid frame."""
    m = np.isfinite(x); xv = np.where(m, x, 0.0)
    cs, cm = np.r_[0, np.cumsum(xv)], np.r_[0, np.cumsum(m)]
    a, b = np.searchsorted(t, lo), np.searchsorted(t, hi)
    n = cm[b] - cm[a]
    return np.where(n > 0, (cs[b] - cs[a]) / np.maximum(n, 1), np.nan)


def _count(ev, lo, hi):
    ev = np.sort(np.asarray(ev, float))
    return (np.searchsorted(ev, hi) - np.searchsorted(ev, lo)).astype(np.float32)


def extract(nwb, h):
    """Everything the reanalysis needs from one open NWB file (pynwb object + raw h5py handle)."""
    T = nwb.trials.to_dataframe()
    starts, stops = T.start_time.values.astype(float), T.stop_time.values.astype(float)
    ev = {k: np.asarray(ts.timestamps[:], float) for k, ts in nwb.acquisition["BehavioralEvents"].time_series.items()}
    samp_on = _first_in(ev["sample_start_times"], starts, stops)
    samp_off = _first_in(ev["sample_stop_times"], starts, stops)
    go = _first_in(ev["go_start_times"], starts, stops)
    win = {"pre": (samp_on - 0.5, samp_on), "sample": (samp_on, samp_off),
           "delay": (go - 1.0, go), "response": (go, go + 0.6)}
    # tracking
    trk = {}
    for name, ts in nwb.acquisition["BehavioralTimeSeries"].time_series.items():
        part = name.split("_")[-1].replace("Tracking", "").lower()          # jaw / nose / tongue
        trk[part] = (np.asarray(ts.timestamps[:], float), np.asarray(ts.data[:], float))
    feats = {}
    for e, (lo, hi) in win.items():
        ok = np.isfinite(lo) & np.isfinite(hi)
        lo_, hi_ = np.where(ok, lo, 0.0), np.where(ok, hi, 0.0)
        f = {}
        for part in ("jaw", "nose", "tongue"):
            if part not in trk:
                continue
            t, d = trk[part]
            vis = d[:, 2] >= LIK
            x, y = np.where(vis, d[:, 0], np.nan), np.where(vis, d[:, 1], np.nan)
            sp = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-4)]
            if part == "tongue":
                f["tongue_visible"] = _wmean(t, vis.astype(float), lo_, hi_)
                f["tongue_speed"] = np.nan_to_num(_wmean(t, sp, lo_, hi_), nan=0.0)   # no tongue in view = 0
            else:
                f[f"{part}_speed"] = _wmean(t, sp, lo_, hi_)
                if part == "jaw":
                    f["jaw_y"] = _wmean(t, y, lo_, hi_)
        f["licks_left"] = _count(ev.get("left_lick_times", []), lo_, hi_)
        f["licks_right"] = _count(ev.get("right_lick_times", []), lo_, hi_)
        for k, v in f.items():
            feats[f"move__{e}__{k}"] = np.where(ok, v, np.nan).astype(np.float32)
    # units (read with h5py so the large waveform columns are never touched)
    st, si = h["units/spike_times"][:], h["units/spike_times_index"][:]
    edges = np.r_[0, si]
    nU = len(si)
    counts = {e: np.zeros((len(T), nU), np.float32) for e in EPOCHS}
    for u in range(nU):
        s = st[edges[u]:edges[u + 1]]
        for e, (lo, hi) in win.items():
            ok = np.isfinite(lo) & np.isfinite(hi)
            c = np.searchsorted(s, np.where(ok, hi, 0)) - np.searchsorted(s, np.where(ok, lo, 0))
            counts[e][:, u] = np.where(ok, c, np.nan)
    # brain area of each unit: the probe target (electrodes table) and the CCF annotation (anno_name)
    E = nwb.electrodes.to_dataframe()
    loc = [json.loads(L).get("brain_regions", "") if str(L).startswith("{") else str(L) for L in E.location.astype(str)]
    eidx = h["units/electrodes"][:]
    if "units/electrodes_index" in h:
        ei = np.r_[0, h["units/electrodes_index"][:]]
        first = np.array([eidx[ei[u]] if ei[u + 1] > ei[u] else -1 for u in range(nU)])
    else:
        first = eidx[:nU]
    target = np.array([loc[int(i)] if 0 <= int(i) < len(loc) else "" for i in first])
    probe = np.array([str(E.group_name.values[int(i)]) if 0 <= int(i) < len(E) else "" for i in first])
    units = {"target": target, "probe": probe}
    for c in ("unit_quality", "anno_name", "classification"):
        v = _col(h, c)
        if v is not None:
            units[c] = np.array([_s(x) for x in v])
    for c in ("unit_posy", "avg_firing_rate", "presence_ratio", "isi_violation", "drift_metric", "max_drift",
              "cumulative_drift", "unit_amp"):
        v = _col(h, c)
        if v is not None:
            units[c] = np.asarray(v, float)
    trials = {"instruction": np.array([_s(x) for x in T.trial_instruction]), "outcome": np.array([_s(x) for x in T.outcome]),
              "early_lick": np.array([_s(x) for x in T.early_lick]), "task": np.array([_s(x) for x in T.task]),
              "photostim": np.array([_s(x) not in ("N/A", "nan", "None", "") for x in T.photostim_onset]),
              "auto_water": T.auto_water.values.astype(float), "free_water": T.free_water.values.astype(float),
              "start_time": starts}
    out = {f"counts__{e}": v for e, v in counts.items()}
    out.update(feats)
    out.update({f"unit__{k}": v for k, v in units.items()})
    out.update({f"trial__{k}": v for k, v in trials.items()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--part", type=int, default=0, help="run several copies in parallel: --part 0 --of 4, --part 1 --of 4, ...")
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    from dandi.dandiapi import DandiAPIClient
    import h5py, pynwb, remfile
    OUT.mkdir(parents=True, exist_ok=True)
    with DandiAPIClient() as c:
        assets = sorted(c.get_dandiset(DANDISET, VERSION).get_assets(), key=lambda x: x.path)[: a.limit or None]
        assets = assets[a.part::a.of]
        jobs = [(x.path, x.get_content_url(follow_redirects=1, strip_query=True)) for x in assets]
    t0 = time.time()
    for i, (path, url) in enumerate(jobs):
        name = Path(path).stem.split("_behavior")[0]
        f = OUT / f"{name}.npz"
        if f.exists():
            print(f"[{i + 1}/{len(jobs)}] {name}: skip", flush=True); continue
        try:
            h = h5py.File(remfile.File(url), "r")
            nwb = pynwb.NWBHDF5IO(file=h, load_namespaces=True).read()
            d = extract(nwb, h)
            tmp = OUT / f"{name}.part.npz"                    # write then rename: an interrupted save is never kept
            np.savez_compressed(tmp, subject=path.split("/")[0], **d)
            tmp.rename(f)
            nU = d["counts__pre"].shape[1]; nT = d["counts__pre"].shape[0]
            msg = f"{nT} trials, {nU} units, targets {sorted(set(d['unit__target']))}"
        except Exception as e:
            msg = f"FAILED ({type(e).__name__}: {e}); run again to retry"
        print(f"[{i + 1}/{len(jobs)}] {name}: {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
