"""Allen Institute Visual Behavior Neuropixels (different institution, rig, pipeline, task) in the atlas row format,
for the pooled cross-dataset test (claude/preregistration-pooled-cross-dataset.md).

No AllenSDK needed: files are read straight from the public S3 bucket and the NWB files with h5py. Each session's NWB
(about 3 GB) is downloaded, reduced to per-trial spike counts, and DELETED, so only one session is on disk at a time.

  python3 part6/allen_rows.py probe            # find the newest manifest and print what it contains (quick)
  python3 part6/allen_rows.py run              # all non-abnormal sessions (resumable; about 2 min download each)
  python3 part6/allen_rows.py run --limit 3    # quick check on 3 sessions

Task: change detection. Images flash every 750 ms; on go trials the image changes and a lick in the response window
is rewarded; catch trials have no change. Variables, matched to the IBL atlas names (pre-registered mapping):
  stimulus      image change (go) vs no change (catch), 0-100 ms after the (sham) change time
  choice        lick vs no lick in the response window (go + catch trials), 150-350 ms after the change time
  feedback      reward vs no reward on go trials (hit vs miss), 500-700 ms after the change time
  prev_choice   the previous trial's lick vs no lick, 500 ms before the change time
All use Harris's linear-shift valid test; codes are planted exactly as in the atlas (new codes, never the real label).
"""
from __future__ import annotations

import argparse, io, json, os, re, sys, time, urllib.parse, urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_atlas"
sys.path.insert(0, str(HERE.parent / "twin"))
BUCKET = "https://visual-behavior-neuropixels-data.s3.us-west-2.amazonaws.com"
PREFIX = "visual-behavior-neuropixels"
R_GRID = (0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3)
LAB = "allen_vbn"
WINDOWS = {"stimulus": (0.0, 0.1), "choice": (0.15, 0.35), "feedback": (0.5, 0.7), "prev_choice": (-0.5, 0.0)}


def _get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _list(prefix):
    keys, token = [], None
    while True:
        q = f"{BUCKET}/?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        x = _get(q).decode()
        keys += re.findall(r"<Key>([^<]+)</Key>", x)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            return keys
        token = m.group(1)


def manifest():
    keys = [k for k in _list(f"{PREFIX}/manifests/") if k.endswith(".json")]
    if not keys:
        sys.exit("No manifest found in the bucket listing; paste this output to Claude.")
    def ver(k):
        v = re.findall(r"v(\d+)\.(\d+)\.(\d+)", k)
        return tuple(map(int, v[-1])) if v else (0, 0, 0)
    key = max(keys, key=ver)
    return key, json.loads(_get(f"{BUCKET}/{key}"))


def _url(entry):
    u = entry.get("url") if isinstance(entry, dict) else entry
    if u.startswith("s3://"):
        b, k = u[5:].split("/", 1)
        return f"https://{b}.s3.us-west-2.amazonaws.com/{k}"
    return u


def _find(d, pred, path=()):
    """Walk a nested manifest, yielding (path, value) for entries where pred(path, value)."""
    if isinstance(d, dict):
        if pred(path, d):
            yield path, d
        for k, v in d.items():
            yield from _find(v, pred, path + (str(k),))


def metadata(man, name, cache):
    f = cache / f"{name}.csv"
    if not f.exists():
        hits = [v for p, v in _find(man, lambda p, v: "url" in v and str(v.get("url", "")).endswith(f"{name}.csv"))]
        if not hits:
            sys.exit(f"{name}.csv not found in the manifest")
        f.write_bytes(_get(_url(hits[0]), timeout=600))
    import pandas as pd
    return pd.read_csv(f)


def session_urls(man):
    hits = {}
    for p, v in _find(man, lambda p, v: "url" in v and re.search(r"ecephys_session_(\d+)\.nwb$", str(v.get("url", "")))):
        sid = int(re.search(r"ecephys_session_(\d+)\.nwb$", v["url"]).group(1))
        hits[sid] = _url(v)
    return hits


# ------------------------------------------------------------------ NWB -> per-trial counts
def read_nwb(path, unit_regions):
    """Returns trials dict and {region: (unit ids, list of spike-time arrays)} using h5py only."""
    import h5py
    with h5py.File(path, "r") as h:
        tr = h["intervals/trials"]
        cols = {k: np.asarray(tr[k]) for k in tr.keys() if isinstance(tr[k], h5py.Dataset) and tr[k].ndim == 1}
        ids = np.asarray(h["units/id"])
        st = np.asarray(h["units/spike_times"]); ix = np.asarray(h["units/spike_times_index"])
    starts = np.r_[0, ix[:-1]]
    regions = {}
    for j, uid in enumerate(ids):
        r = unit_regions.get(int(uid))
        if r is None:
            continue
        regions.setdefault(r, []).append(st[starts[j]:ix[j]])
    return cols, regions


def trial_labels(c):
    """Pre-registered variable definitions from the NWB trials table."""
    n = len(c["start_time"])
    go = c.get("go", c.get("is_change", np.zeros(n))).astype(bool)
    catch = c.get("catch", np.zeros(n)).astype(bool)
    aborted = c.get("aborted", np.zeros(n)).astype(bool)
    keep = (go | catch) & ~aborted
    tchange = c.get("change_time_no_display_delay", c.get("change_time", np.full(n, np.nan))).astype(float)
    hit = c.get("hit", np.zeros(n)).astype(bool); fa = c.get("false_alarm", np.zeros(n)).astype(bool)
    lick = hit | fa
    idx = np.flatnonzero(keep & np.isfinite(tchange))
    lab = {
        "stimulus": (idx, go[idx].astype(float)),
        "choice": (idx, lick[idx].astype(float)),
        "feedback": (idx[go[idx]], hit[idx[go[idx]]].astype(float)),
        "prev_choice": (idx[1:], lick[idx[:-1]].astype(float)),
    }
    return tchange, lab


def counts(spikes, times, lo, hi):
    out = np.empty((len(times), len(spikes)))
    for j, s in enumerate(spikes):
        out[:, j] = np.searchsorted(s, times + hi) - np.searchsorted(s, times + lo)
    return out


def session_rows(sid, path, unit_regions, rng, n_plant, min_units, max_units, burn=10):
    from twin.stats import pooled_corr
    from twin.check import plant
    from twin.tests import p_linear_shift
    cols, regions = read_nwb(path, unit_regions)
    tchange, labs = trial_labels(cols)
    rows = []
    for var, (idx, y) in labs.items():
        idx, y = idx[burn:], y[burn:]
        T = len(y)
        if T < 150 or np.std(y) == 0:
            continue
        lo, hi = WINDOWS[var]
        for r, spikes in sorted(regions.items()):
            if len(spikes) < min_units:
                continue
            X = counts(spikes, tchange[idx], lo, hi); X = X[:, X.std(0) > 0]
            if X.shape[1] < min_units:
                continue
            if X.shape[1] > max_units:
                X = X[:, rng.choice(X.shape[1], max_units, replace=False)]
            N = X.shape[1]
            p_valid = p_linear_shift(X, y)
            S = max(10, T // 10); L = T - 2 * S
            ws = pooled_corr(X[S:S + L], np.column_stack([y[S + k:S + k + L] for k in range(-S, S + 1)]))
            excess = float((ws[S] - np.mean(np.delete(ws, S))) * T / L)
            curve = {}
            for rr in R_GRID:
                ps = []
                for _ in range(n_plant):
                    yp = np.roll(y, int(rng.integers(T // 4, 3 * T // 4)))
                    ps.append(round(p_linear_shift(plant(X, yp, rr, 0.2, rng), yp), 4))
                curve[str(rr)] = ps
            rows.append(dict(session=f"allen_{sid}", subject=str(unit_regions.get("_mouse", "")), lab=LAB, region=r,
                             variable=var, T=T, N=N, p_shuffle=np.nan, p_valid=p_valid, p_shift=p_valid,
                             drift_fp_shuffle=np.nan, r_hat=float(np.sqrt(max(excess, 0) / (T * N * 0.2))),
                             power_curve=json.dumps(curve)))
    return rows


def cmd_probe(a):
    key, man = manifest()
    print("manifest:", key)
    print("top-level keys:", list(man.keys())[:20])
    s = session_urls(man)
    print(f"session NWB files found: {len(s)}; example: {next(iter(s.items())) if s else None}")
    meta = [v.get("url") for p, v in _find(man, lambda p, v: "url" in v and str(v.get('url', '')).endswith('.csv'))]
    print("metadata CSVs:", [m.rsplit('/', 1)[-1] for m in meta][:20])


def cmd_run(a):
    import pandas as pd
    cache = Path(a.cache).expanduser(); cache.mkdir(parents=True, exist_ok=True)
    key, man = manifest()
    print("manifest:", key, flush=True)
    S = metadata(man, "ecephys_sessions", cache)
    U = metadata(man, "units", cache)
    sid_col = "ecephys_session_id" if "ecephys_session_id" in S.columns else S.columns[0]
    bad = S.get("abnormal_histology", pd.Series(False, index=S.index)).fillna(False).astype(bool) | \
          S.get("abnormal_activity", pd.Series(False, index=S.index)).fillna(False).astype(bool)
    S = S[~bad]
    urls = session_urls(man)
    good_q = (U.get("quality", "good") == "good") if "quality" in U.columns else np.ones(len(U), bool)
    U = U[good_q & U.structure_acronym.notna()]
    uid_col = "unit_id" if "unit_id" in U.columns else U.columns[0]
    OUT.mkdir(exist_ok=True)
    outf = OUT / "allen_rows.csv"
    done = set(pd.read_csv(outf).session) if outf.exists() else set()
    todo = [int(s) for s in S[sid_col] if f"allen_{int(s)}" not in done and int(s) in urls]
    if a.limit:
        todo = todo[: a.limit]
    print(f"{len(S)} usable sessions, {len(done)} done, {len(todo)} to run", flush=True)
    rng = np.random.default_rng(7)
    for k, sid in enumerate(todo):
        t0 = time.time()
        nwb = cache / f"ecephys_session_{sid}.nwb"
        try:
            if not nwb.exists():
                tmp = nwb.with_suffix(".part")
                urllib.request.urlretrieve(urls[sid], tmp); tmp.rename(nwb)
            uu = U[U.ecephys_session_id == sid]
            ureg = dict(zip(uu[uid_col].astype(int), uu.structure_acronym.astype(str)))
            mrow = S[S[sid_col] == sid]
            ureg["_mouse"] = str(mrow.mouse_id.iloc[0]) if "mouse_id" in mrow.columns else str(sid)
            rows = session_rows(sid, nwb, ureg, rng, a.plant, a.min_units, a.max_units)
            if rows:
                pd.DataFrame(rows).to_csv(outf, mode="a", header=not outf.exists(), index=False)
            print(f"[{k + 1}/{len(todo)}] session {sid}: {len(rows)} region-variables ({time.time() - t0:.0f}s)", flush=True)
        except Exception as e:
            print(f"[{k + 1}/{len(todo)}] session {sid}: ERROR {e}", flush=True)
        finally:
            if nwb.exists() and not a.keep:
                nwb.unlink()                                   # keep disk use to one session at a time


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    r = sub.add_parser("run"); r.add_argument("--cache", default="~/Downloads/ALLEN_VBN")
    r.add_argument("--limit", type=int, default=0); r.add_argument("--plant", type=int, default=24)
    r.add_argument("--min-units", type=int, default=10); r.add_argument("--max-units", type=int, default=60)
    r.add_argument("--keep", action="store_true", help="keep the NWB files (about 3 GB each)")
    a = ap.parse_args()
    dict(probe=cmd_probe, run=cmd_run)[a.cmd](a)


if __name__ == "__main__":
    main()
