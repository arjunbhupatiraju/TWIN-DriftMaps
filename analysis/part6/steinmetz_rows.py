"""Steinmetz et al. 2019 (Nature; 39 sessions, 10 mice; recorded 2016-2018 before the IBL standardized rig) in the
atlas row format, for the cross-dataset replication test (claude/preregistration-cross-dataset.md).

  python3 part6/steinmetz_rows.py --data ~/Downloads/MULTIVERSE/steinmetz_data

Variables matched to the IBL atlas: stimulus side (0-100 ms after stimulus), choice (100 ms before response),
feedback (0-200 ms after feedback), prev_choice (500 ms before stimulus). Steinmetz has no blocks, so every variable
uses Harris's linear-shift valid test. Codes are planted exactly as in the atlas (new codes, never the real label).
"""
from __future__ import annotations

import argparse, json, sys, urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_atlas"
sys.path.insert(0, str(HERE.parent / "twin"))
URLS = ["https://osf.io/agvxh/download", "https://osf.io/uv3mw/download", "https://osf.io/ehmw2/download"]
BIN, STIM_BIN = 0.01, 50
R_GRID = (0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3)
LAB = "steinmetz2019"


def _bin(t):
    return STIM_BIN + np.round(np.asarray(t, float).ravel() / BIN).astype(int)


def labels(d):
    cl, cr = np.asarray(d["contrast_left"], float), np.asarray(d["contrast_right"], float)
    resp = np.asarray(d["response"], float).ravel(); fb = np.asarray(d["feedback_type"], float).ravel()
    rt = np.asarray(d["response_time"], float).ravel(); ft = np.asarray(d["feedback_time"], float).ravel()
    nb = d["spks"].shape[2]; T = len(resp); go = resp != 0
    rbin, fbin = _bin(rt), _bin(ft)
    pre = (np.full(T, STIM_BIN - 50), np.full(T, STIM_BIN))
    prev = np.r_[np.nan, resp[:-1]]
    return {
        "stimulus": ((cl != cr) & go & (rt > 0.1), np.sign(cr - cl), (np.full(T, STIM_BIN), np.full(T, STIM_BIN + 10))),
        "choice": (go & (rbin - 10 >= STIM_BIN) & (rbin <= nb), (resp > 0).astype(float), (rbin - 10, rbin)),
        "feedback": (fbin + 20 <= nb, (fb > 0).astype(float), (fbin, fbin + 20)),
        "prev_choice": (np.isin(prev, (-1, 1)), (prev > 0).astype(float), pre),
    }


def counts(S, window, mask):
    lo, hi = (np.clip(np.asarray(w, int), 0, S.shape[2]) for w in window)
    tr = np.flatnonzero(mask)
    return np.array([S[:, t, lo[t]:hi[t]].sum(1) for t in tr], float)


def _session(i, d, seed, n_plant, min_units, max_units):
    from twin.stats import pooled_corr
    from twin.check import plant
    from twin.tests import p_linear_shift
    rng = np.random.default_rng(seed)
    areas = np.asarray(d["brain_area"]).astype(str)
    rows = []
    for var, (mask, yfull, win) in labels(d).items():
        mask = mask & np.isfinite(yfull)
        mask[:10] = False                                   # skip the first trials, as burn-in
        T = int(mask.sum())
        y = yfull[mask]
        if T < 150 or np.std(y) == 0:
            continue
        for r in sorted(set(areas) - {"root", ""}):
            sel = areas == r
            if sel.sum() < min_units:
                continue
            X = counts(d["spks"][sel], win, mask); X = X[:, X.std(0) > 0]
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
                curve[str(rr)] = [round(p_linear_shift(plant(X, (yp := np.roll(y, int(rng.integers(T // 4, 3 * T // 4)))),
                                                             rr, 0.2, rng), yp), 4) for _ in range(n_plant)]
            rows.append(dict(session=f"steinmetz_{i:02d}", subject=str(d["mouse_name"]), lab=LAB, region=r, variable=var,
                             T=T, N=N, p_shuffle=np.nan, p_valid=p_valid, p_shift=p_valid, drift_fp_shuffle=np.nan,
                             r_hat=float(np.sqrt(max(excess, 0) / (T * N * 0.2))), power_curve=json.dumps(curve)))
    return rows


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/Downloads/MULTIVERSE/steinmetz_data")
    ap.add_argument("--plant", type=int, default=24); ap.add_argument("--min-units", type=int, default=10)
    ap.add_argument("--max-units", type=int, default=60); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    data = Path(a.data).expanduser(); data.mkdir(parents=True, exist_ok=True)
    for j, url in enumerate([] if a.no_download else URLS):
        f = data / f"steinmetz_part{j}.npz"
        if not (f.exists() and f.stat().st_size > 1e6):
            print(f"downloading {f.name} (a few hundred MB) ...", flush=True); urllib.request.urlretrieve(url, f)
    sessions = []
    for f in sorted(data.glob("steinmetz_part*.npz")):
        sessions.extend(np.load(f, allow_pickle=True)["dat"])
    print(f"{len(sessions)} Steinmetz sessions", flush=True)
    OUT.mkdir(exist_ok=True)
    out = OUT / "steinmetz_rows.csv"
    if out.exists():
        out.unlink()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, i, d, 1000 + i, a.plant, a.min_units, a.max_units): i for i, d in enumerate(sessions)}
        for k, fu in enumerate(as_completed(futs)):
            rows = fu.result()
            if rows:
                pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
            print(f"[{k + 1}/{len(sessions)}] session {futs[fu]}: {len(rows)} region-variables", flush=True)
    print("saved", out)


if __name__ == "__main__":
    main()
