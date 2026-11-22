"""Second dataset: Steinmetz et al. 2019 (Nature), 39 Neuropixels sessions, 10 mice, visual discrimination.

Uses the Neuromatch Academy release of the data (3 npz files, 10 ms bins, stimulus onset at 0.5 s).
Same logic as the IBL analysis, with variables this task offers:

    stim         right vs left stimulus (higher contrast side)  window  0 to 100 ms after stimulus   (randomised trial by trial: a control)
    choice       right vs left wheel turn                       window  100 ms before the response
    feedback     reward vs no reward                            window  0 to 200 ms after feedback
    prev_choice  previous trial's choice                        window  500 ms before stimulus
    reward_rate  recent reward rate (exponential, alpha 0.1)    window  500 ms before stimulus      (slow, like the IBL prior)
    ctl_<var>    fake version: the same variable taken from a session of a DIFFERENT mouse (does not exist here)

Per session x region x variable, three procedures:
    p_naive   standard statistic with a trial-shuffle null       (the common procedure)
    p_plain   standard statistic with an imposter null          (the calibrated safeguard)
    p_new     the drift-aware test (driftaware) with the imposter null

Imposter labels come only from sessions of other mice.
"""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "driftaware" / "src"))
from driftaware import drift_test, imposter_labels, shuffle_labels  # noqa: E402

URLS = ["https://osf.io/agvxh/download", "https://osf.io/uv3mw/download", "https://osf.io/ehmw2/download"]
BIN = 0.01
STIM_BIN = 50
VARS = ("stim", "choice", "feedback", "prev_choice", "reward_rate")
ALL_VARS = VARS + tuple(f"ctl_{v}" for v in VARS)
MIN_UNITS = 5
MIN_PER_CLASS = 20
N_NULL = 200
SKIP_AREAS = {"root", ""}


def download(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for j, url in enumerate(URLS):
        f = data_dir / f"steinmetz_part{j}.npz"
        if f.exists() and f.stat().st_size > 1e6:
            continue
        print(f"downloading {f.name} (a few hundred MB) ...", flush=True)
        urllib.request.urlretrieve(url, f)
    return sorted(data_dir.glob("steinmetz_part*.npz"))


def load_sessions(data_dir):
    out = []
    for f in sorted(Path(data_dir).glob("steinmetz_part*.npz")):
        out.extend(np.load(f, allow_pickle=True)["dat"])
    return out


def _bin(t):
    return STIM_BIN + np.round(np.asarray(t, float).ravel() / BIN).astype(int)


def labels(d):
    """{var: (trial mask, full-length label, window)} for one session. Windows are (start_bin, end_bin) per trial."""
    cl, cr = np.asarray(d["contrast_left"], float), np.asarray(d["contrast_right"], float)
    resp = np.asarray(d["response"], float).ravel()
    fb = np.asarray(d["feedback_type"], float).ravel()
    rt = np.asarray(d["response_time"], float).ravel()
    ft = np.asarray(d["feedback_time"], float).ravel()
    nb = d["spks"].shape[2]
    T = len(resp)
    go = resp != 0
    rbin, fbin = _bin(rt), _bin(ft)
    pre = (np.zeros(T, int), np.full(T, STIM_BIN))
    out = {}
    out["stim"] = ((cl != cr) & go & (rt > 0.1), (cr > cl).astype(float), (np.full(T, STIM_BIN), np.full(T, STIM_BIN + 10)))
    out["choice"] = (go & (rbin - 10 >= STIM_BIN) & (rbin <= nb), (resp > 0).astype(float), (rbin - 10, rbin))
    out["feedback"] = (fbin + 20 <= nb, (fb > 0).astype(float), (fbin, fbin + 20))
    prev = np.r_[np.nan, resp[:-1]]
    out["prev_choice"] = (np.isin(prev, (-1, 1)), (prev > 0).astype(float), pre)
    rr = np.empty(T)
    cur = 0.5
    for t in range(T):
        rr[t] = cur
        cur += 0.1 * ((fb[t] > 0) - cur)
    m = np.ones(T, bool)
    m[:10] = False
    out["reward_rate"] = (m, rr, pre)
    return out


def counts(S, window, mask):
    """S (neurons, trials, bins) -> (trials in mask, neurons) spike counts summed over each trial's window."""
    lo, hi = (np.clip(np.asarray(w, int), 0, S.shape[2]) for w in window)
    trials = np.flatnonzero(mask)
    C = np.empty((len(trials), S.shape[0]))
    for j, t in enumerate(trials):
        C[j] = S[:, t, lo[t]:hi[t]].sum(1)
    return C


def _binary_ok(y):
    return min((y == 0).sum(), (y == 1).sum()) >= MIN_PER_CLASS


def session_rows(i, d, pools, seed):
    rng = np.random.default_rng(seed)
    mouse = str(d["mouse_name"])
    areas = np.asarray(d["brain_area"]).astype(str)
    labs = labels(d)
    rows = []
    for var in ALL_VARS:
        base = var[4:] if var.startswith("ctl_") else var
        mask, yfull, win = labs[base]
        T = int(mask.sum())
        other = [p for (m, p) in pools[base] if m != mouse]
        draw = imposter_labels(other, T)
        y = draw(rng) if var.startswith("ctl_") else yfull[mask]
        cont = base == "reward_rate"
        if T < 2 * MIN_PER_CLASS or (not cont and not _binary_ok(y)) or np.std(y) == 0:
            continue
        for r in sorted(set(areas) - SKIP_AREAS):
            sel = areas == r
            if sel.sum() < MIN_UNITS:
                continue
            X = counts(d["spks"][sel], win, mask)
            X = X[:, X.std(0) > 0]
            if X.shape[1] < MIN_UNITS:
                continue
            s = int(rng.integers(1 << 31))
            rows.append(dict(
                session=i, mouse=mouse, var=var, region=r, n_units=X.shape[1], n_trials=T,
                p_naive=drift_test(X, y, shuffle_labels(y), N_NULL, seed=s, method="plain").p,
                p_plain=drift_test(X, y, draw, N_NULL, seed=s, method="plain").p,
                p_new=drift_test(X, y, draw, N_NULL, seed=s, method="drift").p))
    return rows


def build_pools(sessions):
    pools = {v: [] for v in VARS}
    for d in sessions:
        labs = labels(d)
        for v in VARS:
            m, y, _ = labs[v]
            pools[v].append((str(d["mouse_name"]), y[m]))
    return pools


def run(data_dir, out_csv, n_jobs=4, limit=0, log=print):
    import pandas as pd
    from joblib import Parallel, delayed
    sessions = load_sessions(data_dir)
    log(f"{len(sessions)} sessions, {len({str(d['mouse_name']) for d in sessions})} mice")
    rt = np.concatenate([np.asarray(d["response_time"], float).ravel() for d in sessions])
    log(f"sanity: median response time {np.median(rt):.2f} s after stimulus (expect roughly 0.3 to 1.5)")
    pools = build_pools(sessions)
    todo = list(enumerate(sessions))[: limit or None]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(session_rows)(i, d, pools, 9000 + i) for i, d in todo)
    df = pd.DataFrame([r for rows in res for r in rows])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"done in {(time.time() - t0) / 60:.1f} min: {len(df)} rows -> {out_csv}")
    return df
