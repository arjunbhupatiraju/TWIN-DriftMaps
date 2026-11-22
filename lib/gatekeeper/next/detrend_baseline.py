"""The obvious reviewer alternative: "just detrend each neuron".

Each neuron (and the label) is regressed on a slow cosine basis of trial index (periods >= CUTOFF trials, the
usual high-pass used in fMRI and calcium imaging), then the pooled statistic S = sum of squared correlations is
computed on the residuals. Two nulls:
    p_detrend_param   chi-square (N degrees of freedom), as a regression F-test would assume
    p_detrend_pseudo  the same pseudo-session labels as every other method
Compared on the same sessions, regions and labels with the new test.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "driftaware" / "src"))
import compare_methods as cm  # noqa: E402
from driftaware import drift_test  # noqa: E402
from gatekeeper.mv_data import MVSession  # noqa: E402

CUTOFF = 100


def cosine_basis(T, cutoff=CUTOFF):
    K = int(np.floor(2 * T / cutoff))
    t = np.arange(T)
    B = [np.ones(T)] + [np.cos(np.pi * k * (2 * t + 1) / (2 * T)) for k in range(1, K + 1)]
    return np.column_stack(B)


def detrend(A, B):
    coef, *_ = np.linalg.lstsq(B, A, rcond=None)
    return A - B @ coef


def detrend_stat(X, Y, B):
    T = X.shape[0]
    Xr = detrend(X, B)
    Yr = detrend(Y, B)
    Xr = Xr / (Xr.std(0) + 1e-12)
    Yr = Yr / (Yr.std(0) + 1e-12)
    dof = T - B.shape[1]
    return dof * ((Xr.T @ Yr / T) ** 2).sum(0)       # approx chi2_N under independent residuals


def session_rows(path, n_null, seed, split):
    s = MVSession(path)
    if split != "all" and s.split != split:
        return []
    v = cm.mask_of(s)
    if v is None or v.sum() < 100:
        return []
    rng = np.random.default_rng(seed)
    rows = []
    for var in cm.VARS:
        Y = cm.label_matrix(s, var, v, n_null, rng)
        if np.std(Y[:, 0]) == 0:
            continue
        B = cosine_basis(len(Y))
        for r in s.regions:
            X = s.X(r, "prior_a")
            if X is None:
                continue
            X = X[v]
            X = X[:, X.std(0) > 0]
            N = X.shape[1]
            if N < cm.MIN_UNITS:
                continue
            S = detrend_stat(X, Y, B)
            rows.append(dict(sid=s.sid, subject=s.subject, var=var, region=r, n_units=N, n_trials=int(v.sum()),
                             p_detrend_param=float(chi2.sf(S[0], N)), p_detrend_pseudo=cm._p(S),
                             p_new=drift_test(X, Y[:, 0], Y[:, 1:]).p))
    return rows


def run(cache_dir, out_csv, split="confirm", n_null=200, n_jobs=-1, limit=0, log=print):
    import pandas as pd
    from joblib import Parallel, delayed
    paths = sorted(Path(cache_dir).expanduser().glob("*.npz"))
    paths = [p for p in paths if split == "all" or MVSession(p).split == split]
    todo = paths[:limit] if limit else paths
    log(f"{len(todo)} sessions ({split})")
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(session_rows)(p, n_null, 4242 + i, split) for i, p in enumerate(todo))
    df = pd.DataFrame([r for rows in res for r in rows])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"done in {(time.time() - t0) / 60:.1f} min: {len(df)} rows -> {out_csv}")
    return df


def report(csv, out_dir):
    import pandas as pd
    from driftaware import region_map
    df = pd.read_csv(csv)
    rows = []
    for var in ("bprior", "ctl_bprior", "prior", "ctl_prior"):
        sub = df[df["var"] == var]
        for m, name in (("p_detrend_param", "detrend + parametric test"), ("p_detrend_pseudo", "detrend + pseudo-session"),
                        ("p_new", "NEW drift-aware (pseudo)")):
            rm = region_map(sub, m)
            rows.append(dict(variable=var, method=name, session_region_rate=round(float((sub[m] < 0.05).mean()), 3),
                             regions_FDR5=int(rm.significant.sum()), regions_tested=len(rm)))
    t = pd.DataFrame(rows)
    text = f"# Detrending baseline ({df.sid.nunique()} sessions)\n\nFakes (ctl_) should be about 0.05 and 0 regions.\n\n" + t.to_string(index=False)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "DETREND_SUMMARY.md").write_text(text)
    print(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="confirm")
    ap.add_argument("--out", default="results/detrend")
    ap.add_argument("--nulls", type=int, default=200)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    csv = Path(a.out) / f"detrend_rows_{a.split}.csv"
    run(a.cache, csv, a.split, a.nulls, a.jobs, a.limit)
    report(csv, a.out)
