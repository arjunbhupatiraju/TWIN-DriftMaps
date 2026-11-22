"""PART 3 replication on INDEPENDENT data: Steinmetz et al. 2019 (39 sessions, 10 mice, different lab, task and rig).

Predictions fixed BEFORE looking at these data (from the IBL results of Oct 23-24):
  P1 behaviour: with the mouse's drifting bias modelled, plain "repeat my choice" weights at lags >= 2 are about 0,
     while "repeat the REWARDED choice" weights are positive for lags 1-3.
  P2 neural: unsigned outcome memory (was I rewarded) is widespread one trial back and still present two back.
  P3 neural: memory of WHICH choice was rewarded is absent beyond one trial (lags 2-4: ~0 regions), while the same
     memory planted at the behaviour-implied strength (and at half and a quarter of it) is detected widely.
Same methods as IBL: drift-aware lag tests (shared slow component + per-neuron AR whitening), imposter null from
other mice's real behaviour, pre-stimulus window (-500 to -100 ms), confounds removed symmetrically (face motion,
wheel, licks, pupil in the same window). No block structure here, so no prior term.

    python3 part3/steinmetz_replication.py --data ~/Downloads/MULTIVERSE/steinmetz_data --limit 3
    python3 part3/steinmetz_replication.py --data ~/Downloads/MULTIVERSE/steinmetz_data
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "next", ROOT / "driftaware" / "src", HERE):
    sys.path.insert(0, str(p))
import history_joint as hj  # noqa: E402
from history_map import lags  # noqa: E402

PRE = (0, 40)                 # bins (10 ms) from trial start; stimulus at bin 50  ->  -500 to -100 ms
WIN_RATIO = np.array([0.472, 0.371, 0.245, 0.259, 0.092, 0.170]) / 0.472    # IBL behaviour, fixed in advance
N_NULL = 100
MIN_IMPOSTERS = 20
SKIP = {"root", ""}


def load_sessions(data_dir):
    out = []
    for f in sorted(Path(data_dir).expanduser().glob("steinmetz_part*.npz")):
        out.extend(np.load(f, allow_pickle=True)["dat"])
    return out


def task(d):
    cl = np.asarray(d["contrast_left"], float).ravel(); cr = np.asarray(d["contrast_right"], float).ravel()
    resp = np.asarray(d["response"], float).ravel()
    fb = np.asarray(d["feedback_type"], float).ravel()
    contrast = cr - cl
    choice = np.where(resp > 0, 1, np.where(resp < 0, 0, -1))                   # 1 / 0 / -1 (no-go)
    # which response is "right"? pick the mapping that agrees with feedback on unambiguous trials
    nz = (contrast != 0) & (choice >= 0)
    if nz.sum() > 20 and np.mean((choice[nz] == 1) == (contrast[nz] > 0)) < np.mean((choice[nz] == 0) == (contrast[nz] > 0)):
        choice = np.where(choice >= 0, 1 - choice, -1)
    side = np.where(contrast > 0, 1.0, np.where(contrast < 0, 0.0, np.nan))
    eq = np.isnan(side) & (choice >= 0) & (fb != 0)
    side[eq] = np.where(fb[eq] > 0, choice[eq], 1 - choice[eq])                 # rewarded side when contrasts equal
    return contrast, choice, fb, side


def seq(d):
    contrast, choice, fb, side = task(d)
    return hj.seq_matrix(side, choice, fb, np.full(len(choice), 0.5))


def confounds(d, T):
    cols = []
    for key, f in (("face", lambda a: a.mean(-1)), ("wheel", lambda a: np.abs(a).mean(-1)),
                   ("licks", lambda a: a.sum(-1)), ("pupil", lambda a: a.mean(-1))):
        if key not in d:
            continue
        a = np.asarray(d[key], float)
        if a.ndim == 3:
            a = a[0]
        if a.shape[0] != T:
            continue
        x = f(a[:, PRE[0]:PRE[1]])
        if np.isfinite(x).mean() > 0.8 and np.nanstd(x) > 0:
            cols.append(np.where(np.isfinite(x), x, np.nanmedian(x)))
    return np.column_stack(cols) if cols else None


# ------------------------------------------------------------------------------------------- behaviour (P1)
def behaviour(sessions):
    import pandas as pd
    import behaviour_rw as brw
    rows = []
    for i, d in enumerate(sessions):
        contrast, choice, fb, side = task(d)
        v = choice >= 0
        v[:10] = False
        if v.sum() < 100:
            continue
        try:
            rows.append(dict(session=i, mouse=str(d["mouse_name"]), **brw.session_fit(contrast, choice, fb, v)))
        except Exception as e:
            print("behaviour FAILED", i, e)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------- neural (P2, P3)
def session_rows(i, d, bank, seed=0):
    mouse = str(d["mouse_name"])
    contrast, choice, fb, side = task(d)
    T0 = len(choice)
    lens = sorted([len(M) for m, M in bank if m != mouse], reverse=True)
    if len(lens) < MIN_IMPOSTERS:
        return []
    T = min(T0, lens[MIN_IMPOSTERS - 1])                                         # keep >= 20 imposters
    rng = np.random.default_rng(seed + zlib.crc32(f"{mouse}{i}".encode()) % 100000)
    pool = [M[:T] for m, M in bank if m != mouse and len(M) >= T]
    idx = rng.choice(len(pool), size=min(N_NULL, len(pool)), replace=False)
    owns = np.stack([seq(d)[:T]] + [pool[j] for j in idx])
    v = choice[:T] >= 0
    v[:10] = False
    H = confounds(d, T0)
    H = H[:T] if H is not None else None
    c = np.where(choice >= 0, choice - 0.5, 0.0)[:T]
    rc = c * 2 * (fb[:T] > 0)
    Lc, Lr = lags(c, hj.K), lags(rc, hj.K)
    areas = np.asarray(d["brain_area"]).astype(str)
    S = d["spks"]
    rows = []
    for r in sorted(set(areas) - SKIP):
        sel = areas == r
        if sel.sum() < hj.MIN_UNITS:
            continue
        X = S[sel][:, :T, PRE[0]:PRE[1]].sum(-1).T.astype(float)                 # (T, N)
        keep = X[v].std(0) > 0
        if keep.sum() < hj.MIN_UNITS:
            continue
        X = X[:, keep]
        row = dict(sid=f"s{i}", subject=mouse, region=r, n_units=int(keep.sum()), n_trials=int(v.sum()), n_null=len(idx))
        for name, k, pp, ee in hj.joint_tests(X, owns, v, H=H, drop_prior=True):
            row[f"p_{name}{k}"] = pp
        # planted rewarded-choice memory, neural strength anchored to each neuron's lag-1 choice weight
        Xv = X[v]; sd = Xv.std(0); Z = (Xv - Xv.mean(0)) / sd
        L1 = Lc[v, 0] - Lc[v, 0].mean()
        beta = (L1 @ (Z - Z.mean(0))) / max(L1 @ L1, 1e-9)
        for tag, scale in (("full", 1.0), ("half", 0.5), ("quarter", 0.25)):
            add = np.zeros_like(X)
            for k in range(2, hj.K + 1):
                add += scale * WIN_RATIO[k - 1] * np.outer(Lr[:, k - 1] - Lr[v, k - 1].mean(), beta * sd)
            for name, k, pp, ee in hj.joint_tests(X + add, owns, v, H=H, drop_prior=True, only=("rewarded",)):
                row[f"{tag}_p_rewarded{k}"] = pp
        rows.append(row)
    return rows


def report(beh, df, out_dir):
    from scipy.stats import wilcoxon
    from verify_numbers import region_p, sig
    K = hj.K
    L = ["# Replication on Steinmetz et al. 2019 (independent lab, task and mice)\n",
         "Predictions fixed in advance from the IBL results: P1 behaviour (drift modelled): plain choice weights at lags >= 2 "
         "about 0; rewarded-choice weights > 0 at lags 1-3. P2 neural: outcome memory widespread at lag 1, present at lag 2. "
         "P3 neural: rewarded-choice memory absent at lags 2-4 while planted memory is detected.\n"]
    if beh is not None and len(beh):
        L += [f"## P1 Behaviour ({len(beh)} sessions, {beh.mouse.nunique()} mice), drifting bias modelled\n",
              "| Lag | Plain choice (fixed bias) | Plain choice (drift modelled) | p | Rewarded choice (drift modelled) | Fraction > 0 | p |",
              "|---|---|---|---|---|---|---|"]
        for k in range(1, K + 1):
            a = beh[f"rw_choice{k}"]; w = beh[f"rw_win{k}"]
            L.append(f"| {k} | {beh[f'fixed_choice{k}'].median():+.3f} | {a.median():+.3f} | {wilcoxon(a).pvalue:.1e} | "
                     f"{w.median():+.3f} | {(w > 0).mean():.2f} | {wilcoxon(w).pvalue:.1e} |")
    if df is not None and len(df):
        L += [f"\n## P2/P3 Neural ({df.sid.nunique()} sessions, {df.subject.nunique()} mice, {df.region.nunique()} regions)\n",
              "Pre-stimulus window, confounds removed, imposter null. Regions significant at FDR 5% (>= 2 sessions).\n",
              "| Lag | Whole trial | Outcome only | Choice only | Rewarded choice: REAL | Planted full | Planted half | Planted quarter | Real session rate |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k in range(1, K + 1):
            cells = []
            for col in (f"p_joint{k}", f"p_feedback{k}", f"p_choice{k}", f"p_rewarded{k}",
                        f"full_p_rewarded{k}", f"half_p_rewarded{k}", f"quarter_p_rewarded{k}"):
                pr = region_p(df, col); cells.append(f"{len(sig(pr))} / {len(pr)}")
            L.append(f"| {k} | " + " | ".join(cells) + f" | {(df[f'p_rewarded{k}'] < .05).mean():.3f} |")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "STEINMETZ_REPLICATION.md").write_text(text + "\n")
    print(text)


def main(data_dir, out_dir, limit, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    sessions = load_sessions(data_dir)
    print(f"{len(sessions)} sessions, {len({str(d['mouse_name']) for d in sessions})} mice", flush=True)
    keys = set(sessions[0].keys()) if hasattr(sessions[0], "keys") else set()
    print("confound signals available:", sorted(keys & {"face", "wheel", "licks", "pupil"}), flush=True)
    beh = behaviour(sessions)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    beh.to_csv(Path(out_dir) / "behaviour.csv", index=False)
    print("behaviour done", flush=True)
    bank = [(str(d["mouse_name"]), seq(d)) for d in sessions]
    out = Path(out_dir) / "sessions"; out.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(len(sessions))[: limit or None] if not (out / f"s{i}.csv").exists()]
    t0 = time.time()

    def work(i):
        try:
            rows = session_rows(i, sessions[i], bank)
        except Exception as e:
            print(f"FAILED s{i}: {type(e).__name__}: {e}", flush=True); rows = []
        pd.DataFrame(rows).to_csv(out / f"s{i}.csv", index=False)
        return len(rows)

    for j, n in enumerate(Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(i) for i in todo)):
        print(f"[{j + 1}/{len(todo)}] {n} regions  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fs = [f for f in out.glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None
    report(beh, df, out_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/Downloads/MULTIVERSE/steinmetz_data")
    ap.add_argument("--out", default="results/steinmetz_replication")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    main(a.data, a.out, a.limit, a.jobs)
