"""PART 3 step 1c: is the brain-wide memory of past outcomes real memory, or just what the mouse is DOING now?

Step 1b found that almost every region carries information about the last trial, about 40% about two trials
back, and a few midbrain/hindbrain regions three trials back, mostly about past FEEDBACK (reward or not).
Before calling that memory, rule out the obvious confounds in the pre-stimulus window of the current trial:
   licking (reward consumption), wheel movement, face motion, pupil (arousal), and time since the last
   feedback (error trials have a longer timeout, so lag-1 feedback and elapsed time are almost the same thing).
These are measured per trial in the same window (-600 to -100 ms before stimulus) and removed from the
neurons and from every label (real and imposter) in the whitened space, so the null stays exact.

Two variants per region and session:
   A  "kernel"   : other lags controlled, Bayes prior NOT controlled (full history kernel, incl. choice memory
                   that the prior would otherwise absorb)
   B  "strict"   : other lags + Bayes prior + all confounds controlled

    python3 part3/history_confounds.py fetch --limit 3    (needs internet; small downloads; resumable)
    python3 part3/history_confounds.py fetch
    python3 part3/history_confounds.py run --limit 3
    python3 part3/history_confounds.py run
    python3 part3/history_confounds.py report
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

AUX = "data/part3_aux2"
WIN = (-0.60, -0.10)
CONF_NAMES = ("licks", "wheel_speed", "pupil", "face_motion", "time_since_feedback")


# ------------------------------------------------------------------------------------------- fetch
def _count(times, onsets, win=WIN):
    t = np.sort(np.asarray(times, float))
    return (np.searchsorted(t, onsets + win[1]) - np.searchsorted(t, onsets + win[0])).astype(float)


def _wmean(t, d, onsets, win=WIN):
    n = min(len(t), len(d)); t = np.asarray(t[:n], float); d = np.asarray(d[:n], float)
    out = np.full(len(onsets), np.nan)
    lo = np.searchsorted(t, onsets + win[0]); hi = np.searchsorted(t, onsets + win[1])
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = np.nanmean(d[a:b])
    return out


def fetch_one(one, path, aux_dir):
    from gatekeeper.mv_data import MVSession
    s = MVSession(path)
    out = Path(aux_dir) / f"{s.sid}.npz"
    if out.exists():
        return "skip"
    tr = one.load_object(s.sid, "trials")
    on_all = np.asarray(tr["stimOn_times"], float)
    keep = np.isfinite(on_all)
    on = on_all[keep]
    if len(on) != len(s.contrast):
        np.savez(out, ok=False)
        return "trial count mismatch"
    fbt = np.asarray(tr["feedback_times"], float)[keep]
    since = on - np.r_[np.nan, fbt[:-1]]
    got = {"time_since_feedback": since}
    try:
        lk = one.load_dataset(s.sid, "licks.times.npy", collection="alf")
        got["licks"] = _count(lk, on)
    except Exception:
        pass
    try:
        wh = one.load_object(s.sid, "wheel", collection="alf")
        pos, ts = np.asarray(wh["position"], float), np.asarray(wh["timestamps"], float)
        sp = np.r_[0, np.abs(np.diff(pos)) / np.maximum(np.diff(ts), 1e-4)]
        got["wheel_speed"] = _wmean(ts, sp, on)
    except Exception:
        pass
    for name, ds in (("pupil", "features"), ("face_motion", "ROIMotionEnergy")):
        try:
            if ds == "features":
                cam = one.load_object(s.sid, "leftCamera", attribute=["features", "times"])
                f = cam["features"]
                d = np.asarray(f["pupilDiameter_smooth" if "pupilDiameter_smooth" in f else "pupilDiameter_raw"], float)
                t = cam["times"]
            else:
                d = one.load_dataset(s.sid, "leftCamera.ROIMotionEnergy.npy")
                t = one.load_dataset(s.sid, "leftCamera.times.npy")
            x = _wmean(t, d, on)
            if np.isfinite(x).mean() > 0.8:
                got[name] = x
        except Exception:
            pass
    np.savez(out, ok=True, **got)
    return " ".join(f"{k}={'yes' if k in got else 'no'}" for k in CONF_NAMES)


def fetch(cache, aux_dir, limit):
    from one.api import ONE
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab",
              password="international", silent=True)
    Path(aux_dir).mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[: limit or None]
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            msg = fetch_one(one, p, aux_dir)
        except Exception as e:
            msg = f"FAILED ({type(e).__name__}: {e})"
        print(f"[{i + 1}/{len(paths)}] {p.stem}  {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def confounds(aux_dir, sid, T):
    f = Path(aux_dir) / f"{sid}.npz"
    if not f.exists():
        return None, []
    z = np.load(f)
    if not bool(z["ok"]):
        return None, []
    cols, names = [], []
    for k in CONF_NAMES:
        if k in z.files and len(z[k]) == T:
            x = np.asarray(z[k], float)
            x = np.where(np.isfinite(x), x, np.nanmedian(x))
            if k in ("licks", "wheel_speed", "face_motion"):
                x = np.log1p(np.maximum(x, 0))
            if k == "time_since_feedback":
                x = np.log(np.clip(x, 0.05, 60))
            cols.append(x); names.append(k)
    return (np.column_stack(cols) if cols else None), names


# ------------------------------------------------------------------------------------------- run
def session_rows(path, bank, aux_dir, n_null=hj.N_NULL, seed=0):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    s = MVSession(path)
    v = cm.mask_of(s)
    if v is None or v.sum() < 200:
        return []
    Tf = len(s.contrast)
    H, names = confounds(aux_dir, s.sid, Tf)
    if H is None:
        return []
    rng = np.random.default_rng(seed + zlib.crc32(s.sid.encode()) % 100000)
    pool = [b["M"][:Tf] for sid, b in bank.items() if b["subject"] != s.subject and b["T"] >= Tf]
    if len(pool) < 30:
        return []
    idx = rng.choice(len(pool), size=min(n_null, len(pool)), replace=False)
    owns = np.stack([bank[s.sid]["M"]] + [pool[j] for j in idx])
    rows = []
    for r in s.regions:
        X = s.X(r, hj.WINDOW)
        if X is None:
            continue
        keep = X[v].std(0) > 0
        if keep.sum() < hj.MIN_UNITS:
            continue
        row = dict(sid=s.sid, subject=s.subject, split=s.split, region=r, n_units=int(keep.sum()),
                   confounds="+".join(names))
        for tag, kw in (("a", dict(drop_prior=True)), ("b", dict(H=H))):
            for name, k, pp, ee in hj.joint_tests(X[:, keep], owns, v, **kw):
                row[f"{tag}_p_{name}{k}"] = pp
                row[f"{tag}_eff_{name}{k}"] = ee
        rows.append(row)
    return rows


def run(cache, aux_dir, out_dir, limit, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / "sessions"
    out.mkdir(parents=True, exist_ok=True)
    print("loading behaviour of all sessions (imposter pool) ...", flush=True)
    bank = hj.behaviour_bank(cache)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(paths) - len(todo)} already done, {len(todo)} to run", flush=True)
    t0 = time.time()

    def work(p):
        try:
            rows = session_rows(p, bank, aux_dir)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True)
            rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    for i, n in enumerate(Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)):
        print(f"[{i + 1}/{len(todo)}] {n} regions  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def report(out_dir, base_dir="results/history_joint"):
    import pandas as pd
    from verify_numbers import region_p, sig
    import slowmode as sm
    files = [f for f in (Path(out_dir) / "sessions").glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    base = None
    bf = [f for f in (Path(base_dir) / "sessions").glob("*.csv") if f.stat().st_size > 1]
    if bf:
        base = pd.concat([pd.read_csv(f) for f in bf], ignore_index=True)
        base = base[base.sid.isin(df.sid.unique())]                       # same sessions
    regs = sorted(df.region.unique())
    div = sm.division_of(regs)
    conf_counts = df.drop_duplicates("sid").confounds.value_counts().head(5)
    L = [f"# Is it memory or current behaviour? ({df.sid.nunique()} sessions, {df.subject.nunique()} mice)\n",
         "Same sessions in every column. Region rule: >= 2 sessions; Fisher; BH (FDR 5%).\n",
         "Confound sets available (top): " + "; ".join(f"{k} ({v})" for k, v in conf_counts.items()) + "\n",
         "## Whole past trial, regions significant\n",
         "| Lag | Step 1b (lags + prior) | A: lags only (no prior) | B: lags + prior + confounds | B rate |",
         "|---|---|---|---|---|"]
    sets = {}
    for k in range(1, hj.K + 1):
        cells = []
        if base is not None:
            pb = region_p(base, f"p_joint{k}"); cells.append(f"{len(sig(pb))} / {len(pb)}")
        else:
            cells.append("-")
        for tag in ("a", "b"):
            pr = region_p(df, f"{tag}_p_joint{k}"); sets[(tag, "joint", k)] = sig(pr)
            cells.append(f"{len(sets[(tag, 'joint', k)])} / {len(pr)}")
        L.append(f"| {k} | " + " | ".join(cells) + f" | {(df[f'b_p_joint{k}'] < .05).mean():.3f} |")
    for var in hj.VARS:
        L += [f"\n## {var} only, regions significant\n", "| Lag | A: no prior | B: strict |", "|---|---|---|"]
        for k in range(1, hj.K + 1):
            c = []
            for tag in ("a", "b"):
                pr = region_p(df, f"{tag}_p_{var}{k}"); sets[(tag, var, k)] = sig(pr)
                c.append(f"{len(sets[(tag, var, k)])} / {len(pr)}")
            L.append(f"| {k} | " + " | ".join(c) + " |")
    rows = []
    for r in regs:
        d = df[df.region == r]
        kb = [k for k in range(1, hj.K + 1) if r in sets[("b", "joint", k)]]
        ka = [k for k in range(1, hj.K + 1) if r in sets[("a", "joint", k)]]
        rows.append(dict(region=r, division=div.get(r, r), sessions=d.sid.nunique(), mice=d.subject.nunique(),
                         strict_lags=",".join(map(str, kb)), strict_persistence=max(kb) if kb else 0,
                         kernel_lags=",".join(map(str, ka)), kernel_persistence=max(ka) if ka else 0,
                         choice_lags_A=",".join(str(k) for k in range(1, hj.K + 1) if r in sets[("a", "choice", k)]),
                         feedback_lags_B=",".join(str(k) for k in range(1, hj.K + 1) if r in sets[("b", "feedback", k)]),
                         **{f"effB{k}": d[f"b_eff_joint{k}"].median() for k in range(1, hj.K + 1)}))
    t = pd.DataFrame(rows)
    t2 = t[t.sessions >= 2]
    L += ["\n## Persistence by division after all controls (B)\n",
          "| Division | Regions | Lag 1 | Lag 2 | Lag 3+ |", "|---|---|---|---|---|"]
    for dv, h in t2.groupby("division"):
        L.append(f"| {dv} | {len(h)} | {sum(r in sets[('b', 'joint', 1)] for r in h.region)} | "
                 f"{sum(r in sets[('b', 'joint', 2)] for r in h.region)} | {sum(h.strict_persistence >= 3)} |")
    L += ["\n## Regions with memory beyond the last trial after all controls (B)\n",
          "| Region | Division | Sessions | Mice | Lags (B) | Lags (A, no prior) | Choice lags (A) | Feedback lags (B) | Effect lag 1..6 (B) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t2[t2.strict_persistence >= 2].sort_values(["strict_persistence", "division"], ascending=[False, True]).iterrows():
        L.append(f"| {r.region} | {r.division} | {r.sessions} | {r.mice} | {r.strict_lags} | {r.kernel_lags} | "
                 f"{r.choice_lags_A} | {r.feedback_lags_B} | " + " ".join(f"{r[f'effB{k}']:.2f}" for k in range(1, hj.K + 1)) + " |")
    text = "\n".join(L)
    (Path(out_dir) / "HISTORY_CONFOUNDS.md").write_text(text + "\n")
    t.to_csv(Path(out_dir) / "history_confounds_regions.csv", index=False)
    print(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "run", "report"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--aux", default=AUX)
    ap.add_argument("--out", default="results/history_confounds")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.step == "fetch":
        fetch(a.cache, a.aux, a.limit)
    elif a.step == "run":
        run(a.cache, a.aux, a.out, a.limit, a.jobs)
    else:
        report(a.out)
