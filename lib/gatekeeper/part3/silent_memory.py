"""PART 3 steps 3b + 3c: is the memory of past CHOICES silent between trials, and does it come back?

Behaviour (step 3a, 459 sessions): choices depend on the mouse's own choices 1-6 trials back (median weights
+0.30 +0.23 +0.16 +0.10 +0.17 +0.13, little changed by the block prior). Neural (step 1c): pre-stimulus activity
carries the past choice for ONE trial only, in every region.

3b  POWER ("would we have seen it?"). In each real recording, keep everything real and ADD a synthetic choice
    memory for lags 2-6: each neuron gets its own measured lag-1 choice weight, scaled by the behavioural ratio
    w_k / w_1 (and, conservatively, by half of it). Then run exactly the same test. If the test finds the planted
    memory in many regions while the real data show none, the real memory is silent rather than missed.
3c  REACTIVATION ("does it come back?"). Run the same lag test in three windows of the SAME trials:
       prior_a  (-600 to -100 ms before stimulus)    between trials
       stim_b   (+50 to +150 ms after stimulus)      when new evidence arrives
       choice_a (-100 to 0 ms before movement)       when the decision is executed
    always removing the CURRENT trial's stimulus, choice and reaction time plus the movement/pupil/timing
    confounds (symmetrically, from neurons and every label), so a past choice cannot sneak in through the
    current one.

    python3 part3/silent_memory.py power --limit 3
    python3 part3/silent_memory.py power
    python3 part3/silent_memory.py reactivate --limit 3
    python3 part3/silent_memory.py reactivate
    python3 part3/silent_memory.py rewarded --every 2
    python3 part3/silent_memory.py report
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

BEHAV_RATIO = np.array([1.0, 0.225, 0.162, 0.095, 0.168, 0.132]) / np.array([1.0, 0.300, 0.300, 0.300, 0.300, 0.300])
BEHAV_RATIO[0] = 1.0                       # lag-1 is left as it really is
WINDOWS = ("prior_a", "stim_b", "choice_a")
# behaviour with the drifting bias modelled (step 3a-2): weight of the REWARDED choice 1..6 trials back
WIN_RATIO = np.array([0.472, 0.371, 0.245, 0.259, 0.092, 0.170]) / 0.472


def _setup(path, bank, aux_dir, n_null, seed):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from history_confounds import confounds
    s = MVSession(path)
    v = cm.mask_of(s)
    if v is None or v.sum() < 200:
        return None
    Tf = len(s.contrast)
    rng = np.random.default_rng(seed + zlib.crc32(s.sid.encode()) % 100000)
    pool = [b["M"][:Tf] for sid, b in bank.items() if b["subject"] != s.subject and b["T"] >= Tf]
    if len(pool) < 30:
        return None
    idx = rng.choice(len(pool), size=min(n_null, len(pool)), replace=False)
    owns = np.stack([bank[s.sid]["M"]] + [pool[j] for j in idx])
    Hc, _ = confounds(aux_dir, s.sid, Tf)
    c = np.asarray(s.contrast, float)
    ch = np.where(s.choice >= 0, s.choice - 0.5, 0.0)
    lrt = np.where(np.isfinite(s.rt) & (s.rt > 0), np.log(np.clip(s.rt, 0.05, 10)), np.nan)
    lrt = np.where(np.isfinite(lrt), lrt, np.nanmedian(lrt))
    cur = np.column_stack([c, np.abs(c), ch, lrt, np.asarray(s.feedback, float)])   # current trial, incl. its outcome
    H = cur if Hc is None else np.column_stack([cur, Hc])
    return s, v, owns, H, rng


def power_rows(path, bank, aux_dir, n_null=hj.N_NULL, seed=0):
    from scipy.signal import lfilter  # noqa: F401
    S = _setup(path, bank, aux_dir, n_null, seed)
    if S is None:
        return []
    s, v, owns, H, rng = S
    c = np.where(s.choice >= 0, s.choice - 0.5, 0.0)
    Lc = lags(c, hj.K)
    rows = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        keep = X[v].std(0) > 0
        if keep.sum() < hj.MIN_UNITS:
            continue
        X = X[:, keep].astype(float)
        Xv = X[v]; mu, sd = Xv.mean(0), Xv.std(0)
        Z = (Xv - mu) / sd
        L1 = Lc[v, 0] - Lc[v, 0].mean()
        beta = (L1 @ (Z - Z.mean(0))) / (L1 @ L1)                     # each neuron's measured lag-1 choice weight
        row = dict(sid=s.sid, subject=s.subject, region=r, n_units=int(keep.sum()))
        real = hj.joint_tests(X, owns, v, drop_prior=True, only=("choice",))
        for name, k, pp, ee in real:
            row[f"real_p_choice{k}"] = pp
        for scale_name, scale in (("full", 1.0), ("half", 0.5)):
            add = np.zeros_like(X)
            for k in range(2, hj.K + 1):
                add += scale * BEHAV_RATIO[k - 1] * np.outer(Lc[:, k - 1] - Lc[v, k - 1].mean(), beta * sd)
            res = hj.joint_tests(X + add, owns, v, drop_prior=True, only=("choice",))
            for name, k, pp, ee in res:
                row[f"{scale_name}_p_choice{k}"] = pp
        rows.append(row)
    return rows


def rewarded_rows(path, bank, aux_dir, n_null=hj.N_NULL, seed=0):
    """Real test of reward-gated choice memory (which choice was rewarded k trials back), plus the same test after
    planting it at lags 2-6 with the behavioural decay of the rewarded-choice weights, using each neuron's measured
    lag-1 CHOICE weight as the neural strength (full and half)."""
    S = _setup(path, bank, aux_dir, n_null, seed)
    if S is None:
        return []
    s, v, owns, H, rng = S
    c = np.where(s.choice >= 0, s.choice - 0.5, 0.0)
    rc = c * 2 * (np.asarray(s.feedback) > 0)                       # +1 rewarded right, -1 rewarded left, 0 error
    Lc = lags(c, hj.K); Lr = lags(rc, hj.K)
    rows = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        keep = X[v].std(0) > 0
        if keep.sum() < hj.MIN_UNITS:
            continue
        X = X[:, keep].astype(float)
        Xv = X[v]; sd = Xv.std(0)
        Z = (Xv - Xv.mean(0)) / sd
        L1 = Lc[v, 0] - Lc[v, 0].mean()
        beta = (L1 @ (Z - Z.mean(0))) / (L1 @ L1)
        row = dict(sid=s.sid, subject=s.subject, region=r, n_units=int(keep.sum()))
        for name, k, pp, ee in hj.joint_tests(X, owns, v, drop_prior=True, only=("rewarded",)):
            row[f"real_p_rewarded{k}"] = pp
        for scale_name, scale in (("full", 1.0), ("half", 0.5)):
            add = np.zeros_like(X)
            for k in range(2, hj.K + 1):
                add += scale * WIN_RATIO[k - 1] * np.outer(Lr[:, k - 1] - Lr[v, k - 1].mean(), beta * sd)
            for name, k, pp, ee in hj.joint_tests(X + add, owns, v, drop_prior=True, only=("rewarded",)):
                row[f"{scale_name}_p_rewarded{k}"] = pp
        rows.append(row)
    return rows


def reactivation_rows(path, bank, aux_dir, n_null=hj.N_NULL, seed=0):
    S = _setup(path, bank, aux_dir, n_null, seed)
    if S is None:
        return []
    s, v, owns, H, rng = S
    rows = []
    for r in s.regions:
        row = dict(sid=s.sid, subject=s.subject, region=r)
        okw = 0
        for w in WINDOWS:
            X = s.X(r, w)
            if X is None:
                continue
            keep = X[v].std(0) > 0
            if keep.sum() < hj.MIN_UNITS:
                continue
            vv = v & s.valid["firstMovement"] if w.startswith("choice") else v
            vv = vv & s.valid["feedback"] if w.startswith("feedback") else vv
            res = hj.joint_tests(X[:, keep], owns, vv, H=H, drop_prior=True, only=("choice", "joint", "rewarded"))
            for name, k, pp, ee in res:
                row[f"{w}_p_{name}{k}"] = pp
                row[f"{w}_eff_{name}{k}"] = ee
            okw += 1
        if okw:
            rows.append(row)
    return rows


def check_rows(path, bank, aux_dir, n_null=hj.N_NULL, seed=0):
    """Bug and power check of the reactivation test, in each requested window, on REAL recordings:
    fake  = an unrelated mouse's history used as if it were this mouse's (must be about 0.05, 0 regions)
    planted full / half = rewarded-choice memory at lags 2-6 added to THIS window's activity with the behavioural decay,
                          each neuron weighted by its lag-1 choice weight measured in this window (must be detected)"""
    S = _setup(path, bank, aux_dir, n_null, seed)
    if S is None:
        return []
    s, v, owns, H, rng = S
    fake_owns = owns[1:]                                             # column 0 is now another mouse
    c = np.where(s.choice >= 0, s.choice - 0.5, 0.0)
    rc = c * 2 * (np.asarray(s.feedback) > 0)
    Lc, Lr = lags(c, hj.K), lags(rc, hj.K)
    rows = []
    for r in s.regions:
        for w in WINDOWS:
            X = s.X(r, w)
            if X is None:
                continue
            vv = v & s.valid["firstMovement"] if w.startswith("choice") else v
            vv = vv & s.valid["feedback"] if w.startswith("feedback") else vv
            keep = X[vv].std(0) > 0
            if keep.sum() < hj.MIN_UNITS:
                continue
            X = X[:, keep].astype(float)
            row = dict(sid=s.sid, subject=s.subject, region=r, window=w)
            for name, k, pp, ee in hj.joint_tests(X, owns, vv, H=H, drop_prior=True, only=("rewarded",)):
                row[f"real_p{k}"] = pp
            for name, k, pp, ee in hj.joint_tests(X, fake_owns, vv, H=H, drop_prior=True, only=("rewarded",)):
                row[f"fake_p{k}"] = pp
            Xv = X[vv]; sd = Xv.std(0); Z = (Xv - Xv.mean(0)) / sd
            L1 = Lc[vv, 0] - Lc[vv, 0].mean()
            beta = (L1 @ (Z - Z.mean(0))) / max(L1 @ L1, 1e-9)
            for tag, scale in (("full", 1.0), ("half", 0.5)):
                add = np.zeros_like(X)
                for k in range(2, hj.K + 1):
                    add += scale * WIN_RATIO[k - 1] * np.outer(Lr[:, k - 1] - Lr[vv, k - 1].mean(), beta * sd)
                for name, k, pp, ee in hj.joint_tests(X + add, owns, vv, H=H, drop_prior=True, only=("rewarded",)):
                    row[f"{tag}_p{k}"] = pp
            rows.append(row)
    return rows


def check_report(df, out_dir):
    from verify_numbers import region_p, sig
    L = [f"# Bug and power check of the reactivation test ({df.sid.nunique()} sessions)\n",
         "Rewarded-choice memory k trials back. Regions significant at FDR 5% (session rate). "
         "Fake must be ~0.05 and 0 regions; planted must be detected.\n"]
    for w, d in df.groupby("window"):
        L += [f"## Window {w}\n", "| Lag | Real | Fake | Planted full | Planted half |", "|---|---|---|---|---|"]
        for k in range(1, hj.K + 1):
            cells = []
            for tag in ("real", "fake", "full", "half"):
                col = f"{tag}_p{k}"
                pr = region_p(d, col)
                cells.append(f"{len(sig(pr))} / {len(pr)} ({(d[col] < .05).mean():.3f})")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        L.append("")
    text = "\n".join(L)
    (Path(out_dir) / "REACTIVATION_CHECK.md").write_text(text + "\n")
    print(text)


def run(kind, cache, aux_dir, out_dir, limit, jobs, every, windows=None):
    global WINDOWS
    if windows:
        WINDOWS = tuple(windows.split(","))
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / (kind if not windows else f"{kind}_{'_'.join(WINDOWS)}")
    out.mkdir(parents=True, exist_ok=True)
    print("loading behaviour of all sessions (imposter pool) ...", flush=True)
    bank = hj.behaviour_bank(cache)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[::every][: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(paths) - len(todo)} already done, {len(todo)} to run", flush=True)
    fn = {"power": power_rows, "reactivate": reactivation_rows, "rewarded": rewarded_rows, "check": check_rows}[kind]
    t0 = time.time()

    def work(p):
        try:
            rows = fn(p, bank, aux_dir)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True)
            rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    for i, n in enumerate(Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)):
        print(f"[{i + 1}/{len(todo)}] {n} regions  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if kind == "check":
        df = _load(out)
        if df is not None:
            check_report(df, out)


def _load(d):
    import pandas as pd
    fs = [f for f in Path(d).glob("*.csv") if f.stat().st_size > 1]
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else None


def report(out_dir):
    from verify_numbers import region_p, sig
    L = ["# Is the memory of past choices silent, and does it come back?\n"]
    pw = _load(Path(out_dir) / "power")
    if pw is not None:
        L += [f"## 3b Power: planted behaviour-matched choice memory ({pw.sid.nunique()} sessions, {pw.subject.nunique()} mice)\n",
              "Each region's real lag-1 choice code, copied to lags 2-6 with the behavioural decay (full) or half of it. "
              "Same test as the real data (lags only, no prior). Regions significant at FDR 5%.\n",
              "| Lag | Real data | Planted, behaviour-matched | Planted, half strength | Real: session rate | Planted full: session rate |",
              "|---|---|---|---|---|---|"]
        for k in range(1, hj.K + 1):
            cells = []
            for tag in ("real", "full", "half"):
                pr = region_p(pw, f"{tag}_p_choice{k}"); cells.append(f"{len(sig(pr))} / {len(pr)}")
            L.append(f"| {k} | " + " | ".join(cells) + f" | {(pw[f'real_p_choice{k}'] < .05).mean():.3f} | "
                     f"{(pw[f'full_p_choice{k}'] < .05).mean():.3f} |")
    rw = _load(Path(out_dir) / "rewarded")
    if rw is not None:
        L += [f"\n## 3e Reward-gated choice memory: which choice was REWARDED k trials back ({rw.sid.nunique()} sessions, "
              f"{rw.subject.nunique()} mice)\n",
              "Behaviour (drifting bias modelled) uses this for about 4 trials. Real data vs the same memory planted with the "
              "behavioural decay. Regions significant at FDR 5%.\n",
              "| Lag | Real data | Planted, behaviour-matched | Planted, half strength | Real: session rate | Planted full: session rate |",
              "|---|---|---|---|---|---|"]
        for k in range(1, hj.K + 1):
            cells = []
            for tag in ("real", "full", "half"):
                pr = region_p(rw, f"{tag}_p_rewarded{k}"); cells.append(f"{len(sig(pr))} / {len(pr)}")
            L.append(f"| {k} | " + " | ".join(cells) + f" | {(rw[f'real_p_rewarded{k}'] < .05).mean():.3f} | "
                     f"{(rw[f'full_p_rewarded{k}'] < .05).mean():.3f} |")
    fbk = _load(Path(out_dir) / "reactivate_feedback_a")
    if fbk is not None:
        L += [f"\n## M2 Outcome-time reactivation ({fbk.sid.nunique()} sessions, {fbk.subject.nunique()} mice)\n",
              "Window 0-200 ms after the current trial's feedback; current stimulus, choice, outcome, RT and confounds removed. "
              "Regions significant at FDR 5% (session rate).\n",
              "| Lag | Choice | Rewarded choice | Whole trial |", "|---|---|---|---|"]
        for k in range(1, hj.K + 1):
            cells = []
            for name in ("choice", "rewarded", "joint"):
                c = f"feedback_a_p_{name}{k}"
                if c in fbk:
                    pr = region_p(fbk, c); cells.append(f"{len(sig(pr))} / {len(pr)} ({(fbk[c] < .05).mean():.3f})")
                else:
                    cells.append("-")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
    rc = _load(Path(out_dir) / "reactivate")
    if rc is not None:
        L += [f"\n## 3c Reactivation: past choices in three windows of the same trials ({rc.sid.nunique()} sessions, "
              f"{rc.subject.nunique()} mice)\n",
              "Current stimulus, current choice, reaction time, licking, wheel, pupil and timing removed in every window. "
              "Regions significant at FDR 5% (session-region rejection rate in brackets).\n",
              "| Lag | Choice: between trials | Choice: after stimulus | Choice: before movement | Whole trial: between | Whole trial: after stimulus | Whole trial: before movement |",
              "|---|---|---|---|---|---|---|"]
        sets = {}
        for k in range(1, hj.K + 1):
            cells = []
            for name in ("choice", "joint"):
                for w in WINDOWS:
                    c = f"{w}_p_{name}{k}"
                    if c not in rc:
                        cells.append("-"); continue
                    pr = region_p(rc, c); sets[(name, w, k)] = sig(pr)
                    cells.append(f"{len(sets[(name, w, k)])} / {len(pr)} ({(rc[c] < .05).mean():.3f})")
            L.append(f"| {k} | " + " | ".join(cells) + " |")
        for w in ("stim_b", "choice_a"):
            hits = sorted({r for k in range(2, hj.K + 1) for r in sets.get(("choice", w, k), set())})
            L += [f"\nRegions with choice memory 2+ trials back in {w}: " + (", ".join(hits) if hits else "none")]
    text = "\n".join(L)
    (Path(out_dir) / "SILENT_MEMORY.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["power", "reactivate", "rewarded", "check", "report"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--aux", default="data/part3_aux2")
    ap.add_argument("--out", default="results/silent_memory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--every", type=int, default=1, help="use every n-th session (power only needs a subset)")
    ap.add_argument("--windows", default="", help="comma list of windows for reactivate, e.g. feedback_a")
    a = ap.parse_args()
    if a.step == "report":
        report(a.out)
    else:
        run(a.step, a.cache, a.aux, a.out, a.limit, a.jobs, a.every, a.windows)
