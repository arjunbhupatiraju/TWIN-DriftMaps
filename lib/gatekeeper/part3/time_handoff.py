"""PART 3 model stage 2: does the reward memory leave activity with TIME (the trade-off rule) ?

Pre-registered (claude/preregistration-time-handoff.md, Oct 27, before any gap analysis):
    The last trial's rewarded-choice code in pre-stimulus activity is weaker when more time has passed since the
    previous outcome. Control: the block-prior code (slow; should not depend on the gap unless engagement differs).

Split: per session, median split of the gap (stimulus onset - previous feedback, seconds), done separately for trials
after a reward and after an error. Test: drift-aware joint test (rewarded lag 1), as in step 3e, on each half.

    python3 part3/time_handoff.py sim
    python3 part3/time_handoff.py run --limit 3      then   run
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

N_NULL = 100


def pz(p):
    from scipy.stats import norm
    return norm.isf(np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9))


def gap_split(gap, prev_rewarded, v):
    """Short / long masks: median split of the gap within trials after a reward and within trials after an error."""
    short = np.zeros(len(gap), bool); long_ = np.zeros(len(gap), bool)
    for grp in (prev_rewarded & v, ~prev_rewarded & v):
        idx = np.flatnonzero(grp & np.isfinite(gap))
        if len(idx) < 20:
            continue
        m = np.median(gap[idx])
        short[idx[gap[idx] <= m]] = True
        long_[idx[gap[idx] > m]] = True
    return short, long_


def rc1_test(X, owns, mask, H):
    for name, k, pp, ee in hj.joint_tests(X, owns, mask, H=H, drop_prior=True, only=("rewarded",)):
        if k == 1:
            return pp, ee
    return np.nan, np.nan


def prior_test(X, y, ynull, mask, lag1=None):
    """Block-prior control. lag1 (T, q): last-trial variables (rewarded choice, choice) removed from the neurons
    first, so a decaying lag-1 code cannot leak into the prior control through their correlation."""
    from driftaware import drift_test
    Xm = X[mask].astype(float)
    if lag1 is not None:
        Z = np.column_stack([np.ones(mask.sum()), lag1[mask]])
        Xm = Xm - Z @ np.linalg.lstsq(Z, Xm, rcond=None)[0]
    return drift_test(Xm, y[mask], ynull[mask]).p


def load_gap(aux_dir, sid, T):
    f = Path(aux_dir) / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    if not bool(z["ok"]) or "time_since_feedback" not in z.files or len(z["time_since_feedback"]) != T:
        return None
    return np.asarray(z["time_since_feedback"], float)


def session_rows(path, bank, aux_dir, seed=0):
    import silent_memory as sm
    import compare_methods as cm
    S = sm._setup(path, bank, aux_dir, N_NULL, seed)
    if S is None:
        return []
    s, v, owns, H, rng = S
    T = len(s.contrast)
    gap = load_gap(aux_dir, s.sid, T)
    if gap is None:
        return []
    fb = np.asarray(s.feedback, float)
    prev_rew = np.r_[False, fb[:-1] > 0]
    short, long_ = gap_split(gap, prev_rew, v)
    if short.sum() < 100 or long_.sum() < 100:
        return []
    Y = cm.label_matrix(s, "bprior", np.ones(T, bool), N_NULL, rng)       # real prior + pseudo-session priors
    c = np.where(s.choice >= 0, 2.0 * s.choice - 1, 0.0)
    lag1 = np.column_stack([np.r_[0.0, (c * (fb > 0))[:-1]], np.r_[0.0, c[:-1]]])
    rows = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        keep = X[v].std(0) > 0
        if keep.sum() < hj.MIN_UNITS:
            continue
        X = X[:, keep].astype(float)
        row = dict(sid=s.sid, subject=s.subject, region=r, n_units=int(keep.sum()),
                   gap_short=float(np.median(gap[short])), gap_long=float(np.median(gap[long_])))
        for tag, m in (("short", short), ("long", long_)):
            Xm = X[:, X[m].std(0) > 0]
            if Xm.shape[1] < hj.MIN_UNITS:
                break
            row[f"rc1_p_{tag}"], row[f"rc1_eff_{tag}"] = rc1_test(Xm, owns, m, H)
            row[f"prior_p_{tag}"] = prior_test(Xm, Y[:, 0], Y[:, 1:], m, lag1)
        else:
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------- addendum 1 (follow-ups)
def gap_bins(gap, prev_rewarded, v, nb=5):
    """Quintile index (0 = shortest) of the gap, computed separately after rewards and after errors; -1 = unused."""
    b = np.full(len(gap), -1)
    for grp in (prev_rewarded & v, ~prev_rewarded & v):
        idx = np.flatnonzero(grp & np.isfinite(gap))
        if len(idx) < 5 * nb:
            continue
        r = np.argsort(np.argsort(gap[idx]))
        b[idx] = np.minimum(nb - 1, (r * nb) // len(idx))
    return b


def stim_null(T, n, rng):
    from gatekeeper.multiverse import pseudo_pl
    return np.column_stack([2.0 * (rng.random(T) >= pseudo_pl(T, rng)) - 1 for _ in range(n)])


def followup_rows(path, bank, aux_dir, seed=0):
    """A (dose-response of the lag-1 code over gap quintiles) and B1 (current-stimulus code, short vs long)."""
    import silent_memory as sm
    from driftaware import drift_test
    S = sm._setup(path, bank, aux_dir, N_NULL, seed)
    if S is None:
        return []
    s, v, owns, H, rng = S
    T = len(s.contrast)
    gap = load_gap(aux_dir, s.sid, T)
    if gap is None:
        return []
    fb = np.asarray(s.feedback, float)
    prev_rew = np.r_[False, fb[:-1] > 0]
    bins = gap_bins(gap, prev_rew, v)
    short, long_ = gap_split(gap, prev_rew, v)
    sd = np.sign(np.nan_to_num(np.asarray(s.contrast, float)))
    Ynull = stim_null(T, N_NULL, rng)
    rows = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        keep = X[v].std(0) > 0
        if keep.sum() < hj.MIN_UNITS:
            continue
        X = X[:, keep].astype(float)
        row = dict(sid=s.sid, subject=s.subject, region=r)
        zs = []
        for b in range(5):
            m = bins == b
            Xm = X[:, X[m].std(0) > 0]
            if m.sum() < 40 or Xm.shape[1] < hj.MIN_UNITS:
                zs.append(np.nan); continue
            p, _ = rc1_test(Xm, owns, m, H)
            zs.append(float(pz(p)))
            row[f"bin{b}_z"] = zs[-1]
            row[f"bin{b}_gap"] = float(np.median(gap[m]))
        ok = np.isfinite(zs)
        if ok.sum() >= 3:
            row["slope"] = float(np.polyfit(np.arange(5)[ok], np.asarray(zs)[ok], 1)[0])
        Xs = s.X(r, "stim_a")
        if Xs is not None:
            Xs = Xs[:, keep].astype(float)
            for tag, m in (("short", short), ("long", long_)):
                mm = m & (sd != 0)
                Xm = Xs[mm][:, Xs[mm].std(0) > 0]
                if mm.sum() > 60 and Xm.shape[1] >= hj.MIN_UNITS:
                    row[f"stim_p_{tag}"] = float(drift_test(Xm, sd[mm], Ynull[mm]).p)
        rows.append(row)
    return rows


def behaviour_followup(cache, aux_dir):
    """C (behavioural use of the last reward, short vs long gaps) and B2 (accuracy, RT)."""
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from gatekeeper.multiverse import _sides
    import reward_write as rw
    rows = []
    for p in sorted(Path(cache).expanduser().glob("*.npz")):
        s = MVSession(p)
        v = cm.mask_of(s)
        if v is None:
            continue
        T = len(s.contrast)
        gap = load_gap(aux_dir, s.sid, T)
        if gap is None:
            continue
        fb = np.asarray(s.feedback, float)
        short, long_ = gap_split(gap, np.r_[False, fb[:-1] > 0], v)
        m = short | long_
        if m.sum() < 200:
            continue
        c = np.where(s.choice >= 0, 2.0 * s.choice - 1, 0.0)
        rc = c * (fb > 0)
        c1, rc1 = np.r_[0.0, c[:-1]], np.r_[0.0, rc[:-1]]
        lg = long_.astype(float)
        prior = cm.bayes_prior(_sides(s)) - 0.5
        X = np.column_stack([rw.stim_design(s.contrast), prior, c1, rc1, lg, lg * c1, lg * rc1])
        ok = m & (c != 0)
        w = rw.logit_fit(X[ok], (c[ok] > 0).astype(float), lam=1.0)
        nz = np.nan_to_num(np.asarray(s.contrast, float)) != 0
        lrt = np.log(np.clip(np.asarray(s.rt, float), 0.05, 10))
        rows.append(dict(sid=s.sid, subject=s.subject, w_rc1_short=w[5], w_rc1_long=w[5] + w[8], w_inter=w[8],
                         acc_short=float((fb[short & nz] > 0).mean()), acc_long=float((fb[long_ & nz] > 0).mean()),
                         lrt_short=float(np.nanmedian(lrt[short])), lrt_long=float(np.nanmedian(lrt[long_]))))
    return rows


def run(cache, aux_dir, out_dir, limit, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / "sessions"; out.mkdir(parents=True, exist_ok=True)
    print("loading behaviour of all sessions (imposter pool) ...", flush=True)
    bank = hj.behaviour_bank(cache)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(todo)} to run", flush=True)
    t0 = time.time()

    def work(p):
        try:
            rows = session_rows(p, bank, aux_dir, seed=zlib.crc32(p.stem.encode()) % 100000)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True); rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    try:
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(work)(p) for p in todo)
    except Exception:
        gen = Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)
    for i, n in enumerate(gen):
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"[{i + 1}/{len(todo)}] ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fs = [f for f in out.glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    report(df, out_dir)


def run_followup(cache, aux_dir, out_dir, limit, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / "followup_sessions"; out.mkdir(parents=True, exist_ok=True)
    print("behaviour (C, B2) ...", flush=True)
    beh = pd.DataFrame(behaviour_followup(cache, aux_dir))
    beh.to_csv(Path(out_dir) / "followup_behaviour.csv", index=False)
    print("loading behaviour of all sessions (imposter pool) ...", flush=True)
    bank = hj.behaviour_bank(cache)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(todo)} to run (A, B1)", flush=True)
    t0 = time.time()

    def work(p):
        try:
            rows = followup_rows(p, bank, aux_dir, seed=zlib.crc32(p.stem.encode()) % 100000 + 7)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True); rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    try:
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(work)(p) for p in todo)
    except Exception:
        gen = Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)
    for i, n in enumerate(gen):
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            print(f"[{i + 1}/{len(todo)}] ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fs = [f for f in out.glob("*.csv") if f.stat().st_size > 1]
    neu = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    main = [f for f in (Path(out_dir) / "sessions").glob("*.csv") if f.stat().st_size > 1]
    main = pd.concat([pd.read_csv(f) for f in main], ignore_index=True) if main else None
    report_followup(neu, beh, main, out_dir)


def report_followup(neu, beh, main, out_dir):
    from scipy.stats import ttest_1samp
    from gatekeeper.infoflow_data import split_of
    L = [f"# Addendum 1 (pre-registered follow-ups): does the memory fade from activity while engagement and "
         f"behavioural use stay constant?\n"]
    # A
    d = neu.dropna(subset=["slope"])
    ses = d.groupby(["sid", "subject"]).slope.mean().reset_index()
    ses["split"] = ses.subject.map(split_of)
    tA = ttest_1samp(ses.slope, 0, alternative="less")
    bysplit = ses.groupby("split").slope.mean()
    passA = tA.pvalue < 0.01 and (bysplit < 0).all()
    binz = [neu[f"bin{b}_z"].mean() for b in range(5) if f"bin{b}_z" in neu]
    bing = [neu[f"bin{b}_gap"].median() for b in range(5) if f"bin{b}_gap" in neu]
    L += ["## A. Dose-response (lag-1 code across gap quintiles)\n",
          "| Gap quintile | 1 (shortest) | 2 | 3 | 4 | 5 (longest) |", "|---|---|---|---|---|---|",
          "| Median gap (s) | " + " | ".join(f"{g:.2f}" for g in bing) + " |",
          "| Mean z, rewarded lag 1 | " + " | ".join(f"{z:.3f}" for z in binz) + " |",
          f"\nMean session slope {ses.slope.mean():.4f} per quintile (one-sided p {tA.pvalue:.3g}, {len(ses)} sessions); "
          f"by split: " + ", ".join(f"{k} {v:.4f}" for k, v in bysplit.items()) + f". **A passes: {'YES' if passA else 'no'}**\n"]
    # B1
    b1 = neu.dropna(subset=["stim_p_short", "stim_p_long"]).copy()
    b1["dz"] = pz(b1.stim_p_short) - pz(b1.stim_p_long)
    sb = b1.groupby("sid").dz.mean()
    tB = ttest_1samp(sb, 0, alternative="greater")
    passB = tB.pvalue > 0.05
    L += ["## B. Engagement control\n",
          f"B1, current-stimulus code (stimulus window): mean z short {pz(b1.stim_p_short).mean():.3f}, long "
          f"{pz(b1.stim_p_long).mean():.3f}; session mean (short minus long) {sb.mean():.3f}, one-sided p (> 0) "
          f"{tB.pvalue:.3g}. **B1 passes (no weaker stimulus coding after long gaps): {'YES' if passB else 'no'}**",
          f"\nB2, behaviour: accuracy short {beh.acc_short.mean():.3f}, long {beh.acc_long.mean():.3f}; median log RT "
          f"short {beh.lrt_short.mean():.3f}, long {beh.lrt_long.mean():.3f} (descriptive)\n"]
    # C
    mice = beh.groupby("subject")[["w_rc1_short", "w_rc1_long", "w_inter"]].mean()
    tC = ttest_1samp(mice.w_inter, 0, alternative="less")
    beh_ratio = mice.w_rc1_long.mean() / mice.w_rc1_short.mean()
    if main is not None:
        mm = main.dropna(subset=["rc1_p_short", "rc1_p_long"])
        neu_ratio = pz(mm.rc1_p_long).mean() / pz(mm.rc1_p_short).mean()
    else:
        neu_ratio = np.nan
    passC = tC.pvalue > 0.05 and np.isfinite(neu_ratio) and beh_ratio > neu_ratio
    L += ["## C. Behavioural use of the last reward\n",
          f"Rewarded lag-1 behavioural weight (mean across {len(mice)} mice): short {mice.w_rc1_short.mean():.3f}, long "
          f"{mice.w_rc1_long.mean():.3f}; interaction {mice.w_inter.mean():.3f} (one-sided p < 0: {tC.pvalue:.3g}).",
          f"Long/short ratio: behaviour {beh_ratio:.2f} vs neural visibility {neu_ratio:.2f}. "
          f"**C passes: {'YES' if passC else 'no'}**\n"]
    verdict = ("EXCEPTIONAL (dissociation: fades from activity, engagement and behavioural use constant)" if passA and passB and passC
               else "SUPPORTING BUT PARTIAL" if passA else "NOT CONFIRMED")
    L.append(f"## Verdict (pre-registered): {verdict}")
    text = "\n".join(L)
    (Path(out_dir) / "TIME_HANDOFF_FOLLOWUP.md").write_text(text + "\n")
    print(text)


# ------------------------------------------------------------------------------------------- two-component test
KL = 4


def twopart_session(contrast, choice, feedback, prior, short, long_, v):
    """Per-session weights: short-gap rewarded-choice weights w_1..4 and long x rewarded-choice interactions i_1..4."""
    import reward_write as rw
    from history_map import lags
    c = np.where(np.asarray(choice) >= 0, 2.0 * np.asarray(choice) - 1, 0.0)
    rc = c * (np.asarray(feedback) > 0)
    Lc, Lr = lags(c, KL), lags(rc, KL)
    lg = long_.astype(float)[:, None]
    X = np.column_stack([rw.stim_design(contrast), prior, Lc, Lr, lg[:, 0], lg * Lc, lg * Lr])
    m = (short | long_) & (c != 0) & v
    m[:KL] = False
    w = rw.logit_fit(X[m], (c[m] > 0).astype(float), lam=1.0)
    base = 1 + 2 + 1                                   # intercept, 2 stimulus columns, prior
    wr = w[base + KL: base + 2 * KL]
    ir = w[base + 2 * KL + 1 + KL: base + 2 * KL + 1 + 2 * KL]
    return wr, ir


def twopart_stats(W, I):
    D1 = -I[0] / W[0]
    D24 = -I[1:].sum() / W[1:].sum()
    return D1, D24


def run_twopart(cache, aux_dir, out_dir, n_boot=2000, seed=0):
    import pandas as pd
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from gatekeeper.multiverse import _sides
    from gatekeeper.infoflow_data import split_of
    rows = []
    for p in sorted(Path(cache).expanduser().glob("*.npz")):
        s = MVSession(p)
        v = cm.mask_of(s)
        if v is None:
            continue
        T = len(s.contrast)
        gap = load_gap(aux_dir, s.sid, T)
        if gap is None:
            continue
        fb = np.asarray(s.feedback, float)
        short, long_ = gap_split(gap, np.r_[False, fb[:-1] > 0], v)
        if (short | long_).sum() < 200:
            continue
        prior = cm.bayes_prior(_sides(s)) - 0.5
        wr, ir = twopart_session(s.contrast, s.choice, fb, prior, short, long_, v)
        rows.append(dict(sid=s.sid, subject=s.subject, **{f"w{k + 1}": wr[k] for k in range(KL)},
                         **{f"i{k + 1}": ir[k] for k in range(KL)}))
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "twopart_sessions.csv", index=False)
    report_twopart(df, out_dir, n_boot, seed)


def report_twopart(df, out_dir, n_boot=2000, seed=0):
    from gatekeeper.infoflow_data import split_of
    wc = [f"w{k}" for k in range(1, KL + 1)]; ic = [f"i{k}" for k in range(1, KL + 1)]
    mice = df.groupby("subject")[wc + ic].mean()
    W, I = mice[wc].mean().values, mice[ic].mean().values
    D1, D24 = twopart_stats(W, I)
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(n_boot):
        mb = mice.iloc[rng.integers(0, len(mice), len(mice))]
        B.append(twopart_stats(mb[wc].mean().values, mb[ic].mean().values))
    B = np.asarray(B)
    diff = B[:, 0] - B[:, 1]
    p = float((diff <= 0).mean())
    sp = {}
    for k, g in mice.assign(split=mice.index.map(split_of)).groupby("split"):
        a, b = twopart_stats(g[wc].mean().values, g[ic].mean().values)
        sp[k] = a - b
    ok = (D1 - D24 > 0) and p < 0.05 and all(v > 0 for v in sp.values())
    L = [f"# Pre-registered two-component test ({len(df)} sessions, {len(mice)} mice)\n",
         "Does a longer gap before trial t weaken the LAST reward's effect more (relatively) than the effect of rewards "
         "2-4 trials back? Two components predicts yes (D1 > D24, D24 about 0); single time-decay predicts D24 about D1.\n",
         "| Lag | Short-gap weight W | Long-gap interaction I | Relative drop |", "|---|---|---|---|"]
    for k in range(KL):
        L.append(f"| {k + 1} | {W[k]:.3f} | {I[k]:.3f} | {(-I[k] / W[k]) if W[k] != 0 else float('nan'):.2f} |")
    L += [f"\nD1 (lag 1) = {D1:.2f}; D24 (lags 2-4 combined) = {D24:.2f}, 95% bootstrap interval "
          f"[{np.percentile(B[:, 1], 2.5):.2f}, {np.percentile(B[:, 1], 97.5):.2f}]",
          f"D1 - D24 = {D1 - D24:.2f}, one-sided bootstrap p = {p:.3g}; by split: " +
          ", ".join(f"{k} {v:.2f}" for k, v in sp.items()),
          f"\nTwo components supported by the pre-registered rule: {'YES' if ok else 'no'}"]
    text = "\n".join(L)
    (Path(out_dir) / "TWO_COMPONENT.md").write_text(text + "\n")
    print(text)


def sim_twopart(reps=60, seed=0):
    """Does the test tell the accounts apart? Simulated sessions (IBL-like length and gaps)."""
    import compare_methods as cm
    import history_joint as hj_
    import reward_write as rw
    from history_map import lags
    out = {}
    for model in ("two components", "single decay"):
        rows = []
        for i in range(reps):
            rng = np.random.default_rng(seed + 997 * i)
            T = 700
            side, _, _, prior = hj_.sim_behaviour(rng, T)
            contrast = (2 * side - 1) * rng.choice([0, .0625, .125, .25, 1.0], T)
            gap = rng.gamma(6, 0.55, T) + 0.8
            age_extra = gap - np.median(gap)
            c = np.zeros(T); rc = np.zeros(T); fb = np.zeros(T)
            base = np.array([0.47, 0.37, 0.25, 0.26]) * 1.6
            S = rw.stim_design(contrast)
            for t in range(T):
                lg = 4 * (S[t, 0] - S[t, 1]) + 1.5 * (prior[t] - 0.5)
                for k in range(1, KL + 1):
                    if t - k < 0:
                        continue
                    if model == "single decay":
                        wk = base[k - 1] * np.exp(-max(age_extra[t], -1.5) / 2.0)
                    else:   # fast time-based part only at lag 1 (half of lag 1), slow trial-based part everywhere
                        fast = 0.5 * base[0] * np.exp(-max(age_extra[t], -1.5) / 1.0) if k == 1 else 0.0
                        wk = (0.5 * base[0] if k == 1 else base[k - 1]) + fast
                    lg += wk * rc[t - k]
                ch = rng.random() < 1 / (1 + np.exp(-lg))
                c[t] = 1 if ch else -1
                fb[t] = 1 if (c[t] > 0) == (side[t] > 0) else -1
                rc[t] = c[t] * (fb[t] > 0)
            v = np.ones(T, bool); v[:90] = False
            short, long_ = gap_split(gap, np.r_[False, fb[:-1] > 0], v)
            wr, ir = twopart_session(contrast, (c > 0).astype(int), fb, prior - 0.5, short, long_, v)
            rows.append(np.r_[wr, ir])
        R = np.asarray(rows)
        W, I = R[:, :KL].mean(0), R[:, KL:].mean(0)
        rng = np.random.default_rng(1)
        B = np.asarray([twopart_stats(*(lambda r: (r[:, :KL].mean(0), r[:, KL:].mean(0)))(R[rng.integers(0, reps, reps)]))
                        for _ in range(1000)])
        D1, D24 = twopart_stats(W, I)
        print(f"{model:15s} D1 {D1:.2f}  D24 {D24:.2f}  D1-D24 {D1 - D24:.2f}  one-sided p {(B[:, 0] - B[:, 1] <= 0).mean():.3f}"
              f"  ({reps} simulated sessions)", flush=True)


def report(df, out_dir):
    from scipy.stats import ttest_1samp
    from gatekeeper.infoflow_data import split_of
    d = df.dropna(subset=["rc1_p_short", "rc1_p_long", "prior_p_short", "prior_p_long"]).copy()
    d["drc"] = pz(d.rc1_p_short) - pz(d.rc1_p_long)
    d["dpr"] = pz(d.prior_p_short) - pz(d.prior_p_long)
    d["did"] = d.drc - d.dpr
    ses = d.groupby(["sid", "subject"])[["drc", "dpr", "did"]].mean().reset_index()
    ses["split"] = ses.subject.map(split_of)
    t = lambda x: ttest_1samp(x, 0, alternative="greater")
    r1, r2, r3 = t(ses.drc), t(ses.did), t(ses.dpr)
    by = {k: g.drc.mean() for k, g in ses.groupby("split")}
    ok = r1.pvalue < .05 and r2.pvalue < .05 and all(v > 0 for v in by.values())
    L = [f"# Pre-registered test: does the reward memory leave activity with time? ({len(ses)} sessions, "
         f"{ses.subject.nunique()} mice, {len(d)} session-regions)\n",
         f"Median gap: short {d.gap_short.median():.2f} s, long {d.gap_long.median():.2f} s.\n",
         "| Quantity (session mean, z short minus z long) | Mean | t-test p (one-sided, > 0) |", "|---|---|---|",
         f"| Rewarded lag-1 code (rule 1) | {ses.drc.mean():.3f} | {r1.pvalue:.3g} |",
         f"| Block-prior code (control) | {ses.dpr.mean():.3f} | {r3.pvalue:.3g} |",
         f"| Difference-in-differences (rule 2) | {ses.did.mean():.3f} | {r2.pvalue:.3g} |",
         f"\nRewarded lag-1 difference by split (rule 3): " + ", ".join(f"{k} {v:.3f}" for k, v in by.items()),
         f"Mean z, rewarded lag 1: short {pz(d.rc1_p_short).mean():.3f}, long {pz(d.rc1_p_long).mean():.3f}; "
         f"prior: short {pz(d.prior_p_short).mean():.3f}, long {pz(d.prior_p_long).mean():.3f}",
         f"\nSupported by the pre-registered rule: {'YES' if ok else 'no'}"]
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "TIME_HANDOFF.md").write_text(text + "\n")
    print(text)


def sim(reps=40, seed=0):
    """Calibration and power: neurons whose lag-1 rewarded-choice code decays with the gap (tau 2 s) vs does not."""
    from joblib import Parallel, delayed
    from history_map import sim_session, lags
    from driftaware import drift_test
    import compare_methods as cm

    def one(i, decay):
        rng = np.random.default_rng(seed + 211 * i)
        T = 600
        X, side, choice, fb, prior = sim_session(rng, T=T, N=25, mode="none")
        gap = rng.gamma(4, 0.6, T) + 0.5                              # seconds, roughly IBL-like spread
        c = 2.0 * choice - 1
        rc = c * (fb > 0)
        L1 = lags(rc, 1)[:, 0]
        amp = np.exp(-gap / 2.0) / np.exp(-gap / 2.0).mean() if decay else np.ones(T)
        X = X + 0.10 * np.outer(L1 * amp, rng.standard_normal(X.shape[1]))
        X = X + 0.06 * np.outer((prior - prior.mean()) / prior.std(), rng.standard_normal(X.shape[1]))
        owns = [hj.seq_matrix(side, choice, fb, prior)]
        for _ in range(60):
            s2, c2, f2, p2 = hj.sim_behaviour(rng, T)
            owns.append(hj.seq_matrix(s2, c2, f2, p2))
        owns = np.stack(owns)
        v = np.ones(T, bool); v[:90] = False
        short, long_ = gap_split(gap, np.r_[False, fb[:-1] > 0], v)
        Yp = np.column_stack([prior] + [cm.bayes_prior((rng.random(T) >= 0.5).astype(float)) for _ in range(60)])
        zs = {}
        for tag, m in (("s", short), ("l", long_)):
            zs["rc" + tag] = pz(rc1_test(X, owns, m, None)[0])
            lag1 = np.column_stack([L1, lags(c, 1)[:, 0]])
            zs["pr" + tag] = pz(prior_test(X, Yp[:, 0], Yp[:, 1:], m, lag1))
        return zs["rcs"] - zs["rcl"], zs["prs"] - zs["prl"], (zs["rcs"] - zs["rcl"]) - (zs["prs"] - zs["prl"])

    for decay in (False, True):
        R = np.array(Parallel(n_jobs=-1)(delayed(one)(i, decay) for i in range(reps)))
        from scipy.stats import ttest_1samp
        print(f"{'decaying' if decay else 'no decay':9s} lag-1 z difference {R[:, 0].mean():.3f} "
              f"(p {ttest_1samp(R[:, 0], 0, alternative='greater').pvalue:.3g}), prior difference {R[:, 1].mean():.3f}, "
              f"diff-in-diff {R[:, 2].mean():.3f} (p {ttest_1samp(R[:, 2], 0, alternative='greater').pvalue:.3g}), "
              f"{reps} simulated sessions", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "run", "followup", "twopart", "sim_twopart"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--aux", default="data/part3_aux2")
    ap.add_argument("--out", default="results/time_handoff")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.step == "sim":
        sim()
    elif a.step == "twopart":
        run_twopart(a.cache, a.aux, a.out)
    elif a.step == "sim_twopart":
        sim_twopart()
    elif a.step == "followup":
        run_followup(a.cache, a.aux, a.out, a.limit, a.jobs)
    else:
        run(a.cache, a.aux, a.out, a.limit, a.jobs)
