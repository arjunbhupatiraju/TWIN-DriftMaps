"""STEP 2, DECOMPOSE: what drives neurons across the brain? Task vs movement vs slow drift, with valid tests.
(pre-registration: claude/preregistration-decompose.md)

Brain-wide movement-encoding studies fit encoding models in which body movements explain a large share of neural
variance in nearly every area. Movement and neurons both drift slowly within a session (engagement, arousal, satiety,
electrode motion), and two slowly drifting signals look related even when they are not. This script asks, for every
IBL session x region x trial epoch, how much neural variance is uniquely explained by
    TASK      signed and absolute contrast, choice, feedback, Bayes block prior, previous choice, previous feedback
    MOVEMENT  the 9 per-trial video and wheel features (same set as step 1), measured in the same window as the spikes
    DRIFT     slow cosine basis over the session (periods >= 200 trials, slower than the block structure); removed
              from neurons and predictors before the task/movement models; its share of neural variance is reported
and tests movement and task encoding two ways:
    standard  interleaved 5-fold CV, no drift term, null = trial shuffles of the movement (or task) predictors
    valid     contiguous 5-fold CV on drift-removed data, null = linear shifts of the predictors (Harris 2020),
              which keep each signal's own autocorrelation
On drift twins of the same neurons (phase-randomized, so no real link to movement or task) we measure how often each
test reports encoding that is not there.

  python3 part6/decompose.py features --limit 3     # needs internet; per-trial movement in 4 windows; deletes raw files
  python3 part6/decompose.py features
  python3 part6/decompose.py run --limit 3
  python3 part6/decompose.py run                    # parallel, resumable
  python3 part6/decompose.py test                   # gate, D1-D4 exactly as pre-registered
"""
from __future__ import annotations

import argparse, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker; the workers run in parallel instead
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "twin"))
import prior_map as PM          # noqa: E402  (_gk, _fisher, _bh)
import prior_movement as MV     # noqa: E402  (_wmean, _speed, _Get: the step-1 feature code)

OUT = HERE / "results_decompose"
FEAT = OUT / "features"
WINDOWS = {"prior_a": ("stimOn", -0.60, -0.10), "stim_a": ("stimOn", 0.00, 0.10),
           "choice_a": ("firstMovement", -0.10, 0.00), "feedback_a": ("feedback", 0.00, 0.20)}
FEATS = ("wheel_velocity", "wheel_speed", "wheel_displacement", "licks", "face_motion", "pupil",
         "paw_l_speed", "paw_r_speed", "nose_speed")
VIDEO = ("face_motion", "pupil", "paw_l_speed", "paw_r_speed", "nose_speed")
ALPHAS = np.logspace(-3, 2, 9)


# ================================================================ features (needs internet)
def features_one(one, path, keep_raw=False):
    from gatekeeper.mv_data import MVSession
    s = MVSession(Path(path))
    out = FEAT / f"{s.sid}.npz"
    if out.exists():
        return "skip"
    tr = one.load_object(s.sid, "trials", collection="alf")
    on_all = np.asarray(tr["stimOn_times"], float)
    keep = np.isfinite(on_all)
    if keep.sum() != len(s.contrast):
        np.savez(out, ok=False)
        return "trial count mismatch"
    ev = {"stimOn": on_all[keep], "firstMovement": np.asarray(tr["firstMovement_times"], float)[keep],
          "feedback": np.asarray(tr["feedback_times"], float)[keep]}
    g = MV._Get(one, s.sid, keep_raw); got = {}

    def put(name, fn):
        for w, (e, lo, hi) in WINDOWS.items():
            t = ev[e]; ok = np.isfinite(t)
            x = np.full(len(t), np.nan)
            try:
                x[ok] = fn(t[ok], (lo, hi))
            except Exception:
                continue
            got[f"{w}__{name}"] = x
    try:
        try:
            ts = np.asarray(g("_ibl_wheel.timestamps.npy", "wheel.timestamps.npy"), float)
            pos = np.asarray(g("_ibl_wheel.position.npy", "wheel.position.npy"), float)
            for k in ("wheel_velocity", "wheel_speed", "wheel_displacement"):
                put(k, lambda t, w, k=k: MV.wheel_feats(ts, pos, t, w)[k])
        except Exception:
            pass
        try:
            lk = np.sort(np.asarray(g("licks.times.npy"), float))
            put("licks", lambda t, w: (np.searchsorted(lk, t + w[1]) - np.searchsorted(lk, t + w[0])).astype(float))
        except Exception:
            pass
        try:
            tc = np.asarray(g("_ibl_leftCamera.times.npy", "leftCamera.times.npy"), float)
            try:
                me = np.asarray(g("leftCamera.ROIMotionEnergy.npy"), float)
                put("face_motion", lambda t, w: MV._wmean(tc, me, t, w))
            except Exception:
                pass
            try:
                f = g("_ibl_leftCamera.features.pqt", "leftCamera.features.pqt")
                pu = np.array(f["pupilDiameter_smooth" if "pupilDiameter_smooth" in f else "pupilDiameter_raw"], float)
                put("pupil", lambda t, w: MV._wmean(tc, pu, t, w))
            except Exception:
                pass
            try:
                df = g("_ibl_leftCamera.lightningPose.pqt", "leftCamera.lightningPose.pqt",
                       "_ibl_leftCamera.dlc.pqt", "leftCamera.dlc.pqt")
                for name, part in (("paw_l_speed", "paw_l"), ("paw_r_speed", "paw_r"), ("nose_speed", "nose_tip")):
                    try:
                        sp = MV._speed(df, part, tc)
                        put(name, lambda t, w, sp=sp: MV._wmean(tc, sp, t, w))
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
    finally:
        g.clean()
    if not got:
        return "no features (not saved; retried on the next pass)"
    np.savez(out, ok=True, **got)
    return " ".join(k for k in FEATS if f"prior_a__{k}" in got)


def cmd_features(a):
    from one.api import ONE
    PM._gk(a.gatekeeper)
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab",
              password="international", silent=True)
    FEAT.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(a.cache).expanduser().glob("*.npz"))[: a.limit or None]
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            msg = features_one(one, p, a.keep_raw)
        except Exception as e:
            msg = f"FAILED ({type(e).__name__}: {e})"
        print(f"[{i + 1}/{len(paths)}] {p.stem}  {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ================================================================ predictors
def movement(sid, window, v, robust=False):
    f = FEAT / f"{sid}.npz"
    if not f.exists():
        return None, []
    z = np.load(f)
    if not bool(z["ok"]):
        return None, []
    cols, names = [], []
    for k in FEATS:
        key = f"{window}__{k}"
        if key in z.files and len(z[key]) == len(v):
            x = np.asarray(z[key], float)[v]
            if np.isfinite(x).mean() < 0.8:
                continue
            x = np.where(np.isfinite(x), x, np.nanmedian(x))
            if k in ("wheel_speed", "licks", "face_motion") or k.endswith("_speed"):
                x = np.log1p(np.maximum(x, 0))
            if robust:                                         # Amendment 2: tame rare extreme trials
                if k in ("wheel_velocity", "wheel_displacement"):
                    x = np.sign(x) * np.log1p(np.abs(x))
                lo, hi = np.percentile(x, [1, 99])
                x = np.clip(x, lo, hi)
            if x.std() > 0:
                cols.append(x); names.append(k)
    return (np.column_stack(cols) if cols else None), names


def task(s, v):
    from twin.task import BlockTask
    c = np.asarray(s.contrast, float)
    side = np.where(c != 0, (c > 0).astype(float), np.nan)
    prior = BlockTask().bayes_prior(side)
    ch = 2 * (np.asarray(s.choice, float) > 0) - 1.0
    fb = np.sign(np.asarray(s.feedback, float))
    cols = [c, np.abs(c), ch, fb, prior, np.r_[0, ch[:-1]], np.r_[0, fb[:-1]]]
    return np.column_stack(cols)[v]


def drift(T):
    K = max(1, int(T // 100))                       # cosine periods 2T/k >= 200 trials
    t = (np.arange(T) + 0.5) / T
    return np.column_stack([np.cos(np.pi * k * t) for k in range(1, K + 1)])


# ================================================================ encoding model
def cv_r2(P, Y, contiguous=True, k=5, seed=0):
    """Population held-out R^2 of a ridge encoding model P -> Y (Y z-scored, neurons weighted equally). Ridge strength
    chosen per neuron by GCV inside each training fold. P may have 0 columns (intercept-only model, R^2 about 0)."""
    T = len(Y)
    if contiguous:
        e = np.linspace(0, T, k + 1).astype(int); folds = [np.arange(a, b) for a, b in zip(e[:-1], e[1:])]
    else:
        perm = np.random.default_rng(seed).permutation(T); folds = np.array_split(perm, k)
    sse = sst = 0.0
    for te in folds:
        tr = np.setdiff1d(np.arange(T), te)
        ym = Y[tr].mean(0); Ytr = Y[tr] - ym; Yte = Y[te] - ym
        sst += (Yte ** 2).sum()
        if P.shape[1] == 0:
            sse += (Yte ** 2).sum(); continue
        mu, sd = P[tr].mean(0), P[tr].std(0)
        keep = sd > 0
        Ptr = (P[tr][:, keep] - mu[keep]) / sd[keep]; Pte = (P[te][:, keep] - mu[keep]) / sd[keep]
        if Ptr.shape[1] == 0:
            sse += (Yte ** 2).sum(); continue
        U, sv, Vt = np.linalg.svd(Ptr, full_matrices=False)
        Uy = U.T @ Ytr; n = len(tr)
        tot = (Ytr ** 2).sum(0); proj = (Uy ** 2).sum(0)
        PV = Pte @ Vt.T
        best_g = np.full(Y.shape[1], np.inf); best = np.zeros_like(Yte)
        for al in ALPHAS * n:
            f = sv ** 2 / (sv ** 2 + al)
            rss = tot - proj + (((1 - f) ** 2)[:, None] * Uy ** 2).sum(0)
            gcv = n * rss / max(n - f.sum(), 1e-9) ** 2
            pred = PV @ ((sv / (sv ** 2 + al))[:, None] * Uy)
            take = gcv < best_g
            best[:, take] = pred[:, take]; best_g = np.where(take, gcv, best_g)
        sse += ((Yte - best) ** 2).sum()
    return float(1 - sse / sst)


def detrend(A, D):
    """Remove the slow cosine components (and the mean) from every column, using the whole session."""
    B = np.column_stack([np.ones(len(A)), D])
    return A - B @ np.linalg.lstsq(B, A, rcond=None)[0]


def effects(Y, Tk, M, D):
    """Effect sizes only (no nulls, no twins): the quantities D2 is computed from. Same definitions as tests()."""
    Yd, Td, Md = detrend(Y, D), detrend(Tk, D), detrend(M, D)
    full = cv_r2(np.column_stack([Td, Md]), Yd)
    return dict(R2_full=full, dT=full - cv_r2(Md, Yd), dM=full - cv_r2(Td, Yd),
                dM_nodrift=cv_r2(np.column_stack([Tk, M]), Y) - cv_r2(Tk, Y),
                dM_std=cv_r2(np.column_stack([Tk, M]), Y, contiguous=False) - cv_r2(Tk, Y, contiguous=False))


def _effects_session(path, gk_root, robust, seed):
    """Same sessions, regions, trials, units (same random subsample) as the registered run, effect sizes only."""
    PM._gk(gk_root)
    from gatekeeper.mv_data import MVSession
    rng = np.random.default_rng(seed)
    s = MVSession(Path(path)); rows = []
    for w in WINDOWS:
        e = WINDOWS[w][0]
        v = (np.asarray(s.choice) >= 0) & np.asarray(s.valid[e], bool)
        M, names = movement(s.sid, w, v, robust)
        T = int(v.sum())
        if M is None or T < 300:
            continue
        Tk, D = task(s, v), drift(T)
        for r in s.regions:
            X = s.X(r, w)
            if X is None:
                continue
            X = np.sqrt(np.maximum(np.asarray(X, float)[v], 0)); X = X[:, X.std(0) > 0]
            if X.shape[1] < 10:
                continue
            if X.shape[1] > 150:
                X = X[:, rng.choice(X.shape[1], 150, replace=False)]
            Y = (X - X.mean(0)) / X.std(0)
            row = dict(session=s.sid, region=r, window=w, T=T, N=X.shape[1])
            row.update(effects(Y, Tk, M, D)); rows.append(row)
    return rows


def cmd_effects(a):
    """Amendment 2, fast path: recompute only the effect sizes behind D2 (minutes, not hours) and evaluate D2 with the
    registered rule. D1 and D3 need the nulls and twins, so they stay from the registered run."""
    import pandas as pd
    robust = a.tag == "robust"
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz"))[: a.limit or None] if (FEAT / f"{p.stem}.npz").exists()]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_effects_session, str(p), a.gatekeeper, robust, 9000 + i): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows += fu.result()
            except Exception as e:
                print(f"{futs[fu].stem}: ERROR {e}", flush=True)
            if (k + 1) % 50 == 0:
                print(f"  {k + 1}/{len(paths)} sessions", flush=True)
    D = pd.DataFrame(rows)
    sfx = f"_{a.tag}" if a.tag else "_plain"
    D.to_csv(OUT / f"decompose_effects{sfx}.csv", index=False)
    rng = np.random.default_rng(0)
    share = lambda g: 1 - g.dM.sum() / g.dM_std.sum() if g.dM_std.sum() > 0 else np.nan
    nd = lambda g: 1 - g.dM.sum() / g.dM_nodrift.sum() if g.dM_nodrift.sum() > 0 else np.nan
    regs = D.region.unique(); byreg = {r: g for r, g in D.groupby("region")}
    bt = [share(pd.concat([byreg[regs[i]] for i in rng.integers(0, len(regs), len(regs))])) for _ in range(a.boot)]
    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
    L = [f"# D2, effect sizes only ({'robust movement predictors, Amendment 2' if robust else 'registered predictors'}): "
         f"{len(D)} session-region-windows, {D.session.nunique()} sessions",
         f"  valid movement unique R2: mean {D.dM.mean():+.4f}, positive in {np.mean(D.dM > 0):.0%}; "
         f"standard: mean {D.dM_std.mean():+.4f}, positive in {np.mean(D.dM_std > 0):.0%}",
         f"  D2 share = 1 - valid/standard: {share(D):.3f} (95% CI {lo:.3f}-{hi:.3f}; criterion lower > 0.10)  "
         f"D2 {'PASS' if lo > 0.10 else 'FAIL'}" + ("" if 0 <= share(D) <= 1 else "  (OUT OF RANGE: not readable as a share)"),
         f"  drift alone (contiguous CV, drift left in vs removed): {nd(D):.3f}"]
    for w, g in D.groupby("window"):
        L.append(f"    {w}: share {share(g):.3f}; valid R2 {g.dM.mean():+.4f}, standard {g.dM_std.mean():+.4f}, "
                 f"task {g.dT.mean():+.4f}")
    print("\n".join(L))
    (OUT / f"D2_EFFECTS{sfx}.md").write_text("\n".join(L) + "\n")


def tests(Y, Tk, M, D, S, n_shuf, rng):
    """Unique variance and both tests for one population. Drift is handled by removing the slow components from the
    neurons and from every predictor (no extrapolation of slow regressors into held-out blocks); linear shifts use the
    central trials [S, T-S) of the detrended data."""
    T = len(Y); c = slice(S, T - S)
    Yd, Td, Md = detrend(Y, D), detrend(Tk, D), detrend(M, D)
    full = cv_r2(np.column_stack([Td, Md]), Yd)
    out = dict(R2_full=full,
               dT=full - cv_r2(Md, Yd),
               dM=full - cv_r2(Td, Yd),
               dD=float(1 - (Yd ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()))      # variance in slow components
    out["dM_nodrift"] = cv_r2(np.column_stack([Tk, M]), Y) - cv_r2(Tk, Y)     # contiguous CV, drift left in
    # standard test: interleaved CV, drift left in, trial-shuffle null
    base_s = cv_r2(Tk, Y, contiguous=False)
    out["dM_std"] = cv_r2(np.column_stack([Tk, M]), Y, contiguous=False) - base_s
    nul = [cv_r2(np.column_stack([Tk, M[rng.permutation(T)]]), Y, contiguous=False) - base_s for _ in range(n_shuf)]
    out["p_std_M"] = PM._p(out["dM_std"], nul)
    # valid test: contiguous CV on detrended data, linear-shift null (movement, then task)
    Yc, Tc, Mc = Yd[c], Td[c], Md[c]
    sh = [s for s in range(-S, S + 1, 2) if s != 0]                     # every 2nd shift (S even): 50 nulls
    base_v = cv_r2(Tc, Yc)
    obs = cv_r2(np.column_stack([Tc, Mc]), Yc) - base_v
    nul = [cv_r2(np.column_stack([Tc, Md[S + s:T - S + s]]), Yc) - base_v for s in sh]
    out["p_valid_M"] = PM._p(obs, nul)
    base_t = cv_r2(Mc, Yc)
    obs = cv_r2(np.column_stack([Tc, Mc]), Yc) - base_t
    nul = [cv_r2(np.column_stack([Td[S + s:T - S + s], Mc]), Yc) - base_t for s in sh]
    out["p_valid_T"] = PM._p(obs, nul)
    return out


# ================================================================ one session
def _session(path, gk_root, windows, min_units, max_units, n_shuf, max_shift, n_twin, seed, robust=False):
    PM._gk(gk_root)
    from gatekeeper.mv_data import MVSession
    from twin.drift import phase_twin
    rng = np.random.default_rng(seed)
    s = MVSession(Path(path))
    rows = []
    for w in windows:
        e = WINDOWS[w][0]
        v = (np.asarray(s.choice) >= 0) & np.asarray(s.valid[e], bool)
        M, names = movement(s.sid, w, v, robust)
        T = int(v.sum())
        if M is None or T < 300:
            continue
        Tk, D = task(s, v), drift(T)
        S = 2 * (min(max_shift, T // 6) // 2)
        for r in s.regions:
            X = s.X(r, w)
            if X is None:
                continue
            X = np.sqrt(np.maximum(np.asarray(X, float)[v], 0)); X = X[:, X.std(0) > 0]
            if X.shape[1] < min_units:
                continue
            if X.shape[1] > max_units:
                X = X[:, rng.choice(X.shape[1], max_units, replace=False)]
            Y = (X - X.mean(0)) / X.std(0)
            row = dict(session=s.sid, subject=s.subject, lab=s.lab, region=r, window=w, T=T, N=X.shape[1],
                       n_feat=len(names), video=int(any(k in VIDEO for k in names)), feats="|".join(names))
            row.update(tests(Y, Tk, M, D, S, n_shuf, rng))
            fs = fv = 0                                             # drift twins: no real link to movement or task
            for _ in range(n_twin):
                Yt = phase_twin(Y, rng); Yt = (Yt - Yt.mean(0)) / Yt.std(0)
                o = tests(Yt, Tk, M, D, S, n_shuf, rng)
                fs += o["p_std_M"] < 0.05; fv += o["p_valid_M"] < 0.05
            row.update(twin_fp_std=fs / max(n_twin, 1), twin_fp_valid=fv / max(n_twin, 1), n_twin=n_twin)
            rows.append(row)
    return rows


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    rows_f = OUT / f"decompose_rows{'_' + a.tag if a.tag else ''}.csv"
    done = set(pd.read_csv(rows_f).session) if rows_f.exists() else set()
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz"))[: a.limit or None]
             if p.stem not in done and (FEAT / f"{p.stem}.npz").exists()]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.gatekeeper, list(WINDOWS), 10, 150, a.shuffles, a.max_shift, a.twins,
                          9000 + i, a.tag == "robust"): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True)
                continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} region-windows", flush=True)


# ================================================================ test (pre-registered)
def _systems(acr):
    try:
        from iblatlas.regions import BrainRegions
        return list(BrainRegions().acronym2acronym(np.asarray(acr), mapping="Cosmos"))
    except Exception:
        return ["all"] * len(acr)


def cmd_test(a):
    import pandas as pd
    sfx = f"_{a.tag}" if a.tag else ""
    D = pd.read_csv(OUT / f"decompose_rows{sfx}.csv")
    rng = np.random.default_rng(0); B = a.boot
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    P(f"# DECOMPOSE: {D.region.nunique()} regions, {len(D)} session-region-windows, {D.session.nunique()} sessions, "
      f"{D.subject.nunique()} mice, {D.lab.nunique()} labs\n")

    # ---- gate F2
    n_cache = len(list(Path(a.cache).expanduser().glob("*.npz")))
    cov = D.session.nunique() / max(n_cache, 1); vid = float(D.groupby("session").video.max().mean())
    F2 = cov >= 0.70 and vid >= 0.70
    P("## Gate F2 (coverage)")
    P(f"  sessions analysed: {D.session.nunique()} of {n_cache} ({cov:.0%}); with a video feature: {vid:.0%} "
      f"(criterion both >= 70%)  Gate F2 {'PASS' if F2 else 'FAIL'}\n")

    # ---- D1: validity on drift twins
    sess = [g for _, g in D.groupby("session")]
    rate = lambda g, c: float((g[c] * g.n_twin).sum() / max(g.n_twin.sum(), 1))
    fv, fs = rate(D, "twin_fp_valid"), rate(D, "twin_fp_std")
    bs = [pd.concat([sess[i] for i in rng.integers(0, len(sess), len(sess))]) for _ in range(B)]
    hv = float(np.percentile([rate(g, "twin_fp_valid") for g in bs], 97.5))
    ls = float(np.percentile([rate(g, "twin_fp_std") for g in bs], 2.5))
    D1a = fv <= 0.07 and hv <= 0.08
    D1b = ls >= 0.20
    P("## D1 (movement-encoding false positives on drift twins: neurons with no real link to movement)")
    P(f"  valid test: {fv:.3f} (95% CI upper {hv:.3f}; criterion <= 0.07 and upper <= 0.08)  D1a {'PASS' if D1a else 'FAIL'}")
    P(f"  standard test: {fs:.3f} (95% CI lower {ls:.3f}; criterion lower >= 0.20)  D1b {'PASS' if D1b else 'FAIL'}")
    for w, g in D.groupby("window"):
        P(f"    {w}: valid {rate(g, 'twin_fp_valid'):.3f}, standard {rate(g, 'twin_fp_std'):.3f}")
    P("")

    # ---- D2: how much apparent movement encoding is shared with slow drift?
    regs = D.region.unique(); byreg = {r: g for r, g in D.groupby("region")}
    share = lambda g: 1 - g.dM.sum() / g.dM_std.sum() if g.dM_std.sum() > 0 else np.nan
    infl = lambda g: g.dM_std.sum() / g.dM.sum() if g.dM.sum() > 0 else np.nan
    bt = []
    for _ in range(B):
        g = pd.concat([byreg[regs[i]] for i in rng.integers(0, len(regs), len(regs))])
        bt.append((share(g), infl(g)))
    bt = np.array(bt, float)
    sh, (slo, shi) = share(D), np.nanpercentile(bt[:, 0], [2.5, 97.5])
    D2 = slo > 0.10
    P("## D2 (share of the movement encoding found by the standard analysis that does not survive the valid one)")
    P(f"  1 - (valid movement unique R2) / (standard movement unique R2): {sh:.3f} (95% CI {slo:.3f}-{shi:.3f}; criterion lower > 0.10)"
      f"  D2 {'PASS' if D2 else 'FAIL'}" + ("" if (F2 and D1a) else " (NOT INTERPRETED: a validity check failed)"))
    for w, g in D.groupby("window"):
        P(f"    {w}: {share(g):.3f}")
    P(f"  descriptive: standard movement R2 (interleaved CV, no drift) / valid movement R2 = {infl(D):.2f} "
      f"(95% CI {np.nanpercentile(bt[:, 1], 2.5):.2f}-{np.nanpercentile(bt[:, 1], 97.5):.2f})")
    nd = lambda g: 1 - g.dM.sum() / g.dM_nodrift.sum() if g.dM_nodrift.sum() > 0 else np.nan
    P(f"  descriptive: drift alone (same contiguous CV, drift left in vs removed): {nd(D):.3f}\n")

    # ---- D3: brain-wide map (descriptive)
    R = D.groupby(["window", "region"]).agg(n_sess=("session", "nunique"), dT=("dT", "mean"), dM=("dM", "mean"),
                                            dD=("dD", "mean"), dM_std=("dM_std", "mean"), R2_full=("R2_full", "mean"),
                                            p_std=("p_std_M", PM._fisher), p_M=("p_valid_M", PM._fisher),
                                            p_T=("p_valid_T", PM._fisher)).reset_index()
    for c in ("p_std", "p_M", "p_T"):
        R["sig_" + c[2:]] = False
        for w, idx in R.groupby("window").groups.items():
            R.loc[idx, "sig_" + c[2:]] = PM._bh(R.loc[idx, c].values, 0.05)
    P("## D3 (descriptive: regions with significant encoding, Fisher across sessions, BH q = 0.05)")
    for w, g in R.groupby("window"):
        P(f"  {w}: movement, standard {g.sig_std.mean():.2f}; movement, valid {g.sig_M.mean():.2f}; "
          f"task, valid {g.sig_T.mean():.2f} ({len(g)} regions)")
    R["system"] = _systems(R.region.values)
    S = R.groupby(["window", "system"])[["dT", "dM", "dD"]].mean()
    P("  mean unique R2 by brain system (task / movement / share of variance in slow drift):")
    for (w, sy), x in S.iterrows():
        P(f"    {w:10s} {sy:10s} {x.dT:+.4f} / {x.dM:+.4f} / {x.dD:+.4f}")

    # ---- D4: link to step 1 (descriptive)
    mv = HERE / "results_movement" / "movement_regions.csv"
    if mv.exists():
        from scipy.stats import spearmanr
        M1 = pd.read_csv(mv, index_col=0)
        g = R[R.window == "prior_a"].set_index("region").join(M1[["p_carries_move"]], how="inner")
        if len(g) > 5:
            rho, p = spearmanr(g.dM, g.p_carries_move)
            P(f"\n## D4 (descriptive): regions with more pre-stimulus movement encoding vs P(carries the prior after "
              f"movement removal): Spearman {rho:+.3f} (p = {p:.3g}, {len(g)} regions)")
    R.to_csv(OUT / f"decompose_regions{sfx}.csv", index=False)
    (OUT / f"DECOMPOSE{sfx}.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/DECOMPOSE{sfx}.md")


def cmd_diagnose(a):
    """EXPLORATORY (Amendment 2): are negative valid movement R2 values driven by rare extreme movement trials?
    For every analysed session-window: the largest |z| of any movement feature, vs the valid unique movement R2."""
    import pandas as pd
    from scipy.stats import spearmanr
    PM._gk(a.gatekeeper)
    from gatekeeper.mv_data import MVSession
    D = pd.read_csv(OUT / "decompose_rows.csv")
    G = D.groupby(["session", "window"]).agg(dM=("dM", "mean"), dM_std=("dM_std", "mean")).reset_index()
    cache = Path(a.cache).expanduser(); zmax, worst = [], []
    for sid, g in G.groupby("session"):
        s = MVSession(cache / f"{sid}.npz")
        for i, row in g.iterrows():
            e = WINDOWS[row.window][0]
            v = (np.asarray(s.choice) >= 0) & np.asarray(s.valid[e], bool)
            M, names = movement(sid, row.window, v)
            Z = np.abs((M - M.mean(0)) / M.std(0))
            zmax.append((i, float(Z.max()))); worst.append((i, names[int(Z.max(0).argmax())]))
    G.loc[[i for i, _ in zmax], "zmax"] = [z for _, z in zmax]
    G.loc[[i for i, _ in worst], "worst"] = [w for _, w in worst]
    rho, p = spearmanr(G.zmax, G.dM)
    print(f"session-windows: {len(G)}; Spearman(largest |z| of a movement feature, valid movement R2) = {rho:+.3f} (p = {p:.2g})")
    G["bin"] = pd.qcut(G.zmax, 4, labels=["lowest", "2nd", "3rd", "highest"])
    print(G.groupby("bin", observed=True)[["zmax", "dM", "dM_std"]].mean().round(4).to_string())
    print("\nby window (mean largest |z|, mean valid movement R2):")
    print(G.groupby("window")[["zmax", "dM"]].mean().round(4).to_string())
    print("\nfeature with the most extreme trial, how often:")
    print(G.worst.value_counts().to_string())


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features"); f.add_argument("--keep-raw", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--shuffles", type=int, default=100); r.add_argument("--max-shift", type=int, default=50)
    r.add_argument("--twins", type=int, default=3)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t = sub.add_parser("test"); t.add_argument("--boot", type=int, default=500)
    dg = sub.add_parser("diagnose")
    ef = sub.add_parser("effects"); ef.add_argument("--boot", type=int, default=500)
    ef.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    for q in (r, t, ef):
        q.add_argument("--tag", default="", help="'robust' = Amendment 2 run (signed log + 1st-99th percentile clipping of movement)")
    for q in (f, r, t, dg, ef):
        q.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
        q.add_argument("--gatekeeper"); q.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    {"features": cmd_features, "run": cmd_run, "test": cmd_test, "diagnose": cmd_diagnose, "effects": cmd_effects}[a.cmd](a)


if __name__ == "__main__":
    main()
