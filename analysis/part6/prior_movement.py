"""PRIOR MAP, step 1: is the hidden block prior just movement?   (pre-registration: claude/preregistration-prior-movement.md)

Mice move in anticipation of the side they expect, and movement drives neurons all over the brain (Musall 2019,
Stringer 2019, Wang ... Druckmann 2025). So part of what PRIOR MAP called "prior code" could be movement. This script
measures movement in the same pre-stimulus window (stimOn -600 to -100 ms) and repeats the PRIOR MAP analysis with
movement removed from the neurons and from every target (real, pseudo-session and planted), so the valid null stays exact.

  python3 part6/prior_movement.py features --limit 3   # needs internet; downloads small files per session, keeps
  python3 part6/prior_movement.py features             #   only per-trial window means, deletes the raw files
  python3 part6/prior_movement.py wheel                # Amendment 2: redo wheel features (wheel trace interpolated)
  python3 part6/prior_movement.py run --limit 3        # quick check
  python3 part6/prior_movement.py run                  # parallel, resumable
  python3 part6/prior_movement.py test                 # gate, G1, V1, M1-M3 exactly as pre-registered

Movement features, per trial, mean over the window (all from public IBL datasets; each kept only if >= 80% of
trials have a value):
  wheel_velocity (signed), wheel_speed, wheel_displacement (signed), licks,
  face_motion (left camera motion energy), body_motion (body camera motion energy), pupil,
  paw_l_speed, paw_r_speed, nose_speed, tongue_speed (left camera pose tracking, DLC or Lightning Pose)
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "twin"))
import prior_map as PM  # noqa: E402  (same decoder, nulls, grid and estimator as the registered PRIOR MAP)

OUT = HERE / "results_movement"
FEAT = OUT / "features"
WIN = (-0.60, -0.10)
FEATS = ("wheel_velocity", "wheel_speed", "wheel_displacement", "licks", "face_motion", "body_motion", "pupil",
         "paw_l_speed", "paw_r_speed", "nose_speed", "tongue_speed")
VIDEO = ("face_motion", "body_motion", "pupil", "paw_l_speed", "paw_r_speed", "nose_speed", "tongue_speed")
DETECTED_REGISTERED = 0.071          # fraction of regions detected in the registered PRIOR MAP run (H2, q = 0.05)


# ================================================================ features (needs internet)
def _wmean(t, d, onsets, win=WIN):
    n = min(len(t), len(d)); t = np.asarray(t[:n], float); d = np.asarray(d[:n], float)
    out = np.full(len(onsets), np.nan)
    lo = np.searchsorted(t, onsets + win[0]); hi = np.searchsorted(t, onsets + win[1])
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = np.nanmean(d[a:b]) if np.isfinite(d[a:b]).any() else np.nan
    return out


def wheel_feats(ts, pos, onsets, win=WIN, n=251):
    """Wheel velocity (signed), speed and displacement over the window. The IBL wheel trace is sampled only when the
    wheel moves, so a window with no samples means the wheel was still: the trace is interpolated onto a 2 ms grid
    (as in ibllib) instead of averaging the raw samples, which would leave still trials empty."""
    ts = np.asarray(ts, float); pos = np.asarray(pos, float)
    m = min(len(ts), len(pos)); ts, pos = ts[:m], pos[:m]
    on = np.asarray(onsets, float); dur = win[1] - win[0]
    out = {k: np.full(len(on), np.nan) for k in ("wheel_velocity", "wheel_speed", "wheel_displacement")}
    ok = np.isfinite(on)
    if not ok.any():
        return out
    grid = on[ok, None] + np.linspace(win[0], win[1], n)[None, :]
    p = np.interp(grid.ravel(), ts, pos).reshape(grid.shape)
    disp = p[:, -1] - p[:, 0]
    out["wheel_displacement"][ok] = disp
    out["wheel_velocity"][ok] = disp / dur
    out["wheel_speed"][ok] = np.abs(np.diff(p, axis=1)).sum(1) / dur
    return out


def _speed(df, part, t, lik=0.9):
    n = min(len(df), len(t))                                   # camera times and pose rows can differ by a few frames
    x, y = (np.array(df[f"{part}_{c}"], float)[:n] for c in "xy")   # np.array copies: pandas may hand back read-only
    if f"{part}_likelihood" in df:
        bad = np.array(df[f"{part}_likelihood"], float)[:n] < lik
        x[bad] = np.nan; y[bad] = np.nan
    dt = np.maximum(np.diff(np.asarray(t[:n], float)), 1e-4)
    return np.r_[np.nan, np.hypot(np.diff(x), np.diff(y)) / dt]


class _Get:
    """Download one dataset, load it, and (unless --keep-raw) delete the downloaded file afterwards."""
    def __init__(self, one, eid, keep):
        self.one, self.eid, self.keep, self.files = one, eid, keep, []

    def __call__(self, *names, collection="alf"):
        import pandas as pd
        for nm in names:
            try:
                p = Path(self.one.load_dataset(self.eid, nm, collection=collection, download_only=True))
            except Exception:
                continue
            self.files.append(p)
            return pd.read_parquet(p) if p.suffix == ".pqt" else np.load(p, allow_pickle=False)
        raise FileNotFoundError(names[0])

    def clean(self):
        if not self.keep:
            for p in self.files:
                try:
                    p.unlink()
                except Exception:
                    pass


def features_one(one, path, keep_raw=False):
    from gatekeeper.mv_data import MVSession
    s = MVSession(Path(path))
    out = FEAT / f"{s.sid}.npz"
    if out.exists():
        return "skip"
    tr = one.load_object(s.sid, "trials", collection="alf")
    on_all = np.asarray(tr["stimOn_times"], float)
    on = on_all[np.isfinite(on_all)]
    if len(on) != len(s.contrast):
        np.savez(out, ok=False, reason="trial count mismatch")
        return "trial count mismatch"
    g = _Get(one, s.sid, keep_raw); got = {}
    try:
        try:
            ts, pos = g("_ibl_wheel.timestamps.npy", "wheel.timestamps.npy"), g("_ibl_wheel.position.npy", "wheel.position.npy")
            got.update(wheel_feats(ts, pos, on))
        except Exception:
            pass
        try:
            lk = np.sort(np.asarray(g("licks.times.npy"), float))
            got["licks"] = (np.searchsorted(lk, on + WIN[1]) - np.searchsorted(lk, on + WIN[0])).astype(float)
        except Exception:
            pass
        try:
            t = np.asarray(g("_ibl_leftCamera.times.npy", "leftCamera.times.npy"), float)
            try:
                got["face_motion"] = _wmean(t, g("leftCamera.ROIMotionEnergy.npy"), on)
            except Exception:
                pass
            try:
                f = g("_ibl_leftCamera.features.pqt", "leftCamera.features.pqt")
                col = "pupilDiameter_smooth" if "pupilDiameter_smooth" in f else "pupilDiameter_raw"
                got["pupil"] = _wmean(t, np.asarray(f[col], float), on)
            except Exception:
                pass
            try:
                df = g("_ibl_leftCamera.lightningPose.pqt", "leftCamera.lightningPose.pqt",
                       "_ibl_leftCamera.dlc.pqt", "leftCamera.dlc.pqt")
                for name, part in (("paw_l_speed", "paw_l"), ("paw_r_speed", "paw_r"), ("nose_speed", "nose_tip"),
                                   ("tongue_speed", "tongue_end_l")):
                    try:
                        got[name] = _wmean(t, _speed(df, part, t), on)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass
        try:
            tb = np.asarray(g("_ibl_bodyCamera.times.npy", "bodyCamera.times.npy"), float)
            got["body_motion"] = _wmean(tb, g("bodyCamera.ROIMotionEnergy.npy"), on)
        except Exception:
            pass
    finally:
        g.clean()
    got = {k: v for k, v in got.items() if len(v) == len(on) and np.isfinite(v).mean() >= 0.8}
    if not got:                                                # nothing saved, so a network failure is retried next time
        return "no features (not saved; retried on the next pass)"
    np.savez(out, ok=True, **got)
    return " ".join(k for k in FEATS if k in got)


def cmd_wheel(a):
    """Amendment 2: recompute the three wheel features for sessions already extracted (small download per session)."""
    from one.api import ONE
    PM._gk(a.gatekeeper)
    from gatekeeper.mv_data import MVSession
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab",
              password="international", silent=True)
    files = sorted(FEAT.glob("*.npz"))[: a.limit or None]
    t0 = time.time()
    for i, f in enumerate(files):
        z = dict(np.load(f))
        if not bool(z["ok"]) or bool(z.get("wheel_v2", False)):
            continue
        try:
            s = MVSession(Path(a.cache).expanduser() / f.name)
            tr = one.load_object(s.sid, "trials", collection="alf")
            on_all = np.asarray(tr["stimOn_times"], float); on = on_all[np.isfinite(on_all)]
            g = _Get(one, s.sid, a.keep_raw)
            try:
                ts = g("_ibl_wheel.timestamps.npy", "wheel.timestamps.npy"); pos = g("_ibl_wheel.position.npy", "wheel.position.npy")
            finally:
                g.clean()
            for k, v in wheel_feats(ts, pos, on).items():
                if len(v) == len(s.contrast) and np.isfinite(v).mean() >= 0.8:
                    z[k] = v
            z["wheel_v2"] = True
            np.savez(f, **z)
            msg = " ".join(k for k in FEATS if k in z)
        except Exception as e:
            msg = f"FAILED ({type(e).__name__}: {e}); run the command again to retry"
        print(f"[{i + 1}/{len(files)}] {f.stem}  {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


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


# ================================================================ movement matrix and removal
def movement_matrix(sid, T):
    """Trials x features (z-scored, with intercept), or (None, []) if the session has no usable features."""
    f = FEAT / f"{sid}.npz"
    if not f.exists():
        return None, []
    z = np.load(f)
    if not bool(z["ok"]):
        return None, []
    cols, names = [], []
    for k in FEATS:
        if k in z.files and len(z[k]) == T:
            x = np.asarray(z[k], float)
            x = np.where(np.isfinite(x), x, np.nanmedian(x))
            if k in ("wheel_speed", "licks", "face_motion", "body_motion") or k.endswith("_speed"):
                x = np.log1p(np.maximum(x, 0))
            if x.std() > 0:
                cols.append((x - x.mean()) / x.std()); names.append(k)
    if not cols:
        return None, []
    return np.column_stack([np.ones(T)] + cols), names


def remove(A, M):
    """Remove from every column of A the part linearly explained by movement M (least squares)."""
    return A - M @ np.linalg.lstsq(M, A, rcond=None)[0]


# ================================================================ one session
def _session(path, gk_root, min_units, max_units, n_null, n_twin, n_plant, burn, seed):
    PM._gk(gk_root)
    from gatekeeper.mv_data import MVSession
    from twin.task import BlockTask
    from twin.drift import phase_twin
    from twin.check import plant
    rng = np.random.default_rng(seed)
    task = BlockTask()
    s = MVSession(Path(path))
    c = np.asarray(s.contrast, float)
    Tfull = len(c)
    Mfull, names = movement_matrix(s.sid, Tfull)
    if Mfull is None:
        return []
    side = np.where(c != 0, (c > 0).astype(float), np.nan)
    prior = task.bayes_prior(side)
    v = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]
    v[:burn] = False
    T = int(v.sum())
    if T < 150:
        return []
    M = Mfull[v]
    bank = [PM._pseudo(task, Tfull, rng) for _ in range(n_null + 1)]
    Zp = np.column_stack([b[0] for b in bank])[v]             # pseudo priors: column 0 is held out for G1
    y = prior[v]
    YZr = remove(np.column_stack([y, Zp]), M)                  # movement removed from every target
    yr, Zr = YZr[:, :1], YZr[:, 1:]
    # M3: how well does movement alone predict the prior? (same decoder, same pseudo-session null)
    sm = PM.cv_decode(M[:, 1:], np.column_stack([y, Zp[:, 1:]]))
    score_mov, p_mov = float(sm[0]), PM._p(sm[0], sm[1:])
    rows = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        X = np.asarray(X, float)[v]; X = X[:, X.std(0) > 0]
        if X.shape[1] < min_units:
            continue
        if X.shape[1] > max_units:
            X = X[:, rng.choice(X.shape[1], max_units, replace=False)]
        N = X.shape[1]
        raw = PM.cv_decode(X, np.column_stack([y, Zp[:, 1:]]))   # same session-region before removal (M2)
        p_raw = PM._p(raw[0], raw[1:])
        Xr = remove(X, M)
        sc = PM.cv_decode(Xr, np.column_stack([yr, Zr]))
        score, nul = sc[0], sc[2:]
        p_valid, p_h0 = PM._p(score, nul), PM._p(sc[1], nul)
        fp = 0                                                 # V1: drift twins (no prior code), same removal
        for _ in range(n_twin):
            st = PM.cv_decode(remove(phase_twin(X, rng), M), np.column_stack([yr, Zr[:, 1:]]))
            fp += PM._p(st[0], st[1:]) < 0.05
        curve, med = {}, {}                                    # power: plant a NEW pseudo prior, then remove movement
        for rr in PM.GRID:
            ps, ss = [], []
            for _ in range(n_plant):
                yp = PM._pseudo(task, Tfull, rng)[0][v]
                st = PM.cv_decode(remove(plant(X, yp, rr, 0.2, rng), M),
                                  np.column_stack([remove(yp[:, None], M), Zr[:, 1:]]))
                ps.append(round(PM._p(st[0], st[1:]), 4)); ss.append(st[0])
            curve[str(rr)] = ps; med[str(rr)] = float(np.median(ss))
        yp = PM._pseudo(task, Tfull, rng)[0][v]                # G1: one fresh planted code at r = 0.1
        st = PM.cv_decode(remove(plant(X, yp, 0.1, 0.2, rng), M), np.column_stack([remove(yp[:, None], M), Zr[:, 1:]]))
        p_h1 = PM._p(st[0], st[1:])
        ref = [float(np.median(nul))] + [med[str(rr)] for rr in PM.GRID]
        r_hat = float(PM.GRID0[int(np.argmin(np.abs(np.array(ref) - score)))])
        rows.append(dict(session=s.sid, subject=s.subject, lab=s.lab, region=r, variable="prior_dec_move", T=T, N=N,
                         n_feat=len(names), feats="|".join(names), video=int(any(k in VIDEO for k in names)),
                         score=float(score), null_mean=float(np.mean(nul)), p_valid=p_valid, score_raw=float(raw[0]),
                         p_raw=p_raw, score_mov=score_mov, p_mov=p_mov, p_h0=p_h0, p_h1=p_h1,
                         twin_fp_valid=fp / max(n_twin, 1), n_twin=n_twin, r_hat=r_hat, power_curve=json.dumps(curve)))
    return rows


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    rows_f = OUT / "movement_rows.csv"
    done = set(pd.read_csv(rows_f).session) if rows_f.exists() else set()
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz"))[: a.limit or None]
             if p.stem not in done and (FEAT / f"{p.stem}.npz").exists()]
    print(f"{len(paths)} sessions with movement features to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.gatekeeper, 10, 150, a.null, a.twins, a.plant, 90, 7000 + i): p
                for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True)
                continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} regions", flush=True)


# ================================================================ test (pre-registered)
def cmd_test(a):
    import pandas as pd
    D = pd.read_csv(OUT / "movement_rows.csv")
    rng = np.random.default_rng(0); B = a.boot
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    base_f = HERE / "results_prior" / (f"prior_rows_{a.baseline}.csv" if a.baseline else "prior_rows.csv")
    base = pd.read_csv(base_f) if base_f.exists() else None
    P(f"# PRIOR MAP with movement removed: {D.region.nunique()} regions, {len(D)} session-regions, "
      f"{D.session.nunique()} sessions, {D.subject.nunique()} mice, {D.lab.nunique()} labs\n")

    # ---- gate F: coverage
    n_base = base.session.nunique() if base is not None else None
    ok = []                                                    # feature names per usable session (files closed)
    for f in sorted(FEAT.glob("*.npz")):
        with np.load(f) as z:
            if bool(z["ok"]):
                ok.append(set(z.files))
    with_video = D.groupby("session").video.first()
    cov = D.session.nunique() / n_base if n_base else float("nan")
    vid = float(with_video.mean())
    F = cov >= 0.70 and vid >= 0.70
    P("## Gate F (movement coverage)")
    P(f"  sessions with movement features, of the {n_base} PRIOR MAP sessions: {D.session.nunique()} ({cov:.0%}); "
      f"with at least one video feature: {vid:.0%} (criterion both >= 70%)")
    cnt = {k: sum(k in z for z in ok) for k in FEATS}
    P("  feature availability (sessions): " + ", ".join(f"{k} {v}" for k, v in cnt.items()))
    P(f"  Gate F {'PASS' if F else 'FAIL: M1 is reported but not interpreted'}\n")

    # ---- G1 and V1: is the movement-removed test still valid?
    regs = sorted(D.region.unique())
    w0 = PM.em_pi(list(PM.region_logliks(D, "p_h0", rng).values()))
    pick = set(rng.choice(regs, int(round(0.3 * len(regs))), replace=False))
    w1 = PM.em_pi(list(PM.region_logliks(D.assign(p_mix=np.where(D.region.isin(pick), D.p_h1, D.p_h0)), "p_mix", rng).values()))
    g1a, g1b = PM.frac(w0, 0.04), PM.frac(w1, 0.04)
    G1 = g1a <= 0.10 and abs(g1b - 0.30) <= 0.10
    P("## G1 (estimator check after movement removal)")
    P(f"  no code anywhere: {g1a:.3f} (criterion <= 0.10); code in 30% of regions: {g1b:.3f} (criterion 0.20-0.40)")
    P(f"  G1 {'PASS' if G1 else 'FAIL: M1 is reported but not interpreted'}\n")
    tw = D.assign(k=D.twin_fp_valid * D.n_twin)
    sess = [g for _, g in tw.groupby("session")]
    fv = tw.k.sum() / max(tw.n_twin.sum(), 1)
    bv = [pd.concat([sess[i] for i in rng.integers(0, len(sess), len(sess))]) for _ in range(B)]
    hi = float(np.percentile([g.k.sum() / max(g.n_twin.sum(), 1) for g in bv], 97.5))
    V1 = fv <= 0.07 and hi <= 0.08
    P("## V1 (false positives of the movement-removed test on drift twins)")
    P(f"  {fv:.3f} (95% CI upper {hi:.3f}; criterion <= 0.07 and upper <= 0.08)  V1 {'PASS' if V1 else 'FAIL'}\n")

    # ---- M1 (confirmatory): does the hidden prior survive movement removal?
    LL = PM.region_logliks(D, "p_valid", rng)
    w = PM.em_pi([LL[r] for r in regs])
    est = {t: PM.frac(w, t) for t in (0.02, 0.04, 0.1)}
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(regs), len(regs))
        wb = PM.em_pi([LL[regs[i]] for i in idx], iters=150)
        boots.append({t: PM.frac(wb, t) for t in est})
    ci = {t: tuple(np.percentile([b[t] for b in boots], [2.5, 97.5])) for t in est}
    lo, hi4 = ci[0.04]
    M1 = lo > DETECTED_REGISTERED
    valid_all = F and G1 and V1
    P("## M1 (confirmatory: power-corrected fraction of regions carrying the prior, movement removed)")
    P(f"  estimated distribution of code size: {dict(zip(PM.GRID0, np.round(w, 3)))}")
    for t in est:
        P(f"  fraction with r >= {t}: {est[t]:.3f} (95% CI {ci[t][0]:.3f}-{ci[t][1]:.3f})")
    P(f"  criterion: lower bound of r >= 0.04 above the registered detected fraction {DETECTED_REGISTERED:.3f}")
    P(f"  M1 {'PASS' if M1 else 'FAIL'}" + ("" if valid_all else " (NOT INTERPRETED: a validity check failed)"))
    P(f"  descriptive verdict as in H3: "
      f"{'BRAIN-WIDE' if lo > 0.5 else 'LOCALIZED' if hi4 < 0.5 else 'INCONCLUSIVE'}\n")

    # retained fraction vs the same session-regions without movement removal (descriptive)
    R = D.groupby("region").agg(n_sess=("session", "nunique"), p_valid=("p_valid", PM._fisher), p_raw=("p_raw", PM._fisher),
                                score=("score", "mean"), score_raw=("score_raw", "mean"))
    R["fdr_move"] = PM._bh(R.p_valid.values, 0.05); R["fdr_raw"] = PM._bh(R.p_raw.values, 0.05)
    post = {r: np.exp(LL[r] + np.log(w + 1e-12) - (LL[r] + np.log(w + 1e-12)).max()) for r in regs}
    R["p_carries_move"] = [float(post[r][np.array(PM.GRID0) >= 0.04].sum() / post[r].sum()) for r in R.index]
    if base is not None:
        key = set(zip(D.session, D.region))
        Bm = base[[k in key for k in zip(base.session, base.region)]]
        LB = PM.region_logliks(Bm, "p_valid", rng)
        rb = sorted(set(LB) & set(LL))
        wB = PM.em_pi([LB[r] for r in rb]); wM = PM.em_pi([LL[r] for r in rb])
        ratio = []
        for _ in range(B):
            idx = rng.integers(0, len(rb), len(rb))
            pb = PM.frac(PM.em_pi([LB[rb[i]] for i in idx], iters=150), 0.04)
            pm = PM.frac(PM.em_pi([LL[rb[i]] for i in idx], iters=150), 0.04)
            ratio.append(pm / pb if pb > 0 else np.nan)
        P("## Retained (descriptive): the same session-regions, before vs after movement removal")
        P(f"  baseline run: {base_f.name}; {len(Bm)} matched session-regions")
        P(f"  fraction with r >= 0.04: before {PM.frac(wB, 0.04):.3f}, after {PM.frac(wM, 0.04):.3f}; "
          f"retained {PM.frac(wM, 0.04) / max(PM.frac(wB, 0.04), 1e-9):.2f} "
          f"(95% CI {np.nanpercentile(ratio, 2.5):.2f}-{np.nanpercentile(ratio, 97.5):.2f})\n")

    # ---- M2 (descriptive): detection before vs after, same session-regions and same nulls
    P("## M2 (descriptive: detection before vs after removal, same trials and nulls)")
    P(f"  single session-regions significant: before {np.mean(D.p_raw < 0.05):.3f}, after {np.mean(D.p_valid < 0.05):.3f}")
    P(f"  regions detected (Fisher + BH q = 0.05): before {int(R.fdr_raw.sum())}, after {int(R.fdr_move.sum())} of {len(R)}; "
      f"kept {int((R.fdr_raw & R.fdr_move).sum())} of the {int(R.fdr_raw.sum())} detected before")
    P(f"  mean decoder score: before {D.score_raw.mean():.4f}, after {D.score.mean():.4f}\n")

    # ---- M3 (descriptive): is movement itself prior-driven?
    S = D.groupby("session").agg(score_mov=("score_mov", "first"), p_mov=("p_mov", "first"),
                                 drop=("score_raw", "mean"), after=("score", "mean"))
    S["drop"] = S["drop"] - S["after"]
    from scipy.stats import spearmanr
    rho, pr = spearmanr(S.score_mov, S["drop"])
    P("## M3 (descriptive: does movement itself carry the prior?)")
    P(f"  sessions where movement alone predicts the prior (valid null, p < 0.05): {np.mean(S.p_mov < 0.05):.3f}; "
      f"median movement-only score {S.score_mov.median():.3f}")
    P(f"  sessions with more prior-driven movement lose more neural prior signal: Spearman rho {rho:+.3f} (p = {pr:.3g})")

    R.to_csv(OUT / "movement_regions.csv"); S.to_csv(OUT / "movement_sessions.csv")
    (OUT / "PRIOR_MOVEMENT.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/PRIOR_MOVEMENT.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("features"); f.add_argument("--keep-raw", action="store_true")
    wh = sub.add_parser("wheel"); wh.add_argument("--keep-raw", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--null", type=int, default=1000); r.add_argument("--plant", type=int, default=48)
    r.add_argument("--twins", type=int, default=5)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    for q in (f, wh, r):
        q.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
        q.add_argument("--gatekeeper"); q.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("test"); t.add_argument("--boot", type=int, default=500)
    t.add_argument("--baseline", default="amend1", help="PRIOR MAP run to compare against ('' = registered run)")
    a = ap.parse_args()
    {"features": cmd_features, "wheel": cmd_wheel, "run": cmd_run, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
