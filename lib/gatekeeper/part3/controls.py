"""PART 3, controls for the slow-mode results. Two confounds could fake them:

  PROBE MOTION   a Neuropixels probe slowly moving in the brain changes the firing of every neuron on it together, so
                 regions recorded on the same probe would share a "slow mode" that is really mechanics.
                 Controls: (a) keep only region pairs recorded on DIFFERENT probes; (b) partial out IBL's own estimate
                 of each probe's drift (spike-sorting drift.um), smoothed like everything else.
  AROUSAL        slow pupil-linked arousal is already known to be brain-wide (Raut et al. 2025, Nature). If the global
                 mode is just arousal, it is not new. Controls: (c) how much of the global mode slow pupil and face
                 motion explain; (d) does the global mode still track reaction time and misses after partialling them.

Every control uses the same null as slowmode.py (slow signals from OTHER sessions, aligned by trial) and the same
symmetric partialling (the extra regressors are removed from BOTH signals, real and borrowed).

Step 1 (needs internet, small downloads, resumable):   python3 part3/controls.py fetch --limit 3
Step 2 (offline):                                      python3 part3/controls.py analyse --limit 3
Simulation check (no data):                            python3 part3/controls.py sim
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import slowmode as sm  # noqa: E402

AUX = "data/part3_aux"
CAM_WINDOW = (-0.40, -0.05)   # same pre-stimulus window as the cached pupil code


# ------------------------------------------------------------------------------------------- step 1: fetch
def _window_mean(t, d, onsets, win=CAM_WINDOW):
    n = min(len(t), len(d))
    t, d = np.asarray(t[:n], float), np.asarray(d[:n], float)
    out = np.full(len(onsets), np.nan)
    lo = np.searchsorted(t, onsets + win[0])
    hi = np.searchsorted(t, onsets + win[1])
    for i, (a, b) in enumerate(zip(lo, hi)):
        if b > a:
            out[i] = np.nanmean(d[a:b])
    return out


def _motion_energy(one, eid, onsets):
    for coll in ("alf", None):
        try:
            me = one.load_dataset(eid, "leftCamera.ROIMotionEnergy.npy", collection=coll)
            t = one.load_dataset(eid, "leftCamera.times.npy", collection=coll)
            out = _window_mean(t, me, onsets)
            if np.isfinite(out).mean() > 0.8:
                return out
        except Exception:
            continue
    return None


def _probe_drift(one, eid, pname, onsets):
    """Per-trial drift (um, mean over depth bins) for one probe, from the spike-sorting drift estimate."""
    colls = sorted([c for c in one.list_collections(eid) if c.startswith(f"alf/{pname}")], key=len, reverse=True)
    for coll in colls:
        try:
            dr = one.load_object(eid, "drift", collection=coll)
            um = np.asarray(dr["um"], float)
            um = um.mean(1) if um.ndim == 2 else um
            t = np.asarray(dr["times"], float)
            n = min(len(t), len(um))
            if n < 10:
                continue
            return np.interp(onsets, t[:n], um[:n]), coll
        except Exception:
            continue
    return None, None


def _probe_regions(one, eid, pname):
    from gatekeeper.ibl import region_labels
    from iblatlas.regions import BrainRegions
    colls = sorted([c for c in one.list_collections(eid) if c.startswith(f"alf/{pname}")], key=len, reverse=True)
    for coll in colls:
        rl = region_labels(one, eid, coll)
        if rl is not None:
            beryl = np.asarray(BrainRegions().acronym2acronym(rl[0], mapping="Beryl")).astype(str)
            return sorted(set(beryl.tolist()))
    return []


def fetch_one(one, path, aux_dir):
    from gatekeeper.ibl import _pupil, _trials
    from gatekeeper.mv_data import MVSession
    s = MVSession(path)
    out = Path(aux_dir) / f"{s.sid}.npz"
    if out.exists():
        return "skip"
    _, _, onsets, _ = _trials(one, s.sid)
    if len(onsets) != len(s.contrast):
        np.savez(out, ok=False, why="trial count mismatch")
        return "mismatch"
    pupil, src = _pupil(one, s.sid, onsets)
    me = _motion_energy(one, s.sid, onsets)
    probes = s.meta.get("probes", [])
    drift, regions = {}, {}
    for p in probes:
        d, _ = _probe_drift(one, s.sid, p, onsets)
        if d is not None:
            drift[p] = d
        regions[p] = _probe_regions(one, s.sid, p)
    np.savez(out, ok=True, onsets=onsets,
             pupil=pupil if pupil is not None else np.zeros(0), pupil_source=str(src),
             motion=me if me is not None else np.zeros(0),
             probe_regions=json.dumps(regions), **{f"drift__{p}": d for p, d in drift.items()})
    return f"pupil={'yes' if pupil is not None else 'no'} motion={'yes' if me is not None else 'no'} " \
           f"probes={len(probes)} drift={len(drift)}"


def fetch(cache, aux_dir, split, limit):
    from one.api import ONE
    from gatekeeper.mv_data import MVSession
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab",
              password="international", silent=True)
    Path(aux_dir).mkdir(parents=True, exist_ok=True)
    paths = [p for p in sorted(Path(cache).expanduser().glob("*.npz")) if MVSession(p).split == split][: limit or None]
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            msg = fetch_one(one, p, aux_dir)
        except Exception as e:  # one bad session must not stop the run
            msg = f"FAILED ({type(e).__name__}: {e})"
        print(f"[{i + 1}/{len(paths)}] {p.stem}  {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ------------------------------------------------------------------------------------------- step 2: analyse
def load_aux(aux_dir, sid, v, T):
    f = Path(aux_dir) / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=False)
    if not bool(z["ok"]):
        return None
    a = dict(probe_regions=json.loads(str(z["probe_regions"])), drift={}, arousal=None)
    cols = []
    for k in ("pupil", "motion"):
        x = z[k]
        if len(x) == T:
            cols.append(sm.movavg(x[v]))
    a["arousal"] = np.column_stack(cols) if cols else None
    a["n_arousal"] = len(cols)
    a["has_pupil"] = len(z["pupil"]) == T
    for k in z.files:
        if k.startswith("drift__") and len(z[k]) == T:
            a["drift"][k[7:]] = sm.movavg(z[k][v])
    return a


def summaries_with_aux(cache, aux_dir, split, limit, n_jobs):
    from joblib import Parallel, delayed
    from gatekeeper.mv_data import MVSession
    paths = [p for p in sorted(Path(cache).expanduser().glob("*.npz")) if MVSession(p).split == split][: limit or None]
    paths = [p for p in paths if (Path(aux_dir) / f"{p.stem}.npz").exists()]
    print(f"{len(paths)} sessions with control data ({split})", flush=True)
    S = Parallel(n_jobs=n_jobs, verbose=0)(delayed(sm.session_summary)(p) for p in paths)
    out = []
    for p, x in zip(paths, S):
        if not x:
            continue
        s = MVSession(p)
        v = s.valid["stimOn"]
        x["aux"] = load_aux(aux_dir, s.sid, v, len(s.contrast))
        if x["aux"] is not None:
            out.append(x)
    return out


def region_probe(aux):
    """region -> probe, only for regions recorded on exactly one probe (others are ambiguous)."""
    seen = {}
    for p, regs in aux["probe_regions"].items():
        for r in regs:
            seen.setdefault(r, set()).add(p)
    return {r: next(iter(ps)) for r, ps in seen.items() if len(ps) == 1}


def _aug(Ds, *extra):
    cols = [Ds] + [e if e.ndim == 2 else e[:, None] for e in extra if e is not None]
    return np.column_stack(cols)


def _perm_p(obs, null):
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) <= 20:
        return np.nan, np.nan
    return float((1 + (null >= obs).sum()) / (1 + len(null))), float(np.median(null))


def analyse_controls(S, n_null=200, seed=0, division=None):
    import pandas as pd
    rng = np.random.default_rng(seed)
    div = division or sm.division_of(sorted({r for x in S for r in x["regions"]}))
    by_region = {}
    for i, x in enumerate(S):
        for r, u in x["regions"].items():
            by_region.setdefault(r, []).append((i, u))
    q1, q2, ar = [], [], []
    for i, x in enumerate(S):
        aux, Ds = x["aux"], x["slow_design"]
        rp = region_probe(aux)
        drift_all = np.column_stack(list(aux["drift"].values())) if aux["drift"] else None
        regs = list(x["regions"])
        for a in range(len(regs)):
            for b in range(a + 1, len(regs)):
                A, B = regs[a], regs[b]
                if div.get(A) == div.get(B):
                    continue
                pa, pb = rp.get(A), rp.get(B)
                pair = "different probes" if (pa and pb and pa != pb) else "same probe" if (pa and pb) else "ambiguous"
                # same partner pool as slowmode.py: region B in other sessions, else other sessions' global modes
                pool = [u for j, u in by_region.get(B, []) if j != i] or [y["global_mode"] for j, y in enumerate(S) if j != i]
                idx = rng.integers(0, len(pool), n_null)
                row = dict(sid=x["sid"], A=A, B=B, pair=pair, has_drift=drift_all is not None)
                for name, D in (("plain", Ds), ("drift", _aug(Ds, drift_all) if drift_all is not None else None)):
                    if D is None:
                        continue
                    obs = abs(sm._corr(x["regions"][A], x["regions"][B], D))
                    null = np.array([abs(sm._corr(x["regions"][A], pool[k], D)) for k in idx])
                    row[f"r_{name}"] = obs
                    row[f"p_{name}"], row[f"rnull_{name}"] = _perm_p(obs, null)
                q1.append(row)
        # arousal: how much of the global mode is slow pupil / face motion?
        others = [y["global_mode"] for j, y in enumerate(S) if j != i]
        idx = rng.integers(0, len(others), n_null)
        if aux["arousal"] is not None:
            g = x["global_mode"]
            n = min(len(g), len(aux["arousal"]))
            A1 = np.column_stack([np.ones(n), aux["arousal"][:n]])
            beta, *_ = np.linalg.lstsq(A1, g[:n], rcond=None)
            r2 = 1 - np.var(g[:n] - A1 @ beta) / np.var(g[:n])
            row = dict(sid=x["sid"], subject=x["subject"], has_pupil=aux["has_pupil"], r2_global_by_arousal=float(r2))
            col0 = aux["arousal"][:, 0]
            obs = abs(sm._corr(g, col0, Ds))
            null = np.array([abs(sm._corr(others[k], col0, Ds)) for k in idx])
            row["r_global_vs_first_arousal"] = obs
            row["p_global_vs_first_arousal"], _ = _perm_p(obs, null)
            ar.append(row)
        # Q2 with and without arousal regressors, on the SAME sessions
        for bname, bvec in x["behaviour"].items():
            row = dict(sid=x["sid"], subject=x["subject"], behaviour=bname, has_arousal=aux["arousal"] is not None)
            for name, D in (("plain", Ds), ("arousal", _aug(Ds, aux["arousal"]) if aux["arousal"] is not None else None)):
                if D is None:
                    continue
                obs = abs(sm._corr(x["global_mode"], bvec, D))
                null = np.array([abs(sm._corr(others[k], bvec, D)) for k in idx])
                row[f"r_{name}"] = obs
                row[f"p_{name}"], row[f"rnull_{name}"] = _perm_p(obs, null)
            q2.append(row)
    return pd.DataFrame(q1), pd.DataFrame(q2), pd.DataFrame(ar)


def _fisher(p):
    from scipy.stats import chi2
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan


def report_controls(q1, q2, ar, out_dir):
    L = ["# Part 3 controls: probe motion and arousal\n",
         "Same null as the main test (slow signals from other sessions, aligned by trial); extra regressors are",
         "partialled out of both signals. Under 'nothing real', fraction p < 0.05 is about 0.05.\n",
         "## Q1 sharing across divisions, by probe and with probe drift removed\n",
         "| Pairs | n pairs | sessions | Median abs r | Fraction p < 0.05 (no drift control) | Fraction p < 0.05 (drift removed) | n pairs with drift data |",
         "|---|---|---|---|---|---|---|"]
    if len(q1):
        for pair in ("same probe", "different probes", "ambiguous"):
            d = q1[q1.pair == pair]
            if not len(d):
                continue
            dd = d[d.has_drift] if "p_drift" in d else d.iloc[:0]
            fr_d = f"{(dd.p_drift < .05).mean():.3f}" if len(dd) else "-"
            L.append(f"| {pair} | {len(d)} | {d.sid.nunique()} | {d.r_plain.median():.2f} | {(d.p_plain < .05).mean():.3f} | {fr_d} | {len(dd)} |")
    L += ["\n## Arousal: how much of the global slow mode is slow pupil and face motion?\n"]
    if len(ar):
        L += [f"Sessions with arousal data: {len(ar)} (with pupil: {int(ar.has_pupil.sum())}).",
              f"Variance of the global mode explained by slow pupil + face motion: median {ar.r2_global_by_arousal.median():.2f} "
              f"(IQR {ar.r2_global_by_arousal.quantile(.25):.2f} to {ar.r2_global_by_arousal.quantile(.75):.2f}).",
              f"Global mode vs slow pupil (or face motion if no pupil), beyond other sessions' modes: fraction p < 0.05 = "
              f"{(ar.p_global_vs_first_arousal < .05).mean():.3f}, Fisher p = {_fisher(ar.p_global_vs_first_arousal):.1e}.\n"]
    L += ["## Q2 global mode vs behaviour, before and after removing arousal (same sessions)\n",
          "| Behaviour | Sessions | Fraction p < 0.05 before | Fisher p before | Fraction p < 0.05 after | Fisher p after |",
          "|---|---|---|---|---|---|"]
    if len(q2) and "p_arousal" in q2:
        for b, d in q2[q2.has_arousal].groupby("behaviour"):
            L.append(f"| {b} | {len(d)} | {(d.p_plain < .05).mean():.3f} | {_fisher(d.p_plain):.1e} | "
                     f"{(d.p_arousal < .05).mean():.3f} | {_fisher(d.p_arousal):.1e} |")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "PART3_CONTROLS.md").write_text(text + "\n")
    q1.to_csv(Path(out_dir) / "controls_q1.csv", index=False)
    q2.to_csv(Path(out_dir) / "controls_q2.csv", index=False)
    ar.to_csv(Path(out_dir) / "controls_arousal.csv", index=False)
    print(text)


# ------------------------------------------------------------------------------------------- simulation check
def sim(n_sessions=40, T=600, shared=0.0, motion=0.0, arousal_drives=0.0, seed=0):
    """4 regions on 2 probes (R1, R2 on probe A; R3, R4 on probe B). Each probe has its own slow motion signal that
    enters its regions with weight 'motion' (IBL's drift estimate = that signal + noise). 'shared' = a real common
    state in all regions. 'arousal_drives': the state is pupil, and pupil drives reaction time."""
    rng = np.random.default_rng(seed)

    def slow():
        z = np.zeros(T); e = rng.standard_normal(T)
        for t in range(1, T):
            z[t] = 0.99 * z[t - 1] + np.sqrt(1 - 0.99 ** 2) * e[t]
        return z
    S = []
    for i in range(n_sessions):
        state = slow()
        mot = {"pA": slow(), "pB": slow()}
        regions, allX = {}, []
        for r, p in (("R1", "pA"), ("R2", "pA"), ("R3", "pB"), ("R4", "pB")):
            own = slow()
            d = np.sqrt(shared) * state + motion * mot[p] + np.sqrt(max(1 - shared, 0)) * own
            X = rng.poisson(np.exp(0.5 + 0.4 * d[:, None] * rng.uniform(0.5, 1.5, 15)))
            regions[r] = sm.slow_pc(X)[0]; allX.append(X)
        g, frac = sm.slow_pc(np.hstack(allX))
        pupil = state if arousal_drives else slow()
        rt = 0.3 * arousal_drives * state + rng.standard_normal(T)
        aux = dict(probe_regions={"pA": ["R1", "R2"], "pB": ["R3", "R4"]},
                   drift={p: sm.movavg(m + 0.3 * rng.standard_normal(T)) for p, m in mot.items()},
                   arousal=sm.movavg(pupil + 0.3 * rng.standard_normal(T))[:, None], has_pupil=True)
        S.append(dict(sid=f"s{i}", subject=f"m{i}", T=T, regions=regions, global_mode=g, global_frac=frac,
                      behaviour={"log_rt": sm.movavg(rt)}, slow_design=np.zeros((T, 0)), aux=aux))
    return analyse_controls(S, n_null=100, seed=seed, division={r: r for r in ("R1", "R2", "R3", "R4")})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "analyse", "sim"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--aux", default=AUX)
    ap.add_argument("--out", default="results/part3_controls")
    ap.add_argument("--split", default="explore")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.step == "fetch":
        fetch(a.cache, a.aux, a.split, a.limit)
    elif a.step == "analyse":
        S = summaries_with_aux(a.cache, a.aux, a.split, a.limit, a.jobs)
        report_controls(*analyse_controls(S), a.out)
    else:
        for kw in (dict(), dict(motion=0.8), dict(shared=0.5), dict(shared=0.5, motion=0.8), dict(shared=0.5, arousal_drives=1.0)):
            q1, q2, ar = sim(**kw)
            f = lambda d, c: f"{(d[c] < .05).mean():.3f}" if len(d) and c in d else "-"
            same, diff = q1[q1.pair == "same probe"], q1[q1.pair == "different probes"]
            print(f"{kw}:  Q1 same-probe {f(same, 'p_plain')} -> drift removed {f(same, 'p_drift')} | "
                  f"different-probe {f(diff, 'p_plain')} -> {f(diff, 'p_drift')} | "
                  f"Q2 rt {f(q2, 'p_plain')} -> arousal removed {f(q2, 'p_arousal')} | "
                  f"R2 global by arousal {ar.r2_global_by_arousal.median():.2f}")
