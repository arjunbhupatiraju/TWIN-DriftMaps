"""PART 3 model, stage 1: the write-store-read model of reward memory, tested against the IBL results.

Model (per trial t, real IBL behaviour drives everything):
    behaviour   uses past rewarded choices with the measured kernel K_k (lags 1-4)
    activity    a region's neurons carry
                    beta1 * rc(t-1)                       (the last trial, which IS in activity)
                  + betap * prior(t)                      (the slow block prior / bias state)
                  + sum_k  f_k * beta1 * (K_k / K_1) * rc(t-k),  k = 2..4    (older memory, visible fraction f_k)
                plus realistic drift (shared slow component + private slow + fast noise)
    storage     f_k = 0       memory older than one trial is held silently (synaptic weights)   <- our account
                f_k = 1       memory held in firing in proportion to its use by behaviour (persistent integrator)
                f_k = phi     in between (used to bound how much could be hiding)

Fit and test, with the SAME published-style pipeline as part3/history_findling.py (Lasso, nested CV, pseudo-sessions
with simulated choices):
    calibrate   beta1, betap so that the model reproduces two observed numbers only:
                rc1 session rate 0.225 and rc4 session rate 0.107 (published-style, plain labels)
    predict     (not fitted)  rc2, rc3 plain rates (observed 0.143, 0.128)
                              rc2c, rc3c, rc4c controlled rates (observed 0.090, 0.055, 0.052)
    compare     persistent-integrator alternative (f_k = 1) at the same beta1, betap
    bound       phi sweep: the largest visible fraction of the 2-back memory compatible with the observed rc2c rate

    python3 part3/wsr_model.py calibrate --n 60
    python3 part3/wsr_model.py evaluate  --n 60          (uses the calibrated values saved by calibrate)
"""
from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "next", ROOT / "driftaware" / "src", HERE):
    sys.path.insert(0, str(p))
from history_map import lags  # noqa: E402
import history_findling as hf  # noqa: E402

KERNEL = np.array([0.472, 0.371, 0.245, 0.259])       # IBL rewarded-choice behaviour kernel, lags 1-4
OBS = dict(rc1=0.225, rc2=0.143, rc3=0.128, rc4=0.107, rc2c=0.090, rc3c=0.055, rc4c=0.052)
N_UNITS = 30
N_NULL = 100            # same as the real gap check, so per-session z-scores are on the same scale
DRAWS = [2]
NNULL = [N_NULL]             # independent neural draws per session and setting (averages out simulation noise)
B1_GRID = (0.01, 0.02, 0.035, 0.05, 0.08)
BP_GRID = (0.0, 0.02, 0.05, 0.1, 0.2)


def drift_noise(rng, T, N, drift=0.6):
    """Shared slow drift + private slow drift + fast noise (same family as the project's drift simulations)."""
    def ar(n):
        z = np.zeros((T, n)); e = rng.standard_normal((T, n)); z[0] = e[0]
        for t in range(1, T):
            z[t] = 0.98 * z[t - 1] + np.sqrt(1 - 0.98 ** 2) * e[t]
        return z
    common = ar(1) * 0.6 + ar(N) * 0.8
    return np.sqrt(drift) * common / common.std(0) + np.sqrt(1 - drift) * rng.standard_normal((T, N))


TAU_AK = 5.45            # Findling et al. 2025: median action-kernel time constant (trials)


def action_kernel(c, tau=TAU_AK):
    """The mouse's own slow bias: exponentially weighted average of its PAST choices (not including trial t)."""
    a = np.zeros(len(c)); acc = 0.0; lam = np.exp(-1 / tau)
    for t in range(len(c)):
        a[t] = acc
        acc = lam * acc + (1 - lam) * c[t]
    return a


def neurons(rng, real, beta1, betap, f, N=N_UNITS, betaa=0.0):
    """Simulated region on top of the REAL behaviour of one session. f: visible fractions for lags 2..4.
    betaa: strength of the slow subjective-bias state (action kernel) in activity (v1.1)."""
    rc, prior = real["rc"], real["prior"]
    T = len(rc)
    z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
    L = lags(rc, 4)
    X = drift_noise(rng, T, N)
    X += beta1 * np.outer(z(L[:, 0]), rng.standard_normal(N))
    X += betap * np.outer(z(prior), rng.standard_normal(N))
    if betaa > 0:
        X += betaa * np.outer(z(action_kernel(real["c"])), rng.standard_normal(N))
    for k in range(2, 5):
        if f[k - 2] > 0:
            X += f[k - 2] * beta1 * (KERNEL[k - 1] / KERNEL[0]) * np.outer(z(L[:, k - 1]), rng.standard_normal(N))
    return X


def session_setup(path, seed, n_null=N_NULL):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from gatekeeper.multiverse import _sides
    s = MVSession(path)
    v = cm.mask_of(s)
    if v is None or v.sum() < 150:
        return None
    Tf = len(s.contrast)
    prior = cm.bayes_prior(_sides(s)) - 0.5
    c = np.where(s.choice >= 0, 2.0 * s.choice - 1, 0.0)
    real = dict(rc=c * (np.asarray(s.feedback) > 0), c=c, prior=prior, side=_sides(s),
                fb=np.asarray(s.feedback, float), choice01=np.asarray(s.choice))
    w = hf.behaviour_model(s.contrast, s.choice, s.feedback, prior)
    rng = np.random.default_rng(seed)
    mags = np.abs(np.nan_to_num(np.asarray(s.contrast, float)))
    sims = [hf.simulate_session(w, Tf, mags, rng) for _ in range(n_null + 1)]
    return dict(sid=s.sid, v=v, real=real, w=w, Tf=Tf, mags=mags, sims=sims, n_null=n_null)


def rates_for(S, settings, labels, seed, draws=2):
    """For one session: p-value of each label under each setting and neural draw (published-style decoder)."""
    import compare_methods as cm
    rng = np.random.default_rng(seed)
    Ys = {lab: hf.label_matrix(S["real"], S["w"], S["Tf"], S["mags"], S["v"], lab, S["n_null"], rng, S["sims"])
          for lab in labels}
    out = []
    for j, st in enumerate(settings):
        b1, bp, f = st[:3]
        ba = st[3] if len(st) > 3 else 0.0
        for dr in range(draws):
            X = neurons(np.random.default_rng(seed * 31 + 1000 * j + dr), S["real"], b1, bp, f, betaa=ba)[S["v"]]
            row = dict(sid=S["sid"], draw=dr, b1=b1, bp=bp, ba=ba, f=",".join(f"{x:g}" for x in f))
            for lab, Y in Ys.items():
                R = cm.r2(Y, cm.cv_predictions(X, Y))
                row[lab] = float((1 + (R[1:] >= R[0]).sum()) / len(R))
            out.append(row)
    return out


def pz(p):
    """Per-session z-score from a pseudo-session p-value (same transform for model and data)."""
    from scipy.stats import norm
    return norm.isf(np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9))


def run_grid(cache, n, jobs, settings, labels, out_csv):
    draws, n_null = DRAWS[0], NNULL[0]
    import pandas as pd
    from joblib import Parallel, delayed
    paths = sorted(Path(cache).expanduser().glob("*.npz"))
    step = max(1, len(paths) // n)
    paths = paths[::step][:n]
    t0 = time.time()

    def work(p):
        seed = zlib.crc32(p.stem.encode()) % 100000
        try:
            S = session_setup(p, seed, n_null)
            return [] if S is None else rates_for(S, settings, labels, seed, draws)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True)
            return []

    rows = []
    try:
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(work)(p) for p in paths)
    except Exception:
        gen = Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in paths)
    for i, r in enumerate(gen):
        rows += r
        print(f"[{i + 1}/{len(paths)}] ({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def summarize(df, labels, stat="z"):
    """Per setting: mean per-session z (pooled strength; default) or fraction p < 0.05."""
    if "ba" not in df:
        df = df.assign(ba=0.0)
    g = df.groupby(["b1", "bp", "ba", "f"])
    if stat == "z":
        return g[labels].agg(lambda p: float(pz(p).mean())).reset_index()
    return g[labels].agg(lambda p: float((p < .05).mean())).reset_index()


def observed(res_dir="results"):
    """Observed mean per-session z and fraction for each label, from the real gap-check runs on this machine."""
    import pandas as pd
    out = {}
    for sub in ("history_findling", "history_findling_controlled"):
        fs = [f for f in (Path(res_dir) / sub / "sessions").glob("*.csv") if f.stat().st_size > 1]
        if not fs:
            continue
        df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
        for lab, d in df.groupby("var"):
            z = pz(d.p.values)
            out[lab] = dict(z=float(z.mean()), se=float(z.std(ddof=1) / np.sqrt(len(z))), frac=float((d.p < .05).mean()),
                            n=int(len(d)))
    return out


def calibrate(cache, n, jobs, out_dir):
    labels = ["rc1", "rc2", "rc3", "rc4"]
    settings = [(b1, bp, (0.0, 0.0, 0.0)) for b1 in B1_GRID for bp in BP_GRID]
    df = run_grid(cache, n, jobs, settings, labels, Path(out_dir) / "calibrate.csv")
    S = summarize(df, labels)
    # calibrate on rc1 and rc4 only; rc2, rc3 are held out (predicted)
    S["err"] = (S.rc1 - OBS["rc1"]) ** 2 + (S.rc4 - OBS["rc4"]) ** 2
    best = S.sort_values("err").iloc[0]
    L = ["# Model stage 1: calibration (silent storage, f = 0 beyond lag 1)\n",
         f"{df.sid.nunique()} IBL sessions (real behaviour, simulated region of {N_UNITS} drifting neurons), "
         f"published-style decoder, {N_NULL} pseudo-sessions. Calibrated on rc1 and rc4 ONLY.\n",
         "| beta1 | betap | rc1 | rc2 (held out) | rc3 (held out) | rc4 |", "|---|---|---|---|---|---|"]
    for _, r in S.sort_values(["b1", "bp"]).iterrows():
        L.append(f"| {r.b1} | {r.bp} | {r.rc1:.3f} | {r.rc2:.3f} | {r.rc3:.3f} | {r.rc4:.3f} |")
    L += [f"\nObserved (IBL, 228 sessions): rc1 {OBS['rc1']}, rc2 {OBS['rc2']}, rc3 {OBS['rc3']}, rc4 {OBS['rc4']}",
          f"Best calibration: beta1 = {best.b1}, betap = {best.bp} -> rc1 {best.rc1:.3f}, rc4 {best.rc4:.3f}; "
          f"PREDICTED rc2 {best.rc2:.3f} (obs {OBS['rc2']}), rc3 {best.rc3:.3f} (obs {OBS['rc3']})"]
    text = "\n".join(L)
    (Path(out_dir) / "MODEL_CALIBRATE.md").write_text(text + "\n")
    (Path(out_dir) / "calibrated.json").write_text(json.dumps(dict(b1=float(best.b1), bp=float(best.bp))))
    print(text)


def evaluate(cache, n, jobs, out_dir):
    cal = json.loads((Path(out_dir) / "calibrated.json").read_text())
    b1, bp = cal["b1"], cal["bp"]
    labels = ["rc2", "rc3", "rc4", "rc2c", "rc3c", "rc4c"]
    profiles = {"silent (our account)": (0.0, 0.0, 0.0), "persistent integrator": (1.0, 1.0, 1.0),
                "phi 0.1": (0.1, 0.1, 0.1), "phi 0.25": (0.25, 0.25, 0.25), "phi 0.5": (0.5, 0.5, 0.5)}
    settings = [(b1, bp, f) for f in profiles.values()]
    df = run_grid(cache, n, jobs, settings, labels, Path(out_dir) / "evaluate.csv")
    S = summarize(df, labels)
    name = {",".join(f"{x:g}" for x in f): k for k, f in profiles.items()}
    nses = df.sid.nunique()
    L = ["# Model stage 1: predictions vs IBL (not fitted)\n",
         f"beta1 = {b1}, betap = {bp} (from calibration). {nses} sessions. Session rate p < 0.05.\n",
         "| Storage | rc2 | rc3 | rc4 | rc2c | rc3c | rc4c |", "|---|---|---|---|---|---|---|",
         f"| **observed IBL** | {OBS['rc2']} | {OBS['rc3']} | {OBS['rc4']} | {OBS['rc2c']} | {OBS['rc3c']} | {OBS['rc4c']} |"]
    for _, r in S.iterrows():
        L.append(f"| {name.get(r.f, r.f)} | " + " | ".join(f"{r[l]:.3f}" for l in labels) + " |")
    se = np.sqrt(OBS["rc2c"] * (1 - OBS["rc2c"]) / 1000)
    L.append(f"\nBound: the 2-back visible fraction phi is excluded where the predicted rc2c rate exceeds the observed "
             f"{OBS['rc2c']} by more than sampling error (observed rate based on ~1000+ session-regions; "
             f"model rates on {nses} sessions, SE ~{np.sqrt(0.1 * 0.9 / max(nses, 1)):.3f}).")
    text = "\n".join(L)
    (Path(out_dir) / "MODEL_EVALUATE.md").write_text(text + "\n")
    print(text)


B1_GRID2 = (0.015, 0.02, 0.03)
BP_GRID2 = (0.02, 0.05)
BA_GRID2 = (0.0, 0.03, 0.06, 0.1)
CAL2 = ("rc1", "rc4", "rc2c")                 # tuned on these three
HELD2 = ("rc2", "rc3", "rc3c", "rc4c")         # predicted, never used in tuning


def _obs_or_die(out_dir):
    O = observed()
    if not all(l in O for l in CAL2 + HELD2):
        raise SystemExit("Observed gap-check results not found under results/history_findling(_controlled). "
                         "Run this from the gatekeeper folder where those runs were made.")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "observed.json").write_text(json.dumps(O, indent=1))
    return O


def calibrate2(cache, n, jobs, out_dir):
    """v1.1: activity also carries the mouse's slow subjective bias (action kernel), which our brain-wide decoding
    found. Scored by mean per-session z (pooled strength). Tuned on rc1, rc4, rc2c; the rest are predictions."""
    O = _obs_or_die(out_dir)
    labels = list(CAL2) + list(HELD2)
    settings = [(b1, bp, (0.0, 0.0, 0.0), ba) for b1 in B1_GRID2 for bp in BP_GRID2 for ba in BA_GRID2]
    df = run_grid(cache, n, jobs, settings, labels, Path(out_dir) / "calibrate2.csv")
    S = summarize(df, labels)
    S["err"] = sum((S[l] - O[l]["z"]) ** 2 for l in CAL2)
    S = S.sort_values("err")
    best = S.iloc[0]
    L = ["# Model v1.1: silent storage + slow subjective bias in activity (mean per-session z)\n",
         f"{df.sid.nunique()} IBL sessions x {DRAWS[0]} neural draws, real behaviour, simulated {N_UNITS}-neuron region, "
         f"published-style decoder, {NNULL[0]} pseudo-sessions. Tuned on {', '.join(CAL2)} only; "
         f"held-out predictions: {', '.join(HELD2)}.\n",
         "| beta1 | betap | beta_bias | " + " | ".join(labels) + " |", "|" + "---|" * (3 + len(labels)),
         "| **observed** | | | " + " | ".join(f"{O[l]['z']:.3f} ±{O[l]['se']:.3f}" for l in labels) + " |"]
    for _, r in S.head(8).iterrows():
        L.append(f"| {r.b1} | {r.bp} | {r.ba} | " + " | ".join(f"{r[l]:.3f}" for l in labels) + " |")
    L.append(f"\n(8 best of {len(S)} settings.) Best: beta1 {best.b1}, betap {best.bp}, beta_bias {best.ba}.")
    L.append("Held-out: " + "; ".join(f"{l} predicted {best[l]:.3f} vs observed {O[l]['z']:.3f}" for l in HELD2))
    text = "\n".join(L)
    (Path(out_dir) / "MODEL_CALIBRATE2.md").write_text(text + "\n")
    (Path(out_dir) / "calibrated2.json").write_text(json.dumps(dict(b1=float(best.b1), bp=float(best.bp),
                                                                     ba=float(best.ba))))
    print(text)


def evaluate2(cache, n, jobs, out_dir):
    O = _obs_or_die(out_dir)
    cal = json.loads((Path(out_dir) / "calibrated2.json").read_text())
    b1, bp, ba = cal["b1"], cal["bp"], cal["ba"]
    labels = ["rc2", "rc3", "rc4", "rc2c", "rc3c", "rc4c"]
    profiles = {"silent (our account)": (0.0, 0.0, 0.0), "persistent integrator": (1.0, 1.0, 1.0),
                "phi 0.1": (0.1, 0.1, 0.1), "phi 0.25": (0.25, 0.25, 0.25), "phi 0.5": (0.5, 0.5, 0.5)}
    settings = [(b1, bp, f, ba) for f in profiles.values()]
    df = run_grid(cache, n, jobs, settings, labels, Path(out_dir) / "evaluate2.csv")
    S = summarize(df, labels)
    name = {",".join(f"{x:g}" for x in f): k for k, f in profiles.items()}
    nn = len(df) // max(1, len(profiles))
    L = ["# Model v1.1: storage accounts compared with IBL (mean per-session z; 0 = nothing)\n",
         f"beta1 {b1}, betap {bp}, beta_bias {ba} (from calibrate2). {df.sid.nunique()} sessions x {DRAWS[0]} draws; "
         f"model SE about {1 / np.sqrt(max(nn, 1)):.3f}.\n",
         "| Storage | " + " | ".join(labels) + " |", "|" + "---|" * (1 + len(labels)),
         "| **observed IBL** | " + " | ".join(f"{O[l]['z']:.3f} ±{O[l]['se']:.3f}" for l in labels) + " |"]
    for _, r in S.iterrows():
        L.append(f"| {name.get(r.f, r.f)} | " + " | ".join(f"{r[l]:.3f}" for l in labels) + " |")
    text = "\n".join(L)
    (Path(out_dir) / "MODEL_EVALUATE2.md").write_text(text + "\n")
    print(text)


# ------------------------------------------------------------------------------------------- stage 1b: drift-aware test
B1_DA = [(0.01, 0.02, 0.03, 0.045, 0.065)]


def _owns(S):
    """Drift-aware test inputs: the real behaviour sequence and every pseudo-session, as history_joint expects."""
    import history_joint as hj
    r = S["real"]
    seqs = [hj.seq_matrix(r["side"], r["choice01"], r["fb"], r["prior"] + 0.5)]
    for d in S["sims"][1:]:
        seqs.append(hj.seq_matrix(d["side"], (d["c"] > 0).astype(int), d["fb"], d["prior"] + 0.5))
    return np.stack(seqs)


def da_rows(S, settings, seed, draws):
    import history_joint as hj
    owns = _owns(S)
    v = S["v"]
    out = []
    for j, st in enumerate(settings):
        b1, bp, f = st[:3]
        for dr in range(draws):
            X = neurons(np.random.default_rng(seed * 37 + 1000 * j + dr), S["real"], b1, bp, f)
            row = dict(sid=S["sid"], draw=dr, b1=b1, bp=bp, f=",".join(f"{x:g}" for x in f))
            for name, k, pp, ee in hj.joint_tests(X, owns, v, drop_prior=True, only=("rewarded",)):
                if k <= 4:
                    row[f"rewarded{k}"] = pp
            out.append(row)
    return out


def observed_da(res_dir="results/silent_memory/rewarded"):
    """Observed step-3e drift-aware results (real and planted), as mean per-session z."""
    import pandas as pd
    fs = [f for f in Path(res_dir).glob("*.csv") if f.stat().st_size > 1]
    if not fs:
        fs = [f for f in Path(res_dir).rglob("*.csv") if f.stat().st_size > 1]
    if not fs:
        raise SystemExit(f"Step 3e results not found in {res_dir}. Pass --res3e with the folder of those CSV files.")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    out = {}
    for tag in ("real", "full", "half"):
        for k in range(1, 5):
            col = f"{tag}_p_rewarded{k}"
            if col in df:
                z = pz(df[col].dropna().values)
                out[f"{tag}{k}"] = dict(z=float(z.mean()), se=float(z.std(ddof=1) / np.sqrt(len(z))), n=int(len(z)),
                                        frac=float((df[col].dropna() < .05).mean()))
    return out


def evaluate_da(cache, n, jobs, out_dir, res3e):
    """Stage 1b. Calibrate beta1 so the model's lag-1 drift-aware z matches the observed step-3e lag-1 z (silent
    storage), then compare storage accounts at lags 2-4 with the observed real values. The phi at which the model
    exceeds the observed lag-2 value is the bound on how much 2-back memory can be held in firing."""
    import pandas as pd
    from joblib import Parallel, delayed
    O = observed_da(res3e)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "observed_3e.json").write_text(json.dumps(O, indent=1))
    bp = 0.02
    profiles = {"silent (our account)": (0.0, 0.0, 0.0), "phi 0.1": (0.1, 0.1, 0.1), "phi 0.25": (0.25, 0.25, 0.25),
                "phi 0.5": (0.5, 0.5, 0.5), "persistent integrator": (1.0, 1.0, 1.0)}
    b1s = B1_DA[0]
    settings = [(b1, bp, (0.0, 0.0, 0.0)) for b1 in b1s]
    paths = sorted(Path(cache).expanduser().glob("*.npz"))
    step = max(1, len(paths) // n)
    paths = paths[::step][:n]
    draws, n_null = DRAWS[0], NNULL[0]
    t0 = time.time()

    def work(p):
        seed = zlib.crc32(p.stem.encode()) % 100000
        try:
            S = session_setup(p, seed, n_null)
            if S is None:
                return []
            rows = da_rows(S, settings, seed, draws)
            # the phi profiles at every beta1 too, so the calibrated beta1 can be chosen after the run
            rows += da_rows(S, [(b1, bp, f) for b1 in b1s for f in list(profiles.values())[1:]], seed + 1, draws)
            return rows
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True)
            return []

    rows = []
    try:
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(work)(p) for p in paths)
    except Exception:
        gen = Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in paths)
    for i, r in enumerate(gen):
        rows += r
        print(f"[{i + 1}/{len(paths)}] ({(time.time() - t0) / 60:.1f} min)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(Path(out_dir) / "evaluate_da.csv", index=False)
    labs = [f"rewarded{k}" for k in range(1, 5)]
    Z = df.groupby(["b1", "f"])[labs].agg(lambda p: float(pz(p).mean())).reset_index()
    sil = Z[Z.f == "0,0,0"].copy()
    sil["err"] = (sil.rewarded1 - O["real1"]["z"]) ** 2
    b1 = float(sil.sort_values("err").iloc[0].b1)
    name = {",".join(f"{x:g}" for x in f): k for k, f in profiles.items()}
    nn = len(df[(df.b1 == b1) & (df.f == "0,0,0")])
    L = ["# Model stage 1b: storage accounts under the drift-aware test (mean per-session z; 0 = nothing)\n",
         f"{df.sid.nunique()} sessions x {draws} draws, simulated {N_UNITS}-neuron region, {n_null} pseudo-sessions. "
         f"beta1 = {b1} chosen so the silent model's LAG-1 z matches the observed step-3e lag-1 z; lags 2-4 are "
         f"predictions. Model SE about {1 / np.sqrt(max(nn, 1)):.3f}.\n",
         "| Storage | lag 1 | lag 2 | lag 3 | lag 4 |", "|---|---|---|---|---|",
         "| **observed IBL (step 3e, real)** | " + " | ".join(
             f"{O[f'real{k}']['z']:.3f} ±{O[f'real{k}']['se']:.3f}" if f"real{k}" in O else "-" for k in range(1, 5)) + " |"]
    for prof, f in profiles.items():
        key = ",".join(f"{x:g}" for x in f)
        r = Z[(Z.b1 == b1) & (Z.f == key)]
        if len(r):
            r = r.iloc[0]
            L.append(f"| {prof} | " + " | ".join(f"{r[l]:.3f}" for l in labs) + " |")
    L.append("\nSilent-model lag-1 z for each beta1 (calibration): " +
             ", ".join(f"{row.b1}: {row.rewarded1:.3f}" for _, row in sil.sort_values("b1").iterrows()) +
             f" (observed {O['real1']['z']:.3f})")
    # bound: smallest phi whose predicted lag-2 z exceeds the observed by 2 combined SE
    se_c = np.sqrt(O["real2"]["se"] ** 2 + 1 / max(nn, 1))
    rows_b = [(f[0], Z[(Z.b1 == b1) & (Z.f == ",".join(f"{x:g}" for x in f))].rewarded2) for f in profiles.values()]
    excl = [ph for ph, zz in rows_b if len(zz) and float(zz.iloc[0]) > O["real2"]["z"] + 2 * se_c]
    L.append(f"Bound: visible fractions excluded at lag 2 (prediction above observed + 2 combined SE = "
             f"{O['real2']['z'] + 2 * se_c:.3f}): " + (", ".join(f"{x:g}" for x in sorted(excl)) if excl else "none at this size"))
    if "full2" in O:
        L.append(f"Observed step-3e planted (behaviour-matched, real data): lag 2 full {O['full2']['z']:.3f}, "
                 f"half {O['half2']['z']:.3f}")
    text = "\n".join(L)
    (Path(out_dir) / "MODEL_STAGE1B.md").write_text(text + "\n")
    print(text)


# ------------------------------------------------------------------------------------------- result 2 at full power
B1_CMP = (0.015, 0.02, 0.025, 0.03)
BP_CMP = (0.01, 0.02, 0.035)


def compare(cache, n, jobs, out_dir):
    """Fair model comparison with the published-style decoder: silent and persistent storage EACH get their own
    best beta1, betap (grid) fitted to all four plain lags; goodness of fit = chi-square against the observed values
    using observed and model errors. Run with large n and draws so model noise is below the data's."""
    from scipy.stats import chi2 as chi2d
    O = _obs_or_die(out_dir)
    labels = ["rc1", "rc2", "rc3", "rc4"]
    profiles = {"silent": (0.0, 0.0, 0.0), "persistent": (1.0, 1.0, 1.0)}
    settings = [(b1, bp, f) for f in profiles.values() for b1 in B1_CMP for bp in BP_CMP]
    df = run_grid(cache, n, jobs, settings, labels, Path(out_dir) / "compare.csv")
    S = summarize(df, labels)
    nrow = df.groupby(["b1", "bp", "f"]).size().median()
    mse = 1 / np.sqrt(nrow)
    S["chi2"] = sum((S[l] - O[l]["z"]) ** 2 / (O[l]["se"] ** 2 + mse ** 2) for l in labels)
    name = {",".join(f"{x:g}" for x in f): k for k, f in profiles.items()}
    S["model"] = S.f.map(name)
    best = S.sort_values("chi2").groupby("model").head(1).set_index("model")
    L = ["# Result 2 at full power: silent vs persistent storage, each with its own best fit\n",
         f"{df.sid.nunique()} sessions x {DRAWS[0]} draws; model SE about {mse:.3f} per value; published-style decoder; "
         "fit to all four plain lags.\n",
         "| | beta1 | betap | rc1 | rc2 | rc3 | rc4 | chi-square (4 values) |", "|---|---|---|---|---|---|---|---|",
         "| **observed** | | | " + " | ".join(f"{O[l]['z']:.3f}" for l in labels) + " | |"]
    for m in ("silent", "persistent"):
        r = best.loc[m]
        L.append(f"| {m} | {r.b1} | {r.bp} | " + " | ".join(f"{r[l]:.3f}" for l in labels) + f" | {r.chi2:.1f} |")
    d = best.loc["persistent"].chi2 - best.loc["silent"].chi2
    L.append(f"\nDifference in chi-square (persistent minus silent): {d:.1f}. Both models have the same number of "
             f"fitted parameters, so this is a likelihood-ratio style comparison; a difference above about 4 favours "
             f"silent at roughly p < 0.05, above 9 at p < 0.003. Each model's own fit p-value: "
             f"silent {chi2d.sf(best.loc['silent'].chi2, 2):.3g}, persistent {chi2d.sf(best.loc['persistent'].chi2, 2):.3g} "
             f"(4 values, 2 fitted parameters).")
    text = "\n".join(L)
    (Path(out_dir) / "MODEL_COMPARE.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["calibrate", "evaluate", "calibrate2", "evaluate2", "stage1b", "compare"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--out", default="results/wsr_model")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--null", type=int, default=N_NULL)
    ap.add_argument("--b1", default="", help="comma list of lag-1 code strengths to try (stage1b), e.g. 0.01,0.0125,0.015")
    ap.add_argument("--res3e", default="results/silent_memory/rewarded", help="folder with the step-3e CSV files")
    a = ap.parse_args()
    DRAWS[0] = a.draws
    NNULL[0] = a.null
    if a.b1:
        B1_DA[0] = tuple(float(x) for x in a.b1.split(","))
    if a.step == "stage1b":
        evaluate_da(a.cache, a.n, a.jobs, a.out, a.res3e)
    else:
        {"calibrate": calibrate, "evaluate": evaluate, "calibrate2": calibrate2, "evaluate2": evaluate2,
         "compare": compare}[a.step](a.cache, a.n, a.jobs, a.out)
