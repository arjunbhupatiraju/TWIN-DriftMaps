"""PART 3 idea 3: is the reward memory expressed as an EXPECTATION that scales responses, rather than as a signal of its own?

Every test so far asked whether activity carries past rewards ADDITIVELY. A memory can instead change how the brain
responds to what happens next (reinforcement learning: reward prediction error = outcome - expectation; Bayesian
perception: a stimulus on the expected side is processed differently). That is invisible to additive tests because
it depends on the CURRENT trial.

Expectation built from behaviour (step 3a-2): history support for a side,
    h_t = sum_k w_k * rewarded_side(t-k),  k = 1..4,  w = behavioural rewarded-choice weights (IBL, fixed in advance)
Labels (per trial, real current trial x history):
    stim_b     E_stim   = h_t * stimulus_side_t       (is the stimulus on the recently rewarded side?)
    choice_a   E_choice = h_t * choice_t              (is the mouse choosing the recently rewarded side?)
    feedback_a E_choice and E_choice x outcome_t      (expected vs surprising reward: prediction-error form)
Null: the SAME current trial with the HISTORY taken from another mouse (imposter history), so any effect of the
current trial alone is identical in real and null labels. Removed from neurons and every label (symmetric, in the
whitened space): current stimulus, |contrast|, choice, outcome, reaction time, movement/arousal covariates, and each
label's own additive history (rewarded side lags 1-4, choice lags 1-4, outcome lags 1-4). Drift-aware as before.
Checks: fake (imposter used as real), planted (history scales the current stimulus/outcome response by 25% / 50%).

    python3 part3/expectation.py sim
    python3 part3/expectation.py run --limit 3     then  run --every 2
    python3 part3/expectation.py steinmetz --data <folder>
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
from history_map import lags  # noqa: E402

W = np.array([0.472, 0.371, 0.245, 0.259])                 # behavioural rewarded-choice weights, lags 1-4 (IBL)
KH = 4
N_NULL = 100
MIN_UNITS = 5


def history_support(choice, feedback):
    c = np.where(np.asarray(choice) >= 0, 2.0 * np.asarray(choice) - 1, 0.0)
    rc = c * (np.asarray(feedback) > 0)
    return lags(rc, KH) @ W / W.sum(), rc, c


def own_history(choice, feedback):
    """(T, 3*KH): rewarded side, choice and outcome at lags 1-4 (the additive history of that sequence)."""
    c = np.where(np.asarray(choice) >= 0, 2.0 * np.asarray(choice) - 1, 0.0)
    f = np.asarray(feedback, float)
    return np.hstack([lags(c * (f > 0), KH), lags(c, KH), lags(f, KH)])


def core_test(X, G, C, H, v, pmax=10, k_shared=1, smooth=50):
    """X (Tf, N). G (M, Tf, g) labels (index 0 real). C (M, Tf, q) each label's own controls. H (Tf, h) real covariates.
    Drift-aware statistic per label: variance of each whitened neuron explained by the residualized label columns."""
    from scipy.signal import lfilter
    from driftaware.core import _slow_components, ar_bic
    if v.sum() < 100:                       # too few trials in this window (the 50-trial smoother needs more)
        return np.nan
    Xv = X[v].astype(float); Xv = Xv[:, Xv.std(0) > 0]
    Gv, Cv = G[:, v, :], C[:, v, :]
    Z = (Xv - Xv.mean(0)) / Xv.std(0)
    U = _slow_components(Xv, k_shared, smooth)
    P = lambda A3: A3 - np.einsum("tu,mus->mts", U, np.einsum("tu,mts->mus", U, A3))
    Zc = Z - U @ (U.T @ Z)
    Gc, Cc = P(Gv), P(Cv)
    Hv = np.asarray(H, float)[v]
    Hv = Hv[:, np.isfinite(Hv).all(0) & (Hv.std(0) > 1e-8)]
    Hv = (Hv - Hv.mean(0)) / Hv.std(0)
    Hc = Hv - U @ (U.T @ Hv)
    M = G.shape[0]
    S = np.zeros(M)
    for i in range(Zc.shape[1]):
        phi = ar_bic(Zc[:, i], pmax); a = np.r_[1.0, -phi]; p = len(phi)
        f = lambda A, ax: lfilter(a, [1.0], A - A.mean(ax, keepdims=True), axis=ax)
        e = f(Zc[:, i], 0)[p:]
        Hf = f(Hc, 0)[p:]; Hf = Hf - Hf.mean(0)
        Gf = f(Gc, 1)[:, p:, :]; Cf = f(Cc, 1)[:, p:, :]
        Pm = np.linalg.pinv(Hf)
        e = e - Hf @ (Pm @ e); e = (e - e.mean()) / (e.std() + 1e-12)
        rmH = lambda A3: A3 - np.einsum("tq,mqs->mts", Hf, np.einsum("qt,mts->mqs", Pm, A3))
        Gf, Cf = rmH(Gf), rmH(Cf)
        Gf = Gf - Gf.mean(1, keepdims=True); Cf = Cf - Cf.mean(1, keepdims=True)
        CtC = np.einsum("mts,mtu->msu", Cf, Cf) + 1e-9 * np.eye(Cf.shape[2])
        R = Gf - np.einsum("mts,msc->mtc", Cf, np.linalg.solve(CtC, np.einsum("mts,mtc->msc", Cf, Gf)))
        Q, _ = np.linalg.qr(R)
        S += (np.einsum("mtc,t->mc", Q, e) ** 2).sum(1) / len(e)
    return float((1 + (S[1:] >= S[0]).sum()) / M)


def labels_for(kind, h_hist, cur):
    """h_hist (M, Tf): history support of each label's history source; cur dict of REAL current-trial signs."""
    if kind == "stim":
        return (h_hist * cur["s"][None])[..., None]
    if kind == "choice":
        return (h_hist * cur["c"][None])[..., None]
    if kind == "rpe":                                                    # expectation and its interaction with outcome
        e = h_hist * cur["c"][None]
        return np.stack([e, e * cur["f"][None]], -1)
    raise ValueError(kind)


WINDOW_TESTS = {"stim_b": ("stim",), "choice_a": ("choice",), "feedback_a": ("rpe",)}


def session_tests(Xw, hists, owns, cur, H, vw, rng, plant=True, H_loose=None, H_mid=None):
    """Xw: {window: X}. hists (M+1, Tf) history support (0 = real); owns (M+1, Tf, q). Returns dict of p-values.
    H: strict covariates (incl. the block prior and its products with the current stimulus/choice in IBL).
    H_loose: the same without the block-prior terms (reported as a secondary, less conservative test)."""
    out = {}
    for w, X in Xw.items():
        v = vw[w]
        for kind in WINDOW_TESTS[w]:
            G = labels_for(kind, hists, cur)
            out[f"{w}_{kind}_real"] = core_test(X, G, owns, H, v)
            out[f"{w}_{kind}_fake"] = core_test(X, G[1:], owns[1:], H, v)
            if H_loose is not None:
                out[f"{w}_{kind}_loosereal"] = core_test(X, G, owns, H_loose, v)
                out[f"{w}_{kind}_loosefake"] = core_test(X, G[1:], owns[1:], H_loose, v)
            if H_mid is not None:
                out[f"{w}_{kind}_midreal"] = core_test(X, G, owns, H_mid, v)
                out[f"{w}_{kind}_midfake"] = core_test(X, G[1:], owns[1:], H_mid, v)
            if not plant:
                continue
            # planted: history scales this window's response to the current stimulus / outcome by 25% or 50%
            drive = cur["s"] if kind == "stim" else (cur["f"] if kind == "rpe" else cur["c"])
            Xv = X[v].astype(float); sd = Xv.std(0) + 1e-9
            d0 = drive[v] - drive[v].mean()
            beta = (d0 @ ((Xv - Xv.mean(0)) / sd)) / max(d0 @ d0, 1e-9)       # each neuron's current response
            hz = hists[0] / (hists[0][v].std() + 1e-9)
            for tag, g in (("p25", 0.25), ("p50", 0.5)):
                Xp = X + g * np.outer(hz * (drive - drive[v].mean()), beta * sd)
                out[f"{w}_{kind}_{tag}"] = core_test(Xp, G, owns, H, v)
    return out


# ------------------------------------------------------------------------------------------- simulation
def sim(reps=40, seed=0):
    from joblib import Parallel, delayed
    import history_joint as hj
    from history_map import sim_session

    def one(i, gain):
        rng = np.random.default_rng(seed + i)
        T = 600
        side, choice, fb, prior = hj.sim_behaviour(rng, T)
        h, rc, c = history_support(choice, fb)
        s = 2 * side - 1
        X, *_ = sim_session(rng, T=T, mode="none")
        w = rng.standard_normal(X.shape[1])
        X = X + 0.8 * np.outer(s * (1 + gain * h / (h.std() + 1e-9)), w)           # stimulus response, gain-modulated
        X = X + 0.3 * np.outer(lags(rc, 1)[:, 0], rng.standard_normal(X.shape[1]))  # plus ordinary additive memory
        v = np.ones(T, bool); v[:90] = False
        hists = [h]; owns = [own_history(choice, fb)]
        for _ in range(60):
            s2, c2, f2, _ = hj.sim_behaviour(rng, T)
            hists.append(history_support(c2, f2)[0]); owns.append(own_history(c2, f2))
        cur = dict(s=s, c=c, f=fb)
        H = np.column_stack([s, c, fb])
        G = labels_for("stim", np.stack(hists), cur)
        return core_test(X, G, np.stack(owns), H, v) < .05, core_test(X, G[1:], np.stack(owns)[1:], H, v) < .05

    for gain in (0.0, 0.3):
        R = np.array(Parallel(n_jobs=-1)(delayed(one)(i, gain) for i in range(reps)))
        print(f"gain modulation {gain}: real rejection {R[:, 0].mean():.3f}   fake rejection {R[:, 1].mean():.3f}", flush=True)


# ------------------------------------------------------------------------------------------- IBL
def ibl_bank(cache):
    from gatekeeper.mv_data import MVSession
    bank = {}
    for p in sorted(Path(cache).expanduser().glob("*.npz")):
        s = MVSession(p)
        h, rc, c = history_support(s.choice, s.feedback)
        bank[s.sid] = dict(subject=s.subject, T=len(h), h=h, own=own_history(s.choice, s.feedback))
    return bank


def pooled_only_real():
    import os
    return os.environ.get("EXP_SKIP_PLANT") == "1"


def ibl_rows(path, bank, aux_dir, seed=0, pooled=False):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from history_confounds import confounds
    s = MVSession(path)
    v = cm.mask_of(s)
    if v is None or v.sum() < 200:
        return []
    Tf = len(s.contrast)
    rng = np.random.default_rng(seed + 41 + zlib.crc32(s.sid.encode()) % 100000)
    pool = [b for sid, b in bank.items() if b["subject"] != s.subject and b["T"] >= Tf]
    if len(pool) < 31:
        return []
    idx = rng.choice(len(pool), min(N_NULL + 1, len(pool)), replace=False)
    hists = np.stack([bank[s.sid]["h"]] + [pool[j]["h"][:Tf] for j in idx])
    owns = np.stack([bank[s.sid]["own"]] + [pool[j]["own"][:Tf] for j in idx])
    c = np.where(s.choice >= 0, 2.0 * s.choice - 1, 0.0)
    sd = np.sign(np.asarray(s.contrast, float))
    f = np.asarray(s.feedback, float)
    lrt = np.where(np.isfinite(s.rt) & (s.rt > 0), np.log(np.clip(s.rt, 0.05, 10)), np.nan)
    lrt = np.where(np.isfinite(lrt), lrt, np.nanmedian(lrt))
    Hc, _ = confounds(aux_dir, s.sid, Tf)
    from gatekeeper.multiverse import _sides
    prior = cm.bayes_prior(_sides(s)) - 0.5
    H_loose = np.column_stack([sd, np.abs(s.contrast), c, f, lrt, c * f] + ([Hc] if Hc is not None else []))
    H = np.column_stack([H_loose, prior, prior * sd, prior * c, prior * c * f])   # block expectation removed too
    # "mid": keep the SIGNED expectation, remove only UNSIGNED strength (how strongly history / the block favour a side,
    # i.e. certainty or recent reward rate), which a congruence label can also pick up
    h_real = bank[s.sid]["h"]
    H_mid = np.column_stack([H_loose, prior, np.abs(prior), np.abs(h_real)])
    cur = dict(s=sd, c=c, f=f)
    vw = {"stim_b": v & (sd != 0), "choice_a": v & s.valid["firstMovement"], "feedback_a": v & s.valid["feedback"]}
    rows = []
    groups = [[r] for r in s.regions] if not pooled else [list(s.regions)]
    for regs in groups:
        r = regs[0] if not pooled else "ALL"
        Xw = {}
        for w in WINDOW_TESTS:
            parts = [s.X(q, w) for q in regs]
            parts = [x for x in parts if x is not None]
            if not parts:
                continue
            X = np.hstack(parts)
            if (X[vw[w]].std(0) > 0).sum() >= MIN_UNITS:
                Xw[w] = X[:, X[vw[w]].std(0) > 0].astype(float)
        if not Xw:
            continue
        rows.append(dict(sid=s.sid, subject=s.subject, region=r,
                         **session_tests(Xw, hists, owns, cur, H, vw, rng, plant=not pooled_only_real(), H_loose=H_loose,
                                         H_mid=H_mid)))
    return rows


def report(df, out_dir, title):
    from verify_numbers import region_p, sig
    L = [f"# Idea 3: history as an expectation that scales responses. {title} ({df.sid.nunique()} sessions, "
         f"{df.subject.nunique()} mice)\n",
         "Null: same current trial, history from another mouse. Fake must be ~0.05 / 0 regions; planted (history scales "
         "the current response by 25% / 50%) must be detected. Regions at FDR 5% (session rate).\n",
         "| Window | Test | Real | Fake | Planted 25% | Planted 50% | Real (block prior not removed) | Fake (same) |",
         "|---|---|---|---|---|---|---|---|"]
    names = {"stim": "stimulus on recently rewarded side", "choice": "choosing recently rewarded side",
             "rpe": "expected vs surprising outcome"}
    for w, kinds in WINDOW_TESTS.items():
        for k in kinds:
            cells = []
            for tag in ("real", "fake", "p25", "p50", "loosereal", "loosefake"):
                col = f"{w}_{k}_{tag}"
                if col not in df:
                    cells.append("-"); continue
                pr = region_p(df.dropna(subset=[col]), col)
                cells.append(f"{len(sig(pr))} / {len(pr)} ({(df[col] < .05).mean():.3f})")
            L.append(f"| {w} | {names[k]} | " + " | ".join(cells) + " |")
    real_cols = [f"{w}_{k}_real" for w, ks in WINDOW_TESTS.items() for k in ks]
    for col in real_cols:
        if col in df:
            pr = region_p(df.dropna(subset=[col]), col)
            sg = sorted(sig(pr))
            L.append(f"\nRegions ({col[:-5]}): " + (", ".join(sg) if sg else "none"))
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "EXPECTATION.md").write_text(text + "\n")
    print(text)


def run(cache, aux_dir, out_dir, limit, jobs, every, pooled=False):
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / "sessions"; out.mkdir(parents=True, exist_ok=True)
    bank = ibl_bank(cache)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[::every][: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(todo)} to run", flush=True)
    t0 = time.time()

    def work(p):
        try:
            rows = ibl_rows(p, bank, aux_dir, pooled=pooled)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True); rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    for i, n in enumerate(Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)):
        print(f"[{i + 1}/{len(todo)}] {n} regions  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fs = [f for f in out.glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    if pooled:
        pooled_report(df, out_dir)
    else:
        report(df, out_dir, "IBL")


def pooled_report(df, out_dir):
    from scipy.stats import chi2, binomtest
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    L = [f"# Idea 3 POOLED (all neurons per session): history as an expectation ({df.sid.nunique()} sessions, "
         f"{df.subject.nunique()} mice)\n", "| Window | Test | Condition | Sessions | Fraction p < 0.05 | Binomial p | Fisher p |",
         "|---|---|---|---|---|---|---|"]
    for w, kinds in WINDOW_TESTS.items():
        for k in kinds:
            for tag in ("fake", "real", "p25", "p50", "loosefake", "loosereal", "midfake", "midreal"):
                col = f"{w}_{k}_{tag}"
                if col not in df:
                    continue
                d = df[col].dropna()
                n = int((d < .05).sum())
                L.append(f"| {w} | {k} | {tag} | {len(d)} | {n / max(len(d), 1):.3f} | "
                         f"{binomtest(n, len(d), .05, 'greater').pvalue if len(d) else float('nan'):.1e} | {fisher(d.values):.1e} |")
    text = "\n".join(L)
    (Path(out_dir) / "EXPECTATION_POOLED.md").write_text(text + "\n")
    print(text)


# ------------------------------------------------------------------------------------------- Steinmetz
def run_steinmetz(data_dir, out_dir, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    import steinmetz_replication as sr
    sessions = sr.load_sessions(data_dir)
    bank = []
    for i, d in enumerate(sessions):
        contrast, choice, fb, side = sr.task(d)
        bank.append(dict(i=i, subject=str(d["mouse_name"]), T=len(choice), h=history_support(choice, fb)[0],
                         own=own_history(choice, fb)))

    def work(i, d):
        contrast, choice, fb, side = sr.task(d)
        mouse = str(d["mouse_name"])
        lens = sorted([b["T"] for b in bank if b["subject"] != mouse], reverse=True)
        if len(lens) < 21:
            return []
        T = min(len(choice), lens[20])
        pool = [b for b in bank if b["subject"] != mouse and b["T"] >= T]
        rng = np.random.default_rng(51 + i)
        idx = rng.choice(len(pool), min(N_NULL + 1, len(pool)), replace=False)
        hists = np.stack([bank[i]["h"][:T]] + [pool[j]["h"][:T] for j in idx])
        owns = np.stack([bank[i]["own"][:T]] + [pool[j]["own"][:T] for j in idx])
        c = np.where(choice >= 0, 2.0 * choice - 1, 0.0)[:T]
        sd = np.sign(contrast)[:T]; f = fb[:T]
        Hc = sr.confounds(d, len(choice))
        rt = np.asarray(d["response_time"], float).ravel()[:T]
        H = np.column_stack([sd, np.abs(contrast[:T]), c, f, np.log(np.clip(rt, 0.05, 10)), c * f] +
                            ([Hc[:T]] if Hc is not None else []))
        v = (choice[:T] >= 0); v[:10] = False
        _bin = lambda t: 50 + np.round(np.asarray(t, float).ravel() / 0.01).astype(int)
        rbin = _bin(d["response_time"])[:T]
        fbin = _bin(d["feedback_time"])[:T]
        nb = d["spks"].shape[2]
        vw = {"stim_b": v & (sd != 0), "choice_a": v & (rbin - 10 >= 50) & (rbin <= nb), "feedback_a": v & (fbin + 20 <= nb)}
        wins = {"stim_b": (np.full(T, 55), np.full(T, 65)), "choice_a": (rbin - 10, rbin), "feedback_a": (fbin, fbin + 20)}
        areas = np.asarray(d["brain_area"]).astype(str)
        rows = []
        for r in sorted(set(areas) - sr.SKIP):
            sel = areas == r
            if sel.sum() < MIN_UNITS:
                continue
            S = d["spks"][sel][:, :T]
            Xw = {}
            for w, (lo, hi) in wins.items():
                lo = np.clip(lo, 0, nb); hi = np.clip(hi, 0, nb)
                X = np.stack([S[:, t, lo[t]:hi[t]].sum(1) for t in range(T)]).astype(float)
                if (X[vw[w]].std(0) > 0).sum() >= MIN_UNITS:
                    Xw[w] = X[:, X[vw[w]].std(0) > 0]
            if Xw:
                rows.append(dict(sid=f"s{i}", subject=mouse, region=r,
                                 **session_tests(Xw, hists, owns, dict(s=sd, c=c, f=f), H, vw, rng)))
        return rows

    res = Parallel(n_jobs=jobs, verbose=5)(delayed(work)(i, d) for i, d in enumerate(sessions))
    report(pd.DataFrame([r for rows in res for r in rows]), out_dir, "Steinmetz 2019")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "run", "steinmetz"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--aux", default="data/part3_aux2")
    ap.add_argument("--data", default="~/Downloads/MULTIVERSE/steinmetz_data")
    ap.add_argument("--out", default="results/expectation")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--pooled", action="store_true", help="one test per session with all neurons pooled")
    ap.add_argument("--skip-plant", action="store_true", help="skip the planted checks (already measured)")
    a = ap.parse_args()
    if a.skip_plant:
        import os
        os.environ["EXP_SKIP_PLANT"] = "1"
    if a.step == "sim":
        sim()
    elif a.step == "run":
        run(a.cache, a.aux, a.out + ("/ibl_pooled" if a.pooled else "/ibl"), a.limit, a.jobs, a.every, a.pooled)
    else:
        run_steinmetz(a.data, a.out + "/steinmetz", a.jobs)
