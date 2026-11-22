"""PART 3 step 1b: how many trials back does each region remember ANYTHING about the past trial?

Step 1 tested the stimulus side BEYOND the mouse's own choices and feedback. Because mice are mostly correct,
side and choice agree on most trials, so that test only used error trials and was very conservative.
The zebrafish history signal is the side the animal experienced and acted on, and the mouse behavioural
"action kernel" (Findling 2025) is about past choices. So here the label at lag k is the WHOLE trial t-k:
[stimulus side, choice, feedback] (3 columns), and we ask whether pre-stimulus activity on trial t carries
information about it, beyond:
   - the same three variables at the other lags 1..K, and
   - the Bayes block prior,
all taken from the label's OWN sequence.
Statistic: drift-aware (shared slow PC + per-neuron AR whitening, applied to neurons and labels alike), then
sum over neurons of the variance explained by the 3 residualized label columns.
Null: IMPOSTER sessions = real behaviour sequences of OTHER mice (first T trials), which keep realistic choice
and feedback statistics but cannot be linked to this brain.
Per variable (side, choice, feedback) single-column tests at each lag are also reported.

    python3 part3/history_joint.py sim
    python3 part3/history_joint.py run --limit 3
    python3 part3/history_joint.py run
    python3 part3/history_joint.py report
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

K = 6
N_NULL = 100
MIN_UNITS = 5
WINDOW = "prior_a"
VARS = ("side", "choice", "feedback")


def seq_matrix(side, choice, feedback, prior):
    """(T, 3K+1): [side lags 1..K, choice lags 1..K, feedback lags 1..K, prior], centered."""
    s = np.nan_to_num(np.asarray(side, float) - 0.5)
    c = np.where(np.asarray(choice) >= 0, np.asarray(choice, float) - 0.5, 0.0)
    f = np.asarray(feedback, float) / 2.0
    return np.column_stack([lags(s, K), lags(c, K), lags(f, K), np.asarray(prior, float) - 0.5])


def col(var, k):
    return VARS.index(var) * K + (k - 1)


def joint_tests(X, owns, v, H=None, drop_prior=False, only=None, pmax=10, k_shared=1, smooth=50):
    """owns: (M+1, Tfull, 3K+1), index 0 = real. Returns p and effect for: joint lag k (3 columns) and each
    single variable at lag k. H (Tfull, q): optional real per-trial covariates (movement, licking, pupil, timing)
    removed from the neurons AND from every label in the whitened space (symmetric). drop_prior: do not control
    the Bayes prior (only the other lags)."""
    if drop_prior:
        owns = owns[:, :, :-1]
    from scipy.signal import lfilter
    from driftaware.core import _slow_components, ar_bic
    Xv = X[v].astype(float)
    Xv = Xv[:, Xv.std(0) > 0]
    O = owns[:, v, :]
    M, T, Q = O.shape
    Z = (Xv - Xv.mean(0)) / Xv.std(0)
    U = _slow_components(Xv, k_shared, smooth)
    Zc = Z - U @ (U.T @ Z)
    Oc = O - np.einsum("tu,mus->mts", U, np.einsum("tu,mts->mus", U, O))
    Hc = None
    if H is not None:
        Hv = np.asarray(H, float)[v]
        Hv = Hv[:, np.isfinite(Hv).all(0) & (Hv.std(0) > 1e-8)]
        Hv = (Hv - Hv.mean(0)) / Hv.std(0)
        Hc = Hv - U @ (U.T @ Hv)
    tests = [("joint", k, [col(vv, k) for vv in VARS]) for k in range(1, K + 1)]
    tests += [(vv, k, [col(vv, k)]) for vv in VARS for k in range(1, K + 1)]
    # "rewarded": the side of the choice that was REWARDED k trials back (= side + choice, since side = choice on
    # correct trials), i.e. reward-gated choice memory. Controls: everything except side_k and choice_k.
    tests += [("rewarded", k, [col("side", k), col("choice", k)], np.array([1.0, 1.0])) for k in range(1, K + 1)]
    if only is not None:
        tests = [t for t in tests if t[0] in only]
    S = np.zeros((len(tests), M))
    for i in range(Zc.shape[1]):
        phi = ar_bic(Zc[:, i], pmax)
        a = np.r_[1.0, -phi]; p = len(phi)
        e = lfilter(a, [1.0], Zc[:, i] - Zc[:, i].mean())[p:]
        e = (e - e.mean()) / (e.std() + 1e-12)
        Of = lfilter(a, [1.0], Oc - Oc.mean(1, keepdims=True), axis=1)[:, p:, :]
        Of = Of - Of.mean(1, keepdims=True)
        if Hc is not None and Hc.shape[1]:
            Hf = lfilter(a, [1.0], Hc - Hc.mean(0), axis=0)[p:]
            Hf = Hf - Hf.mean(0)
            Pm = np.linalg.pinv(Hf)
            e = e - Hf @ (Pm @ e)
            e = (e - e.mean()) / (e.std() + 1e-12)
            Of = Of - np.einsum("tq,mqs->mts", Hf, np.einsum("qt,mts->mqs", Pm, Of))
        for ti, tst in enumerate(tests):
            cols = tst[2]
            ctrl = [j for j in range(Q) if j not in cols]
            C = Of[:, :, ctrl]
            G = Of[:, :, cols]                                                       # (M, T', c)
            if len(tst) > 3:
                G = (G @ tst[3])[..., None]
            CtC = np.einsum("mts,mtu->msu", C, C) + 1e-9 * np.eye(len(ctrl))
            CtG = np.einsum("mts,mtc->msc", C, G)
            R = G - np.einsum("mts,msc->mtc", C, np.linalg.solve(CtC, CtG))
            Qm, _ = np.linalg.qr(R)                                                  # orthonormal basis per label
            S[ti] += (np.einsum("mtc,t->mc", Qm, e) ** 2).sum(1) / len(e)
    p = (1 + (S[:, 1:] >= S[:, [0]]).sum(1)) / M
    eff = S[:, 0] / (S[:, 1:].mean(1) + 1e-12)
    return [(t[0], t[1], float(pp), float(ee)) for t, pp, ee in zip(tests, p, eff)]


# ------------------------------------------------------------------------------------------- simulation
def sim_behaviour(rng, T):
    from gatekeeper.multiverse import pseudo_pl
    from compare_methods import bayes_prior
    pl = pseudo_pl(T, rng)
    side = (rng.random(T) >= pl).astype(float)
    choice = np.zeros(T, int)
    hist = 0.0
    for t in range(T):  # choices follow the stimulus 80% of the time plus a decaying choice-history bias
        pr = 0.8 if side[t] == 1 else 0.2
        pr = 1 / (1 + np.exp(-(np.log(pr / (1 - pr)) + hist)))
        choice[t] = rng.random() < pr
        hist = 0.6 * hist + 0.6 * (2 * choice[t] - 1)
    fb = np.where(choice == side, 1.0, -1.0)
    return side, choice, fb, bayes_prior(side)


def sim_check(reps=40, n_null=100, seed=0):
    from joblib import Parallel, delayed
    from history_map import sim_session  # drift generator

    def one(i, mode):
        rng = np.random.default_rng(seed + 1000 * i)
        T = 600
        side, choice, fb, prior = sim_behaviour(rng, T)
        X, *_ = sim_session(rng, T=T, mode="none")                   # drift-only neurons
        w = rng.standard_normal(X.shape[1])
        c = choice - 0.5
        if mode == "last_choice":
            X = X + 0.5 * np.outer(lags(c, 1)[:, 0], w)
        elif mode == "choice_integrator":
            kern = np.exp(-np.arange(10) / 2.5)
            X = X + 0.5 * np.outer(lags(c, 10) @ kern / kern.sum() * 2, w)
        v = np.ones(T, bool); v[:90] = False
        owns = [seq_matrix(side, choice, fb, prior)]
        for _ in range(n_null):
            owns.append(seq_matrix(*sim_behaviour(rng, T)))
        res = joint_tests(X, np.stack(owns), v)
        return [pp < .05 for name, k, pp, ee in res if name == "joint"]

    for mode in ("none", "last_choice", "choice_integrator"):
        R = np.array(Parallel(n_jobs=-1)(delayed(one)(i, mode) for i in range(reps)))
        print(f"{mode:18s} joint lag test, reject rate lags 1..{K}: {np.round(R.mean(0), 2)}", flush=True)


# ------------------------------------------------------------------------------------------- real data
def behaviour_bank(cache):
    import compare_methods as cm
    from gatekeeper.multiverse import _sides
    from gatekeeper.mv_data import MVSession
    bank = {}
    for p in sorted(Path(cache).expanduser().glob("*.npz")):
        s = MVSession(p)
        side = _sides(s)
        bank[s.sid] = dict(subject=s.subject, T=len(s.contrast),
                           M=seq_matrix(side, s.choice, s.feedback, cm.bayes_prior(side)))
    return bank


def session_rows(path, bank, n_null=N_NULL, seed=0):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    s = MVSession(path)
    v = cm.mask_of(s)
    if v is None or v.sum() < 200:
        return []
    Tf = len(s.contrast)
    rng = np.random.default_rng(seed + zlib.crc32(s.sid.encode()) % 100000)
    pool = [b["M"][:Tf] for sid, b in bank.items() if b["subject"] != s.subject and b["T"] >= Tf]
    if len(pool) < 30:
        return []
    idx = rng.choice(len(pool), size=min(n_null, len(pool)), replace=False)
    owns = np.stack([bank[s.sid]["M"]] + [pool[j] for j in idx])
    rows = []
    for r in s.regions:
        X = s.X(r, WINDOW)
        if X is None:
            continue
        keep = X[v].std(0) > 0
        if keep.sum() < MIN_UNITS:
            continue
        row = dict(sid=s.sid, subject=s.subject, split=s.split, region=r, n_units=int(keep.sum()),
                   n_trials=int(v.sum()), n_null=len(idx))
        for name, k, pp, ee in joint_tests(X[:, keep], owns, v):
            row[f"p_{name}{k}"] = pp
            row[f"eff_{name}{k}"] = ee
        rows.append(row)
    return rows


def run(cache, out_dir, limit, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / "sessions"
    out.mkdir(parents=True, exist_ok=True)
    print("loading behaviour of all sessions (imposter pool) ...", flush=True)
    bank = behaviour_bank(cache)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(paths) - len(todo)} already done, {len(todo)} to run", flush=True)
    t0 = time.time()

    def work(p):
        try:
            rows = session_rows(p, bank)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True)
            rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    for i, n in enumerate(Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)):
        print(f"[{i + 1}/{len(todo)}] {n} regions  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def report(out_dir):
    import pandas as pd
    from verify_numbers import region_p, sig
    import slowmode as sm
    files = [f for f in (Path(out_dir) / "sessions").glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    regs = sorted(df.region.unique())
    div = sm.division_of(regs)
    L = [f"# How far back does each region remember the past trial? ({df.sid.nunique()} sessions, "
         f"{df.subject.nunique()} mice)\n",
         "Label at lag k = trial t-k's [stimulus side, choice, feedback], beyond the other lags and the block prior. "
         "Drift-aware test; imposter null (other mice's real behaviour). Region rule: >= 2 sessions; Fisher; BH.\n",
         "## Regions significant at FDR 5%\n",
         "| Lag | Whole trial: rate | Whole trial: regions | Side only | Choice only | Feedback only |",
         "|---|---|---|---|---|---|"]
    sets = {}
    for k in range(1, K + 1):
        cells = []
        for name in ("joint",) + VARS:
            pr = region_p(df, f"p_{name}{k}")
            sets[(name, k)] = sig(pr)
            cells.append(f"{len(sets[(name, k)])} / {len(pr)}")
        L.append(f"| {k} | {(df[f'p_joint{k}'] < .05).mean():.3f} | " + " | ".join(cells) + " |")
    rows = []
    for r in regs:
        d = df[df.region == r]
        ks = [k for k in range(1, K + 1) if r in sets[("joint", k)]]
        rows.append(dict(region=r, division=div.get(r, r), sessions=d.sid.nunique(), mice=d.subject.nunique(),
                         lags=",".join(map(str, ks)), persistence=max(ks) if ks else 0,
                         choice_lags=",".join(str(k) for k in range(1, K + 1) if r in sets[("choice", k)]),
                         side_lags=",".join(str(k) for k in range(1, K + 1) if r in sets[("side", k)]),
                         feedback_lags=",".join(str(k) for k in range(1, K + 1) if r in sets[("feedback", k)]),
                         **{f"eff{k}": d[f"eff_joint{k}"].median() for k in range(1, K + 1)}))
    t = pd.DataFrame(rows)
    t2 = t[t.sessions >= 2]
    L += ["\n## Persistence by division (regions with >= 2 sessions)\n",
          "| Division | Regions | Lag 1 | Lag 2 | Lag 3+ | Median persistence of significant regions |", "|---|---|---|---|---|---|"]
    for dv, h in t2.groupby("division"):
        s1 = sum(r in sets[("joint", 1)] for r in h.region)
        s2 = sum(r in sets[("joint", 2)] for r in h.region)
        s3 = sum(h.persistence >= 3)
        sg = h[h.persistence > 0]
        L.append(f"| {dv} | {len(h)} | {s1} | {s2} | {s3} | {sg.persistence.median() if len(sg) else 0:.0f} |")
    L += ["\n## Regions remembering beyond the last trial (persistence >= 2)\n",
          "| Region | Division | Sessions | Mice | Whole-trial lags | Choice lags | Side lags | Feedback lags | Effect lag 1..6 |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in t2[t2.persistence >= 2].sort_values(["persistence", "division"], ascending=[False, True]).iterrows():
        L.append(f"| {r.region} | {r.division} | {r.sessions} | {r.mice} | {r.lags} | {r.choice_lags} | {r.side_lags} | "
                 f"{r.feedback_lags} | " + " ".join(f"{r[f'eff{k}']:.2f}" for k in range(1, K + 1)) + " |")
    text = "\n".join(L)
    (Path(out_dir) / "HISTORY_JOINT.md").write_text(text + "\n")
    t.to_csv(Path(out_dir) / "history_joint_regions.csv", index=False)
    print(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "run", "report"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--out", default="results/history_joint")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    if a.step == "sim":
        sim_check(reps=a.reps)
    elif a.step == "run":
        run(a.cache, a.out, a.limit, a.jobs)
    else:
        report(a.out)
