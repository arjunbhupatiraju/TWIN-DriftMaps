"""PART 3 gap check: do the history lags survive a PUBLISHED-STYLE method, not just a simple standard test?

Step 1 compared our drift-aware lag test with a simple per-neuron test (about 100 regions per lag vs 14 / 0 / 0 ...).
A reviewer will ask what the field's best published pipeline gives on the same lags. This runs our
reimplementation of Findling et al. 2025 (Nature) on the rewarded-choice lags, exactly as that paper decodes the
prior, with the same pseudo-session logic they used for the action-kernel prior:

    label        rewarded choice at lag k (+1 rewarded right, -1 rewarded left, 0 error), k = 1..4
    window       -600 to -100 ms before stimulus onset (their inter-trial window); first 90 trials excluded
    decoder      Lasso, nested 5-fold cross-validation, cross-validated R2 (compare_methods.cv_predictions)
    null         pseudo-sessions: new block and stimulus sequence from the task's generative process, and the mouse's
                 choices SIMULATED from a model fitted to that session (stimulus, block prior, choice and
                 rewarded-choice lags 1-4), so pseudo labels have the real behaviour's history structure
    region map   Fisher across sessions, Benjamini-Hochberg 5% (as in the paper)
    fake         ctl_rc2: a pseudo-session label used as if real (must give ~0 regions)

The published-style method decodes each label on its own (it does not remove the other lags or the prior), which
is how such claims are made. Comparing its map with step 1c (where lags 2+ give 0 regions once lag 1 and the
prior are removed) shows whether the drop comes from drift, from correlated history variables, or both.

    python3 part3/history_findling.py sim
    python3 part3/history_findling.py run --limit 3
    python3 part3/history_findling.py run --every 2
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

KMAX = 4
VARS = ("rc1", "rc2", "rc3", "rc4", "ctl_rc2")
# controlled: each lag with lag 1..k-1, choice lags 1..k and the block prior removed from the label (real and pseudo alike)
VARS_C = ("rc2c", "rc3c", "rc4c", "ctl_rc2c")
ACTIVE = [VARS]
N_NULL = 100


def _logit_fit(X, y, lam=1.0, iters=50):
    X = np.column_stack([np.ones(len(y)), X])
    w = np.zeros(X.shape[1]); P = lam * np.eye(X.shape[1]); P[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        step = np.linalg.solve((X * (p * (1 - p))[:, None]).T @ X + P + 1e-9 * np.eye(len(w)), X.T @ (p - y) + P @ w)
        w -= step
        if np.abs(step).max() < 1e-7:
            break
    return w


def _stim(contrast):
    c = np.nan_to_num(np.asarray(contrast, float))
    f = lambda x: np.tanh(5 * x) / np.tanh(5)
    return np.column_stack([f(np.clip(c, 0, None)), f(np.clip(-c, 0, None))])


def behaviour_model(contrast, choice, feedback, prior):
    """Logistic choice model of one session: stimulus, block prior, choice lags 1-4, rewarded-choice lags 1-4."""
    c = np.where(np.asarray(choice) >= 0, 2.0 * np.asarray(choice) - 1, 0.0)
    rc = c * (np.asarray(feedback) > 0)
    X = np.column_stack([_stim(contrast), prior, lags(c, KMAX), lags(rc, KMAX)])
    ok = c != 0; ok[:KMAX] = False
    return _logit_fit(X[ok], (c[ok] > 0).astype(float))


def simulate_session(w, Tf, mags, rng):
    """A pseudo-session: block and stimulus sequence from the task's generative process, choices from model w."""
    from gatekeeper.multiverse import pseudo_pl
    from compare_methods import bayes_prior
    pl = pseudo_pl(Tf, rng)
    side = (rng.random(Tf) >= pl).astype(float)
    contrast = (2 * side - 1) * rng.choice(mags, Tf)
    prior = bayes_prior(side) - 0.5
    S = _stim(contrast)
    c = np.zeros(Tf); rc = np.zeros(Tf)
    for t in range(Tf):
        hc = [c[t - k] if t - k >= 0 else 0.0 for k in range(1, KMAX + 1)]
        hr = [rc[t - k] if t - k >= 0 else 0.0 for k in range(1, KMAX + 1)]
        x = np.r_[1.0, S[t], prior[t], hc, hr]
        pr = 1 / (1 + np.exp(-np.clip(x @ w, -30, 30)))
        c[t] = 1.0 if rng.random() < pr else -1.0
        correct = (c[t] > 0) == (side[t] > 0)
        rc[t] = c[t] * correct
    fb = np.where(rc != 0, 1.0, -1.0)
    return dict(rc=rc, c=c, prior=prior, side=side, fb=fb)


def _k(var):
    return int("".join(ch for ch in var if ch.isdigit()))


def controlled(rc, c, prior, k, v):
    """Rewarded choice at lag k with everything it is correlated with removed: rewarded choice at lags 1..k-1,
    choice at lags 1..k, and the block prior on the current trial (least squares on the analysed trials)."""
    y = lags(rc, k)[:, k - 1]
    Z = [np.ones(len(rc)), np.asarray(prior, float)] + [lags(c, k)[:, j] for j in range(k)]
    Z += [lags(rc, k)[:, j] for j in range(k - 1)]
    Z = np.column_stack(Z)[v]
    yv = y[v]
    return yv - Z @ np.linalg.lstsq(Z, yv, rcond=None)[0]


def label_matrix(real, w, Tf, mags, v, var, n_null, rng, sims=None):
    """real: dict(rc, c, prior) of the real session. Column 0 = tested label, 1.. = pseudo-session labels."""
    k = _k(var)
    if sims is None:
        sims = [simulate_session(w, Tf, mags, rng) for _ in range(n_null + 1)]
    ctrl = var.endswith("c")
    lab = (lambda d: controlled(d["rc"], d["c"], d["prior"], k, v)) if ctrl else (lambda d: lags(d["rc"], k)[:, k - 1][v])
    first = lab(real) if not var.startswith("ctl_") else lab(sims[0])
    return np.column_stack([first] + [lab(d) for d in sims[1:]])


def session_rows(path, n_null, seed, window="prior_a"):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from gatekeeper.multiverse import _sides
    s = MVSession(path)
    v = cm.mask_of(s)
    if v is None or v.sum() < 100:
        return []
    Tf = len(s.contrast)
    prior = cm.bayes_prior(_sides(s)) - 0.5
    w = behaviour_model(s.contrast, s.choice, s.feedback, prior)
    mags = np.abs(np.nan_to_num(np.asarray(s.contrast, float)))
    c = np.where(s.choice >= 0, 2.0 * s.choice - 1, 0.0)
    rc_real = c * (np.asarray(s.feedback) > 0)
    real = dict(rc=rc_real, c=c, prior=prior)
    rng = np.random.default_rng(seed)
    rows = []
    sims = [simulate_session(w, Tf, mags, rng) for _ in range(n_null + 1)]     # one set of pseudo-sessions per session
    for var in ACTIVE[0]:
        Y = label_matrix(real, w, Tf, mags, v, var, n_null, rng, sims)
        if np.std(Y[:, 0]) == 0:
            continue
        for r in s.regions:
            X = s.X(r, window)
            if X is None:
                continue
            X = X[v]; X = X[:, X.std(0) > 0]
            if X.shape[1] < cm.MIN_UNITS:
                continue
            R = cm.r2(Y, cm.cv_predictions(X, Y))
            rows.append(dict(sid=s.sid, subject=s.subject, split=s.split, var=var, region=r, n_units=X.shape[1],
                             r2=float(R[0]), r2_null_median=float(np.median(R[1:])),
                             p=float((1 + (R[1:] >= R[0]).sum()) / len(R))))
    return rows


def run(cache, out_dir, limit, jobs, every, n_null):
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / "sessions"; out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[::every][: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(todo)} to run, {n_null} pseudo-sessions each", flush=True)
    t0 = time.time()

    def work(p):
        t = time.time()
        try:
            rows = session_rows(p, n_null, 77 + zlib.crc32(p.stem.encode()) % 100000)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True); rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows), (time.time() - t) / 60

    try:
        gen = Parallel(n_jobs=jobs, return_as="generator_unordered")(delayed(work)(p) for p in todo)
    except Exception:
        gen = Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)
    for i, (n, m) in enumerate(gen):          # printed as each session finishes, in any order
        print(f"[{i + 1}/{len(todo)}] {n} rows, {m:.1f} min for this session  ({(time.time() - t0) / 60:.1f} min total)",
              flush=True)
    fs = [f for f in out.glob("*.csv") if f.stat().st_size > 1]
    report(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True), out_dir)


def report(df, out_dir):
    from verify_numbers import region_p, sig
    L = [f"# Published-style method (Findling et al. 2025 reimplementation) on the rewarded-choice lags "
         f"({df.sid.nunique()} sessions, {df.subject.nunique()} mice)\n",
         "Each lag decoded on its own from the -600 to -100 ms window; pseudo-session null with simulated choices; "
         "Fisher across sessions, BH 5%. ctl_rc2 is a fake label (must be ~0).\n",
         "| Label | Regions significant / tested | Session fraction p < 0.05 | Replicated in both splits | Median R2 minus null |",
         "|---|---|---|---|---|"]
    for var in sorted(df["var"].unique(), key=lambda x: (x.startswith("ctl"), x)):
        d = df[df["var"] == var]
        if not len(d):
            continue
        pr = region_p(d, "p")
        sg = sig(pr)
        rep = "-"
        if "split" in d and d.split.nunique() > 1:
            sets = [set(sig(region_p(g, "p"))) for _, g in d.groupby("split")]
            rep = str(len(set.intersection(*sets)))
        L.append(f"| {var} | {len(sg)} / {len(pr)} | {(d.p < .05).mean():.3f} | {rep} | "
                 f"{(d.r2 - d.r2_null_median).median():.4f} |")
    for var in ("rc2", "rc3", "rc4", "rc2c", "rc3c", "rc4c"):
        d = df[df["var"] == var]
        if len(d):
            sg = sorted(sig(region_p(d, "p")))
            L.append(f"\nRegions ({var}): " + (", ".join(sg) if sg else "none"))
    L.append("\nLabels ending in c: the same lag with lags 1..k-1, choice lags 1..k and the block prior removed "
             "from the label (published-style decoder otherwise unchanged).")
    L.append("\nCompare: step 1c (drift-aware, other lags and prior removed) gave 0 regions at lags 2 and above.")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "HISTORY_FINDLING.md").write_text(text + "\n")
    print(text)


def sim(reps=30, seed=0):
    """Neurons encode only the LAST rewarded choice and the block prior (no lag-2 memory):
    the plain lag-2 label should be inflated, the controlled one calibrated. Then a real lag-2 code: both detect it."""
    from joblib import Parallel, delayed
    import compare_methods as cm
    import history_joint as hj
    from history_map import sim_session

    def one(i, beta2):
        rng = np.random.default_rng(seed + 101 * i)
        T = 500
        side, choice, fb, prior = hj.sim_behaviour(rng, T)
        contrast = (2 * side - 1) * rng.choice([0, .0625, .125, .25, 1.0], T)
        X, *_ = sim_session(rng, T=T, N=20, mode="none")
        c = 2.0 * choice - 1; rc = c * (fb > 0)
        L = lags(rc, 2)
        X = X + 0.5 * np.outer(L[:, 0], rng.standard_normal(X.shape[1]))                     # lag-1 code
        X = X + 0.5 * np.outer((prior - prior.mean()) / prior.std(), rng.standard_normal(X.shape[1]))  # prior code
        X = X + beta2 * np.outer(L[:, 1], rng.standard_normal(X.shape[1]))                   # lag-2 code (if any)
        v = np.ones(T, bool); v[:90] = False
        w = behaviour_model(contrast, choice, fb, prior - 0.5)
        real = dict(rc=rc, c=c, prior=prior - 0.5)
        sims = [simulate_session(w, T, np.abs(contrast), rng) for _ in range(61)]
        out = []
        for var in ("rc2", "rc2c"):
            Y = label_matrix(real, w, T, np.abs(contrast), v, var, 60, rng, sims)
            R = cm.r2(Y, cm.cv_predictions(X[v], Y))
            out.append((1 + (R[1:] >= R[0]).sum()) / len(R) < .05)
        return out

    for beta2, lab in ((0.0, "no lag-2 code (only lag 1 + prior)"), (0.3, "real lag-2 code")):
        R = np.array(Parallel(n_jobs=-1)(delayed(one)(i, beta2) for i in range(reps)))
        print(f"{lab:36s} plain rc2 rejects {R[:, 0].mean():.3f}   controlled rc2c rejects {R[:, 1].mean():.3f}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "run"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--out", default="results/history_findling")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--null", type=int, default=N_NULL)
    ap.add_argument("--controlled", action="store_true", help="lags with lag 1..k-1, choice lags and prior removed")
    a = ap.parse_args()
    if a.step == "sim":
        sim()
    else:
        if a.controlled:
            ACTIVE[0] = VARS_C
        run(a.cache, a.out + ("_controlled" if a.controlled else ""), a.limit, a.jobs, a.every, a.null)
