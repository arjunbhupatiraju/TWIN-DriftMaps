"""PART 3 step 2: does the brain's memory of past outcomes DRIVE the next decision?

Step 1c: about 190 regions carry the last trial's outcome, about 77 carry it two trials back, and about 10
(midbrain/hindbrain: MRN, SCm, PPN, PAG, PRNr, PSV, PGRN; plus MOs, PO, CUL4/5) three trials back, after
removing licking, wheel movement, pupil and time since feedback.

Question: on trials where a region's memory of "last trial was rewarded" is STRONGER than the actual history
implies, does the mouse behave as if it was more rewarded (stay with the previous choice more, respond faster)?
That trial-by-trial link is the standard evidence that a neural memory is used, short of a perturbation.

Per session and region:
  1. Pre-stimulus activity (confounds and the shared slow component removed), cross-validated ridge readout of
     the previous trial's feedback  ->  neural estimate f_hat(t).
  2. Memory fluctuation r(t) = f_hat(t) minus what the ACTUAL history (side, choice, feedback at lags 1-3, block
     prior) predicts, with its slow trend removed.
  3. Behaviour on trial t: stay (repeat the previous choice) and log reaction time, regressed on stimulus,
     actual history, the behaviour's own slow trend, and r(t). Statistic: t-value of r.
  4. Null: circular shifts of r (keeps its autocorrelation, breaks its alignment with behaviour).

    python3 part3/history_behaviour.py sim
    python3 part3/history_behaviour.py run --limit 3
    python3 part3/history_behaviour.py run
    python3 part3/history_behaviour.py report
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "next", ROOT / "driftaware" / "src", HERE):
    sys.path.insert(0, str(p))
from history_map import lags  # noqa: E402

N_SHIFT = 200
MIN_UNITS = 5
WINDOW = "prior_a"
FOLDS = 5
SMOOTH = 50


def movavg(x, w=SMOOTH):
    x = np.asarray(x, float); ok = np.isfinite(x); k = np.ones(w)
    return np.convolve(np.where(ok, x, 0), k, "same") / np.maximum(np.convolve(ok.astype(float), k, "same"), 1)


def history_design(side, choice, feedback, prior, k=3):
    s = np.nan_to_num(np.asarray(side, float) - 0.5)
    c = np.where(np.asarray(choice) >= 0, np.asarray(choice, float) - 0.5, 0.0)
    f = np.asarray(feedback, float) / 2.0
    return np.column_stack([lags(s, k), lags(c, k), lags(f, k), np.asarray(prior, float) - 0.5])


def ridge_cv(Z, y, folds=FOLDS, lam=10.0):
    """Cross-validated ridge prediction with contiguous folds."""
    T = len(y); out = np.zeros(T)
    edges = np.linspace(0, T, folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.r_[0:a, b:T]
        Zt = Z[tr]; mu = Zt.mean(0); sd = Zt.std(0) + 1e-9
        A = (Zt - mu) / sd; yt = y[tr] - y[tr].mean()
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yt)
        out[a:b] = ((Z[a:b] - mu) / sd) @ w + y[tr].mean()
    return out


def _resid(y, D):
    D1 = np.column_stack([np.ones(len(y)), D])
    return y - D1 @ np.linalg.lstsq(D1, y, rcond=None)[0]


def tstat(y, D, r):
    D1 = np.column_stack([np.ones(len(y)), D, r])
    b, *_ = np.linalg.lstsq(D1, y, rcond=None)
    e = y - D1 @ b
    dof = max(len(y) - D1.shape[1], 1)
    cov = (e @ e / dof) * np.linalg.pinv(D1.T @ D1)
    return b[-1] / np.sqrt(max(cov[-1, -1], 1e-300))


def memory_link(Z, hist, f1, targets, rng, n_shift=N_SHIFT):
    """Z (T, N) cleaned activity; hist (T, q) actual history; f1 (T,) previous feedback;
    targets: dict name -> (y (T,), D (T, p), ok mask (T,)). Returns {name: (t, p)}."""
    fh = ridge_cv(Z, f1)
    r = _resid(fh, hist)
    r = r - movavg(r)
    r = (r - r.mean()) / (r.std() + 1e-12)
    T = len(r)
    shifts = rng.integers(SMOOTH, T - SMOOTH, n_shift)
    out = {}
    for name, (y, D, ok) in targets.items():
        if ok.sum() < 100:
            continue
        yy, DD = y[ok], D[ok]
        t0 = tstat(yy, DD, r[ok])
        null = np.array([tstat(yy, DD, np.roll(r, s)[ok]) for s in shifts])
        out[name] = (float(t0), float((1 + (np.abs(null) >= abs(t0)).sum()) / (1 + n_shift)))
    return out


def behaviour_targets(contrast, choice, rt, hist, v):
    """stay(t) = choice(t) == choice(t-1); log RT. Designs include stimulus, actual history and slow trend."""
    ch = np.asarray(choice, float)
    prev = np.r_[np.nan, ch[:-1]]
    okc = v & (ch >= 0) & (prev >= 0)
    stay = (ch == prev).astype(float)
    c_rel = np.asarray(contrast, float) * np.where(prev == 1, 1.0, -1.0)     # evidence FOR repeating
    D_stay = np.column_stack([c_rel, np.tanh(5 * c_rel), np.abs(contrast), hist, movavg(np.where(okc, stay, np.nan))])
    lrt = np.where(np.isfinite(rt) & (rt > 0), np.log(np.clip(rt, 0.05, 10)), np.nan)
    okr = v & np.isfinite(lrt)
    D_rt = np.column_stack([np.abs(contrast), np.abs(contrast) ** 0.5, hist, movavg(np.where(okr, lrt, np.nan))])
    return {"stay": (stay, D_stay, okc), "log_rt": (np.nan_to_num(lrt), D_rt, okr)}


# ------------------------------------------------------------------------------------------- simulation
def sim_check(reps=60, seed=0):
    from history_joint import sim_behaviour
    from history_map import sim_session
    from driftaware.core import _slow_components

    def one(i, link):
        rng = np.random.default_rng(seed + 31 * i)
        T = 600
        side, choice, fb, prior = sim_behaviour(rng, T)
        f1 = lags(fb / 2, 1)[:, 0]
        eta = rng.standard_normal(T)                                 # trial-to-trial fluctuation of the memory
        m = f1 + 0.3 * eta
        X, *_ = sim_session(rng, T=T, mode="none")
        X = X + 0.6 * np.outer(m, rng.standard_normal(X.shape[1]))
        # behaviour: stay depends on actual feedback AND (if link) on the memory fluctuation
        p_stay = 1 / (1 + np.exp(-(0.3 + 1.0 * f1 + link * 0.8 * eta)))
        contrast = np.where(side == 1, 1, -1) * rng.choice([0, .0625, .125, .25, 1], T)
        choice2 = np.zeros(T, int)
        for t in range(1, T):
            choice2[t] = choice2[t - 1] if rng.random() < p_stay[t] else 1 - choice2[t - 1]
        rt = np.exp(rng.standard_normal(T) * 0.3 - 1)
        v = np.ones(T, bool); v[:90] = False
        hist = history_design(side, choice2, fb, prior)
        Z = (X - X.mean(0)) / X.std(0); U = _slow_components(X, 1, 50); Z = Z - U @ (U.T @ Z)
        res = memory_link(Z[v], hist[v], f1[v], {k: (a[v], b[v], c[v]) for k, (a, b, c) in
                                                  behaviour_targets(contrast, choice2, rt, hist, v).items()}, rng)
        return res["stay"][1] < .05, res["log_rt"][1] < .05

    from joblib import Parallel, delayed
    for link in (0.0, 1.0):
        R = np.array(Parallel(n_jobs=-1)(delayed(one)(i, link) for i in range(reps)))
        print(f"link={link}: stay rejection {R[:, 0].mean():.3f}   log_rt rejection (no true link) {R[:, 1].mean():.3f}", flush=True)


# ------------------------------------------------------------------------------------------- real data
def session_rows(path, aux_dir, seed=0):
    import zlib
    import compare_methods as cm
    from gatekeeper.multiverse import _sides
    from gatekeeper.mv_data import MVSession
    from driftaware.core import _slow_components
    from history_confounds import confounds
    s = MVSession(path)
    v = cm.mask_of(s)
    if v is None or v.sum() < 200:
        return []
    Tf = len(s.contrast)
    H, names = confounds(aux_dir, s.sid, Tf)
    side = _sides(s)
    hist = history_design(side, s.choice, s.feedback, cm.bayes_prior(side))
    f1 = lags(np.asarray(s.feedback, float) / 2, 1)[:, 0]
    targets = behaviour_targets(s.contrast, s.choice, s.rt, hist, v)
    rng = np.random.default_rng(seed + zlib.crc32(s.sid.encode()) % 100000)
    rows = []
    for r in s.regions:
        X = s.X(r, WINDOW)
        if X is None:
            continue
        X = X[v].astype(float)
        X = X[:, X.std(0) > 0]
        if X.shape[1] < MIN_UNITS:
            continue
        Z = (X - X.mean(0)) / X.std(0)
        U = _slow_components(X, 1, 50); Z = Z - U @ (U.T @ Z)
        if H is not None:
            Z = np.column_stack([_resid(Z[:, i], H[v]) for i in range(Z.shape[1])])
        tv = {k: (a[v], b[v], c[v]) for k, (a, b, c) in targets.items()}
        res = memory_link(Z, hist[v], f1[v], tv, rng)
        row = dict(sid=s.sid, subject=s.subject, split=s.split, region=r, n_units=X.shape[1])
        for k, (t0, p) in res.items():
            row[f"t_{k}"] = t0; row[f"p_{k}"] = p
        rows.append(row)
    return rows


def run(cache, aux_dir, out_dir, limit, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    out = Path(out_dir) / "sessions"
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[: limit or None]
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]
    print(f"{len(paths)} sessions, {len(paths) - len(todo)} already done, {len(todo)} to run", flush=True)
    t0 = time.time()

    def work(p):
        try:
            rows = session_rows(p, aux_dir)
        except Exception as e:
            print(f"FAILED {p.stem}: {type(e).__name__}: {e}", flush=True)
            rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    for i, n in enumerate(Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)):
        print(f"[{i + 1}/{len(todo)}] {n} regions  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def report(out_dir, memory_csv="results/history_confounds/history_confounds_regions.csv"):
    import pandas as pd
    from scipy.stats import chi2, wilcoxon
    from verify_numbers import region_p, sig
    import slowmode as sm
    files = [f for f in (Path(out_dir) / "sessions").glob("*.csv") if f.stat().st_size > 1]
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    mem = None
    if Path(memory_csv).exists():
        m = pd.read_csv(memory_csv)
        mem = dict(lag3=set(m[m.strict_persistence >= 3].region), lag2=set(m[m.strict_persistence == 2].region),
                   lag1=set(m[m.strict_persistence == 1].region), none=set(m[m.strict_persistence == 0].region))
    L = [f"# Does the brain's outcome memory drive the next decision? ({df.sid.nunique()} sessions, {df.subject.nunique()} mice)\n",
         "Memory fluctuation = neural readout of last trial's outcome minus what the actual history predicts. "
         "Positive t = stronger 'rewarded' memory -> more staying / faster. Null: circular shifts.\n",
         "## All session-regions\n", "| Behaviour | Tests | Fraction p < 0.05 | Median t | Fraction t > 0 | Wilcoxon p (t vs 0) | Regions FDR 5% |",
         "|---|---|---|---|---|---|---|"]
    for b in ("stay", "log_rt"):
        d = df.dropna(subset=[f"p_{b}"])
        pr = region_p(d, f"p_{b}")
        L.append(f"| {b} | {len(d)} | {(d[f'p_{b}'] < .05).mean():.3f} | {d[f't_{b}'].median():+.2f} | "
                 f"{(d[f't_{b}'] > 0).mean():.2f} | {wilcoxon(d[f't_{b}']).pvalue:.1e} | {len(sig(pr))} / {len(pr)} |")
    if mem:
        L += ["\n## By how long the region remembers (from step 1c)\n",
              "| Memory group | Behaviour | Session-regions | Fraction p < 0.05 | Median t | Fraction t > 0 | Fisher p |",
              "|---|---|---|---|---|---|---|"]
        for g in ("lag3", "lag2", "lag1", "none"):
            for b in ("stay", "log_rt"):
                d = df[df.region.isin(mem[g])].dropna(subset=[f"p_{b}"])
                if len(d):
                    L.append(f"| {g} | {b} | {len(d)} | {(d[f'p_{b}'] < .05).mean():.3f} | {d[f't_{b}'].median():+.2f} | "
                             f"{(d[f't_{b}'] > 0).mean():.2f} | {fisher(d[f'p_{b}'].values):.1e} |")
    div = sm.division_of(sorted(df.region.unique()))
    df["division"] = df.region.map(div)
    L += ["\n## By division (stay)\n", "| Division | Session-regions | Fraction p < 0.05 | Median t | Fraction t > 0 |", "|---|---|---|---|---|"]
    for dv, d in df.dropna(subset=["p_stay"]).groupby("division"):
        L.append(f"| {dv} | {len(d)} | {(d.p_stay < .05).mean():.3f} | {d.t_stay.median():+.2f} | {(d.t_stay > 0).mean():.2f} |")
    for b in ("stay", "log_rt"):
        pr = region_p(df.dropna(subset=[f"p_{b}"]), f"p_{b}")
        sg = sorted(sig(pr))
        L += [f"\n## Regions significant at FDR 5% ({b})\n", ", ".join(f"{r} ({df[df.region == r][f't_{b}'].median():+.1f})" for r in sg) or "none"]
    text = "\n".join(L)
    (Path(out_dir) / "HISTORY_BEHAVIOUR.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "run", "report"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--aux", default="data/part3_aux2")
    ap.add_argument("--out", default="results/history_behaviour")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--reps", type=int, default=60)
    a = ap.parse_args()
    if a.step == "sim":
        sim_check(reps=a.reps)
    elif a.step == "run":
        run(a.cache, a.aux, a.out, a.limit, a.jobs)
    else:
        report(a.out)
