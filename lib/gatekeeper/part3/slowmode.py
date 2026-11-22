"""PART 3, feasibility test: is there a brain-wide slow mode, and does it track behaviour?

Q1  SHARING: within a session, do slow (minutes-scale) components of DIFFERENT brain divisions co-fluctuate more than
    slow components taken from DIFFERENT sessions (which cannot share a real internal state)?
Q2  BEHAVIOUR: does the session's global slow mode track slow changes in behaviour (accuracy on easy trials, misses,
    reaction time, choice bias) more than the slow modes of OTHER sessions do (session-permutation null)?

Both nulls compare against slow signals from other sessions, aligned by trial number. Two unrelated slow signals
correlate a lot by chance; this null has exactly that property, so the test does not reward nonsense correlations.

Run on the EXPLORE split only. Decisions about Part 3 are made from these results before any confirm-split analysis.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SMOOTH = 50          # trials (about 2 to 4 minutes in the IBL task)
WINDOW = "prior_a"   # inter-trial window, -600 to -100 ms before stimulus onset: least driven by task events
MIN_UNITS = 5
MIN_TRIALS = 300


def movavg(x, w=SMOOTH):
    """Centered moving average that ignores NaN and does not shrink at the edges."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    k = np.ones(w)
    num = np.convolve(np.where(ok, x, 0.0), k, "same")
    den = np.convolve(ok.astype(float), k, "same")
    return num / np.maximum(den, 1)


def slow_pc(X):
    """Top principal component of the slow (smoothed, z-scored) activity of neurons X (T, N).
    Returns (component (T,), fraction of slow variance it explains). Sign: positive mean loading."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    S = np.column_stack([movavg(Z[:, i]) for i in range(Z.shape[1])])
    S = S - S.mean(0)
    U, sv, Vt = np.linalg.svd(S, full_matrices=False)
    u = U[:, 0] * sv[0]
    if Vt[0].mean() < 0:
        u = -u
    return u, float(sv[0] ** 2 / (sv ** 2).sum())


def task_design(s):
    """Per-trial task regressors (full length). Their effect is removed from every neuron before looking for a slow
    mode, so the block structure and other task variables cannot masquerade as an internal state."""
    from gatekeeper.multiverse import _sides
    sys.path.insert(0, str(ROOT / "next"))
    from compare_methods import bayes_prior
    T = len(s.contrast)
    pl = s.meta.get("probabilityLeft")
    pl = np.asarray(pl, float) if pl is not None and len(pl) == T else np.full(T, 0.5)
    block = np.nan_to_num(pl < 0.5, nan=0.5).astype(float) - 0.5
    unbiased = (np.abs(pl - 0.5) < 0.1).astype(float)              # the 50/50 opening block
    bp = bayes_prior(_sides(s)) - 0.5
    c = s.contrast
    ch = np.where(s.choice >= 0, s.choice - 0.5, 0.0)
    fb = s.feedback / 2.0
    prev = lambda x: np.r_[0.0, x[:-1]]
    return np.column_stack([block, unbiased, bp, c, np.abs(c), ch, fb, prev(ch), prev(fb), prev(ch) * prev(fb)])


def residualize(X, D):
    D1 = np.column_stack([np.ones(len(D)), D])
    b, *_ = np.linalg.lstsq(D1, X, rcond=None)
    return X - D1 @ b


def behaviour(s, v):
    """Slow behavioural signals on the analysed trials v (smoothed, NaN-aware)."""
    c = np.abs(s.contrast)
    easy = c >= 0.25
    acc = np.where(easy & (s.choice >= 0), (s.feedback > 0).astype(float), np.nan)
    miss = (s.choice < 0).astype(float)
    rt = np.where(np.isfinite(s.rt) & (s.rt > 0), np.log(np.clip(s.rt, 0.05, 10)), np.nan)
    hard = c <= 0.0625
    pl = s.meta.get("probabilityLeft")
    pl = np.asarray(pl, float) if pl is not None and len(pl) == len(c) else np.full(len(c), 0.5)
    # bias RELATIVE to the block: +1 when the choice goes against the block's likely side, so block switches
    # do not create a slow 'bias' signal by themselves
    against = np.where(pl < 0.5, 1 - s.choice, s.choice).astype(float)
    bias = np.where(hard & (s.choice >= 0) & (np.abs(pl - 0.5) > 0.1), against, np.nan)
    out = {}
    for name, x in (("accuracy_easy", acc), ("miss_rate", miss), ("log_rt", rt), ("bias_against_block", bias)):
        out[name] = movavg(x[v])
    return out


def _corr(a, b, Ds=None):
    """|corr| after partialling the session's slow task design Ds out of BOTH signals (symmetric for real and
    borrowed slow modes)."""
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    if Ds is not None:
        D1 = np.column_stack([np.ones(n), Ds[:n]])
        for name in ("a", "b"):
            x = a if name == "a" else b
            ok = np.isfinite(x)
            if ok.sum() > D1.shape[1] + 10:
                beta, *_ = np.linalg.lstsq(D1[ok], x[ok], rcond=None)
                x = np.where(ok, x - D1 @ beta, np.nan)
            if name == "a":
                a = x
            else:
                b = x
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 50:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def division_of(regions):
    try:
        from iblatlas.regions import BrainRegions
        br = BrainRegions()
        return dict(zip(regions, br.acronym2acronym(np.array(regions), mapping="Cosmos")))
    except Exception:
        return {r: r for r in regions}


def session_summary(path):
    from gatekeeper.mv_data import MVSession
    s = MVSession(path)
    v = s.valid["stimOn"].copy()
    if v.sum() < MIN_TRIALS:
        return None
    regs, allX = {}, []
    D = task_design(s)[v]
    for r in s.regions:
        X = s.X(r, WINDOW)
        if X is None:
            continue
        X = X[v]
        X = X[:, X.std(0) > 0]
        if X.shape[1] < MIN_UNITS:
            continue
        X = residualize(X.astype(float), D)            # remove everything the task explains
        regs[r] = slow_pc(X)[0]
        allX.append(X)
    if not allX:
        return None
    g, frac = slow_pc(np.hstack(allX))
    Ds = np.column_stack([movavg(D[:, k]) for k in range(D.shape[1])])
    return dict(sid=s.sid, subject=s.subject, split=s.split, T=int(v.sum()), regions=regs, global_mode=g,
                global_frac=frac, behaviour=behaviour(s, v), slow_design=Ds)


def analyse(summaries, n_null=200, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed)
    S = [x for x in summaries if x]
    div = division_of(sorted({r for x in S for r in x["regions"]}))
    by_region = {}
    for i, x in enumerate(S):
        for r, u in x["regions"].items():
            by_region.setdefault(r, []).append((i, u))
    q1, q2 = [], []
    for i, x in enumerate(S):
        regs = list(x["regions"])
        # Q1: pairs of regions in different divisions
        for a in range(len(regs)):
            for b in range(a + 1, len(regs)):
                A, B = regs[a], regs[b]
                if div.get(A) == div.get(B):
                    continue
                Ds = x.get("slow_design")
                obs = abs(_corr(x["regions"][A], x["regions"][B], Ds))
                pool = [u for j, u in by_region.get(B, []) if j != i] or [y["global_mode"] for j, y in enumerate(S) if j != i]
                null = np.array([abs(_corr(x["regions"][A], pool[k], Ds)) for k in rng.integers(0, len(pool), n_null)])
                null = null[np.isfinite(null)]
                if np.isfinite(obs) and len(null) > 20:
                    q1.append(dict(sid=x["sid"], A=A, B=B, divA=div.get(A), divB=div.get(B), r=obs,
                                   r_null_median=float(np.median(null)), p=float((1 + (null >= obs).sum()) / (1 + len(null)))))
        # Q2: global mode vs behaviour
        others = [y["global_mode"] for j, y in enumerate(S) if j != i]
        for name, bvec in x["behaviour"].items():
            Ds = x.get("slow_design")
            obs = abs(_corr(x["global_mode"], bvec, Ds))
            null = np.array([abs(_corr(others[k], bvec, Ds)) for k in rng.integers(0, len(others), n_null)])
            null = null[np.isfinite(null)]
            if np.isfinite(obs) and len(null) > 20:
                q2.append(dict(sid=x["sid"], subject=x["subject"], behaviour=name, r=obs,
                               r_null_median=float(np.median(null)), p=float((1 + (null >= obs).sum()) / (1 + len(null)))))
    meta = pd.DataFrame([dict(sid=x["sid"], subject=x["subject"], T=x["T"], n_regions=len(x["regions"]),
                              global_frac=x["global_frac"]) for x in S])
    return meta, pd.DataFrame(q1), pd.DataFrame(q2)


def report(meta, q1, q2, out_dir):
    from scipy.stats import chi2
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    L = [f"# Part 3 feasibility: brain-wide slow mode ({len(meta)} sessions, {meta.subject.nunique()} mice)\n",
         f"Slow = {SMOOTH}-trial moving average; window {WINDOW}. Nulls: slow signals from OTHER sessions (aligned by trial).",
         "Under 'no shared slow state', p-values are uniform: fraction p<0.05 should be about 0.05.\n",
         f"Global slow mode explains a median {meta.global_frac.median():.2f} of slow variance (IQR {meta.global_frac.quantile(.25):.2f} to {meta.global_frac.quantile(.75):.2f}).\n",
         "## Q1: do different brain divisions share the slow mode?\n",
         "| Pairs | Median abs r (same session) | Median abs r (null) | Fraction p < 0.05 | Sessions with Fisher p < 0.05 |",
         "|---|---|---|---|---|"]
    if len(q1):
        per_s = q1.groupby("sid").p.apply(lambda p: fisher(p.values))
        L.append(f"| {len(q1)} | {q1.r.median():.2f} | {q1.r_null_median.median():.2f} | {(q1.p < .05).mean():.3f} | {(per_s < .05).sum()} / {len(per_s)} |")
    L += ["\n## Q2: does the global slow mode track behaviour?\n",
          "| Behaviour | Sessions | Median abs r | Median abs r (null) | Fraction p < 0.05 | Fisher p across sessions |",
          "|---|---|---|---|---|---|"]
    for b, d in q2.groupby("behaviour"):
        L.append(f"| {b} | {len(d)} | {d.r.median():.2f} | {d.r_null_median.median():.2f} | {(d.p < .05).mean():.3f} | {fisher(d.p.values):.1e} |")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "PART3_FEASIBILITY.md").write_text(text + "\n")
    meta.to_csv(Path(out_dir) / "sessions.csv", index=False)
    q1.to_csv(Path(out_dir) / "q1_pairs.csv", index=False)
    q2.to_csv(Path(out_dir) / "q2_behaviour.csv", index=False)
    print(text)


def run(cache, out_dir, split="explore", n_jobs=-1, limit=0):
    from joblib import Parallel, delayed
    from gatekeeper.mv_data import MVSession
    paths = sorted(Path(cache).expanduser().glob("*.npz"))
    paths = [p for p in paths if MVSession(p).split == split][: limit or None]
    print(f"{len(paths)} sessions ({split})", flush=True)
    t0 = time.time()
    S = Parallel(n_jobs=n_jobs, verbose=5)(delayed(session_summary)(p) for p in paths)
    print(f"summaries done in {(time.time() - t0) / 60:.1f} min", flush=True)
    meta, q1, q2 = analyse(S)
    report(meta, q1, q2, out_dir)


# ---------------------------------------------------------------- simulation check
class _FakeSession:
    pass


def sim_check(n_sessions=40, T=600, shared=0.0, behaviour_link=0.0, seed=0):
    """Synthetic sessions: 4 regions in 4 divisions, each with its own slow AR drift; 'shared' mixes in one common
    slow state per session; 'behaviour_link' makes accuracy track that state. Returns the analysis tables."""
    rng = np.random.default_rng(seed)
    S = []
    regs = ["R1", "R2", "R3", "R4"]

    def slow():
        z = np.zeros(T); e = rng.standard_normal(T)
        for t in range(1, T):
            z[t] = 0.99 * z[t - 1] + np.sqrt(1 - 0.99 ** 2) * e[t]
        return z
    for i in range(n_sessions):
        state = slow()
        regions, allX = {}, []
        for r in regs:
            d = np.sqrt(shared) * state + np.sqrt(1 - shared) * slow()
            X = rng.poisson(np.exp(0.5 + 0.4 * d[:, None] * rng.uniform(0.5, 1.5, 15)))
            regions[r] = slow_pc(X)[0]; allX.append(X)
        g, frac = slow_pc(np.hstack(allX))
        p_corr = 1 / (1 + np.exp(-(1.5 + behaviour_link * state)))
        acc = (rng.random(T) < p_corr).astype(float)
        S.append(dict(sid=f"s{i}", subject=f"m{i}", split="explore", T=T, regions=regions, global_mode=g,
                      global_frac=frac, behaviour={"accuracy_easy": movavg(acc)}))
    import gatekeeper  # noqa: F401
    globals()["division_of"] = lambda rs: {r: r for r in rs}
    return analyse(S, n_null=100, seed=seed)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--out", default="results/part3_feasibility")
    ap.add_argument("--split", default="explore")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.sim:
        for shared, link in ((0.0, 0.0), (0.5, 0.0), (0.5, 1.0)):
            meta, q1, q2 = sim_check(shared=shared, behaviour_link=link)
            print(f"shared={shared} link={link}:  Q1 frac p<.05 = {(q1.p < .05).mean():.3f}   Q2 frac p<.05 = {(q2.p < .05).mean():.3f}")
    else:
        run(a.cache, a.out, a.split, a.jobs, a.limit)
