"""PART 3 step 4: WHERE is the guiding memory? Two positive-mechanism tests.

M1  SLOW STORE. The drift-aware test deliberately removes slow, shared activity. Behaviour (step 3a-2) shows each
    mouse has a large, slowly wandering left/right bias. If that bias is partly an integrator of past rewards, the
    memory may live exactly in the slow population signal the test removes. Test, per session and region:
      target = the mouse's fitted bias path b_t (random-walk model of choices) and a reward-integrated value
               V_t (V <- V + alpha * (rewarded_side - V)), with the block prior removed in IBL
      decoder = cross-validated ridge from between-trial activity (NO drift removal: slow signal is the point)
      null   = the SAME decoder on the bias paths / values of OTHER mice (aligned by trial), which have the same
               slow statistics, so slow nonsense correlations are matched (Harris 2020; Findling 2025 style)
    STRICT: the targets (real and null) and the activity are all residualized on the SAME real recent history
    (stimulus side, choice, feedback at lags 1-3, the current choice, and in IBL the block prior). That is a fixed
    operation, so the null stays exact, and it means a region only counts if it carries the slow bias / long-run
    reward value BEYOND what the last three trials and the upcoming choice explain.
M2  OUTCOME-TIME REACTIVATION is tested with silent_memory.py (window feedback_a), not here.

    python3 part3/mechanism.py sim
    python3 part3/mechanism.py ibl --limit 3        then without --limit
    python3 part3/mechanism.py steinmetz --data <folder with steinmetz_part*.npz>
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
import behaviour_rw as brw  # noqa: E402

N_NULL = 100
MIN_UNITS = 5
ALPHA = 0.2
FOLDS = 5
LAM = 10.0


def value_trace(choice, feedback, alpha=ALPHA):
    """Reward-integrated side value before each trial: +1 rewarded right, -1 rewarded left, 0 error/no-go."""
    ch = np.asarray(choice)
    rc = np.where(ch >= 0, 2.0 * ch - 1, 0.0) * (np.asarray(feedback) > 0)
    V = np.zeros(len(rc)); v = 0.0
    for t in range(len(rc)):
        V[t] = v
        v += alpha * (rc[t] - v)
    return V


def bias_path(contrast, choice, feedback, v):
    X = brw.design(contrast, choice, feedback)
    y = (np.asarray(choice) == 1).astype(float)
    lls = [brw.heldout_ll(X, y, v, s) for s in brw.SIGMAS]
    sig = brw.SIGMAS[int(np.argmax(lls))]
    _, b, _ = brw.fit_rw(X, y, v, sig)
    return b, sig


def ridge_scores(Z, Y, folds=FOLDS, lam=LAM):
    """Z (T, N) activity, Y (T, M) targets. Contiguous-fold cross-validated ridge; returns corr(pred, target) per column."""
    T = len(Z)
    P = np.zeros_like(Y)
    edges = np.linspace(0, T, folds + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.r_[0:a, b:T]
        mu, sd = Z[tr].mean(0), Z[tr].std(0) + 1e-9
        A = (Z[tr] - mu) / sd
        ym = Y[tr].mean(0)
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Y[tr] - ym))
        P[a:b] = ((Z[a:b] - mu) / sd) @ W + ym
    Pc = P - P.mean(0); Yc = Y - Y.mean(0)
    return (Pc * Yc).sum(0) / (np.linalg.norm(Pc, axis=0) * np.linalg.norm(Yc, axis=0) + 1e-12)


def resid(Y, D):
    if D is None:
        return Y - Y.mean(0)
    D1 = np.column_stack([np.ones(len(D)), D])
    return Y - D1 @ np.linalg.lstsq(D1, Y, rcond=None)[0]


def test_region(X, targets, D, owns=None):
    """X (T, N) counts on analysed trials; targets (T, M+1), column 0 real.
    D: the real session's recent history (removed from the activity). owns: list of M+1 matrices, each target's
    OWN recent history (the real one for column 0, each other mouse's own for the nulls); every target is
    residualized on its own history, so real and null targets are treated identically."""
    if owns is None:
        Y = resid(targets, D)
    else:
        # each target loses its own history AND the real history removed from the activity (same subspace)
        Y = np.column_stack([resid(targets[:, [j]], np.column_stack([D, owns[j]]) if D is not None else owns[j])[:, 0]
                             for j in range(targets.shape[1])])
    Xr = resid(X.astype(float), D) if D is not None else X.astype(float)
    r = ridge_scores(Xr, Y)
    return float((1 + (r[1:] >= r[0]).sum()) / len(r)), float(r[0]), float(np.median(r[1:]))


# ------------------------------------------------------------------------------------------- simulation
def sim(reps=40, seed=0):
    rng0 = np.random.default_rng(seed)

    def session(rng, T=600):
        z = np.zeros(T); e = rng.standard_normal(T)
        for t in range(1, T):
            z[t] = 0.98 * z[t - 1] + 0.2 * e[t]
        return z

    for link in (0.0, 1.0):
        hits = []
        for i in range(reps):
            rng = np.random.default_rng(seed + i)
            T, N = 600, 25
            b = session(rng)                                         # this mouse's slow bias
            drift = np.column_stack([session(rng) for _ in range(N)]) # neurons' own slow drift (independent)
            X = rng.poisson(np.exp(0.5 + 0.4 * drift + link * 0.3 * np.outer(b / b.std(), rng.standard_normal(N))))
            others = np.column_stack([session(rng) for _ in range(N_NULL)])
            p, *_ = test_region(X, np.column_stack([b, others]), None)
            hits.append(p < .05)
        print(f"neural code of the bias = {link}: rejection {np.mean(hits):.3f}", flush=True)


# ------------------------------------------------------------------------------------------- IBL
def ibl_bank(cache):
    import compare_methods as cm
    from gatekeeper.multiverse import _sides
    from gatekeeper.mv_data import MVSession
    bank = {}
    for p in sorted(Path(cache).expanduser().glob("*.npz")):
        s = MVSession(p)
        v = cm.mask_of(s)
        if v is None or v.sum() < 200:
            continue
        b, sig = bias_path(s.contrast, s.choice, s.feedback, v)
        side = _sides(s)
        from history_behaviour import history_design
        H = history_design(side, s.choice, s.feedback, cm.bayes_prior(side))
        cur = np.where(s.choice >= 0, s.choice - 0.5, 0.0)
        bank[s.sid] = dict(subject=s.subject, b=b, V=value_trace(s.choice, s.feedback), sigma=sig,
                           prior=cm.bayes_prior(side) - 0.5, T=len(b), D=np.column_stack([H, cur]))
    return bank


def behaviour_link(bank, rng, n_null=N_NULL):
    """Is the wandering bias driven by rewards? corr(b, V) vs corr(b, other mice's V), block prior removed."""
    rows = []
    items = list(bank.items())
    for sid, d in items:
        T = d["T"]
        pool = [e["V"][:T] for s2, e in items if e["subject"] != d["subject"] and e["T"] >= T]
        if len(pool) < 20:
            continue
        idx = rng.choice(len(pool), min(n_null, len(pool)), replace=False)
        Vs = np.column_stack([d["V"]] + [pool[j] for j in idx])
        Y = resid(Vs, d["prior"][:, None]); bb = resid(d["b"][:, None], d["prior"][:, None])[:, 0]
        c = (Y - Y.mean(0)).T @ (bb - bb.mean()) / (np.linalg.norm(Y - Y.mean(0), axis=0) * np.linalg.norm(bb - bb.mean()) + 1e-12)
        rows.append(dict(sid=sid, subject=d["subject"], r=c[0], r_null=np.median(c[1:]),
                         p=(1 + (c[1:] >= c[0]).sum()) / len(c)))
    return rows


def ibl_check_rows(path, bank, seed=0):
    """Bug check on REAL recordings. (1) Fake: an extra other-mouse bias path is used as the 'real' target; must give
    about 0.05. (2) Planted: the session's own bias path (after the same residualization) is added to the real activity,
    each neuron weighted by its measured lag-1 choice weight (full, half); must be detected."""
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from history_map import lags
    s = MVSession(path)
    if s.sid not in bank:
        return []
    v = cm.mask_of(s)
    d = bank[s.sid]; T = d["T"]
    rng = np.random.default_rng(seed + 7 + zlib.crc32(s.sid.encode()) % 100000)
    others = [(e["b"][:T], e["D"][:T]) for s2, e in bank.items() if e["subject"] != s.subject and e["T"] >= T]
    if len(others) < 31:
        return []
    idx = rng.choice(len(others), min(N_NULL + 1, len(others)), replace=False)
    D = d["D"][v]
    real_B = np.column_stack([d["b"]] + [others[j][0] for j in idx[1:]])[v]
    real_owns = [D] + [others[j][1][v] for j in idx[1:]]
    fake_B = np.column_stack([others[j][0] for j in idx])[v]                  # column 0 = an unrelated mouse
    fake_owns = [others[j][1][v] for j in idx]
    breal = resid(real_B[:, [0]], D)[:, 0]; breal = breal / (breal.std() + 1e-12)
    c1 = lags(np.where(s.choice >= 0, s.choice - 0.5, 0.0), 1)[v, 0]; c1 = c1 - c1.mean()
    rows = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        X = X[v].astype(float); X = X[:, X.std(0) > 0]
        if X.shape[1] < MIN_UNITS:
            continue
        sd = X.std(0); Z = (X - X.mean(0)) / sd
        beta = (c1 @ Z) / (c1 @ c1)
        row = dict(sid=s.sid, subject=s.subject, region=r)
        row["p_fake"], *_ = test_region(X, fake_B, D, fake_owns)
        for tag, scale in (("full", 1.0), ("half", 0.5)):
            Xp = X + scale * np.outer(breal, beta * sd) * 0.5     # 0.5: lag-1 choice codes a +-0.5 variable
            row[f"p_planted_{tag}"], *_ = test_region(Xp, real_B, D, real_owns)
        # unmistakable plant (0.3 SD per neuron, random signs): if this is not found, the pipeline is broken
        Xp = X + np.outer(breal, rng.standard_normal(X.shape[1]) * 0.3 * sd)
        row["p_planted_obvious"], *_ = test_region(Xp, real_B, D, real_owns)
        rows.append(row)
    return rows


def pooled_session(X, d_b, others_b, D, owns_real, fake_cols, fake_owns, c1, rng, lam):
    """Session-level test with ALL recorded neurons pooled (the slow bias, if it exists, is brain-wide)."""
    sd = X.std(0)
    out = {}
    pr, rr, rn = _test_lam(X, np.column_stack([d_b] + others_b), D, owns_real, lam)
    out.update(p_real=pr, r_real=rr, rnull=rn)
    out["p_fake"], *_ = _test_lam(X, fake_cols, D, fake_owns, lam)
    breal = resid(d_b[:, None], D)[:, 0]; breal = breal / (breal.std() + 1e-12)
    Z = (X - X.mean(0)) / sd
    beta = (c1 @ Z) / (c1 @ c1)
    for tag, scale in (("full", 1.0), ("half", 0.5)):
        Xp = X + scale * np.outer(breal, beta * sd) * 0.5
        out[f"p_planted_{tag}"], *_ = _test_lam(Xp, np.column_stack([d_b] + others_b), D, owns_real, lam)
    return out


def _slow_pcs(X, k=5, w=10):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ker = np.ones(w) / w
    S = np.apply_along_axis(lambda x: np.convolve(x, ker, "same"), 0, Z)
    U, sv, _ = np.linalg.svd(S - S.mean(0), full_matrices=False)
    return U[:, :k] * sv[:k]


def _detrend(Y, period=100):
    T = len(Y)
    nb = max(1, int(2 * T / period))
    t = (np.arange(T) + 0.5) / T
    B = np.column_stack([np.ones(T)] + [np.cos(np.pi * j * t) for j in range(1, nb + 1)])
    return Y - B @ np.linalg.lstsq(B, Y, rcond=None)[0]


def _test_v3(X, targets, D, owns):
    """v3 statistic (see mechanism3.py): top 5 slow population components, very slow trends removed."""
    Y = np.column_stack([resid(targets[:, [j]], np.column_stack([D, owns[j]]))[:, 0] for j in range(targets.shape[1])])
    P = _detrend(_slow_pcs(resid(X.astype(float), D)))
    r = ridge_scores(P, _detrend(Y), lam=1.0)
    return float((1 + (r[1:] >= r[0]).sum()) / len(r)), float(r[0]), float(np.median(r[1:]))


def _test_lam(X, targets, D, owns, lam):
    import os
    if os.environ.get("MV_TEST") == "v3":
        return _test_v3(X, targets, D, owns)
    global LAM
    old = LAM; LAM = lam
    try:
        Y = np.column_stack([resid(targets[:, [j]], np.column_stack([D, owns[j]]))[:, 0] for j in range(targets.shape[1])])
        Xr = resid(X.astype(float), D)
        r = ridge_scores(Xr, Y, lam=lam)
    finally:
        LAM = old
    return float((1 + (r[1:] >= r[0]).sum()) / len(r)), float(r[0]), float(np.median(r[1:]))


def ibl_pooled_rows(path, bank, seed=0):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from history_map import lags
    s = MVSession(path)
    if s.sid not in bank:
        return []
    v = cm.mask_of(s)
    d = bank[s.sid]; T = d["T"]
    rng = np.random.default_rng(seed + 11 + zlib.crc32(s.sid.encode()) % 100000)
    others = [(e["b"][:T], e["D"][:T]) for s2, e in bank.items() if e["subject"] != s.subject and e["T"] >= T]
    if len(others) < 31:
        return []
    idx = rng.choice(len(others), min(N_NULL + 1, len(others)), replace=False)
    D = d["D"][v]
    Xs = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is not None:
            X = X[v].astype(float); Xs.append(X[:, X.std(0) > 0])
    X = np.hstack(Xs) if Xs else None
    if X is None or X.shape[1] < 10:
        return []
    c1 = lags(np.where(s.choice >= 0, s.choice - 0.5, 0.0), 1)[v, 0]; c1 = c1 - c1.mean()
    out = pooled_session(X, d["b"][v], [others[j][0][v] for j in idx[1:]], D, [D] + [others[j][1][v] for j in idx[1:]],
                         np.column_stack([others[j][0] for j in idx])[v], [others[j][1][v] for j in idx], c1, rng,
                         lam=max(10.0, X.shape[1] / 5))
    return [dict(sid=s.sid, subject=s.subject, n_units=X.shape[1], **out)]


def ibl_rows(path, bank, seed=0):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    s = MVSession(path)
    if s.sid not in bank:
        return []
    v = cm.mask_of(s)
    d = bank[s.sid]; T = d["T"]
    rng = np.random.default_rng(seed + zlib.crc32(s.sid.encode()) % 100000)
    others = [(e["b"][:T], e["V"][:T], e["D"][:T]) for s2, e in bank.items() if e["subject"] != s.subject and e["T"] >= T]
    if len(others) < 30:
        return []
    idx = rng.choice(len(others), min(N_NULL, len(others)), replace=False)
    Bt = np.column_stack([d["b"]] + [others[j][0] for j in idx])[v]
    Vt = np.column_stack([d["V"]] + [others[j][1] for j in idx])[v]
    D = d["D"][v]
    owns = [D] + [others[j][2][v] for j in idx]
    rows = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        X = X[v]; X = X[:, X.std(0) > 0]
        if X.shape[1] < MIN_UNITS:
            continue
        pb, rb, nb = test_region(X, Bt, D, owns)
        pv, rv, nv = test_region(X, Vt, D, owns)
        rows.append(dict(sid=s.sid, subject=s.subject, region=r, n_units=X.shape[1], p_bias=pb, r_bias=rb, rnull_bias=nb,
                         p_value=pv, r_value=rv, rnull_value=nv))
    return rows


# ------------------------------------------------------------------------------------------- Steinmetz
def steinmetz_rows(i, d, bank, seed=0):
    import steinmetz_replication as sr
    mouse = str(d["mouse_name"])
    contrast, choice, fb, side = sr.task(d)
    T0 = len(choice)
    lens = sorted([e["T"] for m, e in bank if m != mouse], reverse=True)
    if len(lens) < 20:
        return []
    T = min(T0, lens[19])
    me = [e for m, e in bank if e["i"] == i][0]
    others = [e for m, e in bank if m != mouse and e["T"] >= T]
    rng = np.random.default_rng(seed + i)
    idx = rng.choice(len(others), min(N_NULL, len(others)), replace=False)
    v = (choice[:T] >= 0); v[:10] = False
    Bt = np.column_stack([me["b"][:T]] + [others[j]["b"][:T] for j in idx])[v]
    Vt = np.column_stack([me["V"][:T]] + [others[j]["V"][:T] for j in idx])[v]
    D = me["D"][:T][v]
    owns = [D] + [others[j]["D"][:T][v] for j in idx]
    areas = np.asarray(d["brain_area"]).astype(str)
    rows = []
    for r in sorted(set(areas) - sr.SKIP):
        sel = areas == r
        if sel.sum() < MIN_UNITS:
            continue
        X = d["spks"][sel][:, :T, sr.PRE[0]:sr.PRE[1]].sum(-1).T.astype(float)[v]
        X = X[:, X.std(0) > 0]
        if X.shape[1] < MIN_UNITS:
            continue
        pb, rb, nb = test_region(X, Bt, D, owns)
        pv, rv, nv = test_region(X, Vt, D, owns)
        rows.append(dict(sid=f"s{i}", subject=mouse, region=r, n_units=X.shape[1], p_bias=pb, r_bias=rb, rnull_bias=nb,
                         p_value=pv, r_value=rv, rnull_value=nv))
    return rows


def report(df, link, out_dir, title):
    import pandas as pd
    from scipy.stats import chi2
    from verify_numbers import region_p, sig
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    L = [f"# Mechanism M1, slow store: {title} ({df.sid.nunique()} sessions, {df.subject.nunique()} mice)\n"]
    if link is not None and len(link):
        lk = pd.DataFrame(link)
        L += ["## Is the wandering bias driven by rewards?\n",
              f"corr(bias path, reward-integrated value, alpha {ALPHA}): median {lk.r.median():+.2f} vs other mice's value "
              f"{lk.r_null.median():+.2f}; fraction p < 0.05 {(lk.p < .05).mean():.3f}; Fisher p {fisher(lk.p.values):.1e} "
              f"({len(lk)} sessions).\n"]
    L += ["## Does slow between-trial activity carry the bias / reward value? (null: other mice's paths)\n",
          "| Target | Session-regions | Fraction p < 0.05 | Median r (real) | Median r (null) | Regions FDR 5% |", "|---|---|---|---|---|---|"]
    for t in ("bias", "value"):
        pr = region_p(df, f"p_{t}")
        L.append(f"| {t} | {len(df)} | {(df[f'p_{t}'] < .05).mean():.3f} | {df[f'r_{t}'].median():+.3f} | "
                 f"{df[f'rnull_{t}'].median():+.3f} | {len(sig(pr))} / {len(pr)} |")
    for t in ("bias", "value"):
        pr = region_p(df, f"p_{t}")
        sg = sorted(sig(pr))
        L += [f"\nRegions carrying the {t} (FDR 5%): " + (", ".join(sg) if sg else "none")]
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "MECHANISM_M1.md").write_text(text + "\n")
    df.to_csv(Path(out_dir) / "m1_rows.csv", index=False)
    print(text)


def run_ibl_check(cache, out_dir, every, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    from verify_numbers import region_p, sig
    t0 = time.time()
    bank = ibl_bank(cache)
    paths = [p for p in sorted(Path(cache).expanduser().glob("*.npz")) if p.stem in bank][::every]
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(ibl_check_rows)(p, bank) for p in paths)
    df = pd.DataFrame([r for rows in res for r in rows])
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "check_rows.csv", index=False)
    L = [f"# Bug check for M1 on real IBL recordings ({df.sid.nunique()} sessions, {(time.time() - t0) / 60:.1f} min)\n",
         "| Condition | Session-regions | Fraction p < 0.05 | Regions FDR 5% | Should be |", "|---|---|---|---|---|"]
    for c, should in (("p_fake", "about 0.05, 0 regions"), ("p_planted_obvious", "near 1 (pipeline works)"),
                      ("p_planted_full", "high if a lag-1-strength code would be visible"), ("p_planted_half", "above 0.05")):
        pr = region_p(df, c)
        L.append(f"| {c[2:]} | {len(df)} | {(df[c] < .05).mean():.3f} | {len(sig(pr))} / {len(pr)} | {should} |")
    text = "\n".join(L)
    (Path(out_dir) / "M1_CHECK.md").write_text(text + "\n")
    print(text)


def pooled_report(df, out_dir, title):
    from scipy.stats import chi2, binomtest
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    L = [f"# M1 pooled: does the WHOLE recorded population carry the mouse's slow bias? {title} "
         f"({len(df)} sessions, {df.subject.nunique()} mice)\n",
         "One test per session, all neurons pooled; strict (recent history, current choice, block prior removed); "
         "null = other mice's bias paths. Fake = an unrelated mouse's path used as if real.\n",
         "| Condition | Sessions | Fraction p < 0.05 | Binomial p vs 0.05 | Fisher p |", "|---|---|---|---|---|"]
    for c in ("p_fake", "p_real", "p_planted_half", "p_planted_full"):
        k = int((df[c] < .05).sum())
        L.append(f"| {c[2:]} | {len(df)} | {k / len(df):.3f} | {binomtest(k, len(df), 0.05, 'greater').pvalue:.1e} | {fisher(df[c].values):.1e} |")
    L.append(f"\nMedian decoding r: real {df.r_real.median():+.3f}, null {df.rnull.median():+.3f}.")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "M1_POOLED.md").write_text(text + "\n")
    df.to_csv(Path(out_dir) / "pooled_rows.csv", index=False)
    print(text)


def run_ibl_pooled(cache, out_dir, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    bank = ibl_bank(cache)
    paths = [p for p in sorted(Path(cache).expanduser().glob("*.npz")) if p.stem in bank]
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(ibl_pooled_rows)(p, bank) for p in paths)
    pooled_report(pd.DataFrame([r for rows in res for r in rows]), out_dir, "IBL")


def run_ibl(cache, out_dir, limit, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    t0 = time.time()
    print("fitting each session's bias path (behaviour only) ...", flush=True)
    bank = ibl_bank(cache)
    print(f"{len(bank)} sessions fitted ({(time.time() - t0) / 60:.1f} min)", flush=True)
    link = None   # corr(bias, value) is circular (both are built from the same choices), so it is not reported
    paths = [p for p in sorted(Path(cache).expanduser().glob("*.npz")) if p.stem in bank][: limit or None]
    out = Path(out_dir) / "sessions"; out.mkdir(parents=True, exist_ok=True)
    todo = [p for p in paths if not (out / f"{p.stem}.csv").exists()]

    def work(p):
        try:
            rows = ibl_rows(p, bank)
        except Exception as e:
            print(f"FAILED {p.stem}: {e}", flush=True); rows = []
        pd.DataFrame(rows).to_csv(out / f"{p.stem}.csv", index=False)
        return len(rows)

    for j, n in enumerate(Parallel(n_jobs=jobs, return_as="generator")(delayed(work)(p) for p in todo)):
        print(f"[{j + 1}/{len(todo)}] {n} regions  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    fs = [f for f in out.glob("*.csv") if f.stat().st_size > 1]
    report(pd.concat([pd.read_csv(f) for f in fs], ignore_index=True), link, out_dir, "IBL")


def run_steinmetz(data_dir, out_dir, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    import steinmetz_replication as sr
    sessions = sr.load_sessions(data_dir)
    bank = []
    for i, d in enumerate(sessions):
        contrast, choice, fb, side = sr.task(d)
        v = choice >= 0; v[:10] = False
        b, sig = bias_path(contrast, choice, fb, v)
        from history_behaviour import history_design
        H = history_design(side, choice, fb, np.full(len(choice), 0.5))[:, :-1]
        cur = np.where(choice >= 0, choice - 0.5, 0.0)
        bank.append((str(d["mouse_name"]), dict(i=i, b=b, V=value_trace(choice, fb), T=len(b), subject=str(d["mouse_name"]),
                                                 D=np.column_stack([H, cur]))))
    bank_d = {f"s{e['i']}": dict(subject=e["subject"], b=e["b"], V=e["V"], T=e["T"], prior=np.zeros(e["T"])) for m, e in bank}
    link = None
    res = Parallel(n_jobs=jobs)(delayed(steinmetz_rows)(i, d, bank) for i, d in enumerate(sessions))
    df = pd.DataFrame([r for rows in res for r in rows])
    report(df, link, out_dir, "Steinmetz 2019")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "ibl", "steinmetz", "check", "pooled"])
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--data", default="~/Downloads/MULTIVERSE/steinmetz_data")
    ap.add_argument("--out", default="results/mechanism")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.step == "sim":
        sim()
    elif a.step == "pooled":
        run_ibl_pooled(a.cache, a.out + "/ibl_pooled", a.jobs)
    elif a.step == "check":
        run_ibl_check(a.cache, a.out + "/ibl_check", a.every, a.jobs)
    elif a.step == "ibl":
        run_ibl(a.cache, a.out + "/ibl", a.limit, a.jobs)
    else:
        run_steinmetz(a.data, a.out + "/steinmetz", a.jobs)
