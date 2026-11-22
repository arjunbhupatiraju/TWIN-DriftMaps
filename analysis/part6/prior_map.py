"""PRIOR MAP: how brain-wide is the block prior, really?   (pre-registration: claude/preregistration-prior-map.md)

For every IBL Brain-Wide Map session x region it decodes the Bayes-optimal block prior from the pre-stimulus window
(stimOn -600 to -100 ms) with cross-validated ridge regression (5 contiguous folds, ridge strength chosen by GCV inside
each training fold), scores it by the Pearson r between held-out predictions and the prior, and records:
  p_valid     decoder score vs 200 pseudo-sessions drawn from the IBL task (the valid null, as in the IBL papers)
  p_shuffle   decoder score vs 200 trial shuffles (the common but invalid null; for comparison only)
  p_block     the same valid test for the true block probability (secondary target)
  twin_fp_*   on 40 drift twins of these neurons (no prior code by construction): how often each null fires   [H1]
  power_curve valid-test p after planting a NEW pseudo-session prior into 20% of the real neurons at per-neuron
              r on the TWIN grid (12 plants per r)                                                          [H3, H4]
  p_h0, p_h1  held-out semi-synthetic checks for the estimator: a pseudo-session target (no code) and a fresh
              planted code at r = 0.1                                                                         [G1]

  python3 part6/prior_map.py run        # parallel, resumable
  python3 part6/prior_map.py test       # G1, H1-H4 exactly as pre-registered; writes results_prior/PRIOR_MAP.md
"""
from __future__ import annotations

import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_prior"
ROWS = OUT / "prior_rows.csv"
sys.path.insert(0, str(HERE.parent / "twin"))
GRID = (0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3)
GRID0 = (0.0,) + GRID
ALPHAS = np.logspace(-3, 2, 9)          # ridge strength, in units of the number of training trials


# ---------------------------------------------------------------- decoder
def cv_decode(X, Y, k=5):
    """Cross-validated ridge decoding of every column of Y from X. Contiguous folds; within each training fold X is
    z-scored and the ridge strength is chosen per target by generalized cross-validation. Returns the Pearson r between
    held-out predictions and each target (one value per column of Y)."""
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    T = len(X); pred = np.zeros_like(Y)
    edges = np.linspace(0, T, k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.r_[0:a, b:T]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[a:b] - mu) / sd
        ym = Y[tr].mean(0); Ytr = Y[tr] - ym
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        Uy = U.T @ Ytr                                         # k x m
        n = len(tr); tot = (Ytr ** 2).sum(0); proj = (Uy ** 2).sum(0)
        best, best_g = None, np.full(Y.shape[1], np.inf)
        XteV = Xte @ Vt.T
        for al in ALPHAS * n:
            f = s ** 2 / (s ** 2 + al)
            rss = tot - proj + (((1 - f) ** 2)[:, None] * Uy ** 2).sum(0)
            g = n * rss / (n - f.sum()) ** 2
            P = XteV @ ((s / (s ** 2 + al))[:, None] * Uy) + ym
            if best is None:
                best = P.copy()
            take = g < best_g
            best[:, take] = P[:, take]; best_g = np.where(take, g, best_g)
        pred[a:b] = best
    pc = pred - pred.mean(0); yc = Y - Y.mean(0)
    den = np.sqrt((pc ** 2).sum(0) * (yc ** 2).sum(0))
    return np.where(den > 0, (pc * yc).sum(0) / np.where(den > 0, den, 1), 0.0)


def _p(obs, null):
    null = np.asarray(null)
    return float((1 + np.sum(null >= obs)) / (1 + len(null)))


# ---------------------------------------------------------------- one session
def _gk(root):
    root = Path(root or os.environ.get("GATEKEEPER_ROOT", "~/Downloads/GATEKEEPER/gatekeeper")).expanduser()
    for p in (root, root / "next"):
        sys.path.insert(0, str(p))


def _pseudo(task, T, rng):
    """A pseudo-session from the IBL task: the Bayes prior is computed from the stimulus sides the mouse could see
    (zero-contrast trials carry no side information), exactly as for the real session."""
    d = task.sample(T, rng)
    side = np.where(d["contrast"] != 0, d["side"], np.nan)
    return task.bayes_prior(side), d["p_side"]


def _session(path, gk_root, min_units, max_units, n_null, n_twin, n_plant, burn, seed):
    _gk(gk_root)
    from gatekeeper.mv_data import MVSession
    from twin.task import BlockTask
    from twin.drift import phase_twin
    from twin.check import plant
    rng = np.random.default_rng(seed)
    task = BlockTask()
    s = MVSession(Path(path))
    c = np.asarray(s.contrast, float)
    Tfull = len(c)
    pl = np.asarray(s.meta.get("probabilityLeft", []), float)
    side = np.where(c != 0, (c > 0).astype(float), np.nan)
    prior = task.bayes_prior(side)
    v = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]
    v[:burn] = False
    T = int(v.sum())
    if T < 150:
        return []
    bank = [_pseudo(task, Tfull, rng) for _ in range(n_null + 1)]
    Zp = np.column_stack([b[0] for b in bank])[v]             # pseudo priors: column 0 is held out for G1
    Zb = np.column_stack([b[1] for b in bank])[v]             # pseudo block probabilities
    y = prior[v]
    yb = pl[v] if len(pl) == Tfull else None
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
        Ysh = np.column_stack([rng.permutation(y) for _ in range(n_null)])
        tg = [y[:, None], Zp, Ysh] + ([yb[:, None], Zb[:, 1:]] if yb is not None else [])
        sc = cv_decode(X, np.column_stack(tg))
        score, nul, shf = sc[0], sc[2:2 + n_null], sc[2 + n_null:2 + 2 * n_null]
        p_valid, p_shuffle = _p(score, nul), _p(score, shf)
        p_h0 = _p(sc[1], nul)                                 # G1: a pseudo target (no code) scored like the real one
        p_block = _p(sc[2 + 2 * n_null], sc[3 + 2 * n_null:]) if yb is not None else np.nan
        # H1: drift twins of these neurons carry no prior code; how often does each null fire for the real prior?
        fp_v = fp_s = 0
        for _ in range(n_twin):
            st = cv_decode(phase_twin(X, rng), np.column_stack([y[:, None], Zp[:, 1:], Ysh]))
            fp_v += _p(st[0], st[1:1 + n_null]) < 0.05
            fp_s += _p(st[0], st[1 + n_null:]) < 0.05
        # power: plant a NEW pseudo prior (never the real one) into 20% of the real neurons
        curve, med = {}, {}
        for rr in GRID:
            ps, ss = [], []
            for _ in range(n_plant):
                yp = _pseudo(task, Tfull, rng)[0][v]
                st = cv_decode(plant(X, yp, rr, 0.2, rng), np.column_stack([yp[:, None], Zp[:, 1:]]))
                ps.append(round(_p(st[0], st[1:]), 4)); ss.append(st[0])
            curve[str(rr)] = ps; med[str(rr)] = float(np.median(ss))
        yp = _pseudo(task, Tfull, rng)[0][v]                  # G1: one fresh planted code at r = 0.1
        st = cv_decode(plant(X, yp, 0.1, 0.2, rng), np.column_stack([yp[:, None], Zp[:, 1:]]))
        p_h1 = _p(st[0], st[1:])
        ref = [float(np.median(nul))] + [med[str(rr)] for rr in GRID]
        r_hat = float(GRID0[int(np.argmin(np.abs(np.array(ref) - score)))])
        rows.append(dict(session=s.sid, subject=s.subject, lab=s.lab, region=r, variable="prior_dec", T=T, N=N,
                         score=float(score), null_mean=float(np.mean(nul)), p_valid=p_valid, p_shuffle=p_shuffle,
                         p_block=p_block, p_h0=p_h0, p_h1=p_h1, twin_fp_valid=fp_v / n_twin,
                         twin_fp_shuffle=fp_s / n_twin, n_twin=n_twin, r_hat=r_hat, power_curve=json.dumps(curve)))
    return rows


def _paths(tag):
    global ROWS
    ROWS = OUT / (f"prior_rows_{tag}.csv" if tag else "prior_rows.csv")
    return OUT / (f"PRIOR_MAP_{tag}.md" if tag else "PRIOR_MAP.md")


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    _paths(a.tag)
    done = set(pd.read_csv(ROWS).session) if ROWS.exists() else set()
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz"))[::a.every] if p.stem not in done]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.gatekeeper, a.min_units, a.max_units, a.null, a.twins, a.plant, 90, i): p
                for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True)
                continue
            if rows:
                pd.DataFrame(rows).to_csv(ROWS, mode="a", header=not ROWS.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} regions", flush=True)


# ---------------------------------------------------------------- analysis
def _fisher(ps):
    from scipy.stats import chi2
    ps = np.clip(np.asarray(ps, float), 1e-12, 1)
    return float(chi2.sf(-2 * np.log(ps).sum(), 2 * len(ps)))


def _bh(p, q):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    out = np.zeros(n, bool); out[o[:k]] = True
    return out


def _lik(p_obs, curve, rng):
    u = np.log10(max(p_obs, 1e-4)); h = 0.35; out = []
    for r in GRID0:
        smp = rng.random(400) if r == 0 else np.asarray(curve[str(r)], float)
        z = np.log10(np.clip(smp, 1e-4, 1))
        out.append(np.mean(np.exp(-0.5 * ((u - z) / h) ** 2)) / (h * np.sqrt(2 * np.pi)) + 1e-6)
    return np.log(np.array(out))


def region_logliks(D, pcol, rng):
    """Region x GRID0 log-likelihood: the sessions of a region share one code size."""
    L = {}
    for reg, g in D.groupby("region"):
        L[reg] = np.sum([_lik(p, json.loads(c), rng) for p, c in zip(g[pcol], g.power_curve)], axis=0)
    return L


def em_pi(LL, iters=300):
    """Brain-wide distribution of code size across regions (empirical Bayes, EM over the TWIN grid)."""
    M = np.vstack(LL); M = np.exp(M - M.max(1, keepdims=True)); w = np.full(M.shape[1], 1 / M.shape[1])
    for _ in range(iters):
        post = M * w; post /= post.sum(1, keepdims=True); w = post.mean(0)
    return w


def frac(w, thr):
    return float(sum(wi for r, wi in zip(GRID0, w) if r >= thr - 1e-12))


def cmd_test(a):
    import pandas as pd
    md = _paths(a.tag)
    D = pd.read_csv(ROWS)
    rng = np.random.default_rng(0)
    B = a.boot
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    P(f"# PRIOR MAP: {D.region.nunique()} regions, {len(D)} session-regions, {D.session.nunique()} sessions, "
      f"{D.subject.nunique()} mice, {D.lab.nunique()} labs\n")

    # ---- G1: estimator validity on semi-synthetic data (gate for H3)
    regs = sorted(D.region.unique())
    L0 = region_logliks(D, "p_h0", rng)
    w0 = em_pi([L0[r] for r in regs])
    # planted set: a random 30% of regions use the fresh r = 0.1 planted p (all their sessions), the rest the null p
    pick = set(rng.choice(regs, int(round(0.3 * len(regs))), replace=False))
    Dm = D.assign(p_mix=np.where(D.region.isin(pick), D.p_h1, D.p_h0))
    w1 = em_pi(list(region_logliks(Dm, "p_mix", rng).values()))
    g1a, g1b = frac(w0, 0.04), frac(w1, 0.04)
    G1 = g1a <= 0.10 and abs(g1b - 0.30) <= 0.10
    P("## G1 (estimator check)")
    P(f"  no code anywhere: estimated fraction with r >= 0.04 = {g1a:.3f} (criterion <= 0.10)")
    P(f"  code in 30% of regions: estimated {g1b:.3f} (criterion 0.20-0.40)")
    P(f"  G1 {'PASS' if G1 else 'FAIL: H3 is not interpreted'}\n")

    # ---- H1: valid decoder test is calibrated on drift twins; shuffle null is not
    tw = D.assign(kv=D.twin_fp_valid * D.n_twin, ks=D.twin_fp_shuffle * D.n_twin)
    sess = {k: g for k, g in tw.groupby("session")}; keys = list(sess)
    fv = tw.kv.sum() / tw.n_twin.sum(); fs = tw.ks.sum() / tw.n_twin.sum()
    bv = []
    for _ in range(B):
        g = pd.concat([sess[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        bv.append(g.kv.sum() / g.n_twin.sum())
    hi = float(np.percentile(bv, 97.5))
    H1 = fv <= 0.07 and hi <= 0.08
    P("## H1 (decoder false positives on drift twins, no prior code)")
    P(f"  pseudo-session null: {fv:.3f} (95% CI upper {hi:.3f}; criterion rate <= 0.07 and upper <= 0.08)")
    P(f"  trial-shuffle null: {fs:.3f}")
    P(f"  H1 {'PASS' if H1 else 'FAIL'}" + (" (amended run uses 1 twin per region: H1 is taken from the registered run)" if a.tag else "") + "\n")

    # ---- H2: detected fraction of regions (descriptive)
    R = D.groupby("region").agg(n_sess=("session", "nunique"), labs=("lab", "nunique"),
                                p_valid=("p_valid", lambda x: _fisher(x)), p_shuffle=("p_shuffle", lambda x: _fisher(x)),
                                sig1_valid=("p_valid", lambda x: float(np.mean(np.asarray(x) < 0.05))),
                                sig1_shuffle=("p_shuffle", lambda x: float(np.mean(np.asarray(x) < 0.05))))
    R["fdr_valid"] = _bh(R.p_valid.values, 0.05); R["fdr_shuffle"] = _bh(R.p_shuffle.values, 0.05)
    R["fdr_valid_01"] = _bh(R.p_valid.values, 0.01)
    det = float(R.fdr_valid.mean())
    P("## H2 (detected, descriptive)")
    P(f"  regions significant (Fisher across sessions, BH q = 0.05): valid null {det:.3f} "
      f"({int(R.fdr_valid.sum())}/{len(R)}); q = 0.01: {R.fdr_valid_01.mean():.3f}; trial-shuffle null {R.fdr_shuffle.mean():.3f}")
    P(f"  single sessions significant: valid {np.mean(D.p_valid < 0.05):.3f}, shuffle {np.mean(D.p_shuffle < 0.05):.3f}")
    if D.p_block.notna().any():
        P(f"  secondary target (true block probability), single sessions significant: {np.mean(D.p_block.dropna() < 0.05):.3f}\n")

    # ---- H3: power-corrected fraction of regions carrying the prior
    LL = region_logliks(D, "p_valid", rng)
    w = em_pi([LL[r] for r in regs])
    est = {t: frac(w, t) for t in (0.02, 0.04, 0.1)}
    boots = []
    for _ in range(B):
        idx = rng.integers(0, len(regs), len(regs))
        wb = em_pi([LL[regs[i]] for i in idx], iters=150)
        boots.append({t: frac(wb, t) for t in est})
    ci = {t: (float(np.percentile([b[t] for b in boots], 2.5)), float(np.percentile([b[t] for b in boots], 97.5))) for t in est}
    lo, hi3 = ci[0.04]
    verdict = ("BRAIN-WIDE" if lo > 0.5 else "LOCALIZED" if hi3 < 0.5 else "INCONCLUSIVE")
    P("## H3 (power-corrected: what fraction of regions truly carry the prior?)")
    P(f"  estimated distribution of code size across regions: {dict(zip(GRID0, np.round(w, 3)))}")
    for t in est:
        P(f"  fraction with r >= {t}: {est[t]:.3f} (95% CI {ci[t][0]:.3f}-{ci[t][1]:.3f})")
    gap = [b[0.04] - det for b in boots]
    P(f"  detection gap (estimated r >= 0.04 minus detected at q = 0.05): {est[0.04] - det:+.3f} "
      f"(95% CI {np.percentile(gap, 2.5):+.3f} to {np.percentile(gap, 97.5):+.3f})")
    post = {r: np.exp(LL[r] + np.log(w + 1e-12) - (LL[r] + np.log(w + 1e-12)).max()) for r in regs}
    R["p_carries"] = [float(post[r][np.array(GRID0) >= 0.04].sum() / post[r].sum()) for r in R.index]
    R["undetected_but_likely"] = (~R.fdr_valid) & (R.p_carries > 0.5)
    R["unanswerable"] = (R.p_carries > 0.2) & (R.p_carries < 0.8)
    P(f"  regions not detected but probably carrying it (posterior > 0.5): {int(R.undetected_but_likely.sum())}; "
      f"unanswerable with current data (posterior 0.2-0.8): {int(R.unanswerable.sum())}")
    P(f"  H3 {'VERDICT: ' + verdict if G1 else 'NOT INTERPRETED (G1 failed)'}\n")

    # ---- H4: zero-shot replication of the prior map across labs
    sys.path.insert(0, str(HERE))
    import replication as RP
    C, A = RP.build_attempts(D[["session", "subject", "lab", "region", "variable", "T", "N", "p_valid", "r_hat",
                                "power_curve"]].copy(), rng)
    A = A.reset_index(drop=True)
    _, real, bt = _h4_scores(A, rng, B, RP)
    d = [b["twin3"] - b["p_orig"] for b in bt]
    dn = [b["twin3"] - b["naive"] for b in bt]
    H4 = real["twin3"] >= 0.65 and np.percentile(d, 2.5) > 0
    P("## H4 (does TWIN predict which labs reproduce the prior map?)")
    P(f"  {len(A)} attempts ({A.orig_lab.nunique()} labs, {A.region.nunique()} regions); base rate {A.replicated.mean():.3f}")
    for k, v in real.items():
        lo_, hi_ = np.percentile([b[k] for b in bt], [2.5, 97.5])
        P(f"  AUC {k}: {v:.3f} ({lo_:.3f}-{hi_:.3f})")
    P(f"  twin3 - p_orig: {real['twin3'] - real['p_orig']:+.3f} ({np.percentile(d, 2.5):+.3f} to {np.percentile(d, 97.5):+.3f}); "
      f"twin3 - naive: {real['twin3'] - real['naive']:+.3f} ({np.percentile(dn, 2.5):+.3f} to {np.percentile(dn, 97.5):+.3f})")
    P(f"  H4 {'PASS' if H4 else 'FAIL'} (criterion: AUC twin3 >= 0.65 and twin3 > original p, lab-bootstrap CI above 0)")

    sfx = f"_{a.tag}" if a.tag else ""
    R.to_csv(OUT / f"prior_map_regions{sfx}.csv")
    A.drop(columns=[c for c in A.columns if A[c].dtype == object and c not in ("orig_lab", "rep_lab", "region", "variable")]).to_csv(
        OUT / f"prior_map_attempts{sfx}.csv", index=False)
    md.write_text("\n".join(L) + "\n")


def _h4_scores(A, rng, B, RP):
    from sklearn.linear_model import LogisticRegression
    y = A.replicated.values.astype(float); po = np.zeros(len(A))
    for lab in A.orig_lab.unique():                            # original p: fitted leaving the predicted lab out
        tr, te = (A.orig_lab != lab).values, (A.orig_lab == lab).values
        yt = y[tr]
        po[te] = (LogisticRegression().fit(A.loc[tr, ["orig_logp"]].values, yt).predict_proba(A.loc[te, ["orig_logp"]].values)[:, 1]
                  if 0 < yt.mean() < 1 else yt.mean())
    A = A.assign(p_orig=po)
    preds = ["twin3", "twin2", "twin", "naive", "p_orig"]
    stats = lambda d: {k: RP._auc(d.replicated, d[k]) for k in preds}
    groups = [g for _, g in A.groupby("orig_lab")]
    import pandas as pd
    bt = [stats(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])) for _ in range(B)]
    return A, stats(A), bt


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
    r.add_argument("--gatekeeper"); r.add_argument("--every", type=int, default=1)
    r.add_argument("--min-units", type=int, default=10); r.add_argument("--max-units", type=int, default=150)
    r.add_argument("--null", type=int, default=200); r.add_argument("--twins", type=int, default=40)
    r.add_argument("--plant", type=int, default=12); r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t = sub.add_parser("test"); t.add_argument("--boot", type=int, default=500)
    for q in (r, t):
        q.add_argument("--tag", default="", help="separate output files, e.g. for a pre-registered amendment")
    a = ap.parse_args()
    {"run": cmd_run, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
