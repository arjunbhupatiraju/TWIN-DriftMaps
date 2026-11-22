"""GAP FIX B5: does the hidden, movement-free prior in the brain predict the mouse's choice, trial by trial?
(pre-registration: claude/preregistration-robustness.md, test P1)

B1 (across sessions) asked whether sessions with more neural prior information show more prior-driven behaviour, and
failed narrowly (p = 0.066). This is the stronger, within-session version:
  1. Movement is removed from the neurons and from every target exactly as in step 1 (part6/prior_movement.py).
  2. The Bayes prior is decoded from the pre-stimulus window with cross-validated ridge (contiguous folds): this gives a
     held-out NEURAL prior estimate for every trial. The same is done for 200 pseudo-session priors.
  3. On low-contrast trials (|contrast| <= 0.0625, where the prior matters most), the partial correlation between the
     neural estimate and the choice is computed, controlling for signed contrast, the Bayes prior itself, and the
     mouse's last two choices and last two rewarded choices.
  4. The real partial correlation is compared with the same statistic for the pseudo-session decoders, which see the
     same neurons, drift and arousal, but decode a prior that does not exist.

    python3 part7/prior_choice.py run --limit 3
    python3 part7/prior_choice.py run
    python3 part7/prior_choice.py test
"""
from __future__ import annotations

import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P6 = HERE.parent / "part6"
sys.path.insert(0, str(P6)); sys.path.insert(0, str(HERE.parent / "twin"))
import prior_map as PM        # noqa: E402
import prior_movement as MV   # noqa: E402

OUT = HERE / "results_prior_choice"


def cv_predict(X, Y, k=5):
    """Held-out predictions of every column of Y from X: the PRIOR MAP decoder (contiguous folds, z-scoring and GCV
    ridge inside each training fold), returning predictions instead of their correlation."""
    T = len(X); pred = np.zeros_like(Y)
    edges = np.linspace(0, T, k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.r_[0:a, b:T]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[a:b] - mu) / sd
        ym = Y[tr].mean(0); Ytr = Y[tr] - ym
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
        Uy = U.T @ Ytr; n = len(tr); tot = (Ytr ** 2).sum(0); proj = (Uy ** 2).sum(0)
        XteV = Xte @ Vt.T; best_g = np.full(Y.shape[1], np.inf); best = np.zeros((b - a, Y.shape[1]))
        for al in PM.ALPHAS * n:
            f = s ** 2 / (s ** 2 + al)
            rss = tot - proj + (((1 - f) ** 2)[:, None] * Uy ** 2).sum(0)
            g = n * rss / (n - f.sum()) ** 2
            P = XteV @ ((s / (s ** 2 + al))[:, None] * Uy) + ym
            take = g < best_g
            best[:, take] = P[:, take]; best_g = np.where(take, g, best_g)
        pred[a:b] = best
    return pred


def partial_r(y, Z, C):
    """Correlation of y with every column of Z after removing covariates C (with intercept) from both."""
    C = np.column_stack([np.ones(len(y)), C])
    res = lambda A: A - C @ np.linalg.lstsq(C, A, rcond=None)[0]
    ry, RZ = res(y[:, None])[:, 0], res(Z)
    den = np.sqrt((ry ** 2).sum() * (RZ ** 2).sum(0))
    return np.where(den > 0, (ry[:, None] * RZ).sum(0) / np.where(den > 0, den, 1), 0.0)


def _session(path, gk_root, n_null, burn, max_contrast, seed):
    PM._gk(gk_root)
    from gatekeeper.mv_data import MVSession
    from twin.task import BlockTask
    rng = np.random.default_rng(seed)
    task = BlockTask()
    s = MVSession(Path(path))
    c = np.asarray(s.contrast, float); Tfull = len(c)
    Mfull, _ = MV.movement_matrix(s.sid, Tfull)
    if Mfull is None:
        return []
    prior = task.bayes_prior(np.where(c != 0, (c > 0).astype(float), np.nan))
    ch = 2.0 * (np.asarray(s.choice, float) > 0) - 1
    fb = np.sign(np.asarray(s.feedback, float))
    rw = ch * (fb > 0)
    lag = lambda x, k: np.r_[np.zeros(k), x[:-k]]
    cov = np.column_stack([c, prior, lag(ch, 1), lag(ch, 2), lag(rw, 1), lag(rw, 2)])
    v = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]
    v[:burn] = False
    if v.sum() < 150:
        return []
    M = Mfull[v]
    Zp = np.column_stack([PM._pseudo(task, Tfull, rng)[0] for _ in range(n_null)])[v]
    targets = MV.remove(np.column_stack([prior[v], Zp]), M)
    low = np.abs(c[v]) <= max_contrast
    if low.sum() < 40:
        return []
    rows = []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        X = np.asarray(X, float)[v]; X = X[:, X.std(0) > 0]
        if X.shape[1] < 10:
            continue
        if X.shape[1] > 150:
            X = X[:, rng.choice(X.shape[1], 150, replace=False)]
        pred = cv_predict(MV.remove(X, M), targets)
        pr = partial_r(ch[v][low], pred[low], cov[v][low])
        rows.append(dict(session=s.sid, subject=s.subject, lab=s.lab, region=r, T=int(v.sum()), n_low=int(low.sum()),
                         N=X.shape[1], r_obs=float(pr[0]), r_null_mean=float(pr[1:].mean()), p=PM._p(pr[0], pr[1:]),
                         null=json.dumps([round(float(x), 4) for x in pr[1:]])))
    return rows


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    rows_f = OUT / "prior_choice_rows.csv"
    done = set(pd.read_csv(rows_f).session) if rows_f.exists() else set()
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz"))[: a.limit or None]
             if p.stem not in done and (MV.FEAT / f"{p.stem}.npz").exists()]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.gatekeeper, a.null, 90, 0.0625, 12000 + i): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True)
                continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} regions", flush=True)


def cmd_test(a):
    import pandas as pd
    D = pd.read_csv(OUT / "prior_choice_rows.csv")
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    N = np.array([json.loads(z) for z in D.null])                       # session-regions x null draws
    obs, nul = D.r_obs.mean(), N.mean(0)
    p = float((1 + np.sum(nul >= obs)) / (1 + len(nul)))
    P1 = obs > 0 and p < 0.05
    P(f"# P1: does the movement-free neural prior predict choice, trial by trial? ({len(D)} session-regions, "
      f"{D.session.nunique()} sessions, {D.region.nunique()} regions, {D.subject.nunique()} mice)")
    P(f"  mean partial r (neural prior estimate vs choice, low-contrast trials, controlling contrast, Bayes prior and")
    P(f"  the last 2 choices and rewarded choices): {obs:+.4f}; pseudo-session null mean {nul.mean():+.4f}, 95th pct "
      f"{np.percentile(nul, 95):+.4f}; one-sided p = {p:.4f}")
    P(f"  P1 {'PASS' if P1 else 'FAIL'} (criterion: mean partial r > 0 and above the pseudo-session null, p < 0.05)")
    # descriptive: per session (averaging its regions), and per region (Fisher + BH)
    S = D.groupby("session").r_obs.mean()
    P(f"\n  descriptive: sessions with positive mean partial r: {np.mean(S > 0):.0%} of {len(S)}")
    R = D.groupby("region").agg(n=("session", "nunique"), r=("r_obs", "mean"), p=("p", PM._fisher))
    R["sig"] = PM._bh(R.p.values, 0.05)
    P(f"  regions significant (Fisher across sessions, BH q = 0.05): {int(R.sig.sum())} of {len(R)}")
    top = R.sort_values("p").head(12)
    P("  strongest regions: " + ", ".join(f"{k} (r {x.r:+.3f}, {x.n} sess)" for k, x in top.iterrows()))
    mv = HERE.parent / "part6" / "results_movement" / "movement_regions.csv"
    if mv.exists():
        from scipy.stats import spearmanr
        J = R.join(pd.read_csv(mv, index_col=0)[["p_carries_move"]], how="inner")
        rho, pp = spearmanr(J.r, J.p_carries_move)
        P(f"  regions more likely to carry the hidden prior (step 1) predict choice more: Spearman {rho:+.3f} "
          f"(p = {pp:.3g}, {len(J)} regions)")
    R.to_csv(OUT / "prior_choice_regions.csv")
    (OUT / "PRIOR_CHOICE.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/PRIOR_CHOICE.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--null", type=int, default=200)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
    r.add_argument("--gatekeeper"); r.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("test")
    a = ap.parse_args()
    {"run": cmd_run, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
