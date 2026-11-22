"""REPLICATION: can TWIN predict which neural findings replicate across labs?  (pre-registration:
claude/preregistration-replication.md)  Uses the IBL repeated-site recordings (same probe site, ~10 labs).

  python3 part6/replication.py run        # per session x region x variable: valid-test p, effect size, power curve
  python3 part6/replication.py test       # lab-level claims, replication attempts, pre-registered tests

A "claim" = (lab, region, variable): the lab's sessions, Fisher-combined valid-test p < 0.05.
A "replication attempt" = the same region and variable in another lab (its sessions combined).
Predictions of P(replicating lab finds it significant), from the ORIGINAL lab's effect size:
  twin     posterior over the code size given the ORIGINAL lab's sessions (likelihood from codes planted in its own
           neurons; empirical-Bayes prior fitted across all claims, which corrects the winner's curse), times the
           power of the REPLICATING lab's own sessions at each size (codes planted in its real neurons, valid test)
  naive    textbook power: independent trials, shuffle-test chi-square, same effect, same N, T and sessions
  p_orig   the original lab's combined p alone (logistic)
  const    one overall rate
"""
from __future__ import annotations

import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_replication"
sys.path.insert(0, str(HERE.parent / "twin"))
sys.path.insert(0, str(HERE))
RS = {"VISa", "VISam", "CA1", "DG", "LP", "PO"}
R_GRID = (0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3)


def is_rs(regions):
    h = RS & set(regions)
    return {"CA1", "DG"} <= h and bool({"LP", "PO"} & h) and bool({"VISa", "VISam"} & h)


def _session(path, gk_root, min_units, max_units, n_null, n_plant, burn, seed):
    from atlas import _gk, _labels, VARIABLES
    _gk(gk_root)
    from gatekeeper.mv_data import MVSession
    from twin.task import BlockTask, PseudoSessions
    from twin.stats import pooled_corr
    from twin.check import plant
    from twin.tests import _p, p_linear_shift
    rng = np.random.default_rng(seed)
    s = MVSession(Path(path))
    if not is_rs(s.regions):
        return []
    labs = _labels(s)
    base = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]; base[:burn] = False
    Tfull = len(base)
    task_nulls = {"block": PseudoSessions(BlockTask(), "block"), "side": PseudoSessions(BlockTask(), "side")}
    rows = []
    for var, (window, _, nulltype) in VARIABLES.items():
        y_full = labs[var]
        if y_full is None:
            continue
        v = base & np.isfinite(y_full)
        if var == "stimulus":
            v &= y_full != 0
        ev = {"choice_a": "firstMovement", "feedback_a": "feedback"}.get(window)
        if ev:
            v &= s.valid[ev]
        T = int(v.sum())
        if T < 150 or np.std(y_full[v]) == 0:
            continue
        y = y_full[v]
        pseudo = nulltype in task_nulls
        bank = (np.column_stack([np.asarray(task_nulls[nulltype].draw(Tfull, rng)) for _ in range(400)])[v]
                if pseudo else None)                      # pseudo-session bank, drawn once per session x variable

        def nulls(n):
            return bank[:, rng.integers(0, bank.shape[1], n)]

        def valid_p(Xa, ya):
            if pseudo:
                w = pooled_corr(Xa, np.column_stack([ya, nulls(n_null)]))
                return _p(w[0], w[1:]), float(w[0] - np.mean(w[1:]))
            return p_linear_shift(Xa, ya), None

        for r in sorted(RS & set(s.regions)):
            X = s.X(r, window)
            if X is None:
                continue
            X = X[v]; X = X[:, X.std(0) > 0]
            if X.shape[1] < min_units:
                continue
            if X.shape[1] > max_units:
                X = X[:, rng.choice(X.shape[1], max_units, replace=False)]
            N = X.shape[1]
            p_obs, excess = valid_p(X, y)
            if excess is None:                       # linear-shift effect: excess over the shifted alignments
                L = T - 2 * max(10, T // 10); S = max(10, T // 10)
                Ys = np.column_stack([y[S + k:S + k + L] for k in range(-S, S + 1)])
                w = pooled_corr(X[S:S + L], Ys)
                excess = float((w[S] - np.mean(np.delete(w, S))) * T / L)
            # effect size in "planted r" units: excess S ~ T * N * 0.2 * r^2
            r_hat = float(np.sqrt(max(excess, 0.0) / (T * N * 0.2)))
            curve = {}
            for rr in R_GRID:
                ps = []
                for _ in range(n_plant):
                    # planted label must NOT be the real label (the neurons may already encode it): circular shift
                    # keeps the label's own structure but breaks its alignment with any existing code
                    yp = nulls(1)[:, 0] if pseudo else np.roll(y, int(rng.integers(T // 4, 3 * T // 4)))
                    ps.append(round(valid_p(plant(X, yp, rr, 0.2, rng), yp)[0], 4))
                curve[str(rr)] = ps
            rows.append(dict(session=s.sid, subject=s.subject, lab=s.lab, region=r, variable=var, T=T, N=N,
                             p_valid=p_obs, r_hat=r_hat, power_curve=json.dumps(curve)))
    return rows


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    f = OUT / "rs_rows.csv"
    done = set(pd.read_csv(f).session) if f.exists() else set()
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz")) if p.stem not in done]
    print(f"{len(paths)} sessions to scan on {a.workers} workers (only repeated-site ones are analysed)", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.gatekeeper, a.min_units, a.max_units, a.null, a.plant, 90, i): p
                for i, p in enumerate(paths)}
        n = 0
        for fu in as_completed(futs):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"{futs[fu].stem}: ERROR {e}", flush=True); continue
            if rows:
                n += 1
                pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
                print(f"[{n}] {futs[fu].stem} ({rows[0]['lab']}): {len(rows)} region-variables", flush=True)


# ------------------------------------------------------------------ tests
def _fisher(ps):
    from scipy.stats import chi2
    ps = np.clip(np.asarray(ps, float), 1e-12, 1)
    return chi2.sf(-2 * np.log(ps).sum(axis=-1), 2 * ps.shape[-1])


def _twin_prob(sessions_curves, r, rng, B=400):
    """P(Fisher-combined valid p < 0.05) for a lab's own sessions, each with a planted code of size r
    (interpolated on the planted-r grid)."""
    grid = np.array(R_GRID)
    r = float(np.clip(r, grid[0], grid[-1]))
    j = int(np.searchsorted(grid, r)); j = min(max(j, 1), len(grid) - 1)
    wgt = (r - grid[j - 1]) / (grid[j] - grid[j - 1])
    draws = []
    for c in sessions_curves:
        lo, hi = np.array(c[str(grid[j - 1])]), np.array(c[str(grid[j])])
        pick = np.where(rng.random(B) < wgt, hi[rng.integers(0, len(hi), B)], lo[rng.integers(0, len(lo), B)])
        draws.append(pick)
    return float(np.mean(_fisher(np.column_stack(draws)) < 0.05))


def _naive_prob(Ts, Ns, r, rng, B=2000):
    """Textbook power: independent trials, shuffle-test statistic S ~ chi2_N (noncentrality T*N*0.2*r^2), Fisher."""
    from scipy.stats import chi2, ncx2
    ps = []
    for T, N in zip(Ts, Ns):
        S = ncx2.rvs(N, max(T * N * 0.2 * r * r, 1e-9), size=B, random_state=rng)
        ps.append(chi2.sf(S, N))
    return float(np.mean(_fisher(np.column_stack(ps)) < 0.05))


def _auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    if y.all() or (~y).all():
        return np.nan
    from scipy.stats import rankdata
    rk = rankdata(s)
    return float((rk[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


GRID0 = (0.0,) + R_GRID


def _session_lik(p_obs, curve, rng):
    """Likelihood of a session's observed valid-test p under each code size (r = 0: p uniform), from the planted
    p-value samples of THAT session (so its own drift and size are built in). Gaussian kernel on log10 p."""
    u = np.log10(max(p_obs, 1e-4))
    out = []
    for r in GRID0:
        smp = rng.random(400) if r == 0 else np.asarray(curve[str(r)], float)
        z = np.log10(np.clip(smp, 1e-4, 1))
        h = 0.35
        out.append(np.mean(np.exp(-0.5 * ((u - z) / h) ** 2)) / (h * np.sqrt(2 * np.pi)) + 1e-6)
    return np.array(out)


def _power_at(curves, rng, B=400):
    """P(Fisher-combined valid p < 0.05) for a lab's sessions at each code size on GRID0."""
    out = []
    for r in GRID0:
        cols = [rng.random(B) if r == 0 else np.asarray(c[str(r)])[rng.integers(0, len(c[str(r)]), B)] for c in curves]
        out.append(float(np.mean(_fisher(np.column_stack(cols)) < 0.05)))
    return np.array(out)


def build_attempts(D, rng):
    import pandas as pd
    claims = []
    for (lab, reg, var), g in D.groupby(["lab", "region", "variable"]):
        curves = [json.loads(c) for c in g.power_curve]
        lik = np.prod([_session_lik(pv, c, rng) for pv, c in zip(g.p_valid, curves)], axis=0)
        claims.append(dict(lab=lab, region=reg, variable=var, n_sess=len(g), p=float(_fisher(g.p_valid.values)),
                           r_hat=float(np.median(g.r_hat)), curves=curves, lik=lik, power=_power_at(curves, rng),
                           Ts=list(g["T"]), Ns=list(g["N"])))
    C = pd.DataFrame(claims)
    # empirical-Bayes prior over code sizes, fitted across all claims (EM), so significant claims are not taken at
    # face value (winner's curse); each claim's posterior over r uses its own lab's drift-aware likelihood
    L = np.vstack(C.lik.values); pi = np.full(len(GRID0), 1 / len(GRID0))
    for _ in range(200):
        post = L * pi; post /= post.sum(1, keepdims=True); pi = post.mean(0)
    C["post"] = list(post)
    print("empirical-Bayes prior over code size r:", dict(zip(GRID0, np.round(pi, 3))))
    # ---- TWIN v2 (zero-shot: uses claim p-values only, never replication outcomes)
    #  (1) a separate empirical-Bayes prior over code size for EACH variable (effect sizes differ a lot by variable)
    #  (2) evidence pooled across labs: to predict lab B, combine the original lab's likelihood with the likelihoods of
    #      every OTHER lab that recorded this region x variable (except B), tempered by W_OTHER for between-lab
    #      heterogeneity (fixed a priori, not tuned)
    W_OTHER = 0.5
    logL = np.log(np.vstack(C.lik.values))
    C["loglik"] = list(logL)
    pri = {}
    for var, idx in C.groupby("variable").groups.items():
        Lv = np.exp(logL[list(idx)] - logL[list(idx)].max(1, keepdims=True)); q = np.full(len(GRID0), 1 / len(GRID0))
        for _ in range(200):
            pv = Lv * q; pv /= pv.sum(1, keepdims=True); q = pv.mean(0)
        pri[var] = np.clip(q, 1e-6, None)
    print("TWIN v2 per-variable priors (mean r):", {v: round(float(np.dot(q, GRID0)), 3) for v, q in pri.items()})
    by_rv = {k: g for k, g in C.groupby(["region", "variable"])}
    v3 = _fit_v3(C, by_rv)
    att = []
    for _, o in C.iterrows():
        group = by_rv[(o.region, o.variable)]
        for _, rpl in group[group.lab != o.lab].iterrows():
            others = group[(group.lab != o.lab) & (group.lab != rpl.lab)]
            ll = o.loglik + W_OTHER * (np.sum(np.vstack(others.loglik.values), 0) if len(others) else 0)
            ll = ll + np.log(pri[o.variable]); post2 = np.exp(ll - ll.max()); post2 /= post2.sum()
            att.append(dict(orig_lab=o.lab, rep_lab=rpl.lab, region=o.region, variable=o.variable,
                            orig_sig=o.p < 0.05, orig_logp=-np.log10(max(o.p, 1e-12)), r_hat=o.r_hat,
                            replicated=rpl.p < 0.05, rep_n=len(rpl.Ts), rep_N=float(np.mean(rpl.Ns)),
                            rep_T=float(np.mean(rpl.Ts)), n_other_labs=len(others),
                            twin=float(np.dot(o.post, rpl.power)), twin2=float(np.dot(post2, rpl.power)),
                            twin3=_predict_v3(v3, o, others, rpl),
                            naive=_naive_prob(rpl.Ts, rpl.Ns, o.r_hat, rng)))
    return C, pd.DataFrame(att)


# ---------------------------------------------------------------- TWIN v3: between-lab heterogeneity (calibration fix)
TAUS_V3 = (0.05, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


def _kernel(tau):
    """K[i, j] = P(a lab's code size is GRID0[j] | the shared size is GRID0[i]), log-normal spread tau on r + 0.01."""
    lr = np.log(np.array(GRID0) + 0.01)
    K = np.exp(-0.5 * ((lr[None, :] - lr[:, None]) / tau) ** 2)
    return K / K.sum(1, keepdims=True)


def _fit_v3(C, by_rv):
    """Zero-shot: for each variable, choose the heterogeneity tau and the prior over the SHARED size by maximizing the
    marginal likelihood of all labs' claims in each region x variable. Uses claim p-values only, never replication
    outcomes."""
    Lnorm = {i: np.exp(l - l.max()) for i, l in zip(C.index, C.loglik)}
    fits = {}
    for var in C.variable.unique():
        groups = [g for (r, v), g in by_rv.items() if v == var]
        best = None
        for tau in TAUS_V3:
            K = _kernel(tau)
            M = [np.vstack([K @ Lnorm[i] for i in g.index]) for g in groups]      # per lab: M(R) = sum_j K(R,j) L(j)
            logM = [np.log(np.clip(m, 1e-300, None)).sum(0) for m in M]           # group evidence for each shared R
            pi = np.full(len(GRID0), 1 / len(GRID0))
            for _ in range(100):
                post = [np.exp(lm - lm.max()) * pi for lm in logM]
                post = np.vstack([q / q.sum() for q in post]); pi = np.clip(post.mean(0), 1e-8, None); pi /= pi.sum()
            ll = sum(np.log(np.dot(np.exp(lm - lm.max()), pi)) + lm.max() for lm in logM)
            if best is None or ll > best[0]:
                best = (ll, tau, pi, K)
        fits[var] = dict(tau=best[1], pi=best[2], K=best[3])
    print("TWIN v3 heterogeneity per variable (tau):", {v: f["tau"] for v, f in fits.items()})
    return dict(fits=fits, L={i: np.exp(l - l.max()) for i, l in zip(C.index, C.loglik)})


def _predict_v3(v3, o, others, rpl):
    f = v3["fits"][o.variable]; K = f["K"]
    lm = np.log(np.clip(K @ v3["L"][o.name], 1e-300, None))
    for i in others.index:
        lm = lm + np.log(np.clip(K @ v3["L"][i], 1e-300, None))
    post = np.exp(lm - lm.max()) * f["pi"]; post /= post.sum()
    return float(post @ (K @ rpl.power))           # replicating lab's own size varies around the shared one


def calibration(y, p, bins=10):
    """Expected calibration error, and the calibration slope / intercept (logistic regression of y on logit p)."""
    y = np.asarray(y, float); p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    from sklearn.linear_model import LogisticRegression
    lg = np.log(p / (1 - p))[:, None]
    m = LogisticRegression(C=1e6).fit(lg, y) if 0 < y.mean() < 1 else None
    return float(ece), (float(m.coef_[0, 0]) if m else np.nan), (float(m.intercept_[0]) if m else np.nan)


def calib_report(sub, rng, cluster="region", B=1000):
    """v2 vs v3: ECE, calibration slope, AUC, Brier, with cluster-bootstrap CIs (regions)."""
    groups = {g: d for g, d in sub.groupby(cluster)}; keys = list(groups)
    def m(d):
        e2, s2, _ = calibration(d.replicated, d.twin2); e3, s3, _ = calibration(d.replicated, d.twin3)
        return dict(ece2=e2, ece3=e3, slope2=s2, slope3=s3, auc2=_auc(d.replicated, d.twin2), auc3=_auc(d.replicated, d.twin3),
                    br2=float(np.mean((d.twin2 - d.replicated) ** 2)), br3=float(np.mean((d.twin3 - d.replicated) ** 2)))
    real = m(sub); bs = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        bs.append(m(pd_concat([groups[keys[i]] for i in pick])))
    ci = lambda f: (np.nanpercentile([f(b) for b in bs], 2.5), np.nanpercentile([f(b) for b in bs], 97.5))
    return real, dict(d_ece=ci(lambda b: b["ece2"] - b["ece3"]), slope3=ci(lambda b: b["slope3"]),
                      d_auc=ci(lambda b: b["auc3"] - b["auc2"]), d_brier=ci(lambda b: b["br2"] - b["br3"]))


def _bins_line(y, p):
    import pandas as pd
    d = pd.DataFrame(dict(y=np.asarray(y, float), p=np.asarray(p, float)))
    g = d.groupby(pd.cut(d.p, [0, .2, .4, .6, .8, 1.0001], include_lowest=True), observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("p", "size"))
    return "; ".join(f"{r.pred:.2f}->{r.obs:.2f} (n={int(r.n)})" for _, r in g.iterrows())


def _calib_block(name, sub, rng):
    real, ci = calib_report(sub, rng)
    return [f"## {name}: {len(sub)} attempts, replication rate {sub.replicated.mean():.2f}",
            f"  ECE  v2 {real['ece2']:.3f}  v3 {real['ece3']:.3f}   improvement {real['ece2'] - real['ece3']:+.3f} [{ci['d_ece'][0]:+.3f}, {ci['d_ece'][1]:+.3f}]",
            f"  calibration slope  v2 {real['slope2']:.2f}  v3 {real['slope3']:.2f} [{ci['slope3'][0]:.2f}, {ci['slope3'][1]:.2f}]  (1 = perfect)",
            f"  AUC  v2 {real['auc2']:.3f}  v3 {real['auc3']:.3f}   change {real['auc3'] - real['auc2']:+.3f} [{ci['d_auc'][0]:+.3f}, {ci['d_auc'][1]:+.3f}]",
            f"  Brier v2 {real['br2']:.3f}  v3 {real['br3']:.3f}   gain {real['br2'] - real['br3']:+.3f} [{ci['d_brier'][0]:+.3f}, {ci['d_brier'][1]:+.3f}]",
            f"  bins v2: {_bins_line(sub.replicated, sub.twin2)}",
            f"  bins v3: {_bins_line(sub.replicated, sub.twin3)}"], real, ci


def cmd_confirm5(a):
    """CALIBRATION confirmation (claude/preregistration-calibration-v3.md): TWIN v3 vs v2 on the FRESH set-3 variables.
    --exploratory: the same comparison on every earlier test set (already seen; descriptive only)."""
    import pandas as pd
    rng = np.random.default_rng(5)
    base = HERE / "results_atlas"
    regs = pd.read_csv(base / "atlas_rows.csv", usecols=["session", "region"]).groupby("session").region.apply(set)
    rs = {sid for sid, rr in regs.items() if is_rs(rr)}

    def ibl(f, variables=None):
        D = pd.read_csv(base / f); D = D[~D.session.isin(rs)]
        if variables:
            D = D[D.variable.isin(variables)]
        nl = D.groupby(["region", "variable"]).lab.nunique(); keep = nl[nl >= 2].index
        return D.set_index(["region", "variable"]).loc[lambda x: x.index.isin(keep)].reset_index()
    lines, verdict = [], None
    if not a.exploratory:
        D = ibl("atlas_rows_set3.csv")
        claims, A = build_attempts(D, rng)
        lines.append(f"# CALIBRATION CONFIRMATION (fresh variables {sorted(D.variable.unique())}): {D.session.nunique()} sessions, "
                     f"{D.lab.nunique()} labs, {D.region.nunique()} regions; {len(claims)} claims\n")
        for name, sub in (("PRIMARY", A[A.orig_sig]), ("SECONDARY", A)):
            blk, real, ci = _calib_block(name, sub, rng); lines += blk + [""]
            if name == "PRIMARY":
                verdict = (ci["d_ece"][0] > 0, ci["slope3"][0] <= 1.25 and ci["slope3"][1] >= 0.8, ci["d_auc"][0] > -0.01)
        lines += [f"Q1 (v3 ECE lower than v2, CI > 0): {'PASS' if verdict[0] else 'FAIL'}",
                  f"Q2 (v3 calibration slope CI overlaps [0.8, 1.25]): {'PASS' if verdict[1] else 'FAIL'}",
                  f"Q3 (v3 AUC not worse than v2 by more than 0.01): {'PASS' if verdict[2] else 'FAIL'}"]
    else:
        lines.append("# CALIBRATION, EXPLORATORY (earlier test sets, already seen)\n")
        for label, D in (("Brain-Wide Map, original variables", ibl("atlas_rows.csv")),
                         ("Brain-Wide Map, new variables", ibl("atlas_rows_new.csv"))):
            claims, A = build_attempts(D, rng)
            blk, _, _ = _calib_block(f"{label} PRIMARY", A[A.orig_sig], rng); lines += blk + [""]
        if (base / "allen_rows.csv").exists() and (base / "steinmetz_rows.csv").exists():
            I = pd.concat([ibl("atlas_rows.csv"), ibl("atlas_rows_new.csv", ["prev_choice"])])
            ext = [pd.read_csv(base / f) for f in EXTERNAL.values()]
            regions = set().union(*[set(d.region) for d in ext])
            D = pd.concat([I] + ext); D = D[D.variable.isin(["stimulus", "choice", "feedback", "prev_choice"]) & D.region.isin(regions)]
            claims, A = build_attempts(D, rng)
            A = A[(A.orig_lab.isin(EXTERNAL) | A.rep_lab.isin(EXTERNAL)) & (A.orig_lab != "cortexlab") & (A.rep_lab != "cortexlab")]
            blk, _, _ = _calib_block("Pooled cross-dataset PRIMARY", A[A.orig_sig], rng); lines += blk
    text = "\n".join(lines)
    OUT.mkdir(exist_ok=True)
    (OUT / ("CALIBRATION_EXPLORATORY.md" if a.exploratory else "CALIBRATION_RESULT.md")).write_text(text + "\n")
    print(text)


def _scores(A, rng, B=1000, cluster="orig_lab"):
    """AUC and Brier of each predictor, with a cluster bootstrap (default: over original labs)."""
    from sklearn.linear_model import LogisticRegression
    y = A.replicated.values.astype(float)
    # p_orig and const: fitted leaving out the original lab being predicted (no in-sample advantage)
    po, co = np.zeros(len(A)), np.zeros(len(A))
    for lab in A.orig_lab.unique():
        tr, te = (A.orig_lab != lab).values, (A.orig_lab == lab).values
        yt = y[tr]
        co[te] = yt.mean()
        po[te] = (LogisticRegression().fit(A.loc[tr, ["orig_logp"]].values, yt).predict_proba(A.loc[te, ["orig_logp"]].values)[:, 1]
                  if 0 < yt.mean() < 1 else yt.mean())
    A = A.assign(p_orig=po, const=co)
    preds = ["twin", "naive", "p_orig", "const"] + (["twin2"] if "twin2" in A.columns else []) + (["design"] if "design" in A.columns else [])
    def stats(d):
        return {k: (_auc(d.replicated, d[k]), float(np.mean((d[k] - d.replicated.astype(float)) ** 2))) for k in preds}
    real = stats(A)
    groups = {g: d for g, d in A.groupby(cluster)}; keys = list(groups); boots = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys), replace=True)
        boots.append(stats(pd_concat([groups[keys[i]] for i in pick])))
    return A, real, boots


def pd_concat(x):
    import pandas as pd
    return pd.concat(x)


def cmd_test(a):
    import pandas as pd
    rng = np.random.default_rng(0)
    D = pd.read_csv(OUT / "rs_rows.csv")
    claims, A = build_attempts(D, rng)
    A.to_csv(OUT / "attempts.csv", index=False)
    claims.drop(columns=["curves", "lik", "power", "post"]).to_csv(OUT / "claims.csv", index=False)
    lines = [f"# REPLICATION test: {D.session.nunique()} repeated-site sessions, {D.lab.nunique()} labs, "
             f"{D.subject.nunique()} mice; {len(claims)} lab-level claims, {int(claims.p.lt(0.05).sum())} significant\n"]
    results = {}
    for name, sub in (("PRIMARY: attempts to replicate significant claims", A[A.orig_sig]),
                      ("SECONDARY: all attempts (does the second lab find it?)", A)):
        if len(sub) < 20 or sub.replicated.nunique() < 2:
            lines.append(f"{name}: too few attempts ({len(sub)}) or no variation in outcome; not tested\n"); continue
        sub2, real, boots = _scores(sub, rng)
        lines.append(f"## {name}: {len(sub)} attempts, observed replication rate {sub.replicated.mean():.2f}")
        for k in ("twin", "naive", "p_orig", "const"):
            au = [b[k][0] for b in boots if np.isfinite(b[k][0])]
            lines.append(f"  {k:7s} AUC {real[k][0]:.3f} [{np.percentile(au, 2.5):.3f}, {np.percentile(au, 97.5):.3f}]   "
                         f"Brier {real[k][1]:.3f}   mean predicted {sub2[k].mean():.2f}")
        d_naive = [b['naive'][1] - b['twin'][1] for b in boots]
        d_porig = [b['p_orig'][1] - b['twin'][1] for b in boots]
        au_tw = [b['twin'][0] for b in boots if np.isfinite(b['twin'][0])]
        lines.append(f"  Brier improvement of TWIN over naive power: {real['naive'][1] - real['twin'][1]:+.3f} "
                     f"[{np.percentile(d_naive, 2.5):+.3f}, {np.percentile(d_naive, 97.5):+.3f}]")
        lines.append(f"  Brier improvement of TWIN over original p: {real['p_orig'][1] - real['twin'][1]:+.3f} "
                     f"[{np.percentile(d_porig, 2.5):+.3f}, {np.percentile(d_porig, 97.5):+.3f}]")
        results[name] = (np.percentile(au_tw, 2.5) > 0.5, np.percentile(d_naive, 2.5) > 0, np.percentile(d_porig, 2.5) > 0)
        lines.append("  TWIN AUC by variable: " + ", ".join(
            f"{v} {_auc(sub2[sub2.variable == v].replicated, sub2[sub2.variable == v].twin):.2f} (n={int((sub2.variable == v).sum())})"
            for v in sorted(sub2.variable.unique())))
        # calibration
        bins = pd.cut(sub2.twin, [0, .2, .4, .6, .8, 1.0001], include_lowest=True)
        cal = sub2.groupby(bins, observed=True).agg(predicted=("twin", "mean"), observed=("replicated", "mean"), n=("twin", "size"))
        lines.append("  TWIN calibration (predicted vs observed):")
        for idx, rw in cal.iterrows():
            lines.append(f"    {str(idx):16s} predicted {rw.predicted:.2f}  observed {rw.observed:.2f}  (n={int(rw.n)})")
        # lab vs statistics (descriptive, H3)
        try:
            import statsmodels.formula.api as smf
            sub3 = sub2.assign(rep=sub2.replicated.astype(int), lt=np.log(np.clip(sub2.twin, .01, .99) / (1 - np.clip(sub2.twin, .01, .99))))
            m0 = smf.logit("rep ~ lt", sub3).fit(disp=0)
            m1 = smf.logit("rep ~ lt + C(rep_lab) + C(orig_lab)", sub3).fit(disp=0, method="bfgs", maxiter=500)
            from scipy.stats import chi2
            lrs = 2 * (m1.llf - m0.llf); dfd = m1.df_model - m0.df_model
            lines.append(f"  H3 lab identity beyond TWIN: likelihood-ratio {lrs:.1f} on {dfd:.0f} df, p = {chi2.sf(lrs, dfd):.3f}; "
                         f"pseudo-R2 TWIN alone {m0.prsquared:.3f}, + labs {m1.prsquared:.3f}")
        except Exception as e:
            lines.append(f"  H3 not computed ({e})")
        lines.append("")
    if "PRIMARY: attempts to replicate significant claims" in results:
        h1, h2a, h2b = results["PRIMARY: attempts to replicate significant claims"]
        lines += [f"H1 (TWIN discriminates replication, AUC CI > 0.5): {'PASS' if h1 else 'FAIL'}",
                  f"H2 (TWIN beats naive power AND original p in Brier): {'PASS' if (h2a and h2b) else 'FAIL'} "
                  f"(vs naive {'yes' if h2a else 'no'}, vs original p {'yes' if h2b else 'no'})"]
    text = "\n".join(lines)
    (OUT / "REPLICATION_RESULT.md").write_text(text + "\n")
    print(text)


def _within_var(sub, rng, B=1000):
    """TWIN AUC within each variable, with a cluster bootstrap over original labs."""
    out = {}
    for v in sorted(sub.variable.unique()):
        d = sub[sub.variable == v]
        if d.replicated.nunique() < 2 or len(d) < 20:
            out[v] = (np.nan, np.nan, np.nan, len(d)); continue
        labs = d.orig_lab.unique(); bs = []
        for _ in range(B):
            pick = rng.choice(labs, len(labs), replace=True)
            bs.append(_auc(*(lambda x: (x.replicated, x.twin))(pd_concat([d[d.orig_lab == l] for l in pick]))))
        bs = [b for b in bs if np.isfinite(b)]
        out[v] = (_auc(d.replicated, d.twin), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(d))
    return out


def _add_design(A, C):
    import pandas as pd
    C = C.assign(rep_n=C.n_sess, rep_N=C.Ns.apply(lambda x: float(np.mean(json.loads(x)))),
                 rep_T=C.Ts.apply(lambda x: float(np.mean(json.loads(x)))))
    return A.merge(C.rename(columns={"lab": "rep_lab"})[["rep_lab", "region", "variable", "rep_n", "rep_N", "rep_T"]],
                   on=["rep_lab", "region", "variable"])


def cmd_confirm(a):
    """Pre-registered CONFIRMATION on the Brain-Wide Map sessions that are NOT repeated-site recordings (independent of
    the development data), using the atlas rows (part6/atlas.py run)."""
    import pandas as pd
    rng = np.random.default_rng(1)
    D = pd.read_csv(HERE / "results_atlas" / "atlas_rows.csv")
    regs = D.groupby("session").region.apply(set)
    rs_sessions = {sid for sid, rr in regs.items() if is_rs(rr)}
    D = D[~D.session.isin(rs_sessions)]
    # a region counts only if at least 2 labs recorded it (otherwise there is nothing to replicate)
    nl = D.groupby(["region", "variable"]).lab.nunique()
    keep = nl[nl >= 2].index
    D = D.set_index(["region", "variable"]).loc[lambda x: x.index.isin(keep)].reset_index()
    claims, A = build_attempts(D, rng)
    OUT.mkdir(exist_ok=True)
    A.to_csv(OUT / "confirm_attempts.csv", index=False)
    claims.drop(columns=["curves", "lik", "power", "post"]).to_csv(OUT / "confirm_claims.csv", index=False)
    lines = [f"# REPLICATION CONFIRMATION (Brain-Wide Map, repeated-site sessions excluded): {D.session.nunique()} sessions, "
             f"{D.lab.nunique()} labs, {D.subject.nunique()} mice, {D.region.nunique()} regions; {len(claims)} claims, "
             f"{int(claims.p.lt(0.05).sum())} significant\n"]
    verdict = {}
    for name, sub in (("PRIMARY", A[A.orig_sig]), ("SECONDARY", A)):
        if len(sub) < 20 or sub.replicated.nunique() < 2:
            lines.append(f"{name}: too few attempts"); continue
        sub2, real, boots = _scores(sub, rng)
        lines.append(f"## {name}: {len(sub)} attempts, replication rate {sub.replicated.mean():.2f}")
        for k in ("twin", "naive", "p_orig", "const"):
            au = [b[k][0] for b in boots if np.isfinite(b[k][0])]
            lines.append(f"  {k:7s} AUC {real[k][0]:.3f} [{np.percentile(au, 2.5):.3f}, {np.percentile(au, 97.5):.3f}]   Brier {real[k][1]:.3f}")
        dn = [b['naive'][1] - b['twin'][1] for b in boots]; dp = [b['p_orig'][1] - b['twin'][1] for b in boots]
        lines.append(f"  Brier gain over naive {real['naive'][1] - real['twin'][1]:+.3f} [{np.percentile(dn, 2.5):+.3f}, {np.percentile(dn, 97.5):+.3f}]; "
                     f"over original p {real['p_orig'][1] - real['twin'][1]:+.3f} [{np.percentile(dp, 2.5):+.3f}, {np.percentile(dp, 97.5):+.3f}]")
        # C4: a simple DESIGN model (variable type + replicating lab's sessions, neurons, trials) trained on the
        # repeated-site attempts and frozen, transferred to these data. Is TWIN better than what simple design predicts?
        rs_att = OUT / "attempts.csv"
        if rs_att.exists():
            from sklearn.linear_model import LogisticRegression
            R = pd.read_csv(rs_att)
            if "rep_n" not in R.columns:
                R = _add_design(R, pd.read_csv(OUT / "claims.csv"))
            R = R[R.orig_sig] if name == "PRIMARY" else R
            levels = sorted(R.variable.unique())

            def design_X(d):                              # unseen variable types get the reference (all-zero) coding
                cols = [np.log(d.rep_N.values), np.log(d.rep_T.values), d.rep_n.values.astype(float)]
                cols += [(d.variable.values == lv).astype(float) for lv in levels[1:]]
                return np.column_stack(cols)
            m = LogisticRegression(C=1e4, max_iter=5000).fit(design_X(R), R.replicated.astype(int))
            sub2 = sub2.assign(design=m.predict_proba(design_X(sub2))[:, 1])
            bd = []
            labs_ = sub2.orig_lab.unique()
            for _ in range(1000):
                pick = rng.choice(labs_, len(labs_), replace=True)
                d = pd_concat([sub2[sub2.orig_lab == l] for l in pick])
                bd.append(np.mean((d.design - d.replicated) ** 2) - np.mean((d.twin - d.replicated) ** 2))
            gain = float(np.mean((sub2.design - sub2.replicated) ** 2) - np.mean((sub2.twin - sub2.replicated) ** 2))
            lines.append(f"  design model (trained on repeated site): AUC {_auc(sub2.replicated, sub2.design):.3f}  Brier "
                         f"{np.mean((sub2.design - sub2.replicated) ** 2):.3f}; TWIN Brier gain over it {gain:+.3f} "
                         f"[{np.percentile(bd, 2.5):+.3f}, {np.percentile(bd, 97.5):+.3f}]")
            verdict.setdefault(name, {})["c4"] = np.percentile(bd, 2.5) > 0
            verdict[name]["c4_tie"] = np.percentile(bd, 97.5) >= 0
        wv = _within_var(sub2, rng)
        lines.append("  within-variable TWIN AUC: " + "; ".join(f"{v} {x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}] n={x[3]}" for v, x in wv.items()))
        au_tw = [b['twin'][0] for b in boots if np.isfinite(b['twin'][0])]
        verdict[name] = verdict.get(name, {}) | dict(h1=np.percentile(au_tw, 2.5) > 0.5, h2=np.percentile(dn, 2.5) > 0 and np.percentile(dp, 2.5) > 0,
                             h4=sum(1 for x in wv.values() if np.isfinite(x[1]) and x[1] > 0.5), nvar=sum(1 for x in wv.values() if np.isfinite(x[1])))
        bins = pd.cut(sub2.twin, [0, .2, .4, .6, .8, 1.0001], include_lowest=True)
        cal = sub2.groupby(bins, observed=True).agg(pred=("twin", "mean"), obs=("replicated", "mean"), n=("twin", "size"))
        lines.append("  calibration: " + "; ".join(f"{r.pred:.2f}->{r.obs:.2f} (n={int(r.n)})" for _, r in cal.iterrows()))
        lines.append("")
    if "PRIMARY" in verdict:
        v = verdict["PRIMARY"]
        lines += [f"C1 (TWIN AUC CI > 0.5): {'PASS' if v['h1'] else 'FAIL'}",
                  f"C2 (TWIN beats naive AND original p in Brier): {'PASS' if v['h2'] else 'FAIL'}",
                  f"C3 (within-variable AUC CI > 0.5 in at least half of the testable variables, SECONDARY set): "
                  f"{'PASS' if verdict.get('SECONDARY', {}).get('h4', 0) * 2 >= max(verdict.get('SECONDARY', {}).get('nvar', 1), 1) else 'FAIL'} "
                  f"({verdict.get('SECONDARY', {}).get('h4', 0)} of {verdict.get('SECONDARY', {}).get('nvar', 0)})"]
        if "c4" in v:
            lines.append(f"C4 (TWIN vs the frozen design model, PRIMARY): "
                         f"{'TWIN BETTER' if v['c4'] else ('NO DIFFERENCE' if v['c4_tie'] else 'DESIGN MODEL BETTER')}")
    text = "\n".join(lines)
    (OUT / "CONFIRMATION_RESULT.md").write_text(text + "\n")
    print(text)


def _design_predict(sub, name):
    """Frozen design rule trained on the repeated-site attempts. A variable it has never seen gets the AVERAGE of its
    predictions over all the variable codings it knows (its fairest guess for a new kind of claim)."""
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    R = pd.read_csv(OUT / "attempts.csv")
    if "rep_n" not in R.columns:
        R = _add_design(R, pd.read_csv(OUT / "claims.csv"))
    R = R[R.orig_sig] if name == "PRIMARY" else R
    levels = sorted(R.variable.unique())

    def X(d, var_override=None):
        var = d.variable.values if var_override is None else np.array([var_override] * len(d))
        cols = [np.log(d.rep_N.values), np.log(d.rep_T.values), d.rep_n.values.astype(float)]
        cols += [(var == lv).astype(float) for lv in levels[1:]]
        return np.column_stack(cols)
    m = LogisticRegression(C=1e4, max_iter=5000).fit(X(R), R.replicated.astype(int))
    known = sub.variable.isin(levels).values
    pred = np.zeros(len(sub))
    if known.any():
        pred[known] = m.predict_proba(X(sub[known]))[:, 1]
    if (~known).any():
        pred[~known] = np.mean([m.predict_proba(X(sub[~known], lv))[:, 1] for lv in levels], axis=0)
    return pred


def cmd_confirm2(a):
    """TWIN v2 vs the frozen design rule. Default: the NEW variables (pre-registered confirmation; neither model was built
    with them). --old: the original four variables on the Brain-Wide Map (EXPLORATORY: those results were already seen)."""
    import pandas as pd
    rng = np.random.default_rng(2)
    f = HERE / "results_atlas" / ("atlas_rows.csv" if a.old else "atlas_rows_new.csv")
    D = pd.read_csv(f)
    regs = pd.read_csv(HERE / "results_atlas" / "atlas_rows.csv", usecols=["session", "region"]).groupby("session").region.apply(set)
    rs_sessions = {sid for sid, rr in regs.items() if is_rs(rr)}
    D = D[~D.session.isin(rs_sessions)]
    nl = D.groupby(["region", "variable"]).lab.nunique(); keep = nl[nl >= 2].index
    D = D.set_index(["region", "variable"]).loc[lambda x: x.index.isin(keep)].reset_index()
    claims, A = build_attempts(D, rng)
    tag = "old_variables_EXPLORATORY" if a.old else "new_variables"
    A.to_csv(OUT / f"v2_attempts_{tag}.csv", index=False)
    lines = [f"# TWIN v2 vs design rule ({'EXPLORATORY: original variables, already seen' if a.old else 'CONFIRMATORY: new variables'}): "
             f"{D.session.nunique()} sessions, {D.lab.nunique()} labs, {D.subject.nunique()} mice, {D.region.nunique()} regions, "
             f"variables {sorted(D.variable.unique())}; {len(claims)} claims, {int(claims.p.lt(0.05).sum())} significant\n"]
    verdict = {}
    for name, sub in (("PRIMARY", A[A.orig_sig]), ("SECONDARY", A)):
        if len(sub) < 20 or sub.replicated.nunique() < 2:
            lines.append(f"{name}: too few attempts"); continue
        sub = sub.assign(design=_design_predict(sub, name))
        sub2, real, boots = _scores(sub, rng)
        lines.append(f"## {name}: {len(sub)} attempts, replication rate {sub.replicated.mean():.2f}")
        for k in ("twin2", "design", "twin", "naive", "p_orig", "const"):
            au = [b[k][0] for b in boots if np.isfinite(b[k][0])]
            lines.append(f"  {k:7s} AUC {real[k][0]:.3f} [{np.percentile(au, 2.5):.3f}, {np.percentile(au, 97.5):.3f}]   Brier {real[k][1]:.3f}")
        def gain(other):
            g = [b[other][1] - b['twin2'][1] for b in boots]
            return real[other][1] - real['twin2'][1], np.percentile(g, 2.5), np.percentile(g, 97.5)
        for other in ("design", "twin", "naive", "p_orig"):
            gg = gain(other)
            lines.append(f"  TWIN v2 Brier gain over {other:7s}: {gg[0]:+.3f} [{gg[1]:+.3f}, {gg[2]:+.3f}]")
        au2 = [b['twin2'][0] for b in boots if np.isfinite(b['twin2'][0])]
        sub_w = sub2.assign(twin=sub2.twin2)
        wv = _within_var(sub_w, rng)
        lines.append("  within-variable TWIN v2 AUC: " + "; ".join(f"{v} {x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}] n={x[3]}" for v, x in wv.items()))
        bins = pd.cut(sub2.twin2, [0, .2, .4, .6, .8, 1.0001], include_lowest=True)
        cal = sub2.groupby(bins, observed=True).agg(pred=("twin2", "mean"), obs=("replicated", "mean"), n=("twin2", "size"))
        lines.append("  calibration v2: " + "; ".join(f"{r.pred:.2f}->{r.obs:.2f} (n={int(r.n)})" for _, r in cal.iterrows()))
        verdict[name] = dict(n1=gain("design")[1] > 0, n2=np.percentile(au2, 2.5) > 0.5 and gain("naive")[1] > 0 and gain("p_orig")[1] > 0,
                             tie=gain("design")[2] >= 0)
        lines.append("")
    if "PRIMARY" in verdict and not a.old:
        v = verdict["PRIMARY"]
        lines += [f"N1 (TWIN v2 beats the frozen design rule in Brier, PRIMARY): "
                  f"{'PASS' if v['n1'] else ('FAIL (no difference)' if v['tie'] else 'FAIL (design rule better)')}",
                  f"N2 (TWIN v2 AUC CI > 0.5 and beats naive power and original p): {'PASS' if v['n2'] else 'FAIL'}"]
    text = "\n".join(lines)
    (OUT / f"V2_RESULT_{tag}.md").write_text(text + "\n")
    print(text)


def cmd_confirm3(a):
    """CROSS-DATASET test: IBL Brain-Wide Map labs <-> Steinmetz et al. 2019 (different era, rig and mice).
    Frozen TWIN v2 and the frozen design rule; only attempts that cross datasets are scored."""
    import pandas as pd
    rng = np.random.default_rng(3)
    base = HERE / "results_atlas"
    I = pd.read_csv(base / "atlas_rows.csv")
    regs = I.groupby("session").region.apply(set)
    rs_sessions = {sid for sid, rr in regs.items() if is_rs(rr)}
    parts = [I]
    if (base / "atlas_rows_new.csv").exists():
        N_ = pd.read_csv(base / "atlas_rows_new.csv"); parts.append(N_[N_.variable == "prev_choice"])
    I = pd.concat(parts)
    I = I[~I.session.isin(rs_sessions)]
    S = pd.read_csv(base / "steinmetz_rows.csv")
    VARS = ["stimulus", "choice", "feedback", "prev_choice"]
    D = pd.concat([I, S])
    D = D[D.variable.isin(VARS) & D.region.isin(set(S.region))]
    claims, A = build_attempts(D, rng)
    A = A[(A.orig_lab == "steinmetz2019") | (A.rep_lab == "steinmetz2019")]
    A_all = A
    A = A[(A.orig_lab != "cortexlab") & (A.rep_lab != "cortexlab")]     # Steinmetz data came from cortexlab: excluded
    A.to_csv(OUT / "cross_attempts.csv", index=False)
    lines = [f"# CROSS-DATASET test (IBL <-> Steinmetz 2019): {S.session.nunique()} Steinmetz sessions, {S.subject.nunique()} mice; "
             f"{D[D.lab != 'steinmetz2019'].session.nunique()} IBL sessions; {D.region.nunique()} shared regions; "
             f"{len(A)} cross-dataset attempts (IBL cortexlab excluded; {len(A_all)} with it)\n"]
    verdict = {}
    for name, sub in (("PRIMARY", A[A.orig_sig]), ("SECONDARY", A)):
        if len(sub) < 20 or sub.replicated.nunique() < 2:
            lines.append(f"{name}: too few attempts"); continue
        sub = sub.assign(design=_design_predict(sub, name))
        sub2, real, boots = _scores(sub, rng)
        lines.append(f"## {name}: {len(sub)} attempts, replication rate {sub.replicated.mean():.2f}")
        for k in ("twin2", "design", "twin", "naive", "p_orig", "const"):
            au = [b[k][0] for b in boots if np.isfinite(b[k][0])]
            lines.append(f"  {k:7s} AUC {real[k][0]:.3f} [{np.percentile(au, 2.5):.3f}, {np.percentile(au, 97.5):.3f}]   Brier {real[k][1]:.3f}")
        def gain(other):
            g = [b[other][1] - b['twin2'][1] for b in boots]
            return real[other][1] - real['twin2'][1], np.percentile(g, 2.5), np.percentile(g, 97.5)
        for other in ("design", "naive", "p_orig"):
            gg = gain(other); lines.append(f"  TWIN v2 Brier gain over {other:7s}: {gg[0]:+.3f} [{gg[1]:+.3f}, {gg[2]:+.3f}]")
        for direction, d in (("IBL claim -> Steinmetz replication", sub2[sub2.rep_lab == "steinmetz2019"]),
                             ("Steinmetz claim -> IBL lab replication", sub2[sub2.orig_lab == "steinmetz2019"])):
            if len(d) > 10 and d.replicated.nunique() == 2:
                lines.append(f"  {direction}: n={len(d)}, rate {d.replicated.mean():.2f}, TWIN v2 AUC {_auc(d.replicated, d.twin2):.3f}, "
                             f"design AUC {_auc(d.replicated, d.design):.3f}")
        au2 = [b['twin2'][0] for b in boots if np.isfinite(b['twin2'][0])]
        verdict[name] = dict(x1=np.percentile(au2, 2.5) > 0.5, x2=gain("naive")[1] > 0 and gain("p_orig")[1] > 0,
                             x3=gain("design")[1] > 0, x3tie=gain("design")[2] >= 0)
        lines.append("")
    if "PRIMARY" in verdict:
        v = verdict["PRIMARY"]
        lines += [f"X1 (TWIN v2 AUC CI > 0.5 across datasets): {'PASS' if v['x1'] else 'FAIL'}",
                  f"X2 (TWIN v2 beats naive power and original p): {'PASS' if v['x2'] else 'FAIL'}",
                  f"X3 (TWIN v2 beats the frozen design rule): {'PASS' if v['x3'] else ('no difference' if v['x3tie'] else 'FAIL: design rule better')}"]
    text = "\n".join(lines)
    (OUT / "CROSS_DATASET_RESULT.md").write_text(text + "\n")
    print(text)


EXTERNAL = {"steinmetz2019": "steinmetz_rows.csv", "allen_vbn": "allen_rows.csv"}


def cmd_confirm4(a):
    """POOLED CROSS-DATASET test (pre-registered: claude/preregistration-pooled-cross-dataset.md). IBL <-> Steinmetz,
    IBL <-> Allen and Steinmetz <-> Allen. Frozen TWIN v2 and design rule. Primary bootstrap clusters = brain regions."""
    import pandas as pd
    rng = np.random.default_rng(4)
    base = HERE / "results_atlas"
    I = pd.read_csv(base / "atlas_rows.csv")
    regs = I.groupby("session").region.apply(set)
    rs_sessions = {sid for sid, rr in regs.items() if is_rs(rr)}
    parts = [I]
    if (base / "atlas_rows_new.csv").exists():
        N_ = pd.read_csv(base / "atlas_rows_new.csv"); parts.append(N_[N_.variable == "prev_choice"])
    I = pd.concat(parts); I = I[~I.session.isin(rs_sessions)]
    ext = {lab: pd.read_csv(base / f) for lab, f in EXTERNAL.items() if (base / f).exists()}
    if len(ext) < 2:
        sys.exit(f"needs both external datasets; found {list(ext)}")
    VARS = ["stimulus", "choice", "feedback", "prev_choice"]
    ext_regions = set().union(*[set(d.region) for d in ext.values()])
    D = pd.concat([I] + list(ext.values()))
    D = D[D.variable.isin(VARS) & D.region.isin(ext_regions)]
    claims, A = build_attempts(D, rng)
    X = set(ext)
    A = A[A.orig_lab.isin(X) | A.rep_lab.isin(X)]
    A_all = A
    A = A[(A.orig_lab != "cortexlab") & (A.rep_lab != "cortexlab")]
    A.to_csv(OUT / "pooled_attempts.csv", index=False)
    lines = [f"# POOLED CROSS-DATASET test: external datasets {sorted(X)}; "
             + "; ".join(f"{k}: {d.session.nunique()} sessions, {d.subject.nunique()} mice" for k, d in ext.items())
             + f"; {D.region.nunique()} shared regions; {len(A)} cross-dataset attempts (IBL cortexlab excluded; {len(A_all)} with it)\n"]
    verdict = {}
    for name, sub in (("PRIMARY", A[A.orig_sig]), ("SECONDARY", A)):
        if len(sub) < 20 or sub.replicated.nunique() < 2:
            lines.append(f"{name}: too few attempts"); continue
        sub = sub.assign(design=_design_predict(sub, name))
        for clabel, ccol in (("regions (pre-registered primary)", "region"), ("original labs (as in the Steinmetz test)", "orig_lab")):
            sub2, real, boots = _scores(sub, rng, cluster=ccol)
            lines.append(f"## {name}: {len(sub)} attempts, replication rate {sub.replicated.mean():.2f}; bootstrap over {clabel}")
            for k in ("twin2", "design", "twin", "naive", "p_orig", "const"):
                au = [b[k][0] for b in boots if np.isfinite(b[k][0])]
                lines.append(f"  {k:7s} AUC {real[k][0]:.3f} [{np.percentile(au, 2.5):.3f}, {np.percentile(au, 97.5):.3f}]   Brier {real[k][1]:.3f}")
            def gain(other):
                g = [b[other][1] - b['twin2'][1] for b in boots]
                return real[other][1] - real['twin2'][1], np.percentile(g, 2.5), np.percentile(g, 97.5)
            for other in ("design", "naive", "p_orig"):
                gg = gain(other); lines.append(f"  TWIN v2 Brier gain over {other:7s}: {gg[0]:+.3f} [{gg[1]:+.3f}, {gg[2]:+.3f}]")
            if ccol == "region":
                au2 = [b['twin2'][0] for b in boots if np.isfinite(b['twin2'][0])]
                verdict[name] = dict(p1=np.percentile(au2, 2.5) > 0.5, p2=gain("naive")[1] > 0 and gain("p_orig")[1] > 0,
                                     p3=gain("design")[1] > 0, p3tie=gain("design")[2] >= 0)
                for lab in sorted(X):
                    for direction, d in ((f"IBL/other claim -> {lab}", sub2[sub2.rep_lab == lab]), (f"{lab} claim -> others", sub2[sub2.orig_lab == lab])):
                        if len(d) > 10 and d.replicated.nunique() == 2:
                            lines.append(f"  {direction}: n={len(d)}, rate {d.replicated.mean():.2f}, TWIN v2 AUC {_auc(d.replicated, d.twin2):.3f}, "
                                         f"design AUC {_auc(d.replicated, d.design):.3f}")
            lines.append("")
    if "PRIMARY" in verdict:
        v = verdict["PRIMARY"]
        lines += [f"P1 (TWIN v2 AUC CI > 0.5): {'PASS' if v['p1'] else 'FAIL'}",
                  f"P2 (TWIN v2 beats naive power AND original p): {'PASS' if v['p2'] else 'FAIL'}",
                  f"P3 (TWIN v2 beats the frozen design rule): {'PASS' if v['p3'] else ('no difference' if v['p3tie'] else 'FAIL: design rule better')}"]
    text = "\n".join(lines)
    (OUT / "POOLED_CROSS_DATASET_RESULT.md").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
    r.add_argument("--gatekeeper"); r.add_argument("--min-units", type=int, default=10); r.add_argument("--max-units", type=int, default=60)
    r.add_argument("--null", type=int, default=100); r.add_argument("--plant", type=int, default=30)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    sub.add_parser("test"); sub.add_parser("confirm")
    c2 = sub.add_parser("confirm2"); c2.add_argument("--old", action="store_true")
    sub.add_parser("confirm3"); sub.add_parser("confirm4")
    c5 = sub.add_parser("confirm5"); c5.add_argument("--exploratory", action="store_true")
    a = ap.parse_args()
    dict(run=cmd_run, test=cmd_test, confirm=cmd_confirm, confirm2=cmd_confirm2, confirm3=cmd_confirm3, confirm4=cmd_confirm4,
         confirm5=cmd_confirm5)[a.cmd](a)


if __name__ == "__main__":
    main()
