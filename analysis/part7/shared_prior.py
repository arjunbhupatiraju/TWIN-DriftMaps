"""SHARED PRIOR (pre-registered 2026-11-10): is the block prior carried by one shared, slow, brain-wide state?

Uses the same IBL sessions, pre-stimulus window, trial filters, movement features and removal as step 1
(part6/prior_movement.py): 11 movement features (incl. pupil) are regressed out of the neurons and out of every
target (real prior and pseudo-session priors), so the pseudo-session null stays exact.

Per session (>= 3 regions with >= 10 units):
  region PCs   top 3 PCs of each region's movement-removed, z-scored activity
  S            the first PC of all regions' PCs together: the shared state. Unsupervised: the prior is never used.
T1  |corr(S, prior)| vs the same with 1,000 pseudo-session priors. Pooled: mean over sessions, null = mean of the
    j-th pseudo statistic over sessions.                                         PASS: one-sided p < 0.05
T2  How much of each region's prior information is S? Prior decoding (contiguous CV ridge, as PRIOR MAP) minus the
    pseudo-null mean, before and after projecting S out of the region; control = projecting out a random direction
    of the same shared space (10 per session).  share = 1 - sum(after) / sum(before) over all region-sessions.
                                            PASS: session-bootstrap lower bound of share(S) - share(random) > 0
                                            "mostly": additionally the lower bound of share(S) > 0.5
T3  T1 restricted to sessions where pupil is among the removed features.        PASS: p < 0.05
T4  Does S predict choice on low-contrast trials (|c| <= 0.0625, >= 40 trials) beyond the Bayes prior, signed
    contrast, choice lags 1-2 and rewarded-choice lags 1-2? S is signed by its correlation with the prior; the null
    signs it by its correlation with each pseudo prior. Pooled like T1.          PASS: one-sided p < 0.05
Split (fixed now): labs sorted, permuted with seed 20261110; first half = EXPLORE, rest = CONFIRM.
Confirmatory verdicts come only from CONFIRM.

    cd ~/Downloads/MODEL && python3 part7/shared_prior.py mock           (synthetic check, no data)
    python3 part7/shared_prior.py run --limit 3 --workers 2               (quick check)
    python3 part7/shared_prior.py run --workers 3                         (all sessions; resumable)
    python3 part7/shared_prior.py test --split explore
    python3 part7/shared_prior.py test --split confirm                    (once)
"""
from __future__ import annotations

import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results_shared_prior"
SEED_SPLIT = 20261110
N_PC, N_RAND, N_NULL, N_DEC = 3, 10, 1000, 200


# ================================================================ helpers (pure numpy)
def zs(X):
    X = np.asarray(X, float)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def remove(A, M):
    return A - M @ np.linalg.lstsq(M, A, rcond=None)[0]


def corr(a, B):
    a = (a - a.mean()) / (a.std() + 1e-12); B = zs(np.atleast_2d(B.T).T)
    return (B * a[:, None]).mean(0)


def shared_state(regions):
    """regions: list of trials x units arrays (movement removed). Returns S (z-scored), the stacked region PCs Z, and
    descriptive numbers: variance share of the first PC, mean |corr(S, region PC1)|."""
    pcs = []
    for X in regions:
        U, s, _ = np.linalg.svd(zs(X), full_matrices=False)
        pcs.append(zs(U[:, :N_PC] * s[:N_PC]))
    Z = np.column_stack(pcs)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    S = zs(U[:, 0] * s[0])
    shared = float(np.mean([abs(corr(S, p[:, :1])[0]) for p in pcs]))
    return S, Z, float(s[0] ** 2 / (s ** 2).sum()), shared


def project_out(X, S):
    S = np.atleast_2d(S.T).T
    return X - S @ np.linalg.lstsq(S, X, rcond=None)[0]


def partial_r(x, y, C):
    C = np.column_stack([np.ones(len(x)), C])
    rx = x - C @ np.linalg.lstsq(C, x, rcond=None)[0]; ry = y - C @ np.linalg.lstsq(C, y, rcond=None)[0]
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


def slow_share(x, T):
    sys.path.insert(0, str(ROOT / "part6"))
    import decompose as DC
    D = np.column_stack([np.ones(T), DC.drift(T)])
    return float(1 - remove(x[:, None], D).var() / x.var())


def analyze(regions, names, y, Zp, cv_decode, choice=None, contrast=None, rewarded=None, rng=None):
    """Core analysis for one session. y: prior (movement removed); Zp: T x n pseudo priors (movement removed,
    n >= N_NULL). Returns (session_row, region_rows, nulls)."""
    rng = rng or np.random.default_rng(0)
    T = len(y)
    S, Z, var1, shared = shared_state(regions)
    # T1
    r_obs = abs(corr(S, y[:, None])[0]); r_null = np.abs(corr(S, Zp[:, :N_NULL]))
    # T2
    Y = np.column_stack([y, Zp[:, :N_DEC]])
    W = [zs(Z @ rng.normal(size=Z.shape[1])) for _ in range(N_RAND)]
    reg_rows = []
    for k, (nm, X) in enumerate(zip(names, regions)):
        full = cv_decode(X, Y); f = full[0] - full[1:].mean()
        mS = cv_decode(project_out(X, S), Y); fs = mS[0] - mS[1:].mean()
        fr = []
        for w in W:
            mR = cv_decode(project_out(X, w), Y); fr.append(mR[0] - mR[1:].mean())
        reg_rows.append(dict(region=nm, N=X.shape[1], info_full=float(f), info_minus_S=float(fs),
                             info_minus_rand=float(np.mean(fr))))
    row = dict(T=T, n_regions=len(regions), var_S=var1, sharedness=shared, slow_share_S=slow_share(S, T),
               r_obs=float(r_obs), r_null_mean=float(r_null.mean()))
    nulls = dict(t1=r_null)
    # T4
    if choice is not None:
        lo = np.abs(contrast) <= 0.0625
        if lo.sum() >= 40:
            ch = np.asarray(choice, float)
            sgn = np.sign(corr(y, ch[:, None])[0]) or 1.0
            ch = ch * sgn
            rc = ch * rewarded
            lag = lambda a, k: np.r_[np.zeros(k), a[:-k]]
            C = np.column_stack([contrast, y, lag(ch, 1), lag(ch, 2), lag(rc, 1), lag(rc, 2)])[lo]
            s_obs = S * (np.sign(corr(S, y[:, None])[0]) or 1.0)
            t4 = partial_r(s_obs[lo], ch[lo], C)
            signs = np.sign(corr(S, Zp[:, :N_NULL])); signs[signs == 0] = 1
            t4n = np.array([partial_r((S * g)[lo], ch[lo], C) for g in signs])
            row.update(t4=t4, t4_null_mean=float(t4n.mean()), n_lowc=int(lo.sum())); nulls["t4"] = t4n
    return row, reg_rows, nulls


# ================================================================ one real session
def _session(path, gk_root, burn, seed, out_dir):
    sys.path.insert(0, str(ROOT / "part6")); sys.path.insert(0, str(ROOT / "twin"))
    import prior_map as PM
    import prior_movement as MV
    PM._gk(gk_root)
    from gatekeeper.mv_data import MVSession
    from twin.task import BlockTask
    rng = np.random.default_rng(seed); task = BlockTask()
    s = MVSession(Path(path))
    c = np.asarray(s.contrast, float); Tf = len(c)
    Mf, feats = MV.movement_matrix(s.sid, Tf)
    if Mf is None:
        return None
    side = np.where(c != 0, (c > 0).astype(float), np.nan)
    prior = task.bayes_prior(side)
    choice_raw = np.asarray(s.choice, float)
    v = (choice_raw >= 0) & s.valid["stimOn"]; v[:burn] = False
    T = int(v.sum())
    if T < 150:
        return None
    M = Mf[v]
    bank = np.column_stack([PM._pseudo(task, Tf, rng)[0] for _ in range(N_NULL)])[v]
    YZ = remove(np.column_stack([prior[v], bank]), M); y, Zp = YZ[:, 0], YZ[:, 1:]
    regions, names = [], []
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        X = np.asarray(X, float)[v]; X = X[:, X.std(0) > 0]
        if X.shape[1] < 10:
            continue
        if X.shape[1] > 150:
            X = X[:, rng.choice(X.shape[1], 150, replace=False)]
        regions.append(remove(np.sqrt(X), M)); names.append(r)
    if len(regions) < 3:
        return None
    ch = (2 * choice_raw[v] - 1)
    fb = getattr(s, "feedback", None)
    if fb is not None and len(np.asarray(fb)) == Tf:
        rew = (np.asarray(fb, float)[v] > 0).astype(float)
    else:
        sd = side[v]; rew = np.where(np.isfinite(sd), ((ch > 0) == (sd > 0)).astype(float), 0.0)
    row, regs, nulls = analyze(regions, names, y, Zp, PM.cv_decode, ch, c[v], rew, rng)
    row.update(session=s.sid, subject=str(s.subject), lab=str(s.lab), feats="|".join(feats),
               pupil=int("pupil" in feats))
    for d in regs:
        d.update(session=s.sid, subject=str(s.subject), lab=str(s.lab))
    np.savez_compressed(out_dir / f"{s.sid}.npz", **nulls)
    return row, regs


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True); (OUT / "nulls").mkdir(exist_ok=True)
    fs, fr = OUT / "sessions.csv", OUT / "regions.csv"
    done = set(pd.read_csv(fs).session) if fs.exists() else set()
    feat = ROOT / "part6" / "results_movement" / "features"
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz"))
             if p.stem not in done and (feat / f"{p.stem}.npz").exists()][: a.limit or None]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.gatekeeper, 90, 9100 + i, OUT / "nulls"): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                res = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True); continue
            if res is None:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: skipped (filters)", flush=True); continue
            row, regs = res
            pd.DataFrame(regs).to_csv(fr, mode="a", header=not fr.exists(), index=False)
            pd.DataFrame([row]).to_csv(fs, mode="a", header=not fs.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {row['session']}: {row['n_regions']} regions, |r| {row['r_obs']:.3f} "
                  f"(null {row['r_null_mean']:.3f})", flush=True)


# ================================================================ test
def split_labs(labs):
    labs = sorted(set(labs)); perm = np.random.default_rng(SEED_SPLIT).permutation(len(labs))
    h = len(labs) // 2
    return {labs[i] for i in perm[:h]}, {labs[i] for i in perm[h:]}


def pooled(obs, nulls):
    obs, nulls = np.asarray(obs), np.asarray(nulls)            # sessions, sessions x n
    m, mn = obs.mean(), nulls.mean(0)
    return float(m), float(mn.mean()), float((1 + np.sum(mn >= m)) / (1 + len(mn)))


def evaluate(Sd, Rd, null_dir, label, boot=500, out=None):
    import pandas as pd
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    nl = {s: np.load(null_dir / f"{s}.npz") for s in Sd.session}
    P(f"# SHARED PRIOR, {label}: {len(Sd)} sessions, {Sd.subject.nunique()} mice, {Sd.lab.nunique()} labs, "
      f"{len(Rd)} region-sessions\n")
    P(f"shared state: variance share {Sd.var_S.median():.2f} (median), mean |corr with region PC1| "
      f"{Sd.sharedness.median():.2f}, slow share {Sd.slow_share_S.median():.2f}")
    m, mn, p = pooled(Sd.r_obs, [nl[s]["t1"] for s in Sd.session])
    v = {}
    v["T1"] = p < 0.05
    P(f"\nT1 |corr(S, prior)|: {m:.4f} vs pseudo-session null {mn:.4f}; p = {p:.4f} -> {'PASS' if v['T1'] else 'FAIL'}")
    g = {s: d for s, d in Rd.groupby("session")}; ss = list(g)
    share = lambda d, c: 1 - d[c].sum() / d.info_full.sum()
    sS, sR = share(Rd, "info_minus_S"), share(Rd, "info_minus_rand")
    rng = np.random.default_rng(0); bs = []
    for _ in range(boot):
        d = pd.concat([g[ss[i]] for i in rng.integers(0, len(ss), len(ss))])
        bs.append((share(d, "info_minus_S"), share(d, "info_minus_S") - share(d, "info_minus_rand")))
    bs = np.array(bs); lo = np.percentile(bs, 2.5, axis=0); hi = np.percentile(bs, 97.5, axis=0)
    v["T2"] = lo[1] > 0
    P(f"T2 share of prior information that is S: {sS:.3f} [{lo[0]:.3f}, {hi[0]:.3f}]; random direction {sR:.3f}; "
      f"difference lower bound {lo[1]:.3f} -> {'PASS' if v['T2'] else 'FAIL'}; 'mostly' (lower > 0.5): "
      f"{'yes' if lo[0] > 0.5 else 'no'}  (total prior information {Rd.info_full.sum():.2f})")
    # T5 (Amendment 1, registered 2026-11-11 after EXPLORE, before CONFIRM): the shared state MASKS the prior
    v["T5"] = hi[1] < 0
    P(f"T5 (amendment) masking: share(S) - share(random) = {sS - sR:.3f} [{lo[1]:.3f}, {hi[1]:.3f}]; "
      f"upper bound < 0 -> {'PASS' if v['T5'] else 'FAIL'}; prior information gained by removing S: "
      f"{-sS:.0%} [{-hi[0]:.0%}, {-lo[0]:.0%}]")
    # T6 (Amendment 2, registered 2026-11-11 before CONFIRM): hiding law. Sessions with a stronger shared slow state
    # gain more prior information when S is removed (one quantity predicts errors in both directions)
    from scipy.stats import spearmanr
    gain = Rd.assign(g=Rd.info_minus_S - Rd.info_full, gr=Rd.info_minus_rand - Rd.info_full) \
        .groupby("session")[["g", "gr"]].sum()
    D6 = Sd.set_index("session").join(gain, how="inner")
    D6["drift"] = D6.var_S * D6.slow_share_S; D6["excess"] = D6.g - D6.gr
    rho = spearmanr(D6.drift, D6.excess)[0]
    rp = np.array([spearmanr(rng.permutation(D6.drift.values), D6.excess)[0] for _ in range(2000)])
    p6 = (1 + (rp >= rho).sum()) / (1 + len(rp)); v["T6"] = p6 < 0.05
    P(f"T6 (amendment) hiding law: Spearman(shared slow drift strength, information gained beyond random) = "
      f"{rho:+.3f} across {len(D6)} sessions; permutation p = {p6:.4f} -> {'PASS' if v['T6'] else 'FAIL'}")
    Sp = Sd[Sd.pupil == 1]
    if len(Sp) >= 5:
        m3, mn3, p3 = pooled(Sp.r_obs, [nl[s]["t1"] for s in Sp.session]); v["T3"] = p3 < 0.05
        P(f"T3 T1 in {len(Sp)} sessions with pupil removed: {m3:.4f} vs {mn3:.4f}; p = {p3:.4f} -> "
          f"{'PASS' if v['T3'] else 'FAIL'}")
    else:
        P("T3: fewer than 5 sessions with pupil; not testable")
    St = Sd.dropna(subset=["t4"]) if "t4" in Sd else Sd.iloc[:0]
    if len(St) >= 5:
        m4, mn4, p4 = pooled(St.t4, [nl[s]["t4"] for s in St.session]); v["T4"] = p4 < 0.05
        P(f"T4 S predicts low-contrast choice beyond prior and history ({len(St)} sessions): partial r {m4:+.4f} vs "
          f"{mn4:+.4f}; p = {p4:.4f} -> {'PASS' if v['T4'] else 'FAIL'}")
    else:
        P("T4: fewer than 5 sessions with enough low-contrast trials; not testable")
    if out:
        Path(out).write_text("\n".join(L) + "\n"); print(f"\nWrote {out}")
    return v


def cmd_test(a):
    import pandas as pd
    Sd, Rd = pd.read_csv(OUT / "sessions.csv"), pd.read_csv(OUT / "regions.csv")
    ex, co = split_labs(Sd.lab)
    keep = ex if a.split == "explore" else co if a.split == "confirm" else set(Sd.lab)
    print(f"labs in {a.split}: {sorted(keep)}")
    Sd = Sd[Sd.lab.isin(keep)]; Rd = Rd[Rd.session.isin(Sd.session)]
    evaluate(Sd, Rd, OUT / "nulls", a.split.upper(), a.boot, OUT / f"SHARED_PRIOR_{a.split}.md")


# ================================================================ synthetic check
def _mock_prior(T, rng):
    side, p, t = np.zeros(T), np.zeros(T), 0
    while t < T:
        L = int(np.clip(rng.exponential(60), 20, 100)) if t else 90; q = 0.5 if t == 0 else rng.choice([0.2, 0.8])
        p[t:t + L] = q; t += L
    side = (rng.random(T) < p).astype(float)
    est = np.zeros(T); e = 0.5
    for i in range(T):
        est[i] = e; e = 0.93 * e + 0.07 * side[i]
    return est - 0.5, side


def cmd_mock(a):
    import pandas as pd
    sys.path.insert(0, str(ROOT / "part6"))
    import prior_map as PM
    rng = np.random.default_rng(1)
    for label, gain in (("NO prior state", 0.0), ("SHARED prior state", 0.8)):
        rows, regs, d = [], [], Path(f"/tmp/shared_prior_mock_{int(gain * 10)}"); d.mkdir(exist_ok=True)
        for s in range(16):
            T = 400; y0, side = _mock_prior(T, rng)
            drift = np.cumsum(rng.normal(0, 1, T)); drift = zs(drift[:, None])[:, 0]
            state = zs((gain * zs(y0[:, None])[:, 0] + rng.normal(0, 0.6, T))[:, None])[:, 0]
            region = [rng.poisson(np.exp(1 + 0.15 * np.outer(drift, rng.normal(1, .3, 20))
                                         + 0.25 * np.outer(state, rng.normal(1, .3, 20)) * (gain > 0)
                                         + 0.1 * rng.normal(size=(T, 20)))) for _ in range(5)]
            M = np.column_stack([np.ones(T), rng.normal(size=T)])
            bank = np.column_stack([_mock_prior(T, rng)[0] for _ in range(N_NULL)])
            YZ = remove(np.column_stack([y0, bank]), M)
            c = rng.choice([-1, -.25, -.0625, 0, .0625, .25, 1], T)
            ch = np.where(rng.random(T) < 1 / (1 + np.exp(-(6 * c + 2 * y0 + 0.8 * state * (gain > 0)))), 1., -1.)
            rew = (np.sign(c) == ch).astype(float)
            row, rg, nulls = analyze([remove(np.sqrt(X), M) for X in region], [f"R{k}" for k in range(5)],
                                     YZ[:, 0], YZ[:, 1:], PM.cv_decode, ch, c, rew, rng)
            sid = f"m{s}"; row.update(session=sid, subject=f"mouse{s}", lab=f"lab{s % 4}", pupil=1)
            for q in rg:
                q.update(session=sid)
            rows.append(row); regs += rg; np.savez(d / f"{sid}.npz", **nulls)
        print(f"\n===== mock: {label} =====")
        evaluate(pd.DataFrame(rows), pd.DataFrame(regs), d, "MOCK", 200)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--workers", type=int, default=3); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache"); r.add_argument("--gatekeeper")
    t = sub.add_parser("test"); t.add_argument("--split", choices=["explore", "confirm", "all"], required=True)
    t.add_argument("--boot", type=int, default=500)
    sub.add_parser("mock")
    a = ap.parse_args(); {"run": cmd_run, "test": cmd_test, "mock": cmd_mock}[a.cmd](a)


if __name__ == "__main__":
    main()
