"""GAP FIXES B1 + B2 (pre-registration: claude/preregistration-robustness.md)

B1  Assumption-free null. Drift twins assume stationary drift. Here the REAL neurons of one session are paired with
    the REAL movement of a different mouse's session (same trial window, first T trials). There is no true link, and
    nothing is simulated: both signals keep their own drift, jumps and all. How often does each test call it encoding?
B2  Power. Real movement (the first principal component of the session's drift-removed movement features) is planted
    into drift twins of the neurons at known strengths. How often does the valid test find it?

Uses the step 2 pipeline (part6/decompose.py) with the robust movement predictors (Amendment 2), on every 4th session.

    python3 part7/robustness.py run --limit 3
    python3 part7/robustness.py run
    python3 part7/robustness.py test
"""
from __future__ import annotations

import argparse, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P6 = HERE.parent / "part6"
sys.path.insert(0, str(P6)); sys.path.insert(0, str(HERE.parent / "twin"))
import prior_map as PM   # noqa: E402
import decompose as DC   # noqa: E402

OUT = HERE / "results_robustness"
POWER_R = (0.05, 0.1, 0.2)


def valid_p(Y, Tk, M, D, S):
    """The valid movement test alone (contiguous CV on drift-removed data, linear-shift null)."""
    T = len(Y); c = slice(S, T - S)
    Yd, Td, Md = DC.detrend(Y, D), DC.detrend(Tk, D), DC.detrend(M, D)
    Yc, Tc = Yd[c], Td[c]
    base = DC.cv_r2(Tc, Yc)
    obs = DC.cv_r2(np.column_stack([Tc, Md[c]]), Yc) - base
    nul = [DC.cv_r2(np.column_stack([Tc, Md[S + s:T - S + s]]), Yc) - base for s in range(-S, S + 1, 2) if s != 0]
    return PM._p(obs, nul)


def _partner_movement(cache, partners, w, T):
    """Movement of other mice's sessions in the same window, first T usable trials (None if too short)."""
    from gatekeeper.mv_data import MVSession
    out = []
    for sid in partners:
        s = MVSession(Path(cache) / f"{sid}.npz")
        e = DC.WINDOWS[w][0]
        v = (np.asarray(s.choice) >= 0) & np.asarray(s.valid[e], bool)
        M, _ = DC.movement(sid, w, v, robust=True)
        if M is not None and len(M) >= T:
            out.append(M[:T])
    return out


def _session(path, cache, gk_root, pool, n_partner, n_shuf, max_shift, n_plant, seed):
    PM._gk(gk_root)
    from gatekeeper.mv_data import MVSession
    from twin.drift import phase_twin
    from twin.check import plant
    rng = np.random.default_rng(seed)
    s = MVSession(Path(path))
    others = [sid for sid, subj in pool if subj != s.subject]
    rows = []
    for w in DC.WINDOWS:
        e = DC.WINDOWS[w][0]
        v = (np.asarray(s.choice) >= 0) & np.asarray(s.valid[e], bool)
        M, _ = DC.movement(s.sid, w, v, robust=True)
        T = int(v.sum())
        if M is None or T < 300:
            continue
        Tk, D = DC.task(s, v), DC.drift(T)
        S = 2 * (min(max_shift, T // 6) // 2)
        partners = _partner_movement(cache, rng.choice(others, min(len(others), 4 * n_partner), replace=False), w, T)[:n_partner]
        Md = DC.detrend(M, D); Md = (Md - Md.mean(0)) / (Md.std(0) + 1e-12)
        pc = np.linalg.svd(Md, full_matrices=False)[0][:, 0]           # main axis of this session's movement
        for r in s.regions:
            X = s.X(r, w)
            if X is None:
                continue
            X = np.sqrt(np.maximum(np.asarray(X, float)[v], 0)); X = X[:, X.std(0) > 0]
            if X.shape[1] < 10:
                continue
            if X.shape[1] > 150:
                X = X[:, rng.choice(X.shape[1], 150, replace=False)]
            Y = (X - X.mean(0)) / X.std(0)
            row = dict(session=s.sid, subject=s.subject, lab=s.lab, region=r, window=w, T=T, N=X.shape[1],
                       n_partner=len(partners))
            fv = fs = 0                                                # B1: real neurons x another mouse's movement
            for Mp in partners:
                o = DC.tests(Y, Tk, Mp, D, S, n_shuf, rng)
                fv += o["p_valid_M"] < 0.05; fs += o["p_std_M"] < 0.05
            row.update(cross_fp_valid=fv / max(len(partners), 1), cross_fp_std=fs / max(len(partners), 1))
            for rr in POWER_R:                                         # B2: real movement planted into twins
                hit = 0
                for _ in range(n_plant):
                    Yt = phase_twin(Y, rng)
                    Yt = plant(Yt, pc, rr, 0.2, rng)
                    Yt = (Yt - Yt.mean(0)) / Yt.std(0)
                    hit += valid_p(Yt, Tk, M, D, S) < 0.05
                row[f"power_{rr}"] = hit / n_plant
            row["n_plant"] = n_plant
            rows.append(row)
    return rows


def cmd_run(a):
    import pandas as pd
    PM._gk(a.gatekeeper)
    from gatekeeper.mv_data import MVSession
    OUT.mkdir(exist_ok=True)
    rows_f = OUT / "robustness_rows.csv"
    cache = Path(a.cache).expanduser()
    allp = [p for p in sorted(cache.glob("*.npz")) if (DC.FEAT / f"{p.stem}.npz").exists()]
    pool = [(p.stem, MVSession(p).subject) for p in allp]
    done = set(pd.read_csv(rows_f).session) if rows_f.exists() else set()
    paths = [p for p in allp[::a.every][: a.limit or None] if p.stem not in done]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), str(cache), a.gatekeeper, pool, a.partners, a.shuffles, a.max_shift,
                          a.plants, 11000 + i): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True)
                continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} region-windows", flush=True)


def cmd_test(a):
    import pandas as pd
    D = pd.read_csv(OUT / "robustness_rows.csv")
    rng = np.random.default_rng(0)
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    P(f"# ROBUSTNESS: {len(D)} session-region-windows, {D.session.nunique()} sessions, {D.region.nunique()} regions\n")
    sess = [g for _, g in D.groupby("session")]
    w = lambda g, c: float((g[c] * g.n_partner).sum() / max(g.n_partner.sum(), 1))
    bs = [pd.concat([sess[i] for i in rng.integers(0, len(sess), len(sess))]) for _ in range(a.boot)]
    fv, fs = w(D, "cross_fp_valid"), w(D, "cross_fp_std")
    hv = float(np.percentile([w(g, "cross_fp_valid") for g in bs], 97.5))
    ls = float(np.percentile([w(g, "cross_fp_std") for g in bs], 2.5))
    R1a, R1b = fv <= 0.07 and hv <= 0.08, ls >= 0.20
    P("## R1 (assumption-free null: real neurons paired with another mouse's real movement)")
    P(f"  valid test: {fv:.3f} (95% CI upper {hv:.3f}; criterion <= 0.07 and upper <= 0.08)  R1a {'PASS' if R1a else 'FAIL'}")
    P(f"  standard test: {fs:.3f} (95% CI lower {ls:.3f}; criterion lower >= 0.20)  R1b {'PASS' if R1b else 'FAIL'}")
    for win, g in D.groupby("window"):
        P(f"    {win}: valid {w(g, 'cross_fp_valid'):.3f}, standard {w(g, 'cross_fp_std'):.3f}")
    P("\n## R2 (power of the valid test: real movement planted into drift twins, 20% of neurons; descriptive)")
    for rr in POWER_R:
        c = f"power_{rr}"
        b = [g[c].mean() for g in bs]
        P(f"  per-neuron r = {rr}: detected {D[c].mean():.3f} (95% CI {np.percentile(b, 2.5):.3f}-{np.percentile(b, 97.5):.3f})")
    for win, g in D.groupby("window"):
        P(f"    {win}: " + ", ".join(f"r={rr} {g[f'power_{rr}'].mean():.2f}" for rr in POWER_R))
    (OUT / "ROBUSTNESS.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/ROBUSTNESS.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--every", type=int, default=4); r.add_argument("--partners", type=int, default=2)
    r.add_argument("--shuffles", type=int, default=100); r.add_argument("--max-shift", type=int, default=50)
    r.add_argument("--plants", type=int, default=2)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
    r.add_argument("--gatekeeper"); r.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("test"); t.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    {"run": cmd_run, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
