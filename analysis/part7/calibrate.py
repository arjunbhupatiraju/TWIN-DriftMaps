"""CLOSING THE CORE CLAIM (pre-registration: claude/preregistration-calibrate.md)

C1  CALIBRATION. Is the standard number wrong and the valid number right? Real movement (a random combination of the
    session's own drift-removed movement features) is planted into drift twins of real populations at KNOWN strength
    (true population unique R2 = 0, 0.01, 0.03, 0.06). Both analyses estimate it. Bias = estimate - truth.
C2  MOUSE-LEVEL STATISTICS. Every key interval recomputed by resampling whole mice (sessions of one mouse are not
    independent): calibration, overstatement share, the law below, and the earlier results (D1, R1, MA1, MA2, CA1, CA2, M1).
C3  A SIMPLE LAW. Inflation (standard minus valid movement R2) should grow with how much slow drift the neurons and the
    movement share: predictor = (slow share of neural variance) x (slow share of movement variance).

Data: IBL (every 2nd session, part6 decompose features, robust predictors) and MAP (all sessions, part7). Nothing new is
downloaded.

    python3 part7/calibrate.py run --limit 3
    python3 part7/calibrate.py run --workers 4
    python3 part7/calibrate.py test
    python3 part7/calibrate.py mouse          # C2 for the earlier results (reads their result files)
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
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(P6)); sys.path.insert(0, str(HERE.parent / "twin"))
import prior_map as PM        # noqa: E402
import decompose as DC        # noqa: E402
import map_decompose as MD    # noqa: E402

OUT = HERE / "results_calibrate"
TRUTH = (0.0, 0.01, 0.03, 0.06)
FRAC = 0.5                                     # planted into half of the neurons


# ---------------------------------------------------------------- one population
def slow_share(A, D):
    A = A - A.mean(0)
    tot = (A ** 2).sum(0)
    return float(np.mean(1 - (DC.detrend(A, D) ** 2).sum(0) / np.maximum(tot, 1e-12)))


def valid_R2(Y, Tk, M, D):
    Yd, Td, Md = DC.detrend(Y, D), DC.detrend(Tk, D), DC.detrend(M, D)
    return DC.cv_r2(np.column_stack([Td, Md]), Yd) - DC.cv_r2(Td, Yd)


def std_R2(Y, Tk, M):
    return DC.cv_r2(np.column_stack([Tk, M]), Y, contiguous=False) - DC.cv_r2(Tk, Y, contiguous=False)


def population(Y, Tk, M, D, rng):
    from twin.drift import phase_twin
    out = dict(valid=valid_R2(Y, Tk, M, D), standard=std_R2(Y, Tk, M),
               slow_neural=slow_share(Y, D), slow_move=slow_share(M, D))
    out["inflation"] = out["standard"] - out["valid"]
    out["law"] = out["slow_neural"] * out["slow_move"]
    Md = DC.detrend(M, D); Md = (Md - Md.mean(0)) / (Md.std(0) + 1e-12)
    w = rng.normal(size=Md.shape[1]); g = Md @ w; g = (g - g.mean()) / g.std()     # fast movement composite
    Yt = phase_twin(Y, rng); Yt = (Yt - Yt.mean(0)) / Yt.std(0)
    N = Y.shape[1]; pick = rng.random(N) < FRAC
    cal = {}
    for t in TRUTH:
        rho = t / max(pick.mean(), 1e-9)                                            # per planted neuron
        Yp = Yt.copy()
        Yp[:, pick] = np.sqrt(1 - rho) * Yt[:, pick] + np.sqrt(rho) * g[:, None]
        Yp = (Yp - Yp.mean(0)) / Yp.std(0)
        cal[str(t)] = dict(truth=float(rho * pick.mean()), valid=valid_R2(Yp, Tk, M, D), standard=std_R2(Yp, Tk, M))
    out["calibration"] = json.dumps(cal)
    return out


# ---------------------------------------------------------------- sessions
def _ibl(path, gk_root, seed):
    PM._gk(gk_root)
    from gatekeeper.mv_data import MVSession
    rng = np.random.default_rng(seed)
    s = MVSession(Path(path)); rows = []
    for w in DC.WINDOWS:
        e = DC.WINDOWS[w][0]
        v = (np.asarray(s.choice) >= 0) & np.asarray(s.valid[e], bool)
        M, _ = DC.movement(s.sid, w, v, robust=True)
        T = int(v.sum())
        if M is None or T < 300:
            continue
        Tk, D = DC.task(s, v), DC.drift(T)
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
            row = dict(dataset="IBL", session=s.sid, mouse=str(s.subject), area=r, window=w, T=T, N=Y.shape[1], probe="")
            row.update(population(Y, Tk, M, D, rng)); rows.append(row)
    return rows


def _map(path, seed):
    rng = np.random.default_rng(seed)
    z = np.load(path, allow_pickle=False)
    keep = MD._clean_trials(z); T = int(keep.sum())
    if T < 200:
        return []
    ins = np.where(z["trial__instruction"] == "right", 1.0, -1.0); hit = np.where(z["trial__outcome"] == "hit", 1.0, -1.0)
    Tk = np.column_stack([ins, hit, np.r_[0, ins[:-1]], np.r_[0, hit[:-1]]])[keep]; D = DC.drift(T)
    uq = z["unit__unit_quality"] if "unit__unit_quality" in z.files else np.array(["good"] * len(z["unit__probe"]))
    good = uq == "good"
    if "unit__presence_ratio" in z.files:
        good &= np.nan_to_num(z["unit__presence_ratio"], nan=1.0) >= 0.9
    region = np.array([t.replace("left ", "").replace("right ", "").strip() for t in z["unit__target"]])
    rows = []
    for e in MD.EPOCHS:
        M, _ = MD._movement(z, e, keep)
        C = np.asarray(z[f"counts__{e}"], float)[keep]
        if M is None or not np.isfinite(C).all():
            continue
        for pr in np.unique(z["unit__probe"]):
            idx = np.nonzero((z["unit__probe"] == pr) & good)[0]
            X = np.sqrt(np.maximum(C[:, idx], 0)); ok = X.std(0) > 0; idx, X = idx[ok], X[:, ok]
            if len(idx) < 10:
                continue
            if len(idx) > 150:
                X = X[:, np.sort(rng.choice(len(idx), 150, replace=False))]
            Y = (X - X.mean(0)) / X.std(0)
            row = dict(dataset="MAP", session=Path(path).stem, mouse=str(z["subject"]), area=region[idx[0]], window=e,
                       T=T, N=Y.shape[1], probe=str(pr))
            row.update(population(Y, Tk, M, D, rng)); rows.append(row)
    return rows


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    rows_f = OUT / "calibrate_rows.csv"
    done = set(read_rows(rows_f).session) if rows_f.exists() else set()
    jobs = []
    ibl = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz")) if (DC.FEAT / f"{p.stem}.npz").exists()][::2]
    jobs += [("ibl", p) for p in ibl]
    jobs += [("map", p) for p in sorted(MD.SESS.glob("*.npz")) if not p.name.endswith(".part.npz")]
    jobs = [(k, p) for k, p in jobs if p.stem not in done][: a.limit or None]
    print(f"{len(jobs)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {}
        for i, (k, p) in enumerate(jobs):
            futs[ex.submit(_ibl, str(p), a.gatekeeper, 17000 + i) if k == "ibl" else ex.submit(_map, str(p), 18000 + i)] = (k, p)
        for n, fu in enumerate(as_completed(futs)):
            k, p = futs[fu]
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{n + 1}/{len(jobs)}] {k} {p.stem}: ERROR {e}", flush=True); continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
            print(f"[{n + 1}/{len(jobs)}] {k} {p.stem}: {len(rows)} populations", flush=True)


def read_rows(f):
    """Read calibrate_rows.csv. Amendment 1 (file format only): IBL rows were written without a 'probe' column and MAP
    rows with one (after 'N'), so MAP lines have one extra field under the IBL header. Parse each line by its own length."""
    import csv
    import pandas as pd
    with open(f, newline="") as fh:
        rd = csv.reader(fh); head = next(rd); recs = []
        withp = head if "probe" in head else head[:head.index("N") + 1] + ["probe"] + head[head.index("N") + 1:]
        for r in rd:
            if len(r) == len(head):
                recs.append(dict(zip(head, r)))
            elif len(r) == len(withp):
                recs.append(dict(zip(withp, r)))
            else:
                raise ValueError(f"unexpected line with {len(r)} fields")
    D = pd.DataFrame(recs).replace("", np.nan)
    for c in D.columns:
        if c not in ("dataset", "session", "mouse", "area", "window", "probe", "calibration"):
            D[c] = pd.to_numeric(D[c])
    return D


def cmd_fix(a):
    """Rewrite calibrate_rows.csv with one consistent header (keeps a backup)."""
    f = OUT / "calibrate_rows.csv"; D = read_rows(f)
    f.rename(OUT / "calibrate_rows_backup.csv"); D.to_csv(f, index=False)
    print(f"rewrote {f}: {len(D)} rows, " + ", ".join(f"{k} {len(g)}" for k, g in D.groupby("dataset")))


# ---------------------------------------------------------------- mouse bootstrap
def mouse_boot(D, stat, B=500, seed=0, col="mouse"):
    rng = np.random.default_rng(seed)
    groups = {m: g for m, g in D.groupby(col)}; ms = list(groups)
    import pandas as pd
    out = [stat(pd.concat([groups[ms[i]] for i in rng.integers(0, len(ms), len(ms))])) for _ in range(B)]
    return np.nanpercentile(out, [2.5, 97.5])


def cmd_test(a):
    import pandas as pd
    from scipy.stats import spearmanr
    D = read_rows(OUT / "calibrate_rows.csv")
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    P(f"# CALIBRATION, MOUSE-LEVEL STATISTICS AND THE DRIFT LAW: {len(D)} populations, "
      + ", ".join(f"{k} {g.session.nunique()} sessions / {g.mouse.nunique()} mice" for k, g in D.groupby("dataset")) + "\n")
    C = []
    for _, r in D.iterrows():
        for t, v in json.loads(r.calibration).items():
            C.append(dict(dataset=r.dataset, mouse=r.mouse, level=float(t), truth=v["truth"], valid=v["valid"], standard=v["standard"]))
    C = pd.DataFrame(C)
    P("## C1 (confirmatory): bias of each analysis against a planted, known movement code (estimate minus truth)")
    for ds in ["ALL"] + sorted(C.dataset.unique()):
        g0 = C if ds == "ALL" else C[C.dataset == ds]
        for lv, g in g0.groupby("level"):
            civ = mouse_boot(g, lambda x: (x.valid - x.truth).mean(), a.boot)
            cis = mouse_boot(g, lambda x: (x.standard - x.truth).mean(), a.boot)
            P(f"  {ds:4s} truth {g.truth.mean():.3f}: standard {g.standard.mean():.4f} (bias {(g.standard - g.truth).mean():+.4f}, "
              f"CI {cis[0]:+.4f} to {cis[1]:+.4f}); valid {g.valid.mean():.4f} (bias {(g.valid - g.truth).mean():+.4f}, "
              f"CI {civ[0]:+.4f} to {civ[1]:+.4f})")
    z0 = C[C.level == 0]
    lo0 = mouse_boot(z0, lambda x: x.standard.mean(), a.boot)[0]
    pos = C[C.level > 0]
    adv = lambda x: (x.standard - x.truth).abs().mean() - (x.valid - x.truth).abs().mean()
    lo_adv = mouse_boot(pos, adv, a.boot)[0]
    ok = lo0 > 0 and lo_adv > 0
    P(f"  no true code: standard reports {z0.standard.mean():+.4f} (CI lower {lo0:+.4f}); valid {z0.valid.mean():+.4f}")
    P(f"  true code > 0: mean |error| standard {(pos.standard - pos.truth).abs().mean():.4f}, valid "
      f"{(pos.valid - pos.truth).abs().mean():.4f}; advantage of valid {adv(pos):+.4f} (CI lower {lo_adv:+.4f})")
    P(f"  C1 {'PASS' if ok else 'FAIL'} (criterion: with no true code the standard estimate is above 0 (CI lower > 0), and with a "
      f"true code the valid estimate is closer to the truth (CI lower of the |error| advantage > 0))\n")
    share = lambda x: 1 - x.valid.sum() / x.standard.sum() if x.standard.sum() > 0 else np.nan
    P("## C2 (descriptive): share of standard movement encoding that does not survive, mouse-bootstrap CI")
    for ds, g in D.groupby("dataset"):
        lo, hi = mouse_boot(g, share, a.boot)
        P(f"  {ds}: {share(g):.3f} (95% CI {lo:.3f}-{hi:.3f}; {g.mouse.nunique()} mice)")
    P("\n## C3 (confirmatory): does inflation grow with shared slow drift?")
    rho = lambda x: spearmanr(x.law, x.inflation)[0]
    ok3 = True
    for ds in ["ALL"] + sorted(D.dataset.unique()):
        g = D if ds == "ALL" else D[D.dataset == ds]
        lo, hi = mouse_boot(g, rho, a.boot)
        P(f"  {ds}: Spearman(slow neural share x slow movement share, inflation) {rho(g):+.3f} (mouse-bootstrap CI {lo:+.3f} to {hi:+.3f})")
        if ds == "ALL":
            ok3 = lo > 0
    q = pd.qcut(D.law, 4, labels=["lowest", "2nd", "3rd", "highest"])
    P("  inflation by quarter of shared slow drift: " + ", ".join(f"{k} {v:+.4f}" for k, v in D.groupby(q, observed=True).inflation.mean().items()))
    P(f"  C3 {'PASS' if ok3 else 'FAIL'} (criterion: pooled Spearman mouse-bootstrap lower bound > 0)")
    C.to_csv(OUT / "calibration_long.csv", index=False)
    (OUT / "CALIBRATE.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/CALIBRATE.md")


def cmd_adjust(a):
    """C4 (registered 2026-11-09 after C1 failed, before this was run): share lost after correcting BOTH analyses for
    their own measured bias. Per dataset, est = a + b * truth is fitted to the planted-code calibration for each analysis;
    every population's estimate is mapped back to the truth scale, (est - a) / b, and the share is
    1 - sum(adjusted valid) / sum(adjusted standard). Mouse bootstrap resamples mice in the calibration and the
    populations together (calibration refitted each time). Sensitivity: intercept-only correction (b fixed at 1).
    Criterion (same 10% as MA2): the adjusted share's mouse-bootstrap lower bound > 0.10, in IBL and in MAP."""
    import pandas as pd
    D = read_rows(OUT / "calibrate_rows.csv"); C = pd.read_csv(OUT / "calibration_long.csv")
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))

    def fit(c, col, slope=True):
        if not slope:
            return float((c[col] - c.truth).mean()), 1.0
        b, a_ = np.polyfit(c.truth, c[col], 1)
        return float(a_), float(b)

    def share(d, c, slope=True):
        a_s, b_s = fit(c, "standard", slope); a_v, b_v = fit(c, "valid", slope)
        return 1 - ((d.valid - a_v) / b_v).sum() / ((d.standard - a_s) / b_s).sum()

    P("# C4: share of standard movement encoding that does not survive, after correcting each analysis for its own bias\n")
    res = {}
    for ds in ("IBL", "MAP"):
        d, c = D[D.dataset == ds], C[C.dataset == ds]
        if not len(d):
            continue
        a_s, b_s = fit(c, "standard"); a_v, b_v = fit(c, "valid")
        P(f"  {ds} calibration: standard = {a_s:+.4f} + {b_s:.3f} x truth; valid = {a_v:+.4f} + {b_v:.3f} x truth")
        rng = np.random.default_rng(0); ms = d.mouse.unique()
        gd = {m: g for m, g in d.groupby("mouse")}; gc = {m: g for m, g in c.groupby("mouse")}
        for slope, nm in ((True, "intercept and slope"), (False, "intercept only (sensitivity)")):
            v = share(d, c, slope); bs = []
            for _ in range(a.boot):
                pick = ms[rng.integers(0, len(ms), len(ms))]
                dd = pd.concat([gd[m] for m in pick]); cc = pd.concat([gc[m] for m in pick if m in gc])
                bs.append(share(dd, cc, slope))
            lo, hi = np.nanpercentile(bs, [2.5, 97.5]); res[(ds, slope)] = lo
            P(f"  {ds} adjusted share ({nm}): {v:.3f} (95% CI {lo:.3f}-{hi:.3f}; {len(ms)} mice)")
        P(f"  {ds} unadjusted share in the same populations: {1 - d.valid.sum() / d.standard.sum():.3f}")
    ok = all(res.get((ds, True), -1) > 0.10 for ds in ("IBL", "MAP"))
    P(f"\n  C4 {'PASS' if ok else 'FAIL'} (criterion: adjusted share lower bound > 0.10 in IBL and in MAP)")
    (OUT / "ADJUST.md").write_text("\n".join(L) + "\n"); print(f"\nWrote {OUT / 'ADJUST.md'}")


def cmd_mouse(a):
    """C2 for the earlier results: the same statistics, intervals by resampling whole mice."""
    import pandas as pd
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    rate = lambda c: (lambda g: float((g[c] * g.n_twin).sum() / max(g.n_twin.sum(), 1)))
    P("# Key results with mouse-level 95% CIs (whole mice resampled)\n")
    f = P6 / "results_decompose" / "decompose_rows.csv"
    if f.exists():
        D = pd.read_csv(f)
        for c in ("twin_fp_valid", "twin_fp_std"):
            lo, hi = mouse_boot(D, rate(c), a.boot, col="subject")
            P(f"  IBL D1 {c}: {rate(c)(D):.3f} ({lo:.3f}-{hi:.3f}; {D.subject.nunique()} mice)")
    f = HERE / "results_robustness" / "robustness_rows.csv"
    if f.exists():
        D = pd.read_csv(f); w = lambda c: (lambda g: float((g[c] * g.n_partner).sum() / max(g.n_partner.sum(), 1)))
        for c in ("cross_fp_valid", "cross_fp_std"):
            lo, hi = mouse_boot(D, w(c), a.boot, col="subject")
            P(f"  IBL R1 {c}: {w(c)(D):.3f} ({lo:.3f}-{hi:.3f})")
    f = HERE / "results_map" / "map_rows.csv"
    if f.exists():
        D = pd.read_csv(f)
        for c in ("twin_fp_valid", "twin_fp_std"):
            lo, hi = mouse_boot(D, rate(c), a.boot, col="subject")
            P(f"  MAP MA1 {c}: {rate(c)(D):.3f} ({lo:.3f}-{hi:.3f}; {D.subject.nunique()} mice)")
        sh = lambda g: 1 - g.dM.sum() / g.dM_std.sum()
        lo, hi = mouse_boot(D, sh, a.boot, col="subject")
        P(f"  MAP MA2 share: {sh(D):.3f} ({lo:.3f}-{hi:.3f})")
    f = HERE / "results_map" / "map_rich_rows.csv"
    if f.exists():
        D = pd.read_csv(f); sh = lambda g: 1 - g.dM_rich.sum() / g.dM_std_rich.sum()
        lo, hi = mouse_boot(D, sh, a.boot, col="subject")
        P(f"  MAP RF1 share (rich): {sh(D):.3f} ({lo:.3f}-{hi:.3f})")
    f = HERE / "results_calcium" / "calcium_rows.csv"
    if f.exists():
        D = pd.read_csv(f); D["mouse"] = D.session.str.extract(r"(MP\d+)")[0]
        for c in ("twin_fp_valid", "twin_fp_std"):
            lo, hi = mouse_boot(D, rate(c), a.boot)
            P(f"  Calcium CA1 {c}: {rate(c)(D):.3f} ({lo:.3f}-{hi:.3f}; {D.mouse.nunique()} mice)")
        sh = lambda g: 1 - g.R2_valid.sum() / g.R2_std.sum()
        lo, hi = mouse_boot(D, sh, a.boot)
        P(f"  Calcium CA2 share: {sh(D):.3f} ({lo:.3f}-{hi:.3f})")
    f = P6 / "results_movement" / "movement_rows.csv"
    if f.exists():
        D = pd.read_csv(f).reset_index(drop=True); rng = np.random.default_rng(0)
        lik = np.vstack([PM._lik(p, json.loads(c), rng) for p, c in zip(D.p_valid, D.power_curve)])
        regs = D.region.values; mice = D.subject.values; um = np.unique(mice); ur = np.unique(regs)
        ridx = np.searchsorted(ur, regs)

        def est(wt):
            LL = np.zeros((len(ur), lik.shape[1])); np.add.at(LL, ridx, lik * wt[:, None])
            used = np.bincount(ridx, weights=wt, minlength=len(ur)) > 0
            return PM.frac(PM.em_pi(list(LL[used]), iters=150), 0.04)
        point = est(np.ones(len(D)))
        bs = []
        for _ in range(a.boot // 2):
            cnt = dict(zip(*np.unique(rng.choice(um, len(um)), return_counts=True)))
            bs.append(est(np.array([cnt.get(m, 0) for m in mice], float)))
        P(f"  Step 1 M1 fraction of regions carrying the prior (r >= 0.04): {point:.3f} "
          f"({np.percentile(bs, 2.5):.3f}-{np.percentile(bs, 97.5):.3f}; criterion lower > 0.071)")
    OUT.mkdir(exist_ok=True)
    (OUT / "MOUSE_LEVEL.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/MOUSE_LEVEL.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--workers", type=int, default=4); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache"); r.add_argument("--gatekeeper")
    t = sub.add_parser("test"); t.add_argument("--boot", type=int, default=500)
    m = sub.add_parser("mouse"); m.add_argument("--boot", type=int, default=500)
    sub.add_parser("fix")
    j = sub.add_parser("adjust"); j.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    {"run": cmd_run, "test": cmd_test, "mouse": cmd_mouse, "fix": cmd_fix, "adjust": cmd_adjust}[a.cmd](a)


if __name__ == "__main__":
    main()
