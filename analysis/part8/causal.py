"""CAUSAL VALIDATION (Spine A; pre-registration claude/preregistration-causal-validation.md).

Do drift-corrected encoding maps (Steinmetz et al. 2019 Neuropixels) match the causal map from optogenetic
inactivation in the same task (Zatka-Haas et al. 2021, 52 dorsal-cortex sites) better than standard maps?

    cd ~/Downloads/MODEL
    python3 part8/causal.py mock                      synthetic check (no real data)
    python3 part8/causal.py check  --steinmetz DIR --zh FILE     data layout, areas, site->area map; NO statistics
    python3 part8/causal.py run    --steinmetz DIR --workers 2   per area-session encoding (standard and valid)
    python3 part8/causal.py inact  --zh FILE                     per-site / per-area inactivation effects
    python3 part8/causal.py test                                 A1, A2, A3
Output: part8/results_causal/
"""
from __future__ import annotations

import argparse, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results_causal"
MIN_UNITS, MAX_UNITS, MIN_SESS = 30, 150, 3
N_NULL, ALPHA, K_FOLD = 40, 10.0, 5
WIN = {"choice": ("move", -0.15, 0.0), "contrast": ("stim", 0.0, 0.15)}


# ================================================================ helpers
def drift(T):
    sys.path.insert(0, str(ROOT / "part6"))
    import decompose as DC                      # the paper's drift basis (periods >= 200 trials)
    return np.column_stack([np.ones(T), DC.drift(T)])


def remove(A, D):
    return A - D @ np.linalg.lstsq(D, A, rcond=None)[0]


def zs(A):
    A = np.asarray(A, float); s = A.std(0); s[s == 0] = 1
    return (A - A.mean(0)) / s


def ridge_cv_r(X, y, folds):
    """held-out correlation of a fixed-alpha ridge decoder X -> y, predictions concatenated over folds"""
    pred = np.empty(len(y))
    for te in folds:
        tr = np.setdiff1d(np.arange(len(y)), te)
        Xt, yt = X[tr], y[tr]; mx, my = Xt.mean(0), yt.mean()
        A = Xt - mx; w = np.linalg.solve(A.T @ A + ALPHA * np.eye(A.shape[1]), A.T @ (yt - my))
        pred[te] = (X[te] - mx) @ w + my
    if pred.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(pred, y)[0, 1])


def folds_of(T, contiguous, rng):
    if contiguous:
        e = np.linspace(0, T, K_FOLD + 1).astype(int); return [np.arange(a, b) for a, b in zip(e[:-1], e[1:])]
    p = rng.permutation(T); return np.array_split(p, K_FOLD)


def standard(X, y, rng):
    """interleaved CV, trial-shuffle null. Returns (r_obs, null-corrected r, p)"""
    X = zs(X); T = len(y); f = folds_of(T, False, rng)
    r = ridge_cv_r(X, y, f)
    nul = np.array([ridge_cv_r(X, rng.permutation(y), f) for _ in range(N_NULL)])
    return r, r - nul.mean(), (1 + (nul >= r).sum()) / (1 + N_NULL)


def valid(X, y, rng):
    """drift removed from each signal separately, contiguous CV, linear-shift null (Harris 2020)"""
    T = len(y); D = drift(T)
    Xc = zs(remove(np.asarray(X, float), D)); yc = remove(np.asarray(y, float)[:, None], D)[:, 0]
    M = T // 4
    shifts = np.unique(np.linspace(-M, M, N_NULL + 1).astype(int))
    def at(d):
        idx = np.arange(M, T - M)
        yy = yc[idx + d]; XX = Xc[idx]
        return ridge_cv_r(XX, yy, folds_of(len(idx), True, rng))
    r = at(0); nul = np.array([at(d) for d in shifts if d != 0])
    return r, r - nul.mean(), (1 + (nul >= r).sum()) / (1 + len(nul))


# ================================================================ Steinmetz loader
def _npy(d, n):
    p = d / f"{n}.npy"
    return np.load(p, allow_pickle=True) if p.exists() else None


def sessions(root):
    return sorted(p.parent for p in Path(root).expanduser().rglob("spikes.times.npy"))


def load_session(d):
    import pandas as pd
    st, sc = _npy(d, "spikes.times").ravel(), _npy(d, "spikes.clusters").ravel().astype(int)
    ann = _npy(d, "clusters._phy_annotation")
    good = np.ones(sc.max() + 1, bool) if ann is None else (ann.ravel() >= 2)
    loc = pd.read_csv(d / "channels.brainLocation.tsv", sep="\t")
    area_ch = loc["allen_ontology"].astype(str).values
    pc = _npy(d, "clusters.peakChannel").ravel().astype(int)
    cpr, chp = _npy(d, "clusters.probes"), _npy(d, "channels.probe")
    nch = len(area_ch)
    if chp is not None and cpr is not None and pc.max() <= np.bincount(chp.ravel().astype(int)).max() < nch:
        off = np.concatenate([[0], np.cumsum(np.bincount(chp.ravel().astype(int)))])
        g = off[cpr.ravel().astype(int)] + pc - (1 if pc.min() >= 1 else 0)
    else:
        g = pc - (1 if pc.max() >= nch else 0)
    area = area_ch[np.clip(g, 0, nch - 1)]
    tr = {k: _npy(d, f"trials.{k}") for k in ("visualStim_times", "visualStim_contrastLeft", "visualStim_contrastRight",
                                             "response_choice", "response_times", "feedbackType", "included")}
    tr = {k: (v.ravel() if v is not None else None) for k, v in tr.items()}
    wi, wt = _npy(d, "wheelMoves.intervals"), _npy(d, "wheelMoves.type")
    return dict(st=st, sc=sc, good=good, area=area, tr=tr, wi=wi, wt=None if wt is None else wt.ravel())


def move_onsets(S):
    """first wheel movement starting after stimulus onset (before response time); fallback: response time"""
    stim, resp = S["tr"]["visualStim_times"], S["tr"]["response_times"]
    out = resp.astype(float).copy()
    if S["wi"] is not None:
        on = S["wi"][:, 0]
        for i, (a, b) in enumerate(zip(stim, resp)):
            k = np.nonzero((on > a) & (on <= b))[0]
            if len(k):
                out[i] = on[k[0]]
    return out


def beryl(acr):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    out = []
    for a in acr:
        try:
            out.append(br.acronym2acronym(np.array([a]), mapping="Beryl")[0])
        except Exception:
            out.append(a)
    return np.array(out)


def counts(S, units, t0, w0, w1):
    st, sc = S["st"], S["sc"]
    keep = np.isin(sc, units); st, sc = st[keep], sc[keep]
    col = {u: i for i, u in enumerate(units)}
    X = np.zeros((len(t0), len(units)))
    o = np.argsort(st); st, sc = st[o], sc[o]
    for i, t in enumerate(t0):
        a, b = np.searchsorted(st, [t + w0, t + w1])
        for c in sc[a:b]:
            X[i, col[c]] += 1
    return np.sqrt(X)


def session_rows(d, rng):
    S = load_session(d); tr = S["tr"]
    inc = tr["included"].astype(bool) if tr["included"] is not None else np.ones(len(tr["response_choice"]), bool)
    ch = tr["response_choice"].astype(float)
    cdiff = tr["visualStim_contrastRight"].astype(float) - tr["visualStim_contrastLeft"].astype(float)
    mv = move_onsets(S)
    B = beryl(S["area"])
    clu_area = {}
    for u in np.nonzero(S["good"])[0]:
        clu_area.setdefault(B[u] if u < len(B) else "?", []).append(u)
    rows = []
    for ar, units in clu_area.items():
        if len(units) < MIN_UNITS or ar in ("root", "void", "?", "nan"):
            continue
        units = sorted(rng.choice(units, min(len(units), MAX_UNITS), replace=False))
        for var, (ev, w0, w1) in WIN.items():
            if var == "choice":
                m = inc & (ch != 0); y = np.sign(ch[m]); t0 = mv[m]
            else:
                m = inc; y = cdiff[m]; t0 = tr["visualStim_times"][m]
            if len(y) < 60 or np.std(y) == 0:
                continue
            X = counts(S, units, t0, w0, w1)
            rs, es, ps = standard(X, y, rng); rv, ev_, pv = valid(X, y, rng)
            rows.append(dict(session=d.name, mouse=d.name.split("_")[0], area=ar, var=var, n_units=len(units),
                             T=len(y), r_std=rs, eff_std=es, p_std=ps, r_val=rv, eff_val=ev_, p_val=pv))
    return rows


# ================================================================ Zatka-Haas inactivation
def load_zh(path):
    """Reads the MATLAB table in Inactivation_52Coord.mat (v7.3). Columns identified from their contents
    (inspect_zh.py): 2-column contrast (0/.1/.24/.54), response 1/2/3, feedback -1/1, laser coordinate N (ML, AP mm;
    NaN = no laser), laser index V (0 = none), session T, subject U."""
    import h5py
    h = h5py.File(Path(path).expanduser(), "r"); R = lambda k: h["#refs#/" + k][()]
    con = R("K").reshape(2, -1) if R("K").shape[0] == 2 else R("K").T
    resp_cands = {k: R(k).ravel() for k in ("A", "L")}
    fb_cands = {k: R(k).ravel() for k in ("C", "M")}
    cl, cr = con[0], con[1]
    # pick the response/feedback pair that is consistent (correct = choosing the higher-contrast side)
    best = None
    for rk, r in resp_cands.items():
        for fk, f in fb_cands.items():
            m = (cl != cr) & (r != 3)
            for left_code in (1, 2):
                corr_side = np.where(cl > cr, left_code, 3 - left_code)
                agree = np.mean((r[m] == corr_side[m]) == (f[m] == 1))
                if best is None or agree > best[0]:
                    best = (agree, rk, fk, left_code)
    agree, rk, fk, left_code = best
    coord = R("N"); coord = coord.reshape(2, -1) if coord.shape[0] == 2 else coord.T
    import pandas as pd
    D = pd.DataFrame(dict(cl=cl, cr=cr, resp=resp_cands[rk], ml=coord[0], ap=coord[1], site=R("V").ravel(),
                          session=R("T").ravel(), subject=R("U").ravel()))
    D["choice_left"] = (D.resp == left_code).astype(float); D["choice_right"] = (D.resp == 3 - left_code).astype(float)
    D.loc[D.resp == 3, ["choice_left", "choice_right"]] = 0.0
    info = f"response={rk}, feedback={fk}, left code={left_code}, consistency {agree:.3f}"
    return D, info


_BA = None


def site_area(ml, ap):
    """Allen Beryl area at the dorsal cortical surface (300 um deep) under a bregma-relative (ML, AP) mm coordinate"""
    global _BA
    if _BA is None:
        from iblatlas.atlas import AllenAtlas
        _BA = AllenAtlas(25)
        if not hasattr(_BA, "top"):
            _BA.compute_surface()
    ba = _BA
    x, y = ml / 1000.0, ap / 1000.0
    iy, ix = ba.bc.y2i(y), ba.bc.x2i(x)
    z = ba.top[iy, ix] - 0.0003
    if not np.isfinite(z):
        return "void"
    lab = np.atleast_1d(ba.get_labels(np.array([[x, y, z]]), mapping="Beryl")).ravel()[0]
    if not lab:
        return "void"
    return str(np.atleast_1d(ba.regions.id2acronym(np.atleast_1d(lab), mapping="Beryl")).ravel()[0])


def inact_effects(D, flip=False):
    """per site: laser minus no-laser P(contralateral choice) on trials with the higher contrast contralateral,
    matched by session (session-weighted). contralateral = opposite hemisphere to the laser (ML < 0 = left)."""
    import pandas as pd
    L = D[D.site > 0].copy(); N0 = D[D.site == 0]
    rows = []
    for s, g in L.groupby("site"):
        ml, ap = g.ml.iloc[0], g.ap.iloc[0]
        contra_right = (ml < 0) != flip
        diffs, w = [], []
        for ses, gs in g.groupby("session"):
            b = N0[N0.session == ses]
            pick = (lambda d: d[d.cr > d.cl]) if contra_right else (lambda d: d[d.cl > d.cr])
            cc = "choice_right" if contra_right else "choice_left"
            a, bb = pick(gs), pick(b)
            if len(a) and len(bb):
                diffs.append(a[cc].mean() - bb[cc].mean()); w.append(len(a))
        if diffs:
            rows.append(dict(site=int(s), ml=ml, ap=ap, effect=float(np.average(diffs, weights=w)), n=int(sum(w))))
    return pd.DataFrame(rows)


# ================================================================ commands
def cmd_check(a):
    import pandas as pd
    ss = sessions(a.steinmetz)
    print(f"Steinmetz: {len(ss)} sessions under {a.steinmetz}")
    if ss:
        S = load_session(ss[0]); B = beryl(S["area"])
        print("  first session:", ss[0].name, "| trials", len(S["tr"]["response_choice"]), "| good units",
              int(S["good"].sum()), "| wheelMoves", S["wi"] is not None)
        print("  choice codes", np.unique(S["tr"]["response_choice"]), "| areas", sorted(set(B))[:30])
        cnt = {}
        for d in ss:
            S = load_session(d); B = beryl(S["area"]); u = np.nonzero(S["good"])[0]
            for ar, n in pd.Series(B[u[u < len(B)]]).value_counts().items():
                if n >= MIN_UNITS:
                    cnt[ar] = cnt.get(ar, 0) + 1
        print("  areas with >= 30 good units, by number of sessions:",
              ", ".join(f"{k} {v}" for k, v in sorted(cnt.items(), key=lambda x: -x[1])))
    if a.zh:
        D, info = load_zh(a.zh); print(f"\nZatka-Haas: {len(D)} trials, {D.subject.nunique()} mice, "
                                       f"{D.session.nunique()} sessions; {info}")
        sites = D[D.site > 0].groupby("site")[["ml", "ap"]].first()
        sites["area"] = [site_area(m, p) for m, p in zip(sites.ml, sites.ap)]
        print("  site -> area:", ", ".join(f"{k}:{v}" for k, v in sites.area.value_counts().items()))
        OUT.mkdir(exist_ok=True); sites.to_csv(OUT / "sites.csv")
    print("\nNo encoding or inactivation statistic was computed.")


def cmd_run(a):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    ss = sessions(a.steinmetz); OUT.mkdir(exist_ok=True); rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(session_rows, d, np.random.default_rng(i)): d for i, d in enumerate(ss)}
        for i, f in enumerate(futs):
            try:
                r = f.result(); rows += r; print(f"[{i + 1}/{len(ss)}] {futs[f].name}: {len(r)} area-variables")
            except Exception as e:
                print(f"[{i + 1}/{len(ss)}] {futs[f].name}: ERROR {e!r}")
    pd.DataFrame(rows).to_csv(OUT / "encoding.csv", index=False); print(f"Wrote {OUT / 'encoding.csv'}")


def cmd_inact(a):
    import pandas as pd
    D, info = load_zh(a.zh); print(info)
    # Amendment 1 (bug fix): the hemisphere convention of the laser ML coordinate relative to the contrast columns
    # is not stated in the file. Choose the convention that reproduces the PUBLISHED Zatka-Haas result (VIS
    # inactivation lowers contralateral choices), using only VISp sites; independent of any encoding statistic.
    sites0 = pd.read_csv(OUT / "sites.csv") if (OUT / "sites.csv").exists() else None
    cands = {}
    for flip in (False, True):
        E0 = inact_effects(D, flip)
        if sites0 is not None:
            E0 = E0.merge(sites0[["site", "area"]], on="site", how="left")
        vis = E0[E0.area == "VISp"].effect.mean() if "area" in E0 else E0.effect.mean()
        mos = E0[E0.area == "MOs"].effect.mean() if "area" in E0 else np.nan
        cands[flip] = vis
        print(f"  convention flip={flip}: mean change in P(contralateral choice), VISp sites {vis:+.3f}, MOs sites {mos:+.3f}")
    FLIP = min(cands, key=cands.get)
    print(f"  using flip={FLIP} (VISp effect most negative, as published)")
    E = inact_effects(D, FLIP)
    sites = pd.read_csv(OUT / "sites.csv") if (OUT / "sites.csv").exists() else None
    if sites is None:
        sites = E[["site", "ml", "ap"]].copy(); sites["area"] = [site_area(m, p) for m, p in zip(sites.ml, sites.ap)]
    E = E.merge(sites[["site", "area"]], on="site", how="left")
    E.to_csv(OUT / "inact_sites.csv", index=False)
    # per mouse, for the bootstrap
    rows = []
    for subj, Ds in D.groupby("subject"):
        Es = inact_effects(Ds, FLIP).merge(sites[["site", "area"]], on="site", how="left"); Es["subject"] = subj; rows.append(Es)
    pd.concat(rows).to_csv(OUT / "inact_sites_by_mouse.csv", index=False)
    print(f"Wrote {OUT / 'inact_sites.csv'} and inact_sites_by_mouse.csv ({len(E)} sites)")


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b)[0]) if len(a) >= 3 else np.nan


def evaluate(Enc, Inm, label, boot=1000, seed=0):
    """Enc: encoding rows; Inm: per-mouse per-site inactivation rows. Causal strength = minus the mean effect
    (inactivation lowers contralateral choices), averaged over sites in the area."""
    import pandas as pd
    rng = np.random.default_rng(seed); L = [f"# CAUSAL VALIDATION {label}"]
    def area_tables(E, I):
        cnt = E.groupby(["area", "var"]).session.nunique().reset_index()
        ok = cnt[cnt.session >= MIN_SESS]
        enc = E.merge(ok[["area", "var"]]).groupby(["var", "area"])[["eff_std", "eff_val"]].mean().reset_index()
        cau = (-I.groupby("area").effect.mean()).rename("causal").reset_index()
        return enc.merge(cau, on="area")
    T = area_tables(Enc, Inm)
    mE, mI = Enc.mouse.unique(), Inm.subject.unique()
    gE = {m: d for m, d in Enc.groupby("mouse")}; gI = {m: d for m, d in Inm.groupby("subject")}
    out = {}
    for k, var in (("A1", "choice"), ("A2", "contrast")):
        t = T[T["var"] == var]
        if len(t) < 3:
            L.append(f"{k} ({var}): fewer than 3 overlapping areas; not testable"); continue
        rs, rv = spearman(t.eff_std, t.causal), spearman(t.eff_val, t.causal)
        bs = []
        for _ in range(boot):
            E2 = pd.concat([gE[m] for m in rng.choice(mE, len(mE))]); I2 = pd.concat([gI[m] for m in rng.choice(mI, len(mI))])
            t2 = area_tables(E2, I2); t2 = t2[t2["var"] == var]
            if len(t2) >= 3:
                bs.append(spearman(t2.eff_val, t2.causal) - spearman(t2.eff_std, t2.causal))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5]); out[k] = lo > 0
        L.append(f"{k} ({var}, {len(t)} areas: {', '.join(t.area)}): rho standard {rs:+.2f}, rho valid {rv:+.2f}; "
                 f"difference {rv - rs:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {'PASS' if out[k] else 'NOT RESOLVED'}")
        rk_s, rk_v = t.eff_std.rank(ascending=False), t.eff_val.rank(ascending=False)
        L.append(f"   A3 areas moving >= 3 ranks: {int((abs(rk_s - rk_v) >= 3).sum())}; table:")
        for _, r in t.sort_values("causal", ascending=False).iterrows():
            L.append(f"     {r.area:8s} causal {r.causal:+.3f}  standard {r.eff_std:+.3f}  valid {r.eff_val:+.3f}")
    print("\n".join(L)); return out, L


def cmd_test(a):
    import pandas as pd
    Enc = pd.read_csv(OUT / "encoding.csv"); Inm = pd.read_csv(OUT / "inact_sites_by_mouse.csv")
    _, L = evaluate(Enc, Inm, "(real data)")
    (OUT / "CAUSAL.md").write_text("\n".join(L) + "\n"); print(f"\nWrote {OUT / 'CAUSAL.md'}")


def cmd_mock(a):
    """Synthetic check. 8 areas with a true causal strength c. Encoding = c + drift artefact (standard only).
    Condition 1: drift artefact unrelated to c -> valid should beat standard. Condition 2: no artefact -> no
    difference. Also checks the estimators: twins with no code must give valid effect about 0."""
    import pandas as pd
    rng = np.random.default_rng(3)
    # estimator check on one synthetic session
    T, N = 300, 40; t = np.arange(T)
    slow = np.sin(2 * np.pi * t / T); y = np.sign(rng.normal(size=T) + 1.5 * slow)
    X = rng.normal(size=(T, N)) + np.outer(slow, rng.normal(size=N)) * 2
    print("estimator check, no true code (slow drift shared by X and y):")
    print("  standard: r %.3f, effect %.3f, p %.3f" % standard(X, y, rng))
    print("  valid:    r %.3f, effect %.3f, p %.3f" % valid(X, y, rng))
    Xc = X + np.outer(y, rng.normal(size=N)) * 0.4
    print("  with a true code: standard effect %.3f, valid effect %.3f" % (standard(Xc, y, rng)[1], valid(Xc, y, rng)[1]))
    for name, art in (("drift artefact unrelated to causal map", 1.0), ("no artefact", 0.0)):
        areas = [f"R{i}" for i in range(8)]; c = np.linspace(0, 0.3, 8); artefact = rng.permutation(c) * art * 2
        E, I = [], []
        for m in range(8):
            for i, ar in enumerate(areas):
                for v in ("choice", "contrast"):
                    tv = c[i] + rng.normal(0, 0.03)
                    E.append(dict(mouse=f"m{m}", session=f"s{m}", area=ar, var=v, eff_val=tv, eff_std=tv + artefact[i]))
        for m in range(5):
            for i, ar in enumerate(areas):
                I.append(dict(subject=m, site=i, area=ar, effect=-(c[i] + rng.normal(0, 0.03))))
        print(f"\nMOCK: {name}")
        evaluate(pd.DataFrame(E), pd.DataFrame(I), name, boot=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mock")
    c = sub.add_parser("check"); c.add_argument("--steinmetz", required=True); c.add_argument("--zh")
    r = sub.add_parser("run"); r.add_argument("--steinmetz", required=True); r.add_argument("--workers", type=int, default=2)
    i = sub.add_parser("inact"); i.add_argument("--zh", required=True)
    sub.add_parser("test")
    a = ap.parse_args()
    {"mock": cmd_mock, "check": cmd_check, "run": cmd_run, "inact": cmd_inact, "test": cmd_test}[a.cmd](a)
