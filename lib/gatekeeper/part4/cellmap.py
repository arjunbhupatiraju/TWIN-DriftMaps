"""CELLMAP feasibility gate: does a region's cell-type composition predict its
decision-signal coding beyond brain division and spatial position?

Steps
  sim     synthetic validation (no downloads): calibration + planted signal
  fetch   download ABC MERFISH cell metadata (Mac), build composition table
  inspect print columns / ranges of the downloaded ABC table
  run     the pre-registered gate on real data
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IBL_DIR = os.path.join(ROOT, "manifests", "ibl_bwm_region_results")
OUT = os.path.join(HERE, "results")
VARS = ("choice", "stim", "fback")
ALPHAS = np.logspace(-2, 4, 13)
N_BLOCK, CV_SEEDS, N_PC, MIN_CELLS = 10, 10, 15, 200


# ---------------------------------------------------------------- helpers
def rankit(y):
    return norm.ppf((rankdata(y) - 0.5) / len(y))


def coord_feats(c):
    x, y, z = c[:, 0], c[:, 1], c[:, 2]
    return np.column_stack([x, y, z, x * x, y * y, z * z, x * y, x * z, y * z])


def onehot(lab):
    u = sorted(set(lab))
    return np.array([[l == v for v in u] for l in lab], float)


def blocks(coords, seed):
    # random 10-fold assignment; spatial autocorrelation is handled by the two nulls
    r = np.random.default_rng(1000 + seed)
    return r.permutation(np.arange(len(coords)) % N_BLOCK)


def cv_r2(y, base, comp, splits):
    """Spatial-block CV. Returns (R2 base, R2 base+comp). comp may be None."""
    out = []
    for g in splits:
        pb, pf = np.zeros_like(y), np.zeros_like(y)
        for b in np.unique(g):
            te, tr = g == b, g != b
            sb = StandardScaler().fit(base[tr])
            Bt, Be = sb.transform(base[tr]), sb.transform(base[te])
            m = RidgeCV(alphas=ALPHAS).fit(Bt, y[tr]); pb[te] = m.predict(Be)
            if comp is not None:
                # two-stage: composition predicts what the base model leaves unexplained
                res_tr = y[tr] - m.predict(Bt)
                sc = StandardScaler().fit(comp[tr])
                m2 = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(sc.transform(comp[tr]), res_tr)
                pf[te] = pb[te] + m2.predict(sc.transform(comp[te]))
        ss = ((y - y.mean()) ** 2).sum()
        out.append((1 - ((y - pb) ** 2).sum() / ss,
                    1 - ((y - pf) ** 2).sum() / ss if comp is not None else np.nan))
    return np.mean(out, axis=0)


def gp_surrogates(y, coords, div, n, rng):
    """Smooth-map null: division means + GP residual with fitted length scale."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
    D = onehot(div)
    beta = np.linalg.lstsq(D, y, rcond=None)[0]
    r = y - D @ beta
    gp = GaussianProcessRegressor(C(1.0) * RBF(1.0, (0.05, 20)) + WhiteKernel(0.5),
                                  normalize_y=True, random_state=0).fit(coords, r)
    K = gp.kernel_(coords)
    K = K * (np.var(r) / np.mean(np.diag(K)))
    L = np.linalg.cholesky(K + 1e-8 * np.eye(len(y)))
    S = (D @ beta)[:, None] + L @ rng.standard_normal((len(y), n))
    return S, str(gp.kernel_)


def gate(y, comp, coords, div, n_null=300, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    y = rankit(y)
    base = np.hstack([onehot(div), coord_feats(coords)])
    splits = [blocks(coords, s) for s in range(CV_SEEDS)]
    r_base, r_full = cv_r2(y, base, comp, splits)
    r_comp = cv_r2(y, np.zeros((len(y), 1)) + rng.normal(0, 1e-6, (len(y), 1)), comp, splits)[1]
    r_div = cv_r2(y, onehot(div), None, splits)[0]
    d_obs = r_full - r_base
    nsplit = splits[:3]
    ref = cv_r2(y, base, comp, nsplit); d_ref = ref[1] - ref[0]
    # null 1: permute composition within division
    from joblib import Parallel, delayed
    perms = []
    for _ in range(n_null):
        perm = np.arange(len(y))
        for v in set(div):
            ix = np.where(div == v)[0]; perm[ix] = rng.permutation(ix)
        perms.append(perm)
    one = lambda yy, cc: (lambda ab: ab[1] - ab[0])(cv_r2(yy, base, cc, nsplit))
    d1 = Parallel(n_jobs=-1)(delayed(one)(y, comp[p]) for p in perms)
    # null 2: smooth surrogate targets
    S, kern = gp_surrogates(y, coords, div, n_null, rng)
    d2 = Parallel(n_jobs=-1)(delayed(one)(rankit(S[:, j]), comp) for j in range(n_null))
    d1, d2 = np.array(d1), np.array(d2)
    p1 = (1 + (d1 >= d_ref).sum()) / (n_null + 1)
    p2 = (1 + (d2 >= d_ref).sum()) / (n_null + 1)
    res = dict(n=len(y), r2_div=r_div, r2_base=r_base, r2_comp=r_comp, r2_full=r_full,
               delta=d_obs, delta3=d_ref, p_perm=p1, p_smooth=p2, kernel=kern,
               passed=bool(d_obs >= 0.01 and p2 < 0.05))
    if verbose:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))
    return res


# ---------------------------------------------------------------- simulation
def sim(args):
    rng = np.random.default_rng(1)
    n, ns = 140, 120
    coords = rng.uniform(0, 8, (n, 3))
    div = np.array(rng.choice(list("ABCDEFGH"), n))
    divm = {d: rng.normal(0, 1, ns) for d in set(div)}
    # smooth + division-structured composition (so composition is spatially confounded)
    W = rng.normal(0, 1, (3, ns))
    logit = np.array([divm[d] for d in div]) + (coords - 4) @ W * args.spatial + rng.normal(0, 1.0, (n, ns))
    P = np.exp(logit); P /= P.sum(1, keepdims=True); comp = np.sqrt(P)
    fake, plant = [], []
    for t in range(args.n_targets):
        r = np.random.default_rng(100 + t)
        # null target: division effect + smooth field, NO composition-specific part
        y0 = (np.array([{d: r.normal() for d in set(div)}[d] for d in div]) * 0
              + np.sin(coords @ r.normal(0, 0.4, 3)) + r.normal(0, 0.5, n))
        fake.append(gate(y0, comp, coords, div, args.n_null, seed=t, verbose=args.verbose_sim)["passed"])
        u = (comp - comp.mean(0)) / comp.std(0)
        w = np.zeros(ns); w[r.choice(ns, 5, replace=False)] = r.normal(0, 1, 5)
        sig = u @ w; sig = (sig - sig.mean()) / sig.std()
        y1 = np.sin(coords @ r.normal(0, 0.4, 3)) + args.plant * sig + r.normal(0, 0.5, n)
        plant.append(gate(y1, comp, coords, div, args.n_null, seed=t, verbose=args.verbose_sim)["passed"])
        print(f"target {t}: fake pass={fake[-1]}  planted pass={plant[-1]}", flush=True)
    if args.verbose_sim: pass
    print(f"FAKE pass rate {np.mean(fake):.3f}   PLANTED pass rate {np.mean(plant):.3f} "
          f"(plant strength {args.plant})")


# ---------------------------------------------------------------- data
def fetch(args):
    from abc_atlas_access.abc_atlas_cache.abc_project_cache import AbcProjectCache
    os.makedirs(args.cache, exist_ok=True)
    cache = AbcProjectCache.from_cache_dir(args.cache) if hasattr(AbcProjectCache, "from_cache_dir") \
        else AbcProjectCache.from_s3_cache(args.cache)
    get = lambda d, f: cache.get_metadata_dataframe(directory=d, file_name=f, dtype={"cell_label": str})
    df = get("MERFISH-C57BL6J-638850-CCF", "cell_metadata_with_parcellation_annotation")
    print("parcellation view columns:", list(df.columns))
    if "subclass" not in df.columns:
        cl = get("MERFISH-C57BL6J-638850", "cell_metadata_with_cluster_annotation")
        keep = ["cell_label"] + [c for c in ("class", "subclass", "supertype") if c in cl.columns]
        df = df.merge(cl[keep], on="cell_label", how="left")
    if not any(c in df.columns for c in ("x_ccf", "x")):
        cc = get("MERFISH-C57BL6J-638850-CCF", "ccf_coordinates")
        cc = cc.rename(columns={"x": "x_ccf", "y": "y_ccf", "z": "z_ccf"})
        df = df.merge(cc[["cell_label", "x_ccf", "y_ccf", "z_ccf"]], on="cell_label", how="left")
    keep = [c for c in df.columns if c in ("cell_label", "class", "subclass", "supertype",
            "parcellation_division", "parcellation_structure", "parcellation_substructure",
            "x_ccf", "y_ccf", "z_ccf", "x", "y", "z")]
    df[keep].to_pickle(os.path.join(args.cache, "abc_cells.pkl"))
    print("saved", len(df), "cells; kept columns:", keep)


def _col(df, *names):
    for n in names:
        if n in df.columns: return n
    raise KeyError(f"none of {names} in {list(df.columns)}")


def build_composition(cache, level):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    df = pd.read_pickle(os.path.join(cache, "abc_cells.pkl")).reset_index(drop=True)
    sub = _col(df, "parcellation_substructure", "parcellation_structure")
    acr = df[sub].astype(str).values
    u = np.unique(acr)
    ids = {}
    for a in u:
        try: ids[a] = int(br.acronym2id([a])[0])
        except Exception: ids[a] = 0
    allen = np.array([ids[a] for a in acr])
    ok = allen > 0
    beryl = np.full(len(df), "", object); cosmos = np.full(len(df), "", object)
    beryl[ok] = br.id2acronym(br.remap(allen[ok], "Allen", "Beryl"))
    cosmos[ok] = br.id2acronym(br.remap(allen[ok], "Allen", "Cosmos"))
    df = df.assign(beryl=beryl, cosmos=cosmos)
    df = df[ok & (df.beryl != "root") & (df.beryl != "void")]
    cx, cy, cz = _col(df, "x_ccf", "x"), _col(df, "y_ccf", "y"), _col(df, "z_ccf", "z")
    ml = (df[cz] - 5.7).abs()  # fold hemispheres (CCF z = ML, midline 5.7 mm)
    df = df.assign(_ap=df[cx], _dv=df[cy], _ml=ml)
    lv = _col(df, level)
    counts = pd.crosstab(df.beryl, df[lv])
    cent = df.groupby("beryl")[["_ap", "_dv", "_ml"]].median()
    cos = df.groupby("beryl").cosmos.agg(lambda s: s.mode().iloc[0])
    return counts, cent, cos


def inspect_(args):
    df = pd.read_pickle(os.path.join(args.cache, "abc_cells.pkl"))
    print(df.shape); print(df.dtypes.to_string())
    for c in df.columns:
        if df[c].dtype.kind in "fi": print(c, float(df[c].min()), float(df[c].max()))
    print(df.head(3).T.to_string())


def run(args):
    os.makedirs(OUT, exist_ok=True)
    counts, cent, cos = build_composition(args.cache, args.level)
    counts = counts[counts.sum(1) >= MIN_CELLS]
    P = counts.div(counts.sum(1), axis=0)
    P = P.loc[:, (P.mean(0) > 0.001)]
    report = {}
    for v in VARS:
        t = pd.read_csv(os.path.join(IBL_DIR, f"{v}.csv")).set_index("region")
        for metric in ("decoding_effect", "euclidean_effect"):
            regs = [r for r in t.index if r in P.index and np.isfinite(t.loc[r, metric])]
            y = t.loc[regs, metric].values.astype(float)
            comp = np.sqrt(P.loc[regs].values)
            print(f"\n== {v} / {metric}: {len(regs)} regions, {comp.shape[1]} {args.level} features")
            res = gate(y, comp, cent.loc[regs].values, cos.loc[regs].values, args.n_null)
            report[f"{v}/{metric}"] = res
            iso = cos.loc[regs].values == "Isocortex"
            if metric == "decoding_effect" and iso.sum() >= 30:
                print("   within Isocortex:")
                r = gate(y[iso], comp[iso], cent.loc[regs].values[iso],
                         np.array(["ctx"] * iso.sum()), args.n_null)
                report[f"{v}/{metric}/isocortex"] = r
    prim = [report[f"{v}/decoding_effect"]["passed"] for v in VARS]
    verdict = "PASS" if sum(prim) >= 2 else "FAIL"
    report["verdict"] = verdict
    with open(os.path.join(OUT, f"cellmap_gate_{args.level}.json"), "w") as f:
        json.dump(report, f, indent=1, default=str)
    print(f"\nPRIMARY passes (choice, stim, fback): {prim}  ->  GATE {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sim", "fetch", "inspect", "run"])
    ap.add_argument("--cache", default=os.path.expanduser("~/Downloads/CELLMAP/abc_cache"))
    ap.add_argument("--level", default="subclass")
    ap.add_argument("--n_null", type=int, default=300)
    ap.add_argument("--n_targets", type=int, default=20)
    ap.add_argument("--plant", type=float, default=0.6)
    ap.add_argument("--verbose_sim", action="store_true")
    ap.add_argument("--spatial", type=float, default=0.25)
    a = ap.parse_args()
    {"sim": sim, "fetch": fetch, "inspect": inspect_, "run": run}[a.step](a)
