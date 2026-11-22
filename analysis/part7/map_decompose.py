"""STEP 3, analysis: the step 2 tests on the Druckmann/Svoboda MAP data (pre-registration:
claude/preregistration-map-reanalysis.md). Input: results_map/sessions/*.npz from map_fetch.py.

Per session x probe insertion (one brain area) x epoch (pre, sample, delay, response), on clean trials:
  TASK      instruction (lick left/right), outcome (hit/miss), previous instruction, previous outcome
  MOVEMENT  7 features from the paper's own DeepLabCut tracking: jaw speed, jaw height, nose speed, tongue visible,
            tongue speed, left licks, right licks (robust transform fixed in advance: log1p of speeds and counts,
            then clipping at the session 1st/99th percentiles)
  DRIFT     slow cosine basis over trials (periods >= 200 trials)
  - population tests exactly as step 2 (standard vs valid, drift twins): MA1, MA2
  - per neuron: unique movement explained variance, standard (interleaved CV, drift left in) and valid (contiguous
    CV on drift-removed data), and the standard value on a drift twin (no real link to movement): MA3 and descriptives

    python3 part7/map_decompose.py run --limit 3
    python3 part7/map_decompose.py run
    python3 part7/map_decompose.py test
"""
from __future__ import annotations

import argparse, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "part6")); sys.path.insert(0, str(HERE.parent / "twin"))
import prior_map as PM   # noqa: E402
import decompose as DC   # noqa: E402

SESS = HERE / "results_map" / "sessions"
OUT = HERE / "results_map"
EPOCHS = ("pre", "sample", "delay", "response")
MOVE = ("jaw_speed", "jaw_y", "nose_speed", "tongue_visible", "tongue_speed", "licks_left", "licks_right")


# ---------------------------------------------------------------- per-neuron encoding R2
def r2_neurons(P, Y, contiguous=True, k=5, seed=0):
    """Held-out R2 per neuron (same ridge/GCV/folds as decompose.cv_r2)."""
    T, N = Y.shape
    if contiguous:
        e = np.linspace(0, T, k + 1).astype(int); folds = [np.arange(a, b) for a, b in zip(e[:-1], e[1:])]
    else:
        folds = np.array_split(np.random.default_rng(seed).permutation(T), k)
    sse = np.zeros(N); sst = np.zeros(N)
    for te in folds:
        tr = np.setdiff1d(np.arange(T), te)
        ym = Y[tr].mean(0); Ytr = Y[tr] - ym; Yte = Y[te] - ym
        sst += (Yte ** 2).sum(0)
        mu, sd = P[tr].mean(0), P[tr].std(0); keep = sd > 0
        if P.shape[1] == 0 or not keep.any():
            sse += (Yte ** 2).sum(0); continue
        Ptr = (P[tr][:, keep] - mu[keep]) / sd[keep]; Pte = (P[te][:, keep] - mu[keep]) / sd[keep]
        U, sv, Vt = np.linalg.svd(Ptr, full_matrices=False)
        Uy = U.T @ Ytr; n = len(tr); tot = (Ytr ** 2).sum(0); proj = (Uy ** 2).sum(0); PV = Pte @ Vt.T
        best_g = np.full(N, np.inf); best = np.zeros_like(Yte)
        for al in DC.ALPHAS * n:
            f = sv ** 2 / (sv ** 2 + al)
            rss = tot - proj + (((1 - f) ** 2)[:, None] * Uy ** 2).sum(0)
            g = n * rss / max(n - f.sum(), 1e-9) ** 2
            pred = PV @ ((sv / (sv ** 2 + al))[:, None] * Uy)
            take = g < best_g
            best[:, take] = pred[:, take]; best_g = np.where(take, g, best_g)
        sse += ((Yte - best) ** 2).sum(0)
    return 1 - sse / np.maximum(sst, 1e-12)


def ev_standard(Y, Tk, M):
    return r2_neurons(np.column_stack([Tk, M]), Y, False) - r2_neurons(Tk, Y, False)


def ev_valid(Y, Tk, M, D):
    Yd, Td, Md = DC.detrend(Y, D), DC.detrend(Tk, D), DC.detrend(M, D)
    return r2_neurons(np.column_stack([Td, Md]), Yd) - r2_neurons(Td, Yd)


# ---------------------------------------------------------------- one session
def _clean_trials(z):
    task = z["trial__task"]; out = z["trial__outcome"]
    keep = (np.char.find(task.astype(str), "audio delay") >= 0) & (~z["trial__photostim"].astype(bool))
    keep &= (z["trial__auto_water"] == 0) & (z["trial__free_water"] == 0)
    keep &= np.isin(out, ["hit", "miss"]) & (z["trial__early_lick"] == "no early")
    return keep


def _movement(z, e, keep):
    cols, names = [], []
    for k in MOVE:
        key = f"move__{e}__{k}"
        if key not in z.files:
            continue
        x = np.asarray(z[key], float)[keep]
        if np.isfinite(x).mean() < 0.8:
            continue
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        if k != "jaw_y":
            x = np.log1p(np.maximum(x, 0))
        lo, hi = np.percentile(x, [1, 99]); x = np.clip(x, lo, hi)
        if x.std() > 0:
            cols.append(x); names.append(k)
    return (np.column_stack(cols) if cols else None), names


def _session(path, n_shuf, max_shift, n_twin, seed):
    from twin.drift import phase_twin
    rng = np.random.default_rng(seed)
    z = np.load(path, allow_pickle=False)
    keep = _clean_trials(z)
    T = int(keep.sum())
    if T < 200:
        return [], []
    ins = np.where(z["trial__instruction"] == "right", 1.0, -1.0)
    hit = np.where(z["trial__outcome"] == "hit", 1.0, -1.0)
    Tk_all = np.column_stack([ins, hit, np.r_[0, ins[:-1]], np.r_[0, hit[:-1]]])
    Tk, D = Tk_all[keep], DC.drift(T)
    S = 2 * (min(max_shift, T // 6) // 2)
    uq = z["unit__unit_quality"] if "unit__unit_quality" in z.files else np.array(["good"] * len(z["unit__probe"]))
    good = (uq == "good")
    if "unit__presence_ratio" in z.files:
        good &= np.nan_to_num(z["unit__presence_ratio"], nan=1.0) >= 0.9
    region = np.array([t.replace("left ", "").replace("right ", "").strip() for t in z["unit__target"]])
    rows, neur = [], []
    for e in EPOCHS:
        M, names = _movement(z, e, keep)
        C = np.asarray(z[f"counts__{e}"], float)[keep]
        if M is None or not np.isfinite(C).all():
            continue
        for pr in np.unique(z["unit__probe"]):
            idx = np.nonzero((z["unit__probe"] == pr) & good)[0]
            X = np.sqrt(np.maximum(C[:, idx], 0)); ok = X.std(0) > 0; idx, X = idx[ok], X[:, ok]
            if len(idx) < 10:
                continue
            if len(idx) > 150:
                pick = np.sort(rng.choice(len(idx), 150, replace=False)); idx, X = idx[pick], X[:, pick]
            Y = (X - X.mean(0)) / X.std(0)
            reg = region[idx[0]]
            row = dict(session=Path(path).stem, subject=str(z["subject"]), lab="MAP", region=reg, probe=str(pr), window=e,
                       T=T, N=len(idx), n_feat=len(names), video=1, feats="|".join(names))
            row.update(DC.tests(Y, Tk, M, D, S, n_shuf, rng))
            fs = fv = 0
            for _ in range(n_twin):
                Yt = phase_twin(Y, rng); Yt = (Yt - Yt.mean(0)) / Yt.std(0)
                o = DC.tests(Yt, Tk, M, D, S, n_shuf, rng)
                fs += o["p_std_M"] < 0.05; fv += o["p_valid_M"] < 0.05
            row.update(twin_fp_std=fs / max(n_twin, 1), twin_fp_valid=fv / max(n_twin, 1), n_twin=n_twin)
            rows.append(row)
            Yt = phase_twin(Y, rng); Yt = (Yt - Yt.mean(0)) / Yt.std(0)
            es, ev, et = ev_standard(Y, Tk, M), ev_valid(Y, Tk, M, D), ev_standard(Yt, Tk, M)
            for j, u in enumerate(idx):
                neur.append(dict(session=Path(path).stem, probe=str(pr), region=reg, window=e, unit=int(u),
                                 posy=float(z["unit__unit_posy"][u]) if "unit__unit_posy" in z.files else np.nan,
                                 drift_metric=float(z["unit__drift_metric"][u]) if "unit__drift_metric" in z.files else np.nan,
                                 cumulative_drift=float(z["unit__cumulative_drift"][u]) if "unit__cumulative_drift" in z.files else np.nan,
                                 anno=str(z["unit__anno_name"][u]) if "unit__anno_name" in z.files else "",
                                 ev_std=float(es[j]), ev_valid=float(ev[j]), ev_twin=float(et[j])))
    return rows, neur


def cmd_run(a):
    import pandas as pd
    rows_f, neur_f = OUT / "map_rows.csv", OUT / "map_neurons.csv"
    done = set(pd.read_csv(rows_f).session) if rows_f.exists() else set()
    paths = [p for p in sorted(SESS.glob("*.npz")) if not p.name.endswith(".part.npz")][: a.limit or None]
    paths = [p for p in paths if p.stem not in done]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.shuffles, a.max_shift, a.twins, 14000 + i): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows, neur = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True); continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
                pd.DataFrame(neur).to_csv(neur_f, mode="a", header=not neur_f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} probe-epochs, {len(neur)} neuron-epochs", flush=True)


# ---------------------------------------------------------------- test (pre-registered)
def smoothness(g, col, k=5, n_perm=200, seed=0):
    """Spatial smoothness along the probe: Spearman correlation between each unit's value and the mean of its k
    nearest units (by depth), MINUS the same statistic with the values shuffled across positions (which removes the
    small-sample bias of a neighbour-mean correlation). 0 = no spatial structure."""
    from scipy.stats import rankdata
    g = g.dropna(subset=["posy", col])
    if len(g) < 3 * k:
        return np.nan
    y = g.posy.values; v = rankdata(g[col].values)
    nbr = np.array([np.argsort(np.abs(y - y[i]), kind="stable")[1:k + 1] for i in range(len(g))])
    rho = lambda x: np.corrcoef(x, rankdata(x[nbr].mean(1)))[0, 1]
    rng = np.random.default_rng(seed)
    return float(rho(v) - np.mean([rho(rng.permutation(v)) for _ in range(n_perm)]))


def cmd_test(a):
    import pandas as pd
    from scipy.stats import spearmanr
    D = pd.read_csv(OUT / "map_rows.csv"); Nn = pd.read_csv(OUT / "map_neurons.csv")
    rng = np.random.default_rng(0); B = a.boot
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    P(f"# MAP REANALYSIS: {D.session.nunique()} sessions, {D.subject.nunique()} mice, {len(D.groupby(['session', 'probe']))} probe insertions, "
      f"{len(D)} probe-epochs, {len(Nn)} neuron-epochs; regions {sorted(D.region.unique())}\n")
    G = D.session.nunique() >= 30 and D.region.nunique() >= 3
    P(f"## Gate: >= 30 sessions and >= 3 target areas: {'PASS' if G else 'FAIL'}\n")
    sess = [g for _, g in D.groupby("session")]
    bs = [pd.concat([sess[i] for i in rng.integers(0, len(sess), len(sess))]) for _ in range(B)]
    rate = lambda g, c: float((g[c] * g.n_twin).sum() / max(g.n_twin.sum(), 1))
    fv, fs = rate(D, "twin_fp_valid"), rate(D, "twin_fp_std")
    hv = float(np.percentile([rate(g, "twin_fp_valid") for g in bs], 97.5))
    ls = float(np.percentile([rate(g, "twin_fp_std") for g in bs], 2.5))
    P("## MA1 (movement-encoding false positives on drift twins of the MAP neurons)")
    P(f"  valid {fv:.3f} (upper {hv:.3f}; <= 0.07 and <= 0.08)  MA1a {'PASS' if fv <= 0.07 and hv <= 0.08 else 'FAIL'}")
    P(f"  standard {fs:.3f} (lower {ls:.3f}; >= 0.20)  MA1b {'PASS' if ls >= 0.20 else 'FAIL'}")
    for w, g in D.groupby("window"):
        P(f"    {w}: valid {rate(g, 'twin_fp_valid'):.3f}, standard {rate(g, 'twin_fp_std'):.3f}")
    share = lambda g: 1 - g.dM.sum() / g.dM_std.sum() if g.dM_std.sum() > 0 else np.nan
    byp = {p: g for p, g in D.groupby(["session", "probe"])}; probes = list(byp)
    bt = [share(pd.concat([byp[probes[i]] for i in rng.integers(0, len(probes), len(probes))])) for _ in range(B)]
    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
    P("\n## MA2 (replication of D2 with the pre-registered robust movement predictors)")
    P(f"  share of standard movement encoding that does not survive the valid analysis: {share(D):.3f} "
      f"(95% CI {lo:.3f}-{hi:.3f}; lower > 0.10)  MA2 {'PASS' if lo > 0.10 else 'FAIL'}")
    for w, g in D.groupby("window"):
        P(f"    {w}: {share(g):.3f}")
    for r, g in D.groupby("region"):
        P(f"    {r}: {share(g):.3f} (standard R2 {g.dM_std.mean():+.4f}, valid {g.dM.mean():+.4f}, {len(g.groupby(['session', 'probe']))} insertions)")
    # MA3: spatial smoothness of per-neuron movement EV along the probe
    sm = []
    for (s, p, w), g in Nn.groupby(["session", "probe", "window"]):
        sm.append(dict(session=s, probe=p, window=w, std=smoothness(g, "ev_std"), valid=smoothness(g, "ev_valid"),
                       twin=smoothness(g, "ev_twin")))
    SM = pd.DataFrame(sm).dropna()
    ss = [g for _, g in SM.groupby("session")]
    bm = [pd.concat([ss[i] for i in rng.integers(0, len(ss), len(ss))])[["std", "valid", "twin"]].mean() for _ in range(B)]
    ci = lambda c: np.percentile([b[c] for b in bm], [2.5, 97.5])
    P("\n## MA3 (does slow drift alone make movement-encoding maps look spatially smooth along the probe?)")
    for c, lab in (("std", "real neurons, standard EV"), ("valid", "real neurons, valid EV"),
                   ("twin", "drift twins (no movement link), standard EV")):
        l_, h_ = ci(c)
        P(f"  {lab}: mean smoothness above shuffled positions {SM[c].mean():+.3f} (95% CI {l_:+.3f} to {h_:+.3f})")
    l_t = ci("twin")[0]
    P(f"  MA3 {'PASS' if l_t > 0 else 'FAIL'} (criterion: drift-twin smoothness CI lower bound > 0)")
    P(f"  descriptive: twin smoothness / real standard smoothness = {SM.twin.mean() / SM['std'].mean():.2f}")
    # descriptive: areas, drift metrics, neurons
    P("\n## Descriptive")
    A = Nn.groupby("region")[["ev_std", "ev_valid"]].mean().sort_values("ev_std", ascending=False)
    P("  mean per-neuron unique movement EV by area (standard -> valid):")
    for r, x in A.iterrows():
        P(f"    {r}: {x.ev_std:+.4f} -> {x.ev_valid:+.4f}")
    P(f"  area ranking, standard vs valid: Spearman {spearmanr(A.ev_std, A.ev_valid)[0]:+.2f}")
    for dm in ("drift_metric", "cumulative_drift"):
        if Nn[dm].notna().any():
            rr = [spearmanr(g[dm], g.ev_std - g.ev_valid)[0] for _, g in Nn.groupby(["session", "probe", "window"])
                  if g[dm].notna().sum() > 10 and g[dm].std() > 0]
            P(f"  units with more Kilosort {dm} lose more EV under the valid analysis: median within-probe Spearman "
              f"{np.nanmedian(rr):+.3f}, positive in {np.mean(np.array(rr) > 0):.0%} of {len(rr)} probe-epochs")
    P(f"  neurons with positive unique movement EV: standard {np.mean(Nn.ev_std > 0):.0%}, valid {np.mean(Nn.ev_valid > 0):.0%}")
    SM.to_csv(OUT / "map_smoothness.csv", index=False)
    (OUT / "MAP_REANALYSIS.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/MAP_REANALYSIS.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--shuffles", type=int, default=100); r.add_argument("--max-shift", type=int, default=50)
    r.add_argument("--twins", type=int, default=3); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t = sub.add_parser("test"); t.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    {"run": cmd_run, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
