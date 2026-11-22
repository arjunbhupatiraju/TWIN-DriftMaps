"""FEATURE RICHNESS TEST on the Druckmann/Svoboda MAP data (pre-registration: claude/preregistration-map-rich.md)

Reviewer objection: "the reanalysis used 7 window-averaged tracking features; the paper's richer video features would
explain more real variance, so less movement encoding would be lost under the valid analysis." The per-trial video
embeddings are not public, so this tests the objection directly: the same analysis with a much richer, time-resolved
description of movement inside every epoch.

RICH features, per trial and epoch, with the epoch window cut into 10 equal sub-windows:
  jaw x, jaw y, nose x, nose y (mean position in each sub-window)          40
  jaw speed, nose speed (mean in each sub-window)                          20
  tongue visible fraction (each sub-window)                                10
  left licks, right licks (each sub-window)                                20      = 90 features
Transform fixed in advance: positions clipped at the session 1st/99th percentiles; speeds, fractions and counts log1p then
clipped. Same trials, units, populations, epochs, CV, nulls and twins as part7/map_decompose.py; the registered 7-feature
set (MARKERS) is recomputed on the same populations so the comparison is paired.

    python3 part7/map_rich.py fetch --part 0 --of 4      (and parts 1-3 in three more windows; tracking only)
    python3 part7/map_rich.py run
    python3 part7/map_rich.py test
"""
from __future__ import annotations

import argparse, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "part6")); sys.path.insert(0, str(HERE.parent / "twin"))
import map_fetch as MF        # noqa: E402  (frozen step 3 extraction helpers)
import map_decompose as MD    # noqa: E402  (frozen step 3 trial/unit filters, marker features, per-neuron EV)
import decompose as DC        # noqa: E402

RICH = HERE / "results_map" / "rich"
OUT = HERE / "results_map"
K = 10


# ---------------------------------------------------------------- fetch (tracking only)
def _sub_means(t, x, lo, hi):
    """(trials x K) mean of x in K equal sub-windows of [lo, hi)."""
    edges = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, K + 1)[None, :]
    return np.column_stack([MF._wmean(t, x, edges[:, j], edges[:, j + 1]) for j in range(K)])


def _sub_counts(ev, lo, hi):
    edges = lo[:, None] + (hi - lo)[:, None] * np.linspace(0, 1, K + 1)[None, :]
    return np.column_stack([MF._count(ev, edges[:, j], edges[:, j + 1]) for j in range(K)])


def extract_rich(nwb):
    T = nwb.trials.to_dataframe()
    starts, stops = T.start_time.values.astype(float), T.stop_time.values.astype(float)
    ev = {k: np.asarray(ts.timestamps[:], float) for k, ts in nwb.acquisition["BehavioralEvents"].time_series.items()}
    so = MF._first_in(ev["sample_start_times"], starts, stops)
    sf = MF._first_in(ev["sample_stop_times"], starts, stops)
    go = MF._first_in(ev["go_start_times"], starts, stops)
    win = {"pre": (so - 0.5, so), "sample": (so, sf), "delay": (go - 1.0, go), "response": (go, go + 0.6)}
    trk = {}
    for name, ts in nwb.acquisition["BehavioralTimeSeries"].time_series.items():
        trk[name.split("_")[-1].replace("Tracking", "").lower()] = (np.asarray(ts.timestamps[:], float), np.asarray(ts.data[:], float))
    out = {"n_trials": len(T)}
    for e, (lo, hi) in win.items():
        ok = np.isfinite(lo) & np.isfinite(hi)
        lo_, hi_ = np.where(ok, lo, 0.0), np.where(ok, hi, 0.0)
        blocks, names = [], []
        for part in ("jaw", "nose"):
            if part not in trk:
                continue
            t, d = trk[part]; vis = d[:, 2] >= MF.LIK
            x, y = np.where(vis, d[:, 0], np.nan), np.where(vis, d[:, 1], np.nan)
            sp = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-4)]
            for lab, v in (("x", x), ("y", y), ("speed", sp)):
                blocks.append(_sub_means(t, v, lo_, hi_)); names += [f"{part}_{lab}_{j}" for j in range(K)]
        if "tongue" in trk:
            t, d = trk["tongue"]
            blocks.append(_sub_means(t, (d[:, 2] >= MF.LIK).astype(float), lo_, hi_)); names += [f"tongue_visible_{j}" for j in range(K)]
        for side in ("left", "right"):
            blocks.append(_sub_counts(ev.get(f"{side}_lick_times", []), lo_, hi_)); names += [f"licks_{side}_{j}" for j in range(K)]
        R = np.column_stack(blocks).astype(np.float32)
        R[~ok] = np.nan
        out[f"rich__{e}"] = R
        out[f"rich_names__{e}"] = np.array(names)
    return out


def cmd_fetch(a):
    from dandi.dandiapi import DandiAPIClient
    import h5py, pynwb, remfile
    RICH.mkdir(parents=True, exist_ok=True)
    with DandiAPIClient() as c:
        assets = sorted(c.get_dandiset(MF.DANDISET, MF.VERSION).get_assets(), key=lambda x: x.path)[a.part::a.of]
        jobs = [(x.path, x.get_content_url(follow_redirects=1, strip_query=True)) for x in assets]
    t0 = time.time()
    for i, (path, url) in enumerate(jobs):
        name = Path(path).stem.split("_behavior")[0]
        f = RICH / f"{name}.npz"
        if f.exists() or not (MD.SESS / f"{name}.npz").exists():
            print(f"[{i + 1}/{len(jobs)}] {name}: skip", flush=True); continue
        try:
            h = h5py.File(remfile.File(url), "r")
            d = extract_rich(pynwb.NWBHDF5IO(file=h, load_namespaces=True).read())
            tmp = RICH / f"{name}.part.npz"; np.savez_compressed(tmp, **d); tmp.rename(f)
            msg = f"{d['n_trials']} trials, {d['rich__pre'].shape[1]} rich features per epoch"
        except Exception as e:
            msg = f"FAILED ({type(e).__name__}: {e}); run again to retry"
        print(f"[{i + 1}/{len(jobs)}] {name}: {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------- analysis
def rich_movement(r, e, keep):
    R = np.asarray(r[f"rich__{e}"], float)[keep]; names = r[f"rich_names__{e}"]
    cols = []
    for j, nm in enumerate(names):
        x = R[:, j]
        if np.isfinite(x).mean() < 0.8:
            continue
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        if not (nm.endswith(tuple(f"_x_{i}" for i in range(K))) or nm.endswith(tuple(f"_y_{i}" for i in range(K)))):
            x = np.log1p(np.maximum(x, 0))
        lo, hi = np.percentile(x, [1, 99]); x = np.clip(x, lo, hi)
        if x.std() > 0:
            cols.append(x)
    return np.column_stack(cols) if cols else None


def _session(path, n_shuf, max_shift, n_twin, seed):
    from twin.drift import phase_twin
    rng = np.random.default_rng(seed)
    z = np.load(path, allow_pickle=False); r = np.load(RICH / Path(path).name, allow_pickle=False)
    if int(r["n_trials"]) != len(z["trial__outcome"]):
        return []
    keep = MD._clean_trials(z); T = int(keep.sum())
    if T < 200:
        return []
    ins = np.where(z["trial__instruction"] == "right", 1.0, -1.0); hit = np.where(z["trial__outcome"] == "hit", 1.0, -1.0)
    Tk = np.column_stack([ins, hit, np.r_[0, ins[:-1]], np.r_[0, hit[:-1]]])[keep]
    D = DC.drift(T); S = 2 * (min(max_shift, T // 6) // 2)
    uq = z["unit__unit_quality"] if "unit__unit_quality" in z.files else np.array(["good"] * len(z["unit__probe"]))
    good = uq == "good"
    if "unit__presence_ratio" in z.files:
        good &= np.nan_to_num(z["unit__presence_ratio"], nan=1.0) >= 0.9
    region = np.array([t.replace("left ", "").replace("right ", "").strip() for t in z["unit__target"]])
    rows = []
    for e in MD.EPOCHS:
        Mm, _ = MD._movement(z, e, keep); Mr = rich_movement(r, e, keep)
        C = np.asarray(z[f"counts__{e}"], float)[keep]
        if Mm is None or Mr is None or not np.isfinite(C).all():
            continue
        for pr in np.unique(z["unit__probe"]):
            idx = np.nonzero((z["unit__probe"] == pr) & good)[0]
            X = np.sqrt(np.maximum(C[:, idx], 0)); ok = X.std(0) > 0; idx, X = idx[ok], X[:, ok]
            if len(idx) < 10:
                continue
            if len(idx) > 150:
                pick = np.sort(rng.choice(len(idx), 150, replace=False)); X = X[:, pick]
            Y = (X - X.mean(0)) / X.std(0)
            row = dict(session=Path(path).stem, subject=str(z["subject"]), region=region[idx[0]], probe=str(pr), window=e,
                       T=T, N=Y.shape[1], n_rich=Mr.shape[1], n_markers=Mm.shape[1])
            for lab, M in (("markers", Mm), ("rich", Mr)):                  # paired: same population, two feature sets
                Yd, Td, Md = DC.detrend(Y, D), DC.detrend(Tk, D), DC.detrend(M, D)
                full = DC.cv_r2(np.column_stack([Td, Md]), Yd)
                row[f"dM_{lab}"] = full - DC.cv_r2(Td, Yd)
                row[f"dM_std_{lab}"] = DC.cv_r2(np.column_stack([Tk, M]), Y, contiguous=False) - DC.cv_r2(Tk, Y, contiguous=False)
            fs = fv = 0                                                     # validity with the rich features
            for _ in range(n_twin):
                Yt = phase_twin(Y, rng); Yt = (Yt - Yt.mean(0)) / Yt.std(0)
                o = DC.tests(Yt, Tk, Mr, D, S, n_shuf, rng)
                fs += o["p_std_M"] < 0.05; fv += o["p_valid_M"] < 0.05
            row.update(twin_fp_std=fs / max(n_twin, 1), twin_fp_valid=fv / max(n_twin, 1), n_twin=n_twin)
            rows.append(row)
    return rows


def cmd_run(a):
    import pandas as pd
    rows_f = OUT / "map_rich_rows.csv"
    done = set(pd.read_csv(rows_f).session) if rows_f.exists() else set()
    paths = [p for p in sorted(MD.SESS.glob("*.npz")) if not p.name.endswith(".part.npz")
             and (RICH / p.name).exists() and p.stem not in done][: a.limit or None]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.shuffles, a.max_shift, a.twins, 16000 + i): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True); continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} probe-epochs", flush=True)


def cmd_test(a):
    import pandas as pd
    D = pd.read_csv(OUT / "map_rich_rows.csv")
    rng = np.random.default_rng(0); B = a.boot
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    P(f"# FEATURE RICHNESS (MAP): {D.session.nunique()} sessions, {len(D)} probe-epochs; median {int(D.n_rich.median())} rich "
      f"vs {int(D.n_markers.median())} marker features\n")
    ins = {k: g for k, g in D.groupby(["session", "probe"])}; keys = list(ins)
    boot = [pd.concat([ins[keys[i]] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(B)]
    share = lambda g, l: 1 - g[f"dM_{l}"].sum() / g[f"dM_std_{l}"].sum() if g[f"dM_std_{l}"].sum() > 0 else np.nan
    gain = [g.dM_std_rich.mean() - g.dM_std_markers.mean() for g in boot]
    g_lo = np.percentile(gain, 2.5)
    vgain = [g.dM_rich.mean() - g.dM_markers.mean() for g in boot]
    P("## RF0 (descriptive: do the rich features explain more neural variance?)")
    P(f"  standard unique movement R2: markers {D.dM_std_markers.mean():+.4f}, rich {D.dM_std_rich.mean():+.4f}; "
      f"gain {D.dM_std_rich.mean() - D.dM_std_markers.mean():+.4f} (95% CI {g_lo:+.4f} to {np.percentile(gain, 97.5):+.4f})")
    P(f"  VALID unique movement R2: markers {D.dM_markers.mean():+.4f}, rich {D.dM_rich.mean():+.4f}; gain "
      f"{D.dM_rich.mean() - D.dM_markers.mean():+.4f} (95% CI {np.percentile(vgain, 2.5):+.4f} to {np.percentile(vgain, 97.5):+.4f})")
    P("  (a valid-R2 gain is consistent with the rich features capturing more real movement signal; descriptive only)")
    sr, sm = share(D, "rich"), share(D, "markers")
    br = [share(g, "rich") for g in boot]; bd = [share(g, "rich") - share(g, "markers") for g in boot]
    lo, hi = np.nanpercentile(br, [2.5, 97.5])
    P("\n## RF1 (confirmatory: with rich features, does movement encoding still fail to survive the valid analysis?)")
    P(f"  share lost, rich features: {sr:.3f} (95% CI {lo:.3f}-{hi:.3f}; lower > 0.10)  RF1 {'PASS' if lo > 0.10 else 'FAIL'}"
      + ("" if 0 <= sr <= 1 else "  (OUT OF RANGE)"))
    P(f"  share lost, markers (same populations): {sm:.3f}; difference rich - markers {sr - sm:+.3f} "
      f"(95% CI {np.nanpercentile(bd, 2.5):+.3f} to {np.nanpercentile(bd, 97.5):+.3f})")
    for w, g in D.groupby("window"):
        P(f"    {w}: rich {share(g, 'rich'):.3f}, markers {share(g, 'markers'):.3f}")
    for r_, g in D.groupby("region"):
        P(f"    {r_}: rich {share(g, 'rich'):.3f}, markers {share(g, 'markers'):.3f}")
    rate = lambda g, c: float((g[c] * g.n_twin).sum() / max(g.n_twin.sum(), 1))
    ss = [g for _, g in D.groupby("session")]
    bs = [pd.concat([ss[i] for i in rng.integers(0, len(ss), len(ss))]) for _ in range(B)]
    fv, fs = rate(D, "twin_fp_valid"), rate(D, "twin_fp_std")
    hv = float(np.percentile([rate(g, "twin_fp_valid") for g in bs], 97.5)); ls = float(np.percentile([rate(g, "twin_fp_std") for g in bs], 2.5))
    P("\n## RF2 (validity with rich features, drift twins)")
    P(f"  valid {fv:.3f} (upper {hv:.3f}; <= 0.07 and <= 0.08)  RF2a {'PASS' if fv <= 0.07 and hv <= 0.08 else 'FAIL'}")
    P(f"  standard {fs:.3f} (lower {ls:.3f}; >= 0.20)  RF2b {'PASS' if ls >= 0.20 else 'FAIL'}")
    (OUT / "MAP_RICH.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/MAP_RICH.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--part", type=int, default=0); f.add_argument("--of", type=int, default=1)
    r = sub.add_parser("run")
    r.add_argument("--shuffles", type=int, default=100); r.add_argument("--max-shift", type=int, default=50)
    r.add_argument("--twins", type=int, default=3); r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t = sub.add_parser("test"); t.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    {"fetch": cmd_fetch, "run": cmd_run, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
