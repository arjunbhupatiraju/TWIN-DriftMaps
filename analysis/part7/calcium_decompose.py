"""STEP 4, analysis: do the movement-encoding results hold for calcium imaging?
(pre-registration: claude/preregistration-calcium.md)

Data: Stringer, Pachitariu et al., Science 2019, about 10,000 neurons per recording in mouse visual cortex, two-photon
(deconvolved), spontaneous behaviour, with face-video motion SVDs, running speed and pupil area. That paper reported that
facial behaviour predicts a large share of brain-wide spontaneous activity. Here the same question is asked with the
standard and the drift-valid tests used in steps 2 and 3.

Per recording: activity summed in bins of 4 frames; neurons ordered by imaging plane and position and cut into
populations of 150 neighbouring neurons (up to 40 per recording, chosen at random); movement = first 20 face motion SVD
components + running speed + pupil area (each clipped at its 1st/99th percentile); drift = slow cosine basis over the
recording (periods >= 200 bins). There is no task.
  standard  interleaved 5-fold CV, drift left in, null = 100 shuffles of the movement time points
  valid     contiguous 5-fold CV on drift-removed data, null = 50 linear shifts of the movement relative to the neurons
            (every 16th shift from -800 to +800 bins, never closer than 16 bins; at most T/6)
  twins     3 phase-randomized drift twins per population (no real link to behaviour)

    python3 part7/calcium_decompose.py run
    python3 part7/calcium_decompose.py test
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

RAW = HERE / "results_calcium" / "raw"
OUT = HERE / "results_calcium"
BIN, NPC, POP, MAXPOP = 4, 20, 150, 40
MAX_SHIFT, STEP, PERIOD = 800, 16, 200


# ---------------------------------------------------------------- loading
def _get(obj, *path):
    for p in path:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    return obj


def load(path):
    """Fsp (neurons x frames), behaviour matrix (frames x features), neuron positions (neurons x 3)."""
    try:
        from scipy.io import loadmat
        mt = loadmat(path, squeeze_me=True, struct_as_record=False)
        F = np.asarray(mt["Fsp"], np.float32); med = np.asarray(mt["med"], float)   # float32: half the memory
        beh = mt["beh"]
        svd = np.asarray(_get(beh, "face", "motionSVD"), float)
        run = np.asarray(_get(beh, "runSpeed"), float).ravel()
        try:
            pup = np.asarray(_get(beh, "pupil", "area"), float).ravel()
        except Exception:
            pup = None
    except NotImplementedError:                                    # MATLAB v7.3 files
        import h5py
        h = h5py.File(path, "r")
        F = np.asarray(h["Fsp"], np.float32).T; med = np.asarray(h["med"], float).T
        svd = np.asarray(h["beh/face/motionSVD"], float).T
        run = np.asarray(h["beh/runSpeed"], float).ravel()
        pup = np.asarray(h["beh/pupil/area"], float).ravel() if "beh/pupil/area" in h else None
    if svd.shape[0] != F.shape[1] and svd.shape[1] == F.shape[1]:
        svd = svd.T
    cols = [svd[:, :NPC], run[:, None]] + ([pup[:, None]] if pup is not None and len(pup) == len(run) else [])
    n = min(F.shape[1], *(len(c) for c in cols))
    return F[:, :n], np.column_stack([c[:n] for c in cols]), med


def binned(F, B):
    n = (F.shape[1] // BIN) * BIN
    X = F[:, :n].reshape(F.shape[0], -1, BIN).sum(2, dtype=np.float64).T     # bins x neurons
    M = np.nanmean(B[:n].reshape(-1, BIN, B.shape[1]), 1)                   # bins x features
    keep = []
    for j in range(M.shape[1]):
        x = M[:, j]
        if np.isfinite(x).mean() < 0.8:
            continue
        x = np.where(np.isfinite(x), x, np.nanmedian(x))
        lo, hi = np.percentile(x, [1, 99]); x = np.clip(x, lo, hi)
        if x.std() > 0:
            keep.append(x)
    return X, np.column_stack(keep)


def drift(T):
    K = max(1, int(2 * T // PERIOD))                                        # periods 2T/k >= PERIOD bins
    t = (np.arange(T) + 0.5) / T
    return np.column_stack([np.cos(np.pi * k * t) for k in range(1, K + 1)])


# ---------------------------------------------------------------- tests (no task predictors)
def tests(Y, M, D, S, n_shuf, rng):
    T = len(Y)
    out = {}
    Yd, Md = DC.detrend(Y, D), DC.detrend(M, D)
    out["R2_valid"] = DC.cv_r2(Md, Yd)                                       # unique movement R2 (no task)
    out["R2_std"] = DC.cv_r2(M, Y, contiguous=False)
    out["R2_nodrift"] = DC.cv_r2(M, Y)
    out["drift_share"] = float(1 - (Yd ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    nul = [DC.cv_r2(M[rng.permutation(T)], Y, contiguous=False) for _ in range(n_shuf)]
    out["p_std"] = PM._p(out["R2_std"], nul)
    c = slice(S, T - S)
    obs = DC.cv_r2(Md[c], Yd[c])
    nul = [DC.cv_r2(Md[S + s:T - S + s], Yd[c]) for s in range(-S, S + 1, STEP) if abs(s) >= STEP]
    out["p_valid"] = PM._p(obs, nul)
    return out


def _session(path, n_shuf, n_twin, seed, cfg=None):
    global MAX_SHIFT, STEP, PERIOD
    if cfg:
        MAX_SHIFT, STEP, PERIOD = cfg
    from twin.drift import phase_twin
    rng = np.random.default_rng(seed)
    F, B, med = load(path)
    X, M = binned(F, B)
    del F                                                                   # free the raw frames before testing
    T = len(X)
    D = drift(T); S = STEP * (min(MAX_SHIFT, T // 6) // STEP)
    order = np.lexsort((med[:, 1], med[:, 0], med[:, 2]))                   # by plane, then x, then y
    pops = [order[i:i + POP] for i in range(0, len(order) - POP + 1, POP)]
    pick = rng.choice(len(pops), min(MAXPOP, len(pops)), replace=False)
    rows = []
    for k in pick:
        idx = pops[k]
        Xp = np.sqrt(np.maximum(X[:, idx], 0)); Xp = Xp[:, Xp.std(0) > 0]
        if Xp.shape[1] < 10:
            continue
        Y = (Xp - Xp.mean(0)) / Xp.std(0)
        row = dict(session=Path(path).stem, population=int(k), T=T, N=Y.shape[1], n_feat=M.shape[1],
                   depth=float(np.median(med[idx, 2])))
        row.update(tests(Y, M, D, S, n_shuf, rng))
        fs = fv = 0
        for _ in range(n_twin):
            Yt = phase_twin(Y, rng); Yt = (Yt - Yt.mean(0)) / Yt.std(0)
            o = tests(Yt, M, D, S, n_shuf, rng)
            fs += o["p_std"] < 0.05; fv += o["p_valid"] < 0.05
        row.update(twin_fp_std=fs / max(n_twin, 1), twin_fp_valid=fv / max(n_twin, 1), n_twin=n_twin)
        rows.append(row)
    return rows


def cmd_run(a):
    import pandas as pd
    rows_f = OUT / "calcium_rows.csv"
    done = set(pd.read_csv(rows_f).session) if rows_f.exists() else set()
    paths = [p for p in sorted(RAW.glob("spont_*.mat")) if p.stem not in done]
    print(f"{len(paths)} recordings to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.shuffles, a.twins, 15000 + i, (MAX_SHIFT, STEP, PERIOD)): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {type(e).__name__}: {e}", flush=True); continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} populations", flush=True)


def cmd_test(a):
    import pandas as pd
    D = pd.read_csv(OUT / "calcium_rows.csv")
    rng = np.random.default_rng(0); B = a.boot
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    P(f"# CALCIUM (Stringer et al. 2019 two-photon): {D.session.nunique()} recordings, {len(D)} populations of "
      f"<= {POP} neurons, median {int(D['T'].median())} bins\n")
    G = D.session.nunique() >= 5
    P(f"## Gate: >= 5 recordings: {'PASS' if G else 'FAIL'}\n")
    groups = [g for _, g in D.groupby("session")]
    bs = [pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(B)]
    rate = lambda g, c: float((g[c] * g.n_twin).sum() / max(g.n_twin.sum(), 1))
    fv, fs = rate(D, "twin_fp_valid"), rate(D, "twin_fp_std")
    hv = float(np.percentile([rate(g, "twin_fp_valid") for g in bs], 97.5))
    ls = float(np.percentile([rate(g, "twin_fp_std") for g in bs], 2.5))
    P("## CA1 (behaviour-encoding false positives on drift twins of the imaged neurons)")
    P(f"  valid {fv:.3f} (upper {hv:.3f}; <= 0.07 and <= 0.08)  CA1a {'PASS' if fv <= 0.07 and hv <= 0.08 else 'FAIL'}")
    P(f"  standard {fs:.3f} (lower {ls:.3f}; >= 0.20)  CA1b {'PASS' if ls >= 0.20 else 'FAIL'}")
    share = lambda g: 1 - g.R2_valid.sum() / g.R2_std.sum() if g.R2_std.sum() > 0 else np.nan
    bt = [share(g) for g in bs]
    lo, hi = np.nanpercentile(bt, [2.5, 97.5])
    P("\n## CA2 (share of the behaviour-explained variance found by the standard analysis that does not survive)")
    P(f"  {share(D):.3f} (95% CI {lo:.3f}-{hi:.3f}; lower > 0.10)  CA2 {'PASS' if lo > 0.10 else 'FAIL'}"
      + ("" if 0 <= share(D) <= 1 else "  (OUT OF RANGE)"))
    P(f"  variance explained by behaviour (population R2): standard {D.R2_std.mean():.3f}, contiguous without drift removal "
      f"{D.R2_nodrift.mean():.3f}, valid {D.R2_valid.mean():.3f}; slow-drift share of activity {D.drift_share.mean():.3f}")
    P("\n## Descriptive")
    P(f"  populations with significant behaviour encoding: standard {np.mean(D.p_std < 0.05):.0%}, valid {np.mean(D.p_valid < 0.05):.0%}")
    for s, g in D.groupby("session"):
        P(f"    {s}: R2 standard {g.R2_std.mean():.3f} -> valid {g.R2_valid.mean():.3f}; twins: standard "
          f"{rate(g, 'twin_fp_std'):.2f}, valid {rate(g, 'twin_fp_valid'):.2f}")
    (OUT / "CALCIUM.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/CALCIUM.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--shuffles", type=int, default=100); r.add_argument("--twins", type=int, default=3)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    t = sub.add_parser("test"); t.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    {"run": cmd_run, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
