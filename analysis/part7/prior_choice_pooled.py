"""P2 (pre-registered follow-up to P1, registered after P1 failed): does the movement-free prior, pooled across ALL
recorded regions of a session, predict the mouse's choice trial by trial?
(pre-registration: claude/preregistration-prior-choice-pooled.md)

P1 read the prior from one region at a time and found no link (p = 0.24). Each region's prior code is weak, so a
single-region, single-trial estimate is mostly noise. If behaviour uses the prior, it most likely pools it across
regions. P2 is identical to P1 except that one decoder per session uses the neurons of every eligible region together.

    python3 part7/prior_choice_pooled.py run --limit 3
    python3 part7/prior_choice_pooled.py run
    python3 part7/prior_choice_pooled.py test
"""
from __future__ import annotations

import argparse, json, os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prior_choice as PC                 # noqa: E402  (frozen P1 code: decoder, partial correlation, paths)
PM, MV = PC.PM, PC.MV

OUT = HERE / "results_prior_choice"


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
    fb = np.sign(np.asarray(s.feedback, float)); rw = ch * (fb > 0)
    lag = lambda x, k: np.r_[np.zeros(k), x[:-k]]
    cov = np.column_stack([c, prior, lag(ch, 1), lag(ch, 2), lag(rw, 1), lag(rw, 2)])
    v = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]
    v[:burn] = False
    if v.sum() < 150:
        return []
    low = np.abs(c[v]) <= max_contrast
    if low.sum() < 40:
        return []
    blocks, regs = [], []
    for r in s.regions:                                           # same per-region eligibility and cap as P1
        X = s.X(r, "prior_a")
        if X is None:
            continue
        X = np.asarray(X, float)[v]; X = X[:, X.std(0) > 0]
        if X.shape[1] < 10:
            continue
        if X.shape[1] > 150:
            X = X[:, rng.choice(X.shape[1], 150, replace=False)]
        blocks.append(X); regs.append(r)
    if not blocks:
        return []
    M = Mfull[v]
    Zp = np.column_stack([PM._pseudo(task, Tfull, rng)[0] for _ in range(n_null)])[v]
    targets = MV.remove(np.column_stack([prior[v], Zp]), M)
    pred = PC.cv_predict(MV.remove(np.column_stack(blocks), M), targets)
    pr = PC.partial_r(ch[v][low], pred[low], cov[v][low])
    return [dict(session=s.sid, subject=s.subject, lab=s.lab, n_regions=len(regs), regions="|".join(regs),
                 N=int(sum(b.shape[1] for b in blocks)), T=int(v.sum()), n_low=int(low.sum()), r_obs=float(pr[0]),
                 r_null_mean=float(pr[1:].mean()), p=PM._p(pr[0], pr[1:]),
                 null=json.dumps([round(float(x), 4) for x in pr[1:]]))]


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    rows_f = OUT / "prior_choice_pooled_rows.csv"
    done = set(pd.read_csv(rows_f).session) if rows_f.exists() else set()
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz"))[: a.limit or None]
             if p.stem not in done and (MV.FEAT / f"{p.stem}.npz").exists()]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.gatekeeper, a.null, 90, 0.0625, 13000 + i): p for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True)
                continue
            if rows:
                pd.DataFrame(rows).to_csv(rows_f, mode="a", header=not rows_f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {rows[0]['n_regions'] if rows else 0} regions pooled", flush=True)


def cmd_test(a):
    import pandas as pd
    from scipy.stats import spearmanr
    D = pd.read_csv(OUT / "prior_choice_pooled_rows.csv")
    L = []; P = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    N = np.array([json.loads(z) for z in D.null])
    obs, nul = D.r_obs.mean(), N.mean(0)
    p = float((1 + np.sum(nul >= obs)) / (1 + len(nul)))
    P2 = obs > 0 and p < 0.05
    P(f"# P2: pooled across all regions of a session ({len(D)} sessions, {D.subject.nunique()} mice; "
      f"median {int(D.n_regions.median())} regions and {int(D.N.median())} neurons per session)")
    P(f"  mean partial r (pooled neural prior estimate vs choice, same trials and controls as P1): {obs:+.4f}; "
      f"pseudo-session null mean {nul.mean():+.4f}, 95th pct {np.percentile(nul, 95):+.4f}; one-sided p = {p:.4f}")
    P(f"  P2 {'PASS' if P2 else 'FAIL'} (criterion: mean partial r > 0 and above the pseudo-session null, p < 0.05)")
    P(f"\n  descriptive: sessions with positive partial r: {np.mean(D.r_obs > 0):.0%}; "
      f"sessions individually significant (p < 0.05): {np.mean(D.p < 0.05):.1%} (5% expected by chance)")
    rho, pp = spearmanr(D.N, D.r_obs)
    P(f"  more pooled neurons -> stronger link? Spearman(neurons, partial r) = {rho:+.3f} (p = {pp:.3g})")
    (OUT / "PRIOR_CHOICE_POOLED.md").write_text("\n".join(L) + "\n")
    print(f"\nWrote {OUT}/PRIOR_CHOICE_POOLED.md")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--null", type=int, default=200)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
    r.add_argument("--gatekeeper"); r.add_argument("--limit", type=int, default=0)
    sub.add_parser("test")
    a = ap.parse_args()
    {"run": cmd_run, "test": cmd_test}[a.cmd](a)


if __name__ == "__main__":
    main()
