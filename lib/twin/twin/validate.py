"""Real-data validation on IBL: does the twin predict the false-positive rate the common test actually has?

For each session x region (prior_a window, valid trials):
  empirical FP   fake labels (pseudo-session block priors, unrelated to the neurons) tested on the REAL neurons with
                 the common trial-shuffle test: fraction significant at 5% (ground truth, no twin involved)
  twin FP        the same, on nonparametric twins of the neurons (multivariate phase randomization)
  planner FP     the same, on parametric twins simulated from fitted drift parameters only (what the planner would
                 predict before the experiment). Three versions in the same run:
                   v3  timescale ladder + heterogeneity, bias correction on ladder coefficients (TWIN 0.3)
                   v3b the same with bias correction on the autocorrelation curves
                   v4  v3b + shared drift as several factors of any sign (optional, factors=True)
  calibrated FP  fake labels on the real neurons, tested against the pseudo-session null (should be about 5%)
Summary: correlation and error of twin / planner predictions against the empirical rate, across session-regions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .task import BlockTask, PseudoSessions
from .drift import DriftModel, phase_twin
from .stats import pooled_corr


def _rate(X, labels, rng, n_perm=100, null=None):
    hits = []
    for y in labels.T:
        if null is None:
            Y = np.column_stack([y] + [rng.permutation(y) for _ in range(n_perm)])
        else:
            Y = np.column_stack([y, null.matrix(len(y), n_perm, rng)])
        s = pooled_corr(X, Y)
        hits.append((1 + (s[1:] >= s[0]).sum()) / (n_perm + 1) < 0.05)
    return float(np.mean(hits))


def validate_ibl(cache, out_dir, every=4, min_units=10, n_fake=40, gatekeeper_root=None, burn=90):
    import sys
    import pandas as pd
    root = Path(gatekeeper_root or "~/Downloads/GATEKEEPER/gatekeeper").expanduser()
    for p in (root, root / "next"):
        sys.path.insert(0, str(p))
    from gatekeeper.mv_data import MVSession
    rng = np.random.default_rng(0)
    task = BlockTask(); null = PseudoSessions(task, "prior")
    rows = []
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[::every]
    for i, p in enumerate(paths):
        s = MVSession(p)
        v = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]; v[:burn] = False
        T = int(v.sum())
        if T < 200:
            continue
        for r in s.regions:
            X = s.X(r, "prior_a")
            if X is None or X.shape[1] < min_units:
                continue
            X = X[v]; X = X[:, X.std(0) > 0]
            if X.shape[1] < min_units:
                continue
            fake = null.matrix(T, n_fake, rng)
            emp = _rate(X, fake, rng)
            emp2 = _rate(X, null.matrix(T, n_fake, rng), rng)          # independent repeat: sets the noise ceiling
            tw = np.mean([_rate(phase_twin(X, rng), null.matrix(T, 1, rng), rng) for _ in range(n_fake)])
            N = X.shape[1]
            fits = dict(v3=DriftModel.fit(X, bc="coef", factors=False), v3b=DriftModel.fit(X, factors=False), v4=DriftModel.fit(X, factors=True))
            pl = {k: float(np.mean([_rate(dm.simulate(T, N, rng), null.matrix(T, 1, rng), rng) for _ in range(n_fake)]))
                  for k, dm in fits.items()}
            cal = _rate(X, null.matrix(T, n_fake, rng), rng, null=null)
            d4 = fits["v4"]; phis, a4, _ = d4.components()
            rows.append(dict(sid=s.sid, subject=s.subject, region=r, T=T, N=N, empirical_fp=emp, empirical_fp_repeat=emp2, twin_fp=float(tw),
                             planner_v3_fp=pl["v3"], planner_v3b_fp=pl["v3b"], planner_v4_fp=pl["v4"], calibrated_fp=cal,
                             drift_slow=d4.slow_frac, drift_long=float(sum(x for p, x in zip(phis, a4) if p > 0.98)),
                             drift_het_sd=float(np.std(d4.het)) if d4.het else 0.0,
                             n_factors=len(d4.fac_pi or []), shared_slow=float(np.sum(d4.fac_pi)) if d4.fac_pi else 0.0,
                             sh_v3=float(np.dot(fits["v3"].a, fits["v3"].sh)), mean_count=float(X.mean()), frac_zero=float((X == 0).mean())))
        print(f"[{i + 1}/{len(paths)}] {s.sid}: {len(rows)} session-regions so far", flush=True)
        if rows:                                                  # partial results survive an interrupted run
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(Path(out_dir) / "validate_ibl_partial.csv", index=False)
    df = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "validate_ibl.csv", index=False)
    r_tw = np.corrcoef(df.empirical_fp, df.twin_fp)[0, 1]
    ceil = np.corrcoef(df.empirical_fp, df.empirical_fp_repeat)[0, 1]
    text = "\n".join([
        f"# TWIN validation on IBL ({len(df)} session-regions, {df.sid.nunique()} sessions, {df.subject.nunique()} mice)\n",
        f"Empirical false-positive rate of the common shuffle test (fake block-prior labels on real neurons): "
        f"median {df.empirical_fp.median():.2f}, mean {df.empirical_fp.mean():.2f}",
        f"Nonparametric twin prediction: r = {r_tw:.3f}, mean absolute error {np.abs(df.empirical_fp - df.twin_fp).mean():.3f}",
        f"Noise ceiling (two independent sets of {n_fake} fake labels on the same neurons): r = {ceil:.3f}",
        *[f"Planner {k} ({desc}): r = {np.corrcoef(df.empirical_fp, df[f'planner_{k}_fp'])[0, 1]:.3f}, "
          f"mean absolute error {np.abs(df.empirical_fp - df[f'planner_{k}_fp']).mean():.3f}, "
          f"bias {(df[f'planner_{k}_fp'] - df.empirical_fp).mean():+.3f}"
          for k, desc in (("v3", "TWIN 0.3"), ("v3b", "curve bias correction, TWIN 0.4 default"), ("v4", "+ multi-factor shared drift, optional"))],
        f"Calibrated test on real neurons with fake labels: mean rate {df.calibrated_fp.mean():.3f} (nominal 0.05)",
    ])
    (out / "VALIDATION.md").write_text(text + "\n")
    print("\n" + text)


def validate_comm(cache, out_dir, every=4, min_units=10, max_units=60, pairs_per_session=3, n_null=40,
                  gatekeeper_root=None, burn=90, window="prior_a", slow_trials=5.0):
    """Communication between regions on IBL.

    Negative controls: region A from one session paired with region B from ANOTHER session (truncated to the same
    trial count). They cannot communicate, so every 'significant' result is a false positive.
    Same-session pairs: how much apparent communication survives the drift-safe test, and on which timescale."""
    import sys
    import pandas as pd
    from .comm import comm_check
    root = Path(gatekeeper_root or "~/Downloads/GATEKEEPER/gatekeeper").expanduser()
    for p in (root, root / "next"):
        sys.path.insert(0, str(p))
    from gatekeeper.mv_data import MVSession
    rng = np.random.default_rng(0)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[::every]
    sessions = []
    for p in paths:
        s = MVSession(p)
        v = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]; v[:burn] = False
        if v.sum() < 200:
            continue
        regs = {}
        for r in s.regions:
            X = s.X(r, window)
            if X is None:
                continue
            X = X[v]; X = X[:, X.std(0) > 0]
            if X.shape[1] < min_units:
                continue
            if X.shape[1] > max_units:
                X = X[:, rng.choice(X.shape[1], max_units, replace=False)]
            regs[r] = X
        if regs:
            sessions.append((s.sid, s.subject, regs))
    rows = []
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for i, (sid, subj, regs) in enumerate(sessions):
        names = sorted(regs)
        pairs = [(a, b) for a in names for b in names if a < b]
        if len(pairs) > pairs_per_session:
            pairs = [pairs[j] for j in rng.choice(len(pairs), pairs_per_session, replace=False)]
        todo = [("same_session", sid, a, regs[a], sid, b, regs[b]) for a, b in pairs]
        sid2, subj2, regs2 = sessions[(i + 1) % len(sessions)]
        if regs2 and sid2 != sid:
            a = names[rng.integers(len(names))]; b = sorted(regs2)[rng.integers(len(regs2))]
            T = min(len(regs[a]), len(regs2[b]))
            todo.append(("other_session", sid, a, regs[a][:T], sid2, b, regs2[b][:T]))
        for kind, sa, a, A, sb, b, B in todo:
            res = comm_check(A, B, n_null=n_null, slow_trials=slow_trials, seed=int(rng.integers(1e9)))
            rows.append(dict(kind=kind, session_a=sa, region_a=a, session_b=sb, region_b=b, subject=subj, T=len(A),
                             N_a=A.shape[1], N_b=B.shape[1], **{k: v for k, v in res.items() if k != "verdict"}))
        print(f"[{i + 1}/{len(sessions)}] {sid}: {len(rows)} pairs so far", flush=True)
        pd.DataFrame(rows).to_csv(out / "validate_comm_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "validate_comm.csv", index=False)
    sig = lambda d, k: f"{(d[k] < 0.05).mean():.2f}"
    o, s_ = df[df.kind == "other_session"], df[df.kind == "same_session"]
    text = "\n".join([
        f"# TWIN communication check on IBL ({len(s_)} same-session pairs, {len(o)} cross-session negative controls, "
        f"{df.session_a.nunique()} sessions)\n",
        "Negative controls (regions from different sessions; any 'communication' is false):",
        f"  common pipeline (random-fold CV vs trial shuffle) significant: {sig(o, 'p_common')}",
        f"  blocked CV vs trial shuffle: {sig(o, 'p_shuffle')}",
        f"  TWIN test (blocked CV vs twins of B): {sig(o, 'p_twin')}   (nominal 0.05)",
        "Same-session pairs:",
        f"  common pipeline significant: {sig(s_, 'p_common')}",
        f"  TWIN test significant (coupled): {sig(s_, 'p_twin')}",
        f"  coupled on slow timescales (> {slow_trials:g} trials): {sig(s_, 'p_slow')}",
        f"  coupled on fast (trial-to-trial) timescales: {sig(s_, 'p_fast')}",
        f"  coupled ONLY on slow timescales: {((s_.p_twin < .05) & (s_.p_slow < .05) & (s_.p_fast >= .05)).mean():.2f}",
        f"  median share of excess predictability that is slow: {s_.slow_share.median():.2f}",
    ])
    (out / "VALIDATION_COMM.md").write_text(text + "\n")
    print("\n" + text)


def validate_net(cache, out_dir, every=4, min_units=10, n_fake=40, gatekeeper_root=None, burn=90):
    """TWIN-Net on real IBL data: predicted vs empirical shuffle-test false-positive rate per session-region
    (fake block-prior labels), plus the calibrated and linear-shift rates. Fast: no simulation."""
    import sys, time
    import pandas as pd
    from .net import TwinNet
    from .tests import p_linear_shift
    root = Path(gatekeeper_root or "~/Downloads/GATEKEEPER/gatekeeper").expanduser()
    for p in (root, root / "next"):
        sys.path.insert(0, str(p))
    from gatekeeper.mv_data import MVSession
    net = TwinNet.default()
    rng = np.random.default_rng(0)
    task = BlockTask(); null = PseudoSessions(task, "prior")
    rows = []; tnet = 0.0
    for i, p in enumerate(sorted(Path(cache).expanduser().glob("*.npz"))[::every]):
        s = MVSession(p)
        v = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]; v[:burn] = False
        T = int(v.sum())
        if T < 200:
            continue
        for r in s.regions:
            X = s.X(r, "prior_a")
            if X is None:
                continue
            X = X[v]; X = X[:, X.std(0) > 0]
            if X.shape[1] < min_units:
                continue
            fake = null.matrix(T, n_fake, rng)
            emp = _rate(X, fake, rng)
            emp2 = _rate(X, null.matrix(T, n_fake, rng), rng)
            cal = _rate(X, null.matrix(T, n_fake, rng), rng, null=null)
            shift = float(np.mean([p_linear_shift(X, y) < 0.05 for y in null.matrix(T, n_fake, rng).T]))
            t0 = time.time()
            pred = net.predict(X, [null.draw(T, rng) for _ in range(5)], r=0.1)
            tnet += time.time() - t0
            rows.append(dict(sid=s.sid, subject=s.subject, region=r, T=T, N=X.shape[1], empirical_fp=emp,
                             empirical_fp_repeat=emp2, calibrated_fp=cal, shift_fp=shift,
                             **{f"net_{k}": v_ for k, v_ in pred.items()}))
        print(f"[{i + 1}] {s.sid}: {len(rows)} session-regions so far", flush=True)
    df = pd.DataFrame(rows)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "validate_net.csv", index=False)
    r_net = np.corrcoef(df.empirical_fp, df.net_fp_shuffle)[0, 1]
    ceil = np.corrcoef(df.empirical_fp, df.empirical_fp_repeat)[0, 1]
    text = "\n".join([
        f"# TWIN-Net on IBL ({len(df)} session-regions, {df.sid.nunique()} sessions, {df.subject.nunique()} mice)\n",
        f"Empirical shuffle-test false-positive rate: median {df.empirical_fp.median():.2f}",
        f"TWIN-Net prediction (trained only on simulations): r = {r_net:.3f}, mean absolute error "
        f"{np.abs(df.empirical_fp - df.net_fp_shuffle).mean():.3f}   (noise ceiling r = {ceil:.3f}; simulation planner v3b was 0.814)",
        f"TWIN-Net time per region: {1000 * tnet / max(len(df), 1):.0f} ms",
        f"Calibrated test on real neurons: {df.calibrated_fp.mean():.3f}; TWIN-Net predicted {df.net_fp_pseudo.mean():.3f}",
        f"Linear-shift test on real neurons: {df.shift_fp.mean():.3f}; TWIN-Net predicted {df.net_fp_shift.mean():.3f}",
    ])
    (out / "VALIDATION_NET.md").write_text(text + "\n")
    print("\n" + text)
