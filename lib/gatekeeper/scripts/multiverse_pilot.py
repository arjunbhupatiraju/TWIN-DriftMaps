"""MULTIVERSE pilot (kill test K2): how fragile are region-level encoding claims?

Runs ONLY on the EXPLORE split already cached on disk by INFOFLOW (data/info_cache). No download.

    python3 scripts/multiverse_pilot.py                 # ~10-20 min
    python3 scripts/multiverse_pilot.py --quick         # ~2 min smoke test (fewer resamples)

Claims tested: "region R encodes V" for
    V = stim   stimulus side, spike counts 30-130 ms after stimulus onset (non-zero contrasts,
               valid choice, no movement before 130 ms)
    V = prior  block prior (left- vs right-likely block), counts 400-50 ms BEFORE stimulus onset

Analysis universes (3 methods x 3 nulls x 2 unit sets = 18 per variable):
    method  decode   ridge decoder, exact leave-one-out (closed form, so nulls are cheap)
            single   fraction of neurons with |Mann-Whitney z| > 1.96
            euclid   distance between the two condition means (z-scored neurons)
    null    shuffle  trial labels permuted
            circ     trial labels circularly shifted (keeps slow structure such as blocks)
            pseudo   IBL's null: labels redrawn from the task's own generative process (blocks + sides)
Negative controls: ctl_stim / ctl_prior are FAKE variables drawn from that same process; any region
"encoding" them is a false positive, so their significant share measures how calibrated each null is.
    units   good     IBL 'good' units only
            all      every sorted unit

Session-level p-values are computed ONCE; a "k-mouse study" combines its sessions with Fisher's
method and calls a region significant at BH-FDR q < 0.05 across regions (region needs >= 2 sessions).
Two disjoint groups of k mice are drawn many times; replication = P(significant in B | significant in A).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gatekeeper.infoflow_data import InfoSession  # noqa: E402

CACHE = Path("data/info_cache")
OUT = Path("results/multiverse_pilot")
METHODS = ("decode", "single", "euclid")
NULLS = ("shuffle", "circ", "pseudo")
UNITS = ("good", "all")
VARS = ("stim", "prior", "ctl_stim", "ctl_prior")   # ctl_* = fake variables (negative controls)
IBL_NULL = "pseudo"
MIN_UNITS = 5
MIN_TRIALS_PER_CLASS = 20
KS = (2, 3, 4, 6, 8, 12, 16, 20)


# ----------------------------------------------------------------------------- per-session statistics
def pseudo_pl(T, rng):
    """IBL task generative process: 90 unbiased trials, then alternating 0.2/0.8 blocks whose
    lengths are exponential (mean 60) truncated to [20, 100]."""
    pl = list(np.full(min(90, T), 0.5))
    p = rng.choice([0.2, 0.8])
    while len(pl) < T:
        L = 0
        while not 20 <= L <= 100:
            L = int(rng.exponential(60))
        pl += [p] * L
        p = 1.0 - p
    return np.asarray(pl[:T])


def _pseudo_y(var, m, T_full, rng):
    pl = pseudo_pl(T_full, rng)
    if var in ("stim", "ctl_stim"):
        return (rng.random(T_full) >= pl)[m].astype(int)     # right side, drawn as the task does
    return (pl[m] < 0.5).astype(int)


def _null_labels(y, n_perm, kind, rng, var=None, m=None, T_full=None):
    """(T, 1 + n_perm) label matrix; column 0 = real labels."""
    T = len(y)
    cols = [y]
    if kind == "pseudo":
        for _ in range(n_perm):
            cols.append(_pseudo_y(var, m, T_full, rng))
    elif kind == "shuffle":
        for _ in range(n_perm):
            cols.append(rng.permutation(y))
    else:
        lo = max(1, T // 10)
        for _ in range(n_perm):
            cols.append(np.roll(y, int(rng.integers(lo, T - lo + 1))))
    return np.stack(cols, 1).astype(float)


def _stats(X, Y, lam=1.0):
    """Statistics for every label column of Y at once. X: (T, N) counts. Returns dict method -> (P+1,)."""
    T, N = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    n1 = Y.sum(0)
    n0 = T - n1
    # decode: ridge with exact leave-one-out (hat matrix is label independent)
    A = Z.T @ Z + lam * N * np.eye(N)
    H = Z @ np.linalg.solve(A, Z.T)
    h = np.clip(np.diag(H), None, 0.999)
    Yc = Y - Y.mean(0)
    loo = (H @ Yc - h[:, None] * Yc) / (1 - h[:, None])
    dec = (loo * Yc).sum(0) / (np.linalg.norm(loo, axis=0) * np.linalg.norm(Yc, axis=0) + 1e-12)
    # single neurons: Mann-Whitney z via ranks (tie-corrected variance)
    R = np.apply_along_axis(rankdata, 0, X)
    U = Y.T @ R - (n1 * (n1 + 1) / 2)[:, None]
    tie = np.array([(np.unique(X[:, j], return_counts=True)[1] ** 3).sum() for j in range(N)]) - T
    var = (n1 * n0 / 12)[:, None] * ((T + 1) - tie[None, :] / (T * (T - 1)))
    z = (U - (n1 * n0 / 2)[:, None]) / np.sqrt(np.maximum(var, 1e-12))
    single = (np.abs(z) > 1.96).mean(1)
    # euclid: distance between condition means of z-scored neurons
    mu1 = (Y.T @ Z) / n1[:, None]
    mu0 = ((1 - Y).T @ Z) / n0[:, None]
    euc = np.linalg.norm(mu1 - mu0, axis=1) / np.sqrt(N)
    return {"decode": dec, "single": single, "euclid": euc}


def _pvals(stats):
    return {m: float((1 + (v[1:] >= v[0]).sum()) / len(v)) for m, v in stats.items()}


def _design(s, var):
    if var.startswith("ctl_"):
        m, y, win = _design(s, var[4:])
        if m is None:
            return m, y, win
        rng = np.random.default_rng(abs(hash((s.sid, var))) % 2**32)
        return m, _pseudo_y(var, m, len(s.contrast), rng), win
    if var == "stim":
        c, ch, rt = s.contrast, s.choice, s.rt
        m = (c != 0) & (ch >= 0) & np.isfinite(rt) & (rt > 0.13)
        return m, (c[m] > 0).astype(int), "early"
    pl = s.meta.get("probabilityLeft")
    if pl is None or len(pl) != len(s.contrast):
        return None, None, None
    pl = np.asarray(pl, float)
    m = np.isfinite(pl) & (np.abs(pl - 0.5) > 0.1)
    return m, (pl[m] < 0.5).astype(int), "pre"


def session_table(paths, n_perm, seed=0, log=print):
    """List of rows: subject, sid, var, region, units, method, null, p, n_units."""
    rows = []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i, p in enumerate(paths):
        for units in UNITS:
            s = InfoSession(p, all_units=(units == "all"))
            if s.split != "explore":
                raise RuntimeError("confirm-split session reached the pilot; refusing to continue")
            subj = s.meta.get("subject", "unknown")
            for var in VARS:
                m, y, win = _design(s, var)
                if m is None or min((y == 0).sum(), (y == 1).sum()) < MIN_TRIALS_PER_CLASS:
                    continue
                Ys = {k: _null_labels(y, n_perm, k, rng, var, m, len(s.contrast)) for k in NULLS}
                for r in s.regions:
                    X = s.X(r, win)
                    if X is None or X.shape[1] < MIN_UNITS:
                        continue
                    X = X[m]
                    keep = X.std(0) > 0
                    if keep.sum() < MIN_UNITS:
                        continue
                    X = X[:, keep]
                    for nk, Y in Ys.items():
                        for meth, pv in _pvals(_stats(X, Y)).items():
                            rows.append((subj, s.sid, var, r, units, meth, nk, pv, int(X.shape[1])))
        if (i + 1) % 10 == 0 or i + 1 == len(paths):
            el = (time.time() - t0) / 60
            log(f"  {i+1}/{len(paths)} sessions ({el:.1f} min, ~{el / (i + 1) * (len(paths) - i - 1):.0f} min left)")
    return rows


# ----------------------------------------------------------------------------- studies and replication
def _bh(p, q=0.05):
    p = np.asarray(p)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    ok = p[o] <= thr
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    sig = np.zeros(n, bool)
    sig[o[:k]] = True
    return sig


def study_verdicts(tab, mice, counts=None):
    """tab: dict (universe) -> dict region -> list of (subject, p). Returns universe -> {region: bool}."""
    ms = set(mice)
    out, nsess = {}, {}
    for u, regs in tab.items():
        names, ps = [], []
        for r, lst in regs.items():
            pp = [p for sub, p in lst if sub in ms]
            if len(pp) < 2:
                continue
            stat = -2 * np.sum(np.log(np.clip(pp, 1e-12, 1)))
            names.append(r)
            ps.append(chi2.sf(stat, 2 * len(pp)))
            nsess[(u, r)] = len(pp)
        sig = _bh(ps)
        out[u] = dict(zip(names, sig.tolist()))
    if counts is not None:
        counts.update(nsess)
    return out


def build_tab(rows):
    tab = {}
    for subj, sid, var, r, units, meth, nk, p, nu in rows:
        tab.setdefault((var, meth, nk, units), {}).setdefault(r, []).append((subj, p))
    return tab


def replication(tab, mice, ks, n_rep, seed=1):
    rng = np.random.default_rng(seed)
    mice = sorted(mice)
    res = {}           # universe -> k -> dict(rate, n_claims)
    per_claim = {}     # (universe, region) -> k -> [hits, trials]
    strat = {}         # (universe, sessions-bin) -> [hits, trials]
    for k in ks:
        if 2 * k > len(mice):
            continue
        acc = {u: [0, 0, 0, 0] for u in tab}   # sigA&sigB, sigA, agree, tested
        for _ in range(n_rep):
            perm = rng.permutation(mice)
            ca, cb = {}, {}
            va = study_verdicts(tab, perm[:k], ca)
            vb = study_verdicts(tab, perm[k:2 * k], cb)
            for u in tab:
                for r in set(va[u]) & set(vb[u]):
                    a, b = va[u][r], vb[u][r]
                    for x, yv in ((a, b), (b, a)):      # symmetric
                        if x:
                            acc[u][1] += 1
                            acc[u][0] += yv
                            pc = per_claim.setdefault((u, r), {}).setdefault(k, [0, 0])
                            pc[0] += yv
                            pc[1] += 1
                            st = strat.setdefault((u, sbin(min(ca[(u, r)], cb[(u, r)]))), [0, 0])
                            st[0] += yv
                            st[1] += 1
                    acc[u][2] += a == b
                    acc[u][3] += 1
        for u, (ss, sa, ag, te) in acc.items():
            res.setdefault(u, {})[k] = dict(replication=ss / sa if sa else np.nan,
                                            agreement=ag / te if te else np.nan,
                                            claims_per_study=sa / (2 * n_rep))
    return res, per_claim, strat


SBINS = ("2", "3-4", "5-8", "9+")


def sbin(n):
    return "2" if n <= 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9+"


def ustr(u):
    return f"{u[0]}|{u[1]}|{u[2]}|{u[3]}"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke test: 50 permutations, 50 resamples")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-rep", type=int, default=300)
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    if a.quick:
        a.n_perm, a.n_rep = 50, 50
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    paths = []
    for p in sorted(Path(a.cache).glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        if json.loads(str(z["meta"])).get("split") == "explore":
            paths.append(p)
    print(f"MULTIVERSE pilot: {len(paths)} EXPLORE sessions (confirm split is never opened)", flush=True)
    if not paths:
        sys.exit("no explore sessions found in " + a.cache)

    print("step 1/3: session-level statistics for 12 universes x 2 variables ...", flush=True)
    rows = session_table(paths, a.n_perm)
    mice = sorted({r[0] for r in rows})
    print(f"  {len(rows)} session-level tests, {len(mice)} mice", flush=True)
    tab = build_tab(rows)

    print("step 2/3: full-explore verdicts, null calibration, multiverse agreement ...", flush=True)
    full = study_verdicts(tab, mice)
    summary = {"n_sessions": len(paths), "n_mice": len(mice), "n_perm": a.n_perm, "n_rep": a.n_rep}
    for var in VARS:
        us = [u for u in tab if u[0] == var]
        if not us:
            continue
        regs = sorted(set().union(*[set(full[u]) for u in us]))
        regs = [r for r in regs if all(r in full[u] for u in us)]
        if not regs:
            continue
        V = np.array([[full[u][r] for u in us] for r in regs])     # regions x universes
        col = {u: j for j, u in enumerate(us)}
        print(f"  {var}: {len(regs)} regions; share called significant (good units / all units):")
        print(f"     {'':8s}" + "".join(f"{n:>16s}" for n in NULLS))
        for meth in METHODS:
            cells = "".join(f"{V[:, col[(var, meth, n, 'good')]].mean():>9.0%} / {V[:, col[(var, meth, n, 'all')]].mean():<4.0%}" for n in NULLS)
            print(f"     {meth:8s}{cells}")
        d = dict(regions=len(regs), pct_sig={ustr(u): float(V[:, j].mean()) for j, u in enumerate(us)})
        if var.startswith("ctl_"):
            fp = {n: float(np.mean([V[:, col[(var, m_, n, un)]].mean() for m_ in METHODS for un in UNITS])) for n in NULLS}
            d["false_positive_rate_by_null"] = fp
            print(f"     NEGATIVE CONTROL (fake variable): false-positive share by null: "
                  + ", ".join(f"{n} {v:.0%}" for n, v in fp.items()))
        else:
            maj = V.mean(1) >= 0.5
            agree = (V == maj[:, None]).mean(1)
            three = V[:, [col[(var, m_, IBL_NULL, "good")] for m_ in METHODS]]
            Vn = V[:, [col[u] for u in us if u[2] == IBL_NULL]]
            agree_n = (Vn == (Vn.mean(1) >= 0.5)[:, None]).mean(1)
            d.update(three_methods_all_agree=float((three.all(1) | (~three).all(1)).mean()),
                     mean_multiverse_agreement=float(agree.mean()),
                     frac_claims_agreement_below_70=float((agree < 0.7).mean()),
                     frac_claims_agreement_below_70_same_null=float((agree_n < 0.7).mean()))
            print(f"     three methods agree (pseudo-session null, good units): {d['three_methods_all_agree']:.0%} of regions")
            print(f"     < 70% agreement: {d['frac_claims_agreement_below_70_same_null']:.0%} of regions across methods x units "
                  f"(pseudo null); {d['frac_claims_agreement_below_70']:.0%} across all {len(us)} universes")
        summary[var] = d

    print("step 3/3: resampling mice (two disjoint k-mouse studies), pseudo-session null, good units ...", flush=True)
    rep, per_claim, strat = replication(tab, mice, KS, a.n_rep)
    summary["replication"] = {ustr(u): {str(k): v for k, v in d.items()} for u, d in rep.items()}
    for var in ("stim", "prior"):
        print(f"  {var}: P(replicate | significant)   [significant regions per study]")
        print("     " + "".join(f"{'k=' + str(k):>8s}" for k in KS if (var, 'decode', IBL_NULL, 'good') in rep and k in rep[(var, 'decode', IBL_NULL, 'good')]))
        for meth in METHODS:
            u = (var, meth, IBL_NULL, "good")
            if u not in rep:
                continue
            print(f"     {meth:7s}" + "".join(f"{rep[u][k]['replication']:>8.0%}" for k in sorted(rep[u])))
        u = (var, "decode", IBL_NULL, "good")
        if u in rep:
            print(f"     {'claims':7s}" + "".join(f"{rep[u][k]['claims_per_study']:>8.1f}" for k in sorted(rep[u])))
            row = []
            for b_ in SBINS:
                h, t = strat.get((u, b_), [0, 0])
                row.append(f"{b_} sessions: {h / t:.0%} (n={t})" if t else f"{b_} sessions: -")
            print("     decode, by sessions per region in each study: " + "; ".join(row))
            summary[f"{var}_replication_by_sessions"] = {b_: strat.get((u, b_), [0, 0]) for b_ in SBINS}
    k2 = {}
    for var in ("stim", "prior"):
        vals = [d[4][0] / d[4][1] for (u, r), d in per_claim.items()
                if u[0] == var and u[2] == IBL_NULL and u[3] == "good" and 4 in d and d[4][1] >= 10]
        if vals:
            k2[var] = float(np.mean(np.array(vals) < 0.8))
    summary["K2_frac_claims_P4_below_80"] = k2
    fp = max((summary[v]["false_positive_rate_by_null"][IBL_NULL] for v in ("ctl_stim", "ctl_prior") if v in summary), default=np.nan)
    calibrated = bool(fp <= 0.10)
    agr = [summary[v]["frac_claims_agreement_below_70_same_null"] for v in ("stim", "prior") if v in summary]
    fragile = any(x >= 0.2 for x in k2.values()) or any(x >= 0.2 for x in agr)
    summary["calibration_ok"] = calibrated
    summary["K2_pilot_verdict"] = ("NULL NOT CALIBRATED: do not trust yet" if not calibrated else
                                   "FRAGILE (story A: go)" if fragile else "MOSTLY ROBUST (story B: decide together)")
    print("")
    print(f"CALIBRATION  fake-variable false-positive share with the pseudo-session null: {fp:.0%} "
          f"({'OK' if calibrated else 'TOO HIGH'}; must be <= 10%)")
    for var, x in k2.items():
        print(f"K2  {var}: {x:.0%} of claims replicate < 80% of the time at k = 4 mice")
    for var in ("stim", "prior"):
        if var in summary:
            print(f"K2  {var}: {summary[var]['frac_claims_agreement_below_70_same_null']:.0%} of regions have < 70% agreement across methods x units")
    print(f"K2 PILOT VERDICT: {summary['K2_pilot_verdict']}")
    (out / "pilot.json").write_text(json.dumps(summary, indent=1, default=float))
    _plot(rep, out / "replication.png")
    print(f"saved {out/'pilot.json'} and {out/'replication.png'}")


def _plot(rep, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, var in zip(axs, ("stim", "prior")):
        for u, d in sorted(rep.items()):
            if u[0] != var or u[3] != "good" or u[2] != IBL_NULL:
                continue
            ks = sorted(d)
            ax.plot(ks, [d[k]["replication"] for k in ks], marker="o", label=u[1])
        ax.axhline(0.8, ls="--", c="gray", lw=1)
        ax.axvspan(3, 8, color="gray", alpha=0.1)
        ax.set_title(f"{var}: P(replicate | significant), pseudo-session null")
        ax.set_xlabel("mice per study (shaded: typical papers)")
        ax.set_ylim(0, 1)
    axs[0].set_ylabel("replication probability")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
