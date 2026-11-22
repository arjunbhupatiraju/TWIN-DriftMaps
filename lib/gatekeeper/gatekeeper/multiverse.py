"""MULTIVERSE analysis: how reliable are region-level encoding claims?

Stage 1 (heavy, once): per session x region, p-values for every (variable, window, units, method,
null) from the same label matrices. Saved to session_stats.csv.gz (small; can be shared).
Stage 2 (fast, vectorised): combine sessions into "studies" (any set of mice), call regions
significant (BH-FDR q < 0.05 across regions), and compare studies:
    - full-explore verdicts in every analysis universe (multiverse agreement, which choice flips most)
    - two disjoint k-mouse studies (P(k), N80, stratified by sessions per region)
    - two disjoint sets of LABS vs size-matched random mouse sets (the "another lab" test)
    - K1: our full-explore verdicts and effects vs IBL's published region table
    - K4: estimator check on simulated brains with known truth (separate entry point)

Analysis universe = (var, window, units, method, null, level, inclusion):
    var     stim | choice | feedback | prior          (+ ctl_* fake variables = negative controls)
    window  a (IBL's) | b (shifted)
    units   good | all
    method  decode (ridge, exact leave-one-out) | single (fraction of neurons |z|>1.96) | euclid
    null    shuffle | circ | pseudo (IBL style: task generative process for stim/prior,
            imposter sessions for choice/feedback)
    level   session (Fisher across sessions) | mouse (Fisher within mouse, then across mice)
            | pooled (single-neuron tests pooled across all mice as if independent; binomial)
    incl    region needs >= 2 sessions (IBL's rule) | >= 5 sessions
Default ("IBL-like") universe: decode, pseudo, good, window a, session level, >= 2 sessions.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom, chi2, rankdata, spearmanr

from .mv_data import MVSession

BASES = ("stim", "choice", "feedback", "prior", "stim_all", "sprior")
# stim_all: stimulus side without the reaction-time cut (closer to IBL); sprior: the mouse's SUBJECTIVE
# prior estimated from its own choices (IBL's prior target), continuous. Both added 2026-10-13 after
# exploration showed stim K1 below threshold and block identity to be the wrong prior target.
WINDOW_OF = {"stim_all": "stim", "sprior": "prior"}
VARS = BASES + tuple(f"ctl_{b}" for b in BASES)
WINDOWS = ("a", "b")
UNITS = ("good", "all")
METHODS = ("decode", "single", "euclid")
NULLS = ("shuffle", "circ", "pseudo")
LEVELS = ("session", "mouse")
INCLS = (2, 5)
DEFAULT = dict(window="a", units="good", method="decode", null="pseudo", level="session", incl=2)
MIN_UNITS = 5
MIN_PER_CLASS = 20
WIN_HI = {"stim": {"a": 0.10, "b": 0.15}}
IBL_TABLE = {"stim": "stim", "choice": "choice", "feedback": "fback", "stim_all": "stim"}


# =============================================================================== stage 1
def pseudo_pl(T, rng):
    pl = list(np.full(min(90, T), 0.5))
    p = rng.choice([0.2, 0.8])
    while len(pl) < T:
        L = 0
        while not 20 <= L <= 100:
            L = int(rng.exponential(60))
        pl += [p] * L
        p = 1.0 - p
    return np.asarray(pl[:T])


def _base(var):
    return var[4:] if var.startswith("ctl_") else var


_ALPHA = {}


def _sides(s):
    side = np.where(s.contrast > 0, 1.0, np.where(s.contrast < 0, 0.0, np.nan))
    z = (s.contrast == 0) & (s.choice >= 0) & (s.feedback != 0)
    side[z] = np.where(s.feedback[z] > 0, s.choice[z], 1 - s.choice[z])
    return side


def _prior_trace(side, alpha):
    pi = np.empty(len(side))
    cur = 0.5
    for t in range(len(side)):
        pi[t] = cur
        if np.isfinite(side[t]):
            cur = cur + alpha * (side[t] - cur)
    return pi


def subjective_alpha(s):
    """Exponential-smoothing (stimulus-kernel) prior; alpha fitted to the mouse's choices by maximum likelihood."""
    if s.sid in _ALPHA:
        return _ALPHA[s.sid]
    import warnings
    from sklearn.linear_model import LogisticRegression
    side = _sides(s)
    ok = s.choice >= 0
    c = np.tanh(5 * s.contrast) / np.tanh(5)
    best = (-np.inf, 0.1)
    for a in (0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.3, 0.4, 0.5):
        pi = np.clip(_prior_trace(side, a), 1e-3, 1 - 1e-3)
        X = np.c_[c, np.log(pi / (1 - pi))][ok]
        y = s.choice[ok]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(C=1e4).fit(X, y)
        pr = np.clip(m.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        ll = float(np.sum(y * np.log(pr) + (1 - y) * np.log(1 - pr)))
        if ll > best[0]:
            best = (ll, a)
    _ALPHA[s.sid] = best[1]
    return best[1]


def design(s, var, w):
    """(mask, labels) for a variable in window w, or (None, None)."""
    b = _base(var)
    if b == "sprior":
        if s.meta.get("probabilityLeft") is None:
            return None, None
        v = (s.choice >= 0) & s.valid["stimOn"]
        v[:90] = False
        y = _prior_trace(_sides(s), subjective_alpha(s))
    elif b == "stim_all":
        v = s.valid["stimOn"] & (s.contrast != 0) & (s.choice >= 0)
        y = (s.contrast > 0).astype(int)
    elif b == "stim":
        v = s.valid["stimOn"] & (s.contrast != 0) & (s.choice >= 0) & np.isfinite(s.rt) & (s.rt > WIN_HI["stim"][w])
        y = (s.contrast > 0).astype(int)
    elif b == "choice":
        v = s.valid["firstMovement"] & (s.choice >= 0)
        y = (s.choice == 1).astype(int)
    elif b == "feedback":
        v = s.valid["feedback"] & (s.feedback != 0)
        y = (s.feedback > 0).astype(int)
    else:
        pl = s.meta.get("probabilityLeft")
        if pl is None or len(pl) != len(s.contrast):
            return None, None
        pl = np.asarray(pl, float)
        v = np.isfinite(pl) & (np.abs(pl - 0.5) > 0.1)
        y = (pl < 0.5).astype(int)
    if var.startswith("ctl_"):
        rng = np.random.default_rng(int.from_bytes((s.sid + var + w).encode()[:8].ljust(8, b"0"), "little") % 2**32)
        return v, None, rng     # labels drawn later from the null process (needs the imposter pools)
    return v, y[v]


def _imposter(pool, T, rng):
    start = int(rng.integers(0, len(pool)))
    idx = (start + np.arange(T)) % len(pool)
    return pool[idx]


def null_draw(var, v, T_full, pool, rng):
    b = _base(var)
    if b == "sprior":
        pl = pseudo_pl(T_full, rng)
        fake_side = (rng.random(T_full) >= pl).astype(float)
        return _prior_trace(fake_side, _ALPHA_CUR[0])[v]
    if b in ("stim", "prior", "stim_all"):
        pl = pseudo_pl(T_full, rng)
        if b in ("stim", "stim_all"):
            return (rng.random(T_full) >= pl)[v].astype(int)
        return (pl[v] < 0.5).astype(int)
    return _imposter(pool, int(v.sum()), rng)


_ALPHA_CUR = [0.1]


def label_matrix(y, kind, n_perm, rng, var, v, T_full, pool):
    T = len(y)
    cols = [y]
    for _ in range(n_perm):
        if kind == "shuffle":
            cols.append(rng.permutation(y))
        elif kind == "circ":
            lo = max(1, T // 10)
            cols.append(np.roll(y, int(rng.integers(lo, T - lo + 1))))
        else:
            cols.append(null_draw(var, v, T_full, pool, rng))
    return np.stack(cols, 1).astype(float)


def stats(X, Y, lam=1.0):
    T, N = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    if not np.isin(Y[:, 0], (0.0, 1.0)).all():
        return _stats_cont(X, Z, Y, lam)
    n1 = Y.sum(0)
    n0 = T - n1
    A = Z.T @ Z + lam * N * np.eye(N)
    H = Z @ np.linalg.solve(A, Z.T)
    h = np.clip(np.diag(H), None, 0.999)
    Yc = Y - Y.mean(0)
    loo = (H @ Yc - h[:, None] * Yc) / (1 - h[:, None])
    dec = (loo * Yc).sum(0) / (np.linalg.norm(loo, axis=0) * np.linalg.norm(Yc, axis=0) + 1e-12)
    R = np.apply_along_axis(rankdata, 0, X)
    U = Y.T @ R - (n1 * (n1 + 1) / 2)[:, None]
    tie = np.array([(np.unique(X[:, j], return_counts=True)[1] ** 3).sum() for j in range(N)]) - T
    var = (n1 * n0 / 12)[:, None] * ((T + 1) - tie[None, :] / (T * (T - 1)))
    z = (U - (n1 * n0 / 2)[:, None]) / np.sqrt(np.maximum(var, 1e-12))
    sigfrac = (np.abs(z) > 1.96).mean(1)
    mu1 = (Y.T @ Z) / np.maximum(n1, 1)[:, None]
    mu0 = ((1 - Y).T @ Z) / np.maximum(n0, 1)[:, None]
    euc = np.linalg.norm(mu1 - mu0, axis=1) / np.sqrt(N)
    nsig = int((np.abs(z[0]) > 1.96).sum())
    return {"decode": dec, "single": sigfrac, "euclid": euc}, nsig, float(dec[0])


def _stats_cont(X, Z, Y, lam):
    """Continuous target: decode = leave-one-out ridge correlation; single = fraction of neurons with
    |Spearman z| > 1.96; euclid = norm of the neuron-wise correlation vector."""
    T, N = X.shape
    A = Z.T @ Z + lam * N * np.eye(N)
    H = Z @ np.linalg.solve(A, Z.T)
    h = np.clip(np.diag(H), None, 0.999)
    Yc = Y - Y.mean(0)
    loo = (H @ Yc - h[:, None] * Yc) / (1 - h[:, None])
    dec = (loo * Yc).sum(0) / (np.linalg.norm(loo, axis=0) * np.linalg.norm(Yc, axis=0) + 1e-12)
    R = np.apply_along_axis(rankdata, 0, X)
    R = (R - R.mean(0)) / (R.std(0) + 1e-9)
    RY = np.apply_along_axis(rankdata, 0, Y)
    RY = (RY - RY.mean(0)) / (RY.std(0) + 1e-9)
    rho = (RY.T @ R) / T
    zz = rho * np.sqrt(T - 1)
    single = (np.abs(zz) > 1.96).mean(1)
    Yz = Yc / (Yc.std(0) + 1e-12)
    euc = np.linalg.norm((Yz.T @ Z) / T, axis=1) / np.sqrt(N)
    return {"decode": dec, "single": single, "euclid": euc}, int((np.abs(zz[0]) > 1.96).sum()), float(dec[0])


def _pv(v):
    return float((1 + (v[1:] >= v[0]).sum()) / len(v))


def session_table(paths, split="explore", n_perm=200, vars=VARS, windows=WINDOWS, units=UNITS,
                  nulls=NULLS, seed=0, log=print):
    """Wide table: one row per (session, var, window, units, region) with p_<method>_<null>."""
    import pandas as pd
    sess = []
    for p in paths:
        s = MVSession(p)
        if s.split != split:
            continue
        sess.append(p)
    # imposter pools (choice, feedback) from all sessions of this split
    pools = {}
    for b in ("choice", "feedback"):
        seqs = []
        for p in sess:
            s = MVSession(p)
            v, y = design(s, b, "a")[:2]
            if v is not None:
                seqs.append((s.sid, y))
        pools[b] = seqs
    rows = []
    rng = np.random.default_rng(seed)
    t0 = time.time()
    for i, p in enumerate(sess):
        views = {u: MVSession(p, all_units=(u == "all")) for u in units}
        s = views[units[0]]
        if s.split != split:
            raise RuntimeError("session from another split reached the analysis; refusing")
        Xcache = {}
        for var in vars:
            b = _base(var)
            pool = None
            if b in pools:
                pool = np.concatenate([y for sid, y in pools[b] if sid != s.sid] or [np.zeros(1)])
            for w in windows:
                d = design(s, var, w)
                if d[0] is None:
                    continue
                if var.startswith("ctl_"):
                    v, _, crng = d
                    y = null_draw(var, v, len(s.contrast), pool, crng)
                else:
                    v, y = d
                if _base(var) == "sprior":
                    _ALPHA_CUR[0] = subjective_alpha(s)
                    if len(y) < 2 * MIN_PER_CLASS or np.std(y) == 0:
                        continue
                elif min((y == 0).sum(), (y == 1).sum()) < MIN_PER_CLASS:
                    continue
                Ys = {k: label_matrix(y, k, n_perm, rng, var, v, len(s.contrast), pool) for k in nulls}
                win = f"{WINDOW_OF.get(b, b)}_{w}"
                for u in units:
                    sv = views[u]
                    for r in sv.regions:
                        key = (u, r, win)
                        if key not in Xcache:
                            Xcache[key] = sv.X(r, win)
                        X = Xcache[key]
                        if X is None:
                            continue
                        X = X[v]
                        keep = X.std(0) > 0
                        if keep.sum() < MIN_UNITS:
                            continue
                        X = X[:, keep]
                        row = dict(sid=s.sid, subject=s.subject, lab=s.lab, var=var, window=w, units=u,
                                   region=r, n_units=int(X.shape[1]), n_trials=int(len(y)))
                        for k, Y in Ys.items():
                            st, nsig, eff = stats(X, Y)
                            for m, val in st.items():
                                row[f"p_{m}_{k}"] = _pv(val)
                        row["nsig"], row["eff"] = nsig, eff
                        rows.append(row)
        if (i + 1) % 10 == 0 or i + 1 == len(sess):
            el = (time.time() - t0) / 60
            log(f"  {i+1}/{len(sess)} sessions ({el:.1f} min, ~{el / (i + 1) * (len(sess) - i - 1):.0f} min left)")
    return pd.DataFrame(rows)


# =============================================================================== stage 2
def _bh(p, q=0.05):
    n = len(p)
    if n == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    sig = np.zeros(n, bool)
    sig[o[:k]] = True
    return sig


class Universes:
    """Per-universe (region x mouse) matrices so any study (set of mice) is a matrix-vector product."""

    def __init__(self, df):
        self.mice = sorted(df.subject.unique())
        self.mouse_lab = df.groupby("subject").lab.first().reindex(self.mice).to_numpy()
        self.mats = {}
        mi = {m: j for j, m in enumerate(self.mice)}
        for (var, w, u), g in df.groupby(["var", "window", "units"]):
            regions = sorted(g.region.unique())
            ri = {r: j for j, r in enumerate(regions)}
            R, M = len(regions), len(self.mice)
            rr = g.region.map(ri).to_numpy()
            mm = g.subject.map(mi).to_numpy()
            C = np.zeros((R, M))
            np.add.at(C, (rr, mm), 1)
            K = np.zeros((R, M))
            np.add.at(K, (rr, mm), g.nsig.to_numpy())
            N = np.zeros((R, M))
            np.add.at(N, (rr, mm), g.n_units.to_numpy())
            for meth in METHODS:
                for nl in NULLS:
                    col = f"p_{meth}_{nl}"
                    if col not in g:
                        continue
                    S = np.zeros((R, M))
                    np.add.at(S, (rr, mm), -2 * np.log(np.clip(g[col].to_numpy(), 1e-12, 1)))
                    pm = np.where(C > 0, chi2.sf(S, 2 * np.maximum(C, 1)), 1.0)
                    Lm = np.where(C > 0, -2 * np.log(np.clip(pm, 1e-300, 1)), 0.0)
                    self.mats[(var, w, u, meth, nl)] = dict(regions=regions, S=S, C=C, Lm=Lm, Cm=(C > 0).astype(float))
            self.mats[(var, w, u, "single", "none")] = dict(regions=regions, K=K, N=N, C=C)

    def keys(self, var=None):
        out = []
        for (v, w, u, meth, nl) in self.mats:
            if var is not None and v != var:
                continue
            if nl == "none":
                for inc in INCLS:
                    out.append((v, w, u, meth, nl, "pooled", inc))
            else:
                for lev in LEVELS:
                    for inc in INCLS:
                        out.append((v, w, u, meth, nl, lev, inc))
        return out

    def verdict(self, key, a):
        """a: (M,) 0/1 mouse indicator -> (sig int8 per region: 1/0, -1 untested; sessions per region)."""
        v, w, u, meth, nl, lev, inc = key
        d = self.mats[(v, w, u, meth, nl)]
        n = d["C"] @ a
        tested = n >= inc
        out = np.full(len(n), -1, np.int8)
        if not tested.any():
            return out, n
        if lev == "pooled":
            k, nn = d["K"] @ a, d["N"] @ a
            p = binom.sf(k - 1, np.maximum(nn, 1), 0.05)
        elif lev == "mouse":
            p = chi2.sf(d["Lm"] @ a, 2 * np.maximum(d["Cm"] @ a, 1))
        else:
            p = chi2.sf(d["S"] @ a, 2 * np.maximum(n, 1))
        out[tested] = _bh(p[tested]).astype(np.int8)
        return out, n

    def regions(self, key):
        return self.mats[key[:5]]["regions"]


def default_key(var, **kw):
    d = dict(DEFAULT, **kw)
    return (var, d["window"], d["units"], d["method"], d["null"], d["level"], d["incl"])


def sbin(n):
    return "2" if n <= 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9+"


SBINS = ("2", "3-4", "5-8", "9+")


def compare_pairs(U, keys, pairs, track=()):
    """For each key: replication P(sig B | sig A) (symmetric), agreement, claims per study.
    track: keys for which per-region hits and sessions-bin strata are recorded."""
    res = {k: np.zeros(4) for k in keys}          # hits, claims, agree, tested
    per = {k: {} for k in track}
    strat = {k: {b: [0, 0] for b in SBINS} for k in track}
    for a, b in pairs:
        for k in keys:
            va, na = U.verdict(k, a)
            vb, nb = U.verdict(k, b)
            both = (va >= 0) & (vb >= 0)
            sa, sb = (va == 1) & both, (vb == 1) & both
            r = res[k]
            r[0] += (sa & sb).sum() * 2
            r[1] += sa.sum() + sb.sum()
            r[2] += (va[both] == vb[both]).sum()
            r[3] += both.sum()
            if k in per:
                regs = U.regions(k)
                for idx in np.nonzero(sa | sb)[0]:
                    for x, yv, nx, ny in ((sa[idx], sb[idx], na[idx], nb[idx]), (sb[idx], sa[idx], nb[idx], na[idx])):
                        if x:
                            pc = per[k].setdefault(regs[idx], [0, 0])
                            pc[0] += int(yv)
                            pc[1] += 1
                            st = strat[k][sbin(min(nx, ny))]
                            st[0] += int(yv)
                            st[1] += 1
    out = {k: dict(replication=r[0] / r[1] if r[1] else np.nan, agreement=r[2] / r[3] if r[3] else np.nan,
                   claims_per_study=r[1] / (2 * len(pairs)) if pairs else np.nan) for k, r in res.items()}
    return out, per, strat


def mouse_pairs(M, k, n_rep, rng):
    out = []
    for _ in range(n_rep):
        p = rng.permutation(M)
        a, b = np.zeros(M), np.zeros(M)
        a[p[:k]] = 1
        b[p[k:2 * k]] = 1
        out.append((a, b))
    return out


def lab_pairs(U, n_rep, rng):
    labs = np.unique(U.mouse_lab)
    M = len(U.mice)
    labp, matched = [], []
    for _ in range(n_rep):
        sel = rng.permutation(labs)[: len(labs) // 2]
        a = np.isin(U.mouse_lab, sel).astype(float)
        b = 1 - a
        labp.append((a, b))
        p = rng.permutation(M)
        na = int(a.sum())
        a2, b2 = np.zeros(M), np.zeros(M)
        a2[p[:na]] = 1
        b2[p[na:]] = 1
        matched.append((a2, b2))
    return labp, matched


KS = (2, 3, 4, 6, 8, 12, 16, 20, 24)


def k1_vs_ibl(U, df, manifest_dir):
    import pandas as pd
    out = {}
    for var, fname in IBL_TABLE.items():
        f = Path(manifest_dir) / f"{fname}.csv"
        if not f.exists():
            continue
        ibl = pd.read_csv(f).set_index("region")
        key = default_key(var)
        if key[:5] not in U.mats:
            continue
        v, _ = U.verdict(key, np.ones(len(U.mice)))
        regs = U.regions(key)
        ours = pd.Series({r: v[i] for i, r in enumerate(regs) if v[i] >= 0})
        eff = df[(df["var"] == var) & (df.window == "a") & (df.units == "good")].groupby("region").eff.mean()
        common = [r for r in ours.index if r in ibl.index and np.isfinite(ibl.loc[r, "decoding_effect"])]
        if len(common) < 10:
            continue
        a = ours[common].to_numpy().astype(bool)
        b = ibl.loc[common, "decoding_significant"].to_numpy().astype(bool)
        po = (a == b).mean()
        pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
        rho = spearmanr(eff[common], ibl.loc[common, "decoding_effect"]).correlation
        out[var] = dict(regions=len(common), agreement=float(po), kappa=float((po - pe) / (1 - pe)) if pe < 1 else np.nan,
                        spearman_effect=float(rho), ours_sig=float(a.mean()), ibl_sig=float(b.mean()),
                        passed=bool(rho >= 0.5))
    return out


def run_explore(cache_dir, out_dir, n_perm=200, n_rep=300, manifest_dir=None, log=print, stats_file=None, split="explore"):
    import pandas as pd
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sf = Path(stats_file) if stats_file else out / "session_stats.csv.gz"
    if sf.exists():
        log(f"step 1/5: reusing {sf}")
        df = pd.read_csv(sf)
        missing = [v for v in VARS if v not in set(df["var"])]
        if missing:
            paths = sorted(Path(cache_dir).glob("*.npz"))
            log(f"  computing the added variables {missing} (explore split only) ...")
            extra = session_table(paths, split, n_perm, vars=tuple(missing), log=log)
            df = pd.concat([df, extra], ignore_index=True)
            df.to_csv(sf, index=False)
            log(f"  saved {sf} ({len(df)} rows)")
    else:
        paths = sorted(Path(cache_dir).glob("*.npz"))
        log(f"step 1/5: session-level statistics on the {split.upper()} split ({len(paths)} cached sessions in total) ...")
        df = session_table(paths, split, n_perm, log=log)
        df.to_csv(sf, index=False)
        log(f"  saved {sf} ({len(df)} rows)")
    if df.empty:
        raise SystemExit("no explore data")
    U = Universes(df)
    M = len(U.mice)
    labs = np.unique(U.mouse_lab)
    log(f"  {df.sid.nunique()} sessions, {M} mice, {len(labs)} labs")
    summary = dict(n_sessions=int(df.sid.nunique()), n_mice=M, n_labs=len(labs), n_perm=n_perm, n_rep=n_rep)
    allm = np.ones(M)
    rng = np.random.default_rng(1)

    log("step 2/5: calibration (fake variables) and full-explore verdicts ...")
    calib = {}
    for var in VARS:
        if not var.startswith("ctl_"):
            continue
        for nl in NULLS:
            fr = []
            for k in U.keys(var):
                if k[4] == nl and k[5] == "session":
                    v, _ = U.verdict(k, allm)
                    if (v >= 0).any():
                        fr.append((v == 1).sum() / (v >= 0).sum())
            calib.setdefault(var, {})[nl] = float(np.mean(fr)) if fr else np.nan
    summary["calibration_false_positive_share"] = calib
    for var, d in calib.items():
        log(f"  {var:13s} false-positive share: " + ", ".join(f"{n} {x:.0%}" for n, x in d.items()))
    worst = max(calib[v]["pseudo"] for v in calib)
    # fake-variable flag shares for the pre-registered claims (default units/window, all mice)
    fake = {}
    for var in ("ctl_stim", "ctl_choice", "ctl_feedback", "ctl_prior", "ctl_sprior", "ctl_stim_all"):
        if default_key(var)[:5] not in U.mats:
            continue
        d = {}
        for name, kk in (("pseudo", default_key(var)), ("shuffle", default_key(var, null="shuffle")),
                         ("circ", default_key(var, null="circ")),
                         ("pooled", default_key(var, level="pooled", method="single", null="none"))):
            v_, _ = U.verdict(kk, allm)
            d[name] = float((v_ == 1).sum() / max((v_ >= 0).sum(), 1))
        fake[var] = d
    summary["fake_flag_share"] = fake
    summary["calibration_ok"] = bool(worst <= 0.10)

    flips, agree = {}, {}
    for var in BASES:
        keys = [k for k in U.keys(var)]
        V = {k: U.verdict(k, allm)[0] for k in keys}
        regs = U.regions(default_key(var))
        if not regs:
            continue
        # regions tested in every universe that shares the region list (same var/window/units)
        k0 = default_key(var)
        v0 = V[k0]
        f = {}
        alts = dict(method=[("method", m) for m in ("single", "euclid")], null=[("null", n) for n in ("shuffle", "circ")],
                    units=[("units", "all")], window=[("window", "b")], level=[("level", "mouse")], incl=[("incl", 5)])
        for axis, opts in alts.items():
            for name, val in opts:
                k = default_key(var, **{name: val})
                if k not in V:
                    continue
                r1, r2 = U.regions(k0), U.regions(k)
                common = sorted(set(r1) & set(r2))
                a = np.array([v0[r1.index(r)] for r in common])
                b = np.array([V[k][r2.index(r)] for r in common])
                ok = (a >= 0) & (b >= 0)
                f[f"{axis}:{val}"] = float((a[ok] != b[ok]).mean()) if ok.any() else np.nan
        kp = default_key(var, level="pooled", method="single", null="none")
        if kp in V:
            r1, r2 = U.regions(k0), U.regions(kp)
            common = sorted(set(r1) & set(r2))
            a = np.array([v0[r1.index(r)] for r in common])
            b = np.array([V[kp][r2.index(r)] for r in common])
            ok = (a >= 0) & (b >= 0)
            f["level:pooled-neurons"] = float((a[ok] != b[ok]).mean()) if ok.any() else np.nan
        flips[var] = f
        # multiverse agreement across all universes on regions present everywhere
        allregs = set.intersection(*[set(U.regions(k)) for k in keys])
        regsl = sorted(allregs)
        mat = np.array([[V[k][U.regions(k).index(r)] for k in keys] for r in regsl])
        good = (mat >= 0).all(1)
        mat = mat[good]
        if len(mat):
            maj = mat.mean(1) >= 0.5
            ag = (mat == maj[:, None]).mean(1)
            agree[var] = dict(regions=int(len(mat)), universes=len(keys), mean_agreement=float(ag.mean()),
                              frac_below_70=float((ag < 0.7).mean()), sig_share_default=float((v0 == 1).sum() / max((v0 >= 0).sum(), 1)))
        log(f"  {var}: significant in default universe {(v0 == 1).sum()}/{(v0 >= 0).sum()} regions; "
            f"share of regions flipped by changing ONE choice: " + ", ".join(f"{a}={x:.0%}" for a, x in f.items()))
        if var in agree:
            log(f"     across all {len(keys)} universes: mean agreement {agree[var]['mean_agreement']:.0%}, "
                f"{agree[var]['frac_below_70']:.0%} of regions below 70%")
    summary["flips"], summary["multiverse_agreement"] = flips, agree

    log("step 3/5: K1 - comparison with IBL's published region verdicts ...")
    k1 = k1_vs_ibl(U, df, manifest_dir) if manifest_dir else {}
    summary["K1"] = k1
    for var, d in k1.items():
        log(f"  {var}: {d['regions']} regions; verdict agreement {d['agreement']:.0%} (kappa {d['kappa']:.2f}); "
            f"effect rank correlation {d['spearman_effect']:.2f}; significant ours {d['ours_sig']:.0%} vs IBL {d['ibl_sig']:.0%} "
            f"-> {'PASS' if d['passed'] else 'FAIL'}")

    log("step 4/5: different mice (k-mouse studies) ...")
    main = [default_key(v) for v in BASES if default_key(v)[:5] in U.mats]
    main_mouse = [default_key(v, level="mouse") for v in BASES if default_key(v)[:5] in U.mats]
    main_pool = [default_key(v, level="pooled", method="single", null="none") for v in BASES if default_key(v)[:5] in U.mats]
    other = [default_key(v, method=m) for v in BASES for m in ("single", "euclid") if default_key(v)[:5] in U.mats]
    curves, per_k, strat_all = {}, {}, {k: {b: [0, 0] for b in SBINS} for k in main}
    for k in KS:
        if 2 * k > M:
            continue
        pairs = mouse_pairs(M, k, n_rep, rng)
        res, per, strat = compare_pairs(U, main + main_mouse + main_pool + other, pairs, track=main)
        for key, d in res.items():
            curves.setdefault(key, {})[k] = d
        for key in main:
            per_k.setdefault(key, {})[k] = per[key]
            for b in SBINS:
                strat_all[key][b][0] += strat[key][b][0]
                strat_all[key][b][1] += strat[key][b][1]
    for v in BASES:
        key = default_key(v)
        if key not in curves:
            continue
        ks = sorted(curves[key])
        log(f"  {v}: P(replicate | significant) by mice per study ({', '.join(str(k) for k in ks)})")
        for lab_, kk in (("decode/session", key), ("decode/mouse", default_key(v, level="mouse")),
                         ("single/session", default_key(v, method="single")), ("euclid/session", default_key(v, method="euclid")),
                         ("single/pooled", default_key(v, level="pooled", method="single", null="none"))):
            if kk in curves:
                log(f"     {lab_:15s}" + "".join(f"{curves[kk][k]['replication']:>6.0%}" for k in ks))
        log(f"     {'claims/study':15s}" + "".join(f"{curves[key][k]['claims_per_study']:>6.1f}" for k in ks))
        log("     by sessions per region: " + "; ".join(
            f"{b}: {h / t:.0%} (n={t})" if t else f"{b}: -" for b, (h, t) in strat_all[key].items()))
    # N80 per claim
    n80 = {}
    for key in main:
        d = {}
        for k, per in per_k.get(key, {}).items():
            for r, (h, t) in per.items():
                if t >= 10:
                    d.setdefault(r, {})[k] = h / t
        n80[key[0]] = {r: next((k for k in sorted(v) if v[k] >= 0.8), None) for r, v in d.items()}
    summary["N80"] = n80
    for v, d in n80.items():
        vals = [x for x in d.values() if x is not None]
        log(f"  {v}: N80 reached (<= {max(KS)} mice) for {len(vals)}/{len(d)} claims; median N80 = "
            f"{np.median(vals) if vals else float('nan')}")

    log("step 5/5: different LABS vs size-matched random mice ...")
    lp, mp = lab_pairs(U, n_rep, rng)
    fake_keys = [kk for v in ("ctl_prior", "ctl_choice") for kk in
                 (default_key(v, null="shuffle"), default_key(v, level="pooled", method="single", null="none"))
                 if kk[:5] in U.mats]
    rf, _, _ = compare_pairs(U, fake_keys, lp)
    summary["fake_lab_replication"] = {"|".join(map(str, kk)): rf[kk]["replication"] for kk in fake_keys}
    keys = main + main_mouse + main_pool
    rl, _, _ = compare_pairs(U, keys, lp)
    rm, _, _ = compare_pairs(U, keys, mp)
    labres = {}
    for key in keys:
        labres["|".join(map(str, key))] = dict(across_labs=rl[key]["replication"], matched_random=rm[key]["replication"])
    summary["lab_test"] = labres
    for v in BASES:
        for lab_, kk in (("session", default_key(v)), ("mouse", default_key(v, level="mouse")),
                         ("pooled", default_key(v, level="pooled", method="single", null="none"))):
            if kk in rl:
                log(f"  {v:8s} {lab_:8s} replication across labs {rl[kk]['replication']:.0%} "
                    f"vs random halves {rm[kk]['replication']:.0%}")

    summary["curves"] = {"|".join(map(str, k)): {str(kk): vv for kk, vv in d.items()} for k, d in curves.items()}
    summary["strata"] = {"|".join(map(str, k)): d for k, d in strat_all.items()}
    (out / f"{split}.json").write_text(json.dumps(summary, indent=1, default=lambda x: None if x is None else float(x)))
    _figures(curves, strat_all, flips, out)
    log("")
    log(f"CALIBRATION (pseudo null, fake variables): worst false-positive share {worst:.0%} -> "
        f"{'OK' if summary['calibration_ok'] else 'NOT OK'}")
    if k1:
        log(f"K1 (match IBL): " + ", ".join(f"{v} {'PASS' if d['passed'] else 'FAIL'}" for v, d in k1.items()))
    log(f"saved {out/(split + '.json')}, {sf}, figures in {out}")
    return summary


def _figures(curves, strat, flips, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for v in BASES:
        key = default_key(v)
        if key in curves:
            ks = sorted(curves[key])
            axs[0].plot(ks, [curves[key][k]["replication"] for k in ks], marker="o", label=v)
        if key in strat:
            ys = [h / t if t else np.nan for h, t in strat[key].values()]
            axs[1].plot(SBINS, ys, marker="o", label=v)
    for ax, t in zip(axs[:2], ("by mice per study", "by sessions per region")):
        ax.axhline(0.8, ls="--", c="gray", lw=1)
        ax.set_ylim(0, 1)
        ax.set_title(f"P(replicate | significant), {t}")
        ax.legend(fontsize=8)
    axs[0].axvspan(3, 8, color="gray", alpha=0.1)
    names = sorted({a for d in flips.values() for a in d})
    w = 0.8 / max(len(flips), 1)
    for i, (v, d) in enumerate(flips.items()):
        axs[2].bar(np.arange(len(names)) + i * w, [d.get(n, np.nan) for n in names], w, label=v)
    axs[2].set_xticks(np.arange(len(names)) + 0.4 - w / 2)
    axs[2].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    axs[2].set_title("share of regions whose verdict flips (one change)")
    axs[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "explore.png", dpi=130)


# =============================================================================== K4: simulated brains
def simulate_cache(src_dir, dst_dir, seed, truth, het=0.5, split="explore", vars=("stim", "prior")):
    """Copy the explore cache structure (trials, regions, unit counts, base rates) with planted effects.
    truth: {var: {region: effect}}; per-mouse effect multipliers drawn with this seed (=> new 'animals')."""
    rng = np.random.default_rng(seed)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    mult = {}
    for p in sorted(Path(src_dir).glob("*.npz")):
        s = MVSession(p, all_units=True)
        if s.split != split:
            continue
        with np.load(p, allow_pickle=False) as zf:
            z = {k: zf[k] for k in zf.files}
        arrays = {k: v for k, v in z.items() if not any(k.startswith(f"{b}_") for b in BASES)}
        m = mult.setdefault(s.subject, max(0.0, rng.normal(1.0, het)))
        for b in vars:
            v, y = design(s, b, "a")[:2]
            lab = np.zeros(len(s.contrast))
            if v is not None:
                lab[v] = 2 * y - 1
            for r in s.regions:
                X = z[f"{b}_a__{r}"].astype(float)
                base = np.maximum(X.mean(0), 0.05)
                eff = truth.get(b, {}).get(r, 0.0) * m
                wgt = rng.normal(0, 1, X.shape[1])
                rate = base[None, :] * np.exp(np.clip(eff * wgt[None, :] * lab[:, None], -3, 3))
                arrays[f"{b}_a__{r}"] = rng.poisson(rate).astype(np.int16)
        np.savez_compressed(dst / p.name, **arrays)
    return dst


def run_k4(cache_dir, out_dir, n_perm=100, n_rep=200, log=print):
    """Estimator check: P(k) estimated within one simulated dataset vs the truth (studies drawn from
    two independently simulated datasets = different animals)."""
    import pandas as pd
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    probe = [MVSession(p) for p in sorted(Path(cache_dir).glob("*.npz"))]
    regs = sorted({r for s in probe if s.split == "explore" for r in s.regions})
    truth = {}
    for b in ("stim", "prior"):
        chosen = rng.choice(regs, size=max(1, len(regs) // 4), replace=False)
        truth[b] = {r: float(rng.uniform(0.03, 0.25)) for r in chosen}
    dfs = []
    for i, seed in enumerate((101, 202)):
        d = simulate_cache(cache_dir, out / f"sim{i}", seed, truth)
        log(f"  simulated dataset {i + 1}/2 written; computing statistics ...")
        dfs.append(session_table(sorted(d.glob("*.npz")), "explore", n_perm, vars=("stim", "prior"),
                                 windows=("a",), units=("all",), nulls=("pseudo",), log=log))
    U1, U2 = Universes(dfs[0]), Universes(dfs[1])
    M = len(U1.mice)
    report = {}
    for b in ("stim", "prior"):
        key = (b, "a", "all", "decode", "pseudo", "session", 2)
        if key[:5] not in U1.mats:
            continue
        v, _ = U1.verdict(key, np.ones(M))
        regs1 = U1.regions(key)
        null_regs = [i for i, r in enumerate(regs1) if r not in truth[b] and v[i] >= 0]
        fp = float(np.mean([v[i] == 1 for i in null_regs])) if null_regs else np.nan
        rows = {}

        def cross(UA, UB, pairs):
            h = c = 0
            for a, bb in pairs:
                va, _ = UA.verdict(key, a)
                vb, _ = UB.verdict(key, bb)
                ra, rb = UA.regions(key), UB.regions(key)
                for idx in np.nonzero(va == 1)[0]:
                    r = ra[idx]
                    if r in rb and vb[rb.index(r)] >= 0:
                        c += 1
                        h += int(vb[rb.index(r)] == 1)
            return h, c

        for k in KS:
            if 2 * k > M:
                continue
            pairs = mouse_pairs(M, k, n_rep, rng)
            # estimate: two disjoint k-mouse studies inside ONE dataset (averaged over both datasets);
            # truth: study A from one dataset, study B (disjoint mice) from the OTHER (= new animals)
            e1, _, _ = compare_pairs(U1, [key], pairs)
            e2, _, _ = compare_pairs(U2, [key], pairs)
            h1, c1 = cross(U1, U2, pairs)
            h2, c2 = cross(U2, U1, pairs)
            est = np.nanmean([e1[key]["replication"], e2[key]["replication"]])
            rows[k] = dict(estimated=float(est), true=(h1 + h2) / (c1 + c2) if c1 + c2 else np.nan)
        report[b] = dict(false_positive_share=fp, curve=rows)
        log(f"  {b}: false positives on null regions {fp:.0%}")
        for k, d in rows.items():
            log(f"     k={k:2d}: estimated {d['estimated']:.0%}  true {d['true']:.0%}")
    ok = all(abs(d["estimated"] - d["true"]) <= 0.05 for r in report.values() for d in r["curve"].values()
             if np.isfinite(d["estimated"]) and np.isfinite(d["true"])) and \
        all(r["false_positive_share"] <= 0.10 for r in report.values() if np.isfinite(r["false_positive_share"]))
    report["K4_pass"] = bool(ok)
    (out / "k4.json").write_text(json.dumps(report, indent=1, default=float))
    log(f"K4 (estimator vs truth within 5 points; false positives <= 10%): {'PASS' if ok else 'FAIL'}")
    return report


# =============================================================================== K1 retry: IBL-matched stimulus decoder
# Recipe read from int-brain-lab/paper-brain-wide-map (settings_stimside.py, functions/decoding.py,
# process_outputs.py, meta/bwm_figs.py::group_into_regions):
#   trials   non-zero contrast, valid choice, 0.08 s <= reaction time <= 2.0 s; window 0-100 ms after stimOn
#   decoder  L1 logistic regression (liblinear), balanced class weights, C from a grid by inner CV,
#            score = balanced accuracy on held-out folds (IBL averages 10 repeats; we use 2)
#   null     pseudo-sessions from the task's generative process (IBL: 200; we use 100)
#   region   Fisher-combined p across sessions (>= 2 sessions, >= 5 units), BH-FDR q = 0.01,
#            effect = median(score) - median(null medians)
# Simplification (stated in Methods): C is chosen once per session-region by nested CV on the real
# labels and reused for that session's pseudo-sessions, so the null costs one CV per pseudo-session.
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _cv_bacc(X, y, C, folds, seed):
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced",
                                   tol=1e-3, max_iter=1000).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return balanced_accuracy_score(y, pred)


def _nested_bacc(X, y, seed, outer=5, inner=3, grid=(1e-2, 1e-1, 1.0, 10.0)):
    """IBL-style nested cross-validation: inside every outer training fold, C is chosen by an inner CV
    on that fold only; the held-out fold is predicted with that C. Same procedure for real and null labels.
    (Fix 2026-10-13: the first version chose C once on all trials of the real labels and reused it for the
    nulls, which gives the real score an optimistic, region-dependent head start.)"""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold

    def fit(Xa, ya, C):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return LogisticRegression(penalty="l1", solver="liblinear", C=C, class_weight="balanced",
                                      tol=1e-3, max_iter=1000).fit(Xa, ya)

    pred = np.zeros(len(y))
    for tr, te in StratifiedKFold(outer, shuffle=True, random_state=seed).split(X, y):
        Xt, yt = X[tr], y[tr]
        best, bestC = -1.0, grid[0]
        for C in grid:
            pi = np.zeros(len(yt))
            for itr, ite in StratifiedKFold(inner, shuffle=True, random_state=seed + 1).split(Xt, yt):
                pi[ite] = fit(Xt[itr], yt[itr], C).predict(Xt[ite])
            sc = balanced_accuracy_score(yt, pi)
            if sc > best:
                best, bestC = sc, C
        pred[te] = fit(Xt, yt, bestC).predict(X[te])
    return balanced_accuracy_score(y, pred)


def _ibl_session(p, n_pseudo, seed, split="explore"):
    s = MVSession(p)
    if s.split != split:
        return []
    v = s.valid["stimOn"] & s.valid["firstMovement"] & (s.contrast != 0) & (s.choice >= 0) \
        & np.isfinite(s.rt) & (s.rt >= 0.08) & (s.rt <= 2.0)
    y = (s.contrast[v] > 0).astype(int)
    if min((y == 0).sum(), (y == 1).sum()) < MIN_PER_CLASS:
        return []
    rng = np.random.default_rng(seed)
    nulls = [((rng.random(len(s.contrast)) >= pseudo_pl(len(s.contrast), rng))[v]).astype(int) for _ in range(n_pseudo)]
    nulls = [yn for yn in nulls if min((yn == 0).sum(), (yn == 1).sum()) >= 5]
    rows = []
    for r in s.regions:
        X = s.X(r, "stim_a")
        if X is None:
            continue
        X = X[v]
        X = X[:, X.std(0) > 0]
        if X.shape[1] < MIN_UNITS:
            continue
        X = (X - X.mean(0)) / X.std(0)
        score = _nested_bacc(X, y, seed)
        null = np.array([_nested_bacc(X, yn, seed) for yn in nulls])
        rows.append(dict(sid=s.sid, subject=s.subject, lab=s.lab, var="stim_ibl", window="a", units="good",
                         region=r, n_units=int(X.shape[1]), n_trials=int(len(y)), nsig=0, eff=float(score),
                         null_median=float(np.median(null)),
                         p_decode_pseudo=float((1 + (null >= score).sum()) / (1 + len(null)))))
    return rows


def run_k1_stim(cache_dir, out_dir, manifest_dir, n_pseudo=100, n_rep=300, n_jobs=-1, log=print, split="explore", min_sessions=2):
    import pandas as pd
    from joblib import Parallel, delayed
    from scipy.stats import combine_pvalues
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sf = out / "stim_ibl_stats_v2.csv.gz"
    if sf.exists():
        df = pd.read_csv(sf)
        log(f"reusing {sf}")
    else:
        paths = [p for p in sorted(Path(cache_dir).glob("*.npz")) if MVSession(p).split == split]
        log(f"IBL-matched stimulus decoding (nested CV, v2) on {len(paths)} explore sessions (parallel; ~45-90 min) ...")
        t0 = time.time()
        res = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_ibl_session)(p, n_pseudo, 1000 + i, split) for i, p in enumerate(paths))
        df = pd.DataFrame([r for rows in res for r in rows])
        df.to_csv(sf, index=False)
        log(f"  done in {(time.time() - t0) / 60:.0f} min; saved {sf}")
    # region table exactly as IBL builds it
    reg = []
    for r, g in df.groupby("region"):
        if len(g) < 2:
            continue
        pv = np.clip(g.p_decode_pseudo.to_numpy(), 1.0 / (n_pseudo + 1), 1)
        reg_n = len(g)
        reg.append(dict(region=r, n_sessions=reg_n, p=combine_pvalues(pv, method="fisher")[1],
                        effect=float(np.median(g.eff) - np.median(g.null_median))))
    reg = pd.DataFrame(reg).set_index("region")
    for q in (0.01, 0.05):
        reg[f"sig_q{q}"] = _bh(reg.p.to_numpy(), q)
    ibl = pd.read_csv(Path(manifest_dir) / "stim.csv").set_index("region")
    common = [r for r in reg.index if r in ibl.index and np.isfinite(ibl.loc[r, "decoding_effect"])
              and reg.loc[r, "n_sessions"] >= min_sessions]
    rho = spearmanr(reg.loc[common, "effect"], ibl.loc[common, "decoding_effect"]).correlation
    agree = float((reg.loc[common, "sig_q0.01"].to_numpy() == ibl.loc[common, "decoding_significant"].astype(bool).to_numpy()).mean())
    summary = dict(split=split, min_sessions=min_sessions, regions=len(common), spearman_effect=float(rho), agreement_q01=agree, noise_ceiling=None,
                   ours_sig_q01=float(reg.loc[common, "sig_q0.01"].mean()),
                   ibl_sig=float(ibl.loc[common, "decoding_significant"].mean()), passed=bool(rho >= 0.5))
    rs = []
    rng0 = np.random.default_rng(5)
    mice_ = df.subject.unique()
    for _ in range(200):
        pm = rng0.permutation(mice_)
        A = set(pm[: len(pm) // 2])

        def eff_of(g):
            return g.groupby("region").apply(lambda h: np.median(h.eff) - np.median(h.null_median))
        ea, eb = eff_of(df[df.subject.isin(A)]), eff_of(df[~df.subject.isin(A)])
        c = ea.index.intersection(eb.index)
        rs.append(spearmanr(ea[c], eb[c]).correlation)
    rel_half = float(np.nanmean(rs))
    rel_full = 2 * rel_half / (1 + rel_half)
    summary_ceiling = dict(split_half=rel_half, explore_reliability=rel_full)
    log(f"K1 stimulus (IBL-matched, v2, {split}, regions with >= {min_sessions} sessions): {len(common)} regions; effect rank correlation with IBL {rho:.2f}; "
        f"verdict agreement {agree:.0%}; significant ours {summary['ours_sig_q01']:.0%} vs IBL {summary['ibl_sig']:.0%} "
        f"-> {'PASS' if summary['passed'] else 'FAIL'} (pre-set line: >= 0.50)")
    summary["noise_ceiling"] = summary_ceiling
    log(f"  (descriptive) reliability of our own region ranking within the {split} mice: split-half {rel_half:.2f}, "
        f"full explore ~{rel_full:.2f}; this caps the possible agreement with IBL")
    # replication with the IBL-matched pipeline (only decode / pseudo exists for this variable)
    U = Universes(df)
    M = len(U.mice)
    rng = np.random.default_rng(1)
    key = ("stim_ibl", "a", "good", "decode", "pseudo", "session", 2)
    curve, strat_all = {}, {b: [0, 0] for b in SBINS}
    for k in KS:
        if 2 * k > M:
            continue
        r_, _, st = compare_pairs(U, [key], mouse_pairs(M, k, n_rep, rng), track=[key])
        curve[k] = r_[key]
        for b in SBINS:
            strat_all[b][0] += st[key][b][0]
            strat_all[b][1] += st[key][b][1]
    lp, mp = lab_pairs(U, n_rep, rng)
    rl, _, _ = compare_pairs(U, [key], lp)
    rm, _, _ = compare_pairs(U, [key], mp)
    log("  P(replicate | significant) by mice per study: " + ", ".join(f"k={k} {d['replication']:.0%}" for k, d in curve.items()))
    log("  by sessions per region: " + "; ".join(f"{b}: {h / t:.0%} (n={t})" if t else f"{b}: -" for b, (h, t) in strat_all.items()))
    log(f"  across labs {rl[key]['replication']:.0%} vs size-matched random mice {rm[key]['replication']:.0%}")
    summary.update(curve={str(k): d for k, d in curve.items()}, strata=strat_all,
                   lab_test=dict(across_labs=rl[key]["replication"], matched_random=rm[key]["replication"]))
    (out / "k1_stim.json").write_text(json.dumps(summary, indent=1, default=float))
    reg.to_csv(out / "stim_ibl_regions.csv")
    log(f"saved {out/'k1_stim.json'}")
    return summary


# =============================================================================== pre-registered evaluation
def evaluate_prereg(conf, expl=None, k1s=None):
    """PASS/FAIL for every pre-registered claim (MULTIVERSE pre-registration, locked on OSF), from the confirm run."""
    out = []

    def add(cid, text, value, ok):
        out.append(dict(id=cid, claim=text, value=None if value is None else float(value), passed=bool(ok)))

    ff = conf.get("fake_flag_share", {})
    for v in ("ctl_prior", "ctl_choice"):
        for null, name in (("shuffle", "label-shuffle null"), ("pooled", "pooled-neuron test")):
            x = ff.get(v, {}).get(null)
            add(f"P1:{v[4:]}:{null}", f"{name} flags >= 50% of regions for the fake {v[4:]}", x, x is not None and x >= 0.5)
    worst = max((d.get("pseudo", 1.0) for d in ff.values()), default=None)
    add("P2", "pseudo-session null flags <= 5% of regions for every fake variable", worst, worst is not None and worst <= 0.05)
    for k, x in conf.get("fake_lab_replication", {}).items():
        if k.startswith("ctl_prior|"):
            kind = "shuffle" if "|shuffle|" in k else "pooled"
            add(f"P3:{kind}", f"false fake-prior claims ({kind}) replicate across disjoint labs >= 80%", x, x >= 0.8)
    for v in ("choice", "sprior"):
        st = conf.get("strata", {}).get(f"{v}|a|good|decode|pseudo|session|2")
        if st:
            lo = st["2"][0] / max(st["2"][1], 1)
            hi = (st["5-8"][0] + st["9+"][0]) / max(st["5-8"][1] + st["9+"][1], 1)
            add(f"P4:{v}", f"{v}: replication with >= 5 sessions/region exceeds 2 sessions by >= 10 points", hi - lo, hi - lo >= 0.10)
    for v in ("choice", "feedback"):
        d = conf.get("lab_test", {}).get(f"{v}|a|good|decode|pseudo|session|2")
        if d:
            x = d["across_labs"] - d["matched_random"]
            add(f"P5:{v}", f"{v}: across-lab minus size-matched random replication within +/-5 points", x, abs(x) <= 0.05)
    fc = conf.get("flips", {}).get("choice", {})
    add("P6:window", "50-ms window shift flips >= 15% of choice verdicts", fc.get("window:b"), fc.get("window:b", 0) >= 0.15)
    add("P6:method", "single-neuron test instead of decoding flips >= 15% of choice verdicts", fc.get("method:single"), fc.get("method:single", 0) >= 0.15)
    for v in ("choice", "feedback"):
        x = conf.get("flips", {}).get(v, {}).get("level:mouse")
        add(f"P7:{v}", f"{v}: mouse-level instead of session-level statistics flips <= 5% of verdicts", x, x is not None and x <= 0.05)
    for v in ("choice", "feedback"):
        x = conf.get("K1", {}).get(v, {}).get("spearman_effect")
        add(f"V1:{v}", f"{v}: region ranking agrees with IBL (Spearman >= 0.5)", x, x is not None and x >= 0.5)
    if k1s is not None:
        x = k1s.get("spearman_effect") if k1s.get("regions", 0) >= 20 else None   # needs >= 20 regions to be evaluable
        add("V2:stimulus", "stimulus (IBL-matched decoder, regions with >= 5 sessions) agrees with IBL (Spearman >= 0.5)", x, x is not None and x >= 0.5)
    if expl:
        key = "choice|a|good|decode|pseudo|session|2"
        ce, cc = expl.get("curves", {}).get(key, {}), conf.get("curves", {}).get(key, {})
        common = [k for k in ce if k in cc]
        dev = max(abs(ce[k]["replication"] - cc[k]["replication"]) for k in common) if common else None
        add("C1", "choice replication curve within 10 points of the explore curve at every shared study size", dev, dev is not None and dev <= 0.10)
    return out
