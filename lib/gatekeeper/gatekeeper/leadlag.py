"""Does a region's activity change BEFORE the behavioral state switches?

Pipeline per session
  1. p_t = smoothed P(engaged) from the GLM-HMM. Smoothed (not filtered) posteriors are
     symmetric in time, so they do not bias lags; a filtered posterior lags behavior and
     would make every neural signal look like it "leads".
  2. Switches: p crosses 0.5 and stays on each side for >= `stable` trials.
  3. Region axis: for each region, the engaged-vs-disengaged difference in z-scored unit
     activity, estimated ONLY from confident trials (p > .9 or < .1) farther than `win`
     trials from any switch. The axis is therefore never fit to the transitions it is
     used to time.
  4. Switch-triggered average of the projected signal, oriented so that "toward engaged"
     is positive (disengage switches are sign-flipped), in a window of +-`win` trials.
  5. Timing: fit x(k) = a + b * sigmoid((k - k0) / s). k0 < 0 means the region's change
     is centered BEFORE the behavioral switch. Reference: the same fit to p itself.

Group statistics bootstrap over SESSIONS (the independent unit), with switches pooled
within each resampled session.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

WIN = 15
STABLE = 8


def find_switches(p, stable=STABLE, win=WIN):
    """Indices t where the state changes between t-1 and t, with stable runs on both sides."""
    s = (p > 0.5).astype(int)
    out = []
    for t in np.flatnonzero(np.diff(s)) + 1:
        if t - stable < 0 or t + stable > len(s):
            continue
        if (s[t - stable:t] == s[t - 1]).all() and (s[t:t + stable] == s[t]).all():
            if t - win >= 0 and t + win < len(s):
                out.append((t, 1 if s[t] == 1 else -1))      # +1: disengaged -> engaged
    return out


def region_axis_signal(X, p, switches, win=WIN, min_trials=20):
    """Project (T, N) activity onto the state axis estimated away from switches."""
    T = len(p)
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    far = np.ones(T, bool)
    for t, _ in switches:
        far[max(0, t - win):t + win] = False
    eng = far & (p > 0.9)
    dis = far & (p < 0.1)
    if eng.sum() < min_trials or dis.sum() < min_trials:
        return None
    w = Z[eng].mean(0) - Z[dis].mean(0)
    if np.linalg.norm(w) < 1e-9:
        return None
    x = Z @ (w / np.linalg.norm(w))
    return (x - x[far].mean()) / (x[far].std() + 1e-9)


def triggered(x, switches, win=WIN):
    """(n_switches, 2*win+1) signal around switches, oriented toward 'engaged' = up."""
    return np.array([sgn * x[t - win:t + win + 1] for t, sgn in switches]) if switches else np.zeros((0, 2 * win + 1))


def _sig(k, a, b, k0, s):
    return a + b / (1 + np.exp(-(k - k0) / s))


def onset(avg, win=WIN):
    """Center k0 of a sigmoid fit to a triggered average (NaN if it fails or the step is ~0)."""
    k = np.arange(-win, win + 1)
    b0 = avg[-5:].mean() - avg[:5].mean()
    if abs(b0) < 1e-6:
        return np.nan
    try:
        popt, _ = curve_fit(_sig, k, avg, p0=[avg[:5].mean(), b0, 0.0, 2.0],
                            bounds=([-np.inf, -np.inf, -win, 0.3], [np.inf, np.inf, win, win]), maxfev=5000)
    except Exception:
        return np.nan
    return popt[2] if popt[1] * b0 > 0 else np.nan


def asymmetry(x, p, maxlag=10):
    """Cross-correlation lead index: mean corr(x_t, p_{t+k}) over k>0 minus over k<0.
    > 0 when x predicts future p better than past p (x leads)."""
    xc, pc = x - x.mean(), p - p.mean()
    def cc(k):
        if k > 0:
            return np.corrcoef(xc[:-k], pc[k:])[0, 1]
        if k < 0:
            return np.corrcoef(xc[-k:], pc[:k])[0, 1]
        return np.corrcoef(xc, pc)[0, 1]
    pos = np.mean([cc(k) for k in range(1, maxlag + 1)])
    neg = np.mean([cc(-k) for k in range(1, maxlag + 1)])
    return pos - neg


def session_traces(sess, p, regions, win=WIN):
    """Per session: switch list and triggered matrices for p and each region."""
    sw = find_switches(p, win=win)
    out = dict(sid=sess.sid, n_switches=len(sw), p=triggered(p, sw, win), regions={}, asym={})
    for r in regions:
        if r not in sess.signals:
            continue
        x = region_axis_signal(sess.signals[r], p, sw, win)
        if x is None:
            continue
        out["regions"][r] = triggered(x, sw, win)
        out["asym"][r] = asymmetry(x, p)
    return out


def _pooled(traces, key, idx):
    mats = [traces[i]["p"] if key == "p" else traces[i]["regions"].get(key) for i in idx]
    mats = [m for m in mats if m is not None and len(m)]
    return np.concatenate(mats).mean(0) if mats else None


def group_stats(traces, regions, n_boot=2000, seed=0, win=WIN):
    """Point estimates + session-bootstrap 95% CIs for onset (relative to behavior) and asymmetry."""
    rng = np.random.default_rng(seed)
    n = len(traces)
    idx_all = np.arange(n)

    def est(idx):
        ref = onset(_pooled(traces, "p", idx), win)
        e = {}
        for r in regions:
            avg = _pooled(traces, r, idx)
            e[f"onset_{r}"] = (onset(avg, win) - ref) if avg is not None else np.nan
            a = [traces[i]["asym"][r] for i in idx if r in traces[i]["asym"]]
            e[f"asym_{r}"] = float(np.mean(a)) if a else np.nan
        for i, r1 in enumerate(regions):
            for r2 in regions[i + 1:]:
                e[f"onset_{r1}-{r2}"] = e[f"onset_{r1}"] - e[f"onset_{r2}"]
                e[f"asym_{r1}-{r2}"] = e[f"asym_{r1}"] - e[f"asym_{r2}"]
        return e

    point = est(idx_all)
    boots = [est(rng.integers(0, n, n)) for _ in range(n_boot)]
    res = {}
    for k, v in point.items():
        b = np.array([bb[k] for bb in boots], float)
        b = b[np.isfinite(b)]
        lo, hi = (np.percentile(b, [2.5, 97.5]) if len(b) > 10 else (np.nan, np.nan))
        res[k] = dict(est=float(v), lo=float(lo), hi=float(hi), n_boot_ok=int(len(b)))
    res["n_sessions"] = n
    res["n_switches"] = int(sum(t["n_switches"] for t in traces))
    return res
