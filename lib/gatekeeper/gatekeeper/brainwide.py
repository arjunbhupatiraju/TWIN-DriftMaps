"""Brain-wide version of modules 1-3: every region gets the same validated lead-lag and lapse
tests, with false-discovery-rate control across regions.

Per region r (using only sessions where r has >= min_units good units):
  onset_r      center of r's switch-triggered change minus that of behavior (trials; < 0 = before)
  asym_r       cross-correlation asymmetry (> 0 = r predicts future state better than past)
  onset_r - onset_pupil
  p_lead       intersection-union p-value: max(P_boot[onset >= -LEAD_MARGIN], P_boot[asym <= 0]).
               A region counts as a LEADER only if both hold, so both must be convincing.
  lapse        AUC gain of (behavior + pupil + r) over (behavior + pupil) for a lapse 5 trials ahead
Across regions:
  Benjamini-Hochberg q-values on p_lead (all analyzable regions) and on lapse p (leaders only)
  heterogeneity of onsets (Cochran's Q with bootstrap SEs): do regions change at different times?
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from . import lapse, leadlag
from .decision import HORIZON

# Minimum lead (trials). Behavioral switch times are only known to about +-1 trial, and in
# simulation concurrent regions showed small negative onset biases (~-0.1 to -0.3 trials) that
# few-session bootstraps could call "significant". A region must lead by MORE than this margin.
# Fixed on simulated data only (validation run 1, 2026-10-10), before any real brain-wide data.
LEAD_MARGIN = 1.0


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def region_sessions(sessions, region, min_units):
    return [i for i, s in enumerate(sessions) if region in s.signals and s.signals[region].shape[1] >= min_units]


def all_traces(sessions, posteriors, regions):
    return [leadlag.session_traces(s, p, list(regions) + ["pupil"]) for s, p in zip(sessions, posteriors)]


def _onset_from(traces, key, idx, win):
    avg = leadlag._pooled(traces, key, idx)
    return leadlag.onset(avg, win) if avg is not None else np.nan


def region_stats(traces, region, idx, n_boot=1000, seed=0, win=leadlag.WIN):
    """Point estimates, bootstrap SEs/CIs and one-sided p-values for one region."""
    idx = [i for i in idx if region in traces[i]["regions"]]
    if len(idx) < 2:
        return None
    rng = np.random.default_rng(seed)

    def est(ix):
        ref = _onset_from(traces, "p", ix, win)
        on = _onset_from(traces, region, ix, win) - ref
        pu = _onset_from(traces, "pupil", ix, win) - ref
        a = np.mean([traces[i]["asym"][region] for i in ix])
        return on, a, on - pu

    on0, a0, d0 = est(idx)
    B = np.array([est(list(rng.choice(idx, len(idx)))) for _ in range(n_boot)], float)
    on_b, a_b, d_b = B[:, 0], B[:, 1], B[:, 2]
    ok = np.isfinite(on_b)
    # failed fits (flat average) count against a lead
    p_on = (1 + np.sum(~ok) + np.sum(on_b[ok] >= -LEAD_MARGIN)) / (1 + n_boot)
    p_as = (1 + np.sum(a_b <= 0)) / (1 + n_boot)
    if not np.isfinite(on0):
        p_on = 1.0
    ci = lambda b: (np.nanpercentile(b, 2.5), np.nanpercentile(b, 97.5)) if np.isfinite(b).sum() > 10 else (np.nan, np.nan)
    return dict(region=region, n_sessions=len(idx), n_switches=int(sum(traces[i]["n_switches"] for i in idx)),
                onset=float(on0), onset_lo=float(ci(on_b)[0]), onset_hi=float(ci(on_b)[1]),
                onset_se=float(np.nanstd(on_b)) if ok.sum() > 10 else np.nan,
                asym=float(a0), asym_lo=float(ci(a_b)[0]), asym_hi=float(ci(a_b)[1]),
                onset_minus_pupil=float(d0), omp_lo=float(ci(d_b)[0]), omp_hi=float(ci(d_b)[1]),
                p_onset=float(p_on), p_asym=float(p_as), p_lead=float(max(p_on, p_as)))


def heterogeneity(rows):
    """Cochran's Q on region onsets: are regions' change times different from one another?"""
    r = [x for x in rows if np.isfinite(x["onset"]) and np.isfinite(x["onset_se"]) and x["onset_se"] > 0]
    if len(r) < 3:
        return dict(Q=np.nan, df=0, p=np.nan, n_regions=len(r))
    k = np.array([x["onset"] for x in r])
    w = 1 / np.array([x["onset_se"] for x in r]) ** 2
    kbar = (w * k).sum() / w.sum()
    Q = float((w * (k - kbar) ** 2).sum())
    return dict(Q=Q, df=len(r) - 1, p=float(chi2.sf(Q, len(r) - 1)), n_regions=len(r), weighted_mean_onset=float(kbar))


def region_lapse(sessions, model, region, idx, n_boot=1000, seed=0):
    S = [sessions[i] for i in idx if "pupil" in sessions[i].signals]
    if len(S) < 3:
        return None
    rows = lapse.build_table(S, model, [region, "pupil"], horizon=HORIZON)
    res = lapse.evaluate(rows, [region, "pupil"],
                         feature_sets={"base": ["pupil"], "full": ["pupil", region]})
    d = lapse.session_bootstrap_delta(res, "full", "base", n_boot=n_boot, seed=seed)
    return dict(lapse_auc_gain=d["est"], lapse_lo=d["lo"], lapse_hi=d["hi"], p_lapse=d["p_pos"])


def analyze(sessions, model, regions, min_units=5, min_sessions=5, n_boot=1000, seed=0, alpha=0.05, log=print):
    post = [model.p_engaged(s) for s in sessions]
    traces = all_traces(sessions, post, regions)
    rows = []
    for j, r in enumerate(regions):
        idx = region_sessions(sessions, r, min_units)
        if len(idx) < min_sessions:
            continue
        st = region_stats(traces, r, idx, n_boot=n_boot, seed=seed + j)
        if st is None or st["n_sessions"] < min_sessions:
            continue
        rows.append(st)
        log(f"  {r:10s} sessions {st['n_sessions']:3d}  onset {st['onset']:+6.2f} [{st['onset_lo']:+.2f},{st['onset_hi']:+.2f}]"
            f"  asym {st['asym']:+.3f}  p_lead {st['p_lead']:.4f}")
    q = bh([x["p_lead"] for x in rows])
    for x, qq in zip(rows, q):
        x["q_lead"] = float(qq)
        x["leader"] = bool(qq < alpha and x["onset"] < -LEAD_MARGIN and x["asym"] > 0)
    leaders = [x for x in rows if x["leader"]]
    for j, x in enumerate(leaders):
        lp = region_lapse(sessions, model, x["region"], region_sessions(sessions, x["region"], min_units), n_boot, seed + 1000 + j)
        if lp:
            x.update(lp)
    ql = bh([x.get("p_lapse", 1.0) for x in leaders])
    for x, qq in zip(leaders, ql):
        x["q_lapse"] = float(qq)
        x["lapse_forecast"] = bool(qq < alpha and x.get("lapse_auc_gain", 0) > 0)
    het = heterogeneity(rows)
    return dict(regions=sorted(rows, key=lambda x: x["onset"] if np.isfinite(x["onset"]) else 99),
                heterogeneity=het, n_analyzable=len(rows), alpha=alpha)


def verdict(res, alpha=0.05):
    """Pre-registered brain-wide claims (see docs/PREREGISTRATION_BRAINWIDE.md)."""
    leaders = [x["region"] for x in res["regions"] if x.get("leader")]
    lapse_regions = [x["region"] for x in res["regions"] if x.get("lapse_forecast")]
    het = res["heterogeneity"]
    return dict(
        H1_some_region_leads=dict(verdict="SUPPORTED" if leaders else "NOT SUPPORTED", leaders=leaders),
        H2_regions_change_at_different_times=dict(
            verdict="SUPPORTED" if np.isfinite(het["p"]) and het["p"] < alpha else "NOT SUPPORTED", **het),
        H3_leaders_forecast_lapses=dict(verdict="SUPPORTED" if lapse_regions else "NOT SUPPORTED", regions=lapse_regions),
        n_analyzable_regions=res["n_analyzable"])
