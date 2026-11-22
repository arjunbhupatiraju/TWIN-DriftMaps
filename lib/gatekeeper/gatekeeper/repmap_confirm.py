"""REPMAP confirmatory test (registered; run once on the CONFIRM mice).

Method (fixed after exploration on the explore mice, see the REPMAP registration):
    PC_m   partial conjunction across mice (effect in >= 2 mice; Benjamini & Heller 2008) on mouse-level
           Fisher p-values, testing only regions recorded in >= m mice, BH-FDR across tested regions.
           Headline rule: PC_3.
    F_m    standard Fisher across sessions, same minimum-mice rule. F_1 = the standard pipeline.
Evidence per session: the calibrated MULTIVERSE default (ridge decoding, pseudo-session / imposter null,
good units, IBL window), column p_decode_pseudo of session_stats.csv.gz.

Evaluation: random studies of k mice (k = 8, 12, 16; 300 each). Held-out references on all OTHER mice:
standard F_1 and PC_2, both q = 0.05. Precision = share of a study's claims significant in the reference,
read at matched claims per study (q swept), averaged over k, claim levels and the two references.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .multiverse import _bh

VARS = ("stim", "sprior", "choice", "feedback")
GRID = {"stim": (0.5, 1, 2), "sprior": (0.5, 1, 2), "choice": (3, 6, 12), "feedback": (5, 10, 20)}
QS = (1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
RULES = ("F_1", "F_2", "F_3", "F_4", "PC_2", "PC_3", "PC_4")
REFS = ("F_1", "PC_2")
KS = (8, 12, 16)


def mats(df, var):
    g = df[(df["var"] == var) & (df.window == "a") & (df.units == "good")]
    mice = sorted(df.subject.unique())
    regs = sorted(g.region.unique())
    ri, mi = {r: i for i, r in enumerate(regs)}, {m: j for j, m in enumerate(mice)}
    rr, mm = g.region.map(ri).to_numpy(), g.subject.map(mi).to_numpy()
    p = np.clip(g.p_decode_pseudo.to_numpy(), 1e-12, 1)
    C = np.zeros((len(regs), len(mice)))
    np.add.at(C, (rr, mm), 1)
    S = np.zeros_like(C)
    np.add.at(S, (rr, mm), -2 * np.log(p))
    with np.errstate(all="ignore"):
        pm = np.where(C > 0, chi2.sf(S, 2 * np.maximum(C, 1)), np.nan)
    return C, S, pm


def pvals(X, rule, a):
    C, S, pm = X
    sel = a > 0
    c = C[:, sel]
    N, nm = c.sum(1), (c > 0).sum(1)
    fam, m = rule.split("_")
    m = int(m)
    if fam == "F":
        p = chi2.sf(S[:, sel].sum(1), 2 * np.maximum(N, 1))
        return np.where((N >= 2) & (nm >= m), p, np.nan)
    s = np.sort(np.where(np.isnan(pm[:, sel]), -1, pm[:, sel]), 1)[:, ::-1]
    out = np.full(len(nm), np.nan)
    for i in np.nonzero((nm >= max(m, 2)) & (N >= 2))[0]:
        top = s[i, : nm[i] - 1]
        out[i] = chi2.sf(-2 * np.log(np.clip(top, 1e-300, 1)).sum(), 2 * len(top))
    return out


def verdict(p, q):
    o = np.full(len(p), -1, np.int8)
    t = ~np.isnan(p)
    if t.any():
        o[t] = _bh(p[t], q).astype(np.int8)
    return o


def _prec(curve, g):
    c = np.array(curve)
    o = np.argsort(c[:, 0])
    cl, hi = c[o, 0], c[o, 1]
    return np.nan if (g > cl.max() or g < cl.min()) else float(np.interp(g, cl, hi) / g)


def evaluate(df, n_rep=300, seed=2026, log=print):
    out = {"precision": {}, "regions_testable": {}, "fake_flag_share": {}}
    for var in VARS:
        X = mats(df, var)
        M = X[0].shape[1]
        prec = {r: [] for r in RULES}
        cov = {r: [] for r in RULES}
        for k in KS:
            rng = np.random.default_rng(seed + k)
            acc = {(rf, r, q): np.zeros(2) for rf in REFS for r in RULES for q in QS}
            tested = {r: 0 for r in RULES}
            for _ in range(n_rep):
                p = rng.permutation(M)
                a = np.zeros(M)
                a[p[:k]] = 1
                refs = {rf: verdict(pvals(X, rf, 1 - a), 0.05) for rf in REFS}
                for r in RULES:
                    pp = pvals(X, r, a)
                    tested[r] += int(np.sum(~np.isnan(pp)))
                    for q in QS:
                        cl = verdict(pp, q)
                        for rf in REFS:
                            ok = (cl >= 0) & (refs[rf] >= 0)
                            acc[(rf, r, q)] += [((cl == 1) & (refs[rf] == 1) & ok).sum(), ((cl == 1) & ok).sum()]
            for r in RULES:
                prec[r] += [_prec([[acc[(rf, r, q)][1] / n_rep, acc[(rf, r, q)][0] / n_rep] for q in QS], g)
                            for rf in REFS for g in GRID[var]]
                cov[r].append(tested[r] / n_rep)
            log(f"  {var} k={k} done")
        out["precision"][var] = {r: float(np.nanmean(prec[r])) if np.isfinite(prec[r]).any() else None for r in RULES}
        out["regions_testable"][var] = {r: float(np.mean(cov[r])) for r in RULES}
        # calibration: fake variable, all confirm mice, headline rule and standard
        ctl = df[df["var"] == f"ctl_{var}"]
        if len(ctl):
            Xc = mats(df.assign(var=df["var"].replace({f"ctl_{var}": f"__ctl"})), "__ctl")
            out["fake_flag_share"][var] = {}
            for r in ("F_1", "PC_3"):
                v = verdict(pvals(Xc, r, np.ones(Xc[0].shape[1])), 0.05)
                out["fake_flag_share"][var][r] = float(np.mean(v[v >= 0] == 1)) if (v >= 0).any() else 0.0
    return out


def hypotheses(res):
    P = res["precision"]

    def d(var, a, b):
        x, y = P[var].get(a), P[var].get(b)
        return None if x is None or y is None else x - y

    H = []

    def add(hid, text, val, ok):
        H.append(dict(id=hid, claim=text, value=val, passed=bool(val is not None and ok(val))))

    add("R1", "stimulus: PC_3 precision exceeds the standard (F_1) by >= 0.08", d("stim", "PC_3", "F_1"), lambda v: v >= 0.08)
    add("R2", "subjective prior: PC_3 precision exceeds the standard by >= 0.03", d("sprior", "PC_3", "F_1"), lambda v: v >= 0.03)
    add("R3:choice", "choice: PC_3 precision no more than 0.03 below the standard", d("choice", "PC_3", "F_1"), lambda v: v >= -0.03)
    add("R3:feedback", "feedback: PC_3 precision no more than 0.03 below the standard", d("feedback", "PC_3", "F_1"), lambda v: v >= -0.03)
    add("R4:stim", "stimulus: reliability rises with mouse coverage (PC_4 precision > PC_2)", d("stim", "PC_4", "PC_2"), lambda v: v > 0)
    add("R4:sprior", "subjective prior: reliability rises with mouse coverage (PC_4 precision > PC_2)", d("sprior", "PC_4", "PC_2"), lambda v: v > 0)
    add("R5", "stimulus: at equal coverage the consistency test adds precision (PC_3 - F_3 > 0)", d("stim", "PC_3", "F_3"), lambda v: v > 0)
    worst = max((v.get("PC_3", 0.0) for v in res["fake_flag_share"].values()), default=None)
    add("R6", "calibration: PC_3 flags <= 5% of tested regions for every fake variable (all confirm mice)", worst, lambda v: v <= 0.05)
    return H


def run(stats_file, out_dir, n_rep=300, log=print):
    import pandas as pd
    df = pd.read_csv(stats_file)
    df = df[df["var"].isin(list(VARS) + [f"ctl_{v}" for v in VARS])]
    log(f"{df.subject.nunique()} mice, {df.sid.nunique()} sessions")
    res = evaluate(df, n_rep=n_rep, log=log)
    res["hypotheses"] = hypotheses(res)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "repmap_confirm.json").write_text(json.dumps(res, indent=1))
    log("\nPRECISION AT MATCHED CLAIMS (share of claims confirmed in held-out mice):")
    for var in VARS:
        log(f"  {var:9s} " + "  ".join(f"{r}:{'n/a' if res['precision'][var][r] is None else round(res['precision'][var][r], 3)}"
                                    f"({res['regions_testable'][var][r]:.0f}r)" for r in RULES))
    log("\nPRE-REGISTERED REPMAP HYPOTHESES (confirm mice, run once):")
    for h in res["hypotheses"]:
        v = "n/a" if h["value"] is None else f"{h['value']:+.3f}"
        log(f"  {h['id']:12s} {'PASS' if h['passed'] else 'FAIL'}  value={v:>7s}  {h['claim']}")
    log(f"{sum(h['passed'] for h in res['hypotheses'])}/{len(res['hypotheses'])} passed")
    return res
