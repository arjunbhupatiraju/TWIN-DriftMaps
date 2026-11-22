"""REPMAP follow-up: evaluation with reference strength held fixed (registered as an amendment before running
on the confirm mice).

Why: the registered test (repmap_confirm.py) checked a k-mouse study's claims against ALL other mice. That
reference has ~40 mice in the explore split but ~70-77 in the confirm split, so in confirm it confirms nearly
every weak-but-real effect and every rule scores near ceiling. Here the reference strength is held fixed:
    equal   reference = a second, disjoint study of the SAME k mice, standard F_1 at q = 0.05
            ("would an equally sized lab replicate this claim?" - the paper's main replication question)
    fixed36 reference = 36 mice drawn from the remaining mice, standard F_1 at q = 0.05
            (36 = the largest size available for every k in both splits)
Rules, studies, claim levels and precision are exactly as in repmap_confirm.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .repmap_confirm import GRID, KS, QS, RULES, VARS, _prec, mats, pvals, verdict

REF_MODES = ("equal", "fixed36")
REF_SIZE = 36


def evaluate(df, n_rep=300, seed=4052, log=print):
    out = {m: {"precision": {}} for m in REF_MODES}
    for var in VARS:
        X = mats(df, var)
        M = X[0].shape[1]
        prec = {(mode, r): [] for mode in REF_MODES for r in RULES}
        for k in KS:
            rng = np.random.default_rng(seed + k)
            acc = {(mode, r, q): np.zeros(2) for mode in REF_MODES for r in RULES for q in QS}
            for _ in range(n_rep):
                p = rng.permutation(M)
                a = np.zeros(M)
                a[p[:k]] = 1
                b = np.zeros(M)
                b[p[k:2 * k]] = 1
                c = np.zeros(M)
                c[p[k:k + REF_SIZE]] = 1
                refs = {"equal": verdict(pvals(X, "F_1", b), 0.05), "fixed36": verdict(pvals(X, "F_1", c), 0.05)}
                for r in RULES:
                    pp = pvals(X, r, a)
                    for q in QS:
                        cl = verdict(pp, q)
                        for mode in REF_MODES:
                            ref = refs[mode]
                            ok = (cl >= 0) & (ref >= 0)
                            acc[(mode, r, q)] += [((cl == 1) & (ref == 1) & ok).sum(), ((cl == 1) & ok).sum()]
            for mode in REF_MODES:
                for r in RULES:
                    prec[(mode, r)] += [_prec([[acc[(mode, r, q)][1] / n_rep, acc[(mode, r, q)][0] / n_rep] for q in QS], g)
                                        for g in GRID[var]]
            log(f"  {var} k={k} done")
        for mode in REF_MODES:
            out[mode]["precision"][var] = {r: (float(np.nanmean(prec[(mode, r)])) if np.isfinite(prec[(mode, r)]).any() else None)
                                           for r in RULES}
    return out


def table(res, log=print):
    for mode in REF_MODES:
        log(f"\nreference = {mode}:")
        for var in VARS:
            P = res[mode]["precision"][var]
            log(f"  {var:9s} " + "  ".join(f"{r}:{'n/a' if P[r] is None else f'{P[r]:.3f}'}" for r in RULES))


def run(stats_file, out_json, n_rep=300, log=print):
    import pandas as pd
    df = pd.read_csv(stats_file)
    df = df[df["var"].isin(VARS)]
    log(f"{df.subject.nunique()} mice")
    res = evaluate(df, n_rep=n_rep, log=log)
    res["hypotheses"] = hypotheses(res)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(res, indent=1))
    table(res, log)
    log("\nREPMAP AMENDMENT 1 HYPOTHESES (confirm mice, run once):")
    for h in res["hypotheses"]:
        v = "n/a" if h["value"] is None else f"{h['value']:+.3f}"
        log(f"  {h['id']:12s} {'PASS' if h['passed'] else 'FAIL'}  value={v:>7s}  {h['claim']}")
    log(f"{sum(h['passed'] for h in res['hypotheses'])}/{len(res['hypotheses'])} passed")
    return res


def hypotheses(res):
    E, F = res["equal"]["precision"], res["fixed36"]["precision"]

    def d(P, var, a, b):
        x, y = P[var].get(a), P[var].get(b)
        return None if x is None or y is None else x - y

    H = []

    def add(hid, text, val, ok):
        H.append(dict(id=hid, claim=text, value=val, passed=bool(val is not None and ok(val))))

    add("A1", "equal-size replication, stimulus: PC_3 - F_1 >= +0.04", d(E, "stim", "PC_3", "F_1"), lambda v: v >= 0.04)
    add("A2", "equal-size replication, subjective prior: PC_3 - F_1 >= +0.02", d(E, "sprior", "PC_3", "F_1"), lambda v: v >= 0.02)
    add("A3:choice", "equal-size replication, choice: PC_3 - F_1 >= -0.03", d(E, "choice", "PC_3", "F_1"), lambda v: v >= -0.03)
    add("A3:feedback", "equal-size replication, feedback: PC_3 - F_1 >= -0.03", d(E, "feedback", "PC_3", "F_1"), lambda v: v >= -0.03)
    add("A4:stim", "equal-size replication, stimulus: PC_4 - PC_2 > 0", d(E, "stim", "PC_4", "PC_2"), lambda v: v > 0)
    add("A4:sprior", "equal-size replication, subjective prior: PC_4 - PC_2 > 0", d(E, "sprior", "PC_4", "PC_2"), lambda v: v > 0)
    add("A5", "equal-size replication, stimulus: PC_3 - F_3 > 0", d(E, "stim", "PC_3", "F_3"), lambda v: v > 0)
    add("A6:stim", "fixed 36-mouse reference, stimulus: PC_3 - F_1 >= +0.08", d(F, "stim", "PC_3", "F_1"), lambda v: v >= 0.08)
    add("A6:sprior", "fixed 36-mouse reference, subjective prior: PC_3 - F_1 >= +0.03", d(F, "sprior", "PC_3", "F_1"), lambda v: v >= 0.03)
    return H
