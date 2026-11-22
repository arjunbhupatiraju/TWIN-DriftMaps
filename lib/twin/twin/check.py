"""The claim checker, identifiability and the experiment planner."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from .drift import phase_twin, DriftModel
from .stats import STATS
from .task import PseudoSessions


def wilson(k, n, z=1.96):
    """95% Wilson interval for a rate k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def _p(real, null):
    return float((1 + (null >= real).sum()) / (len(null) + 1))


def plant(X, y, r_target, frac, rng, indicator_tau=0.0):
    """Add a code of label y to a fraction of the real neurons so each coding neuron's correlation with y is about
    r_target. With indicator_tau > 0 (calcium data), the planted spiking code is first passed through the indicator's
    decay across trials, as a real code would be. Returns the modified recording."""
    from .calcium import smear
    X = np.asarray(X, float).copy()
    y = smear(np.asarray(y, float), indicator_tau)
    yz = (y - y.mean()) / (y.std() + 1e-12)
    N = X.shape[1]
    idx = rng.choice(N, max(1, int(round(frac * N))), replace=False)
    sd = X[:, idx].std(0)
    b = r_target / np.sqrt(max(1e-6, 1 - r_target ** 2))
    X[:, idx] += np.outer(yz, b * sd * rng.choice([-1, 1], len(idx)))
    return X


@dataclass
class Report:
    label: str
    stat: str
    T: int
    N: int
    value: float
    p_shuffle: float
    p_calibrated: float
    drift_only_rate: float
    calibrated_rate_on_twins: float
    detection_curve: dict = field(default_factory=dict)
    min_detectable_r: Optional[float] = None
    drift: dict = field(default_factory=dict)
    verdict: str = ""
    n_null: int = 0
    n_twin: int = 0
    drift_only_ci: tuple = ()
    calibrated_ci: tuple = ()
    p_linear_shift: Optional[float] = None

    def dict(self):
        return asdict(self)


def check(X, y, null: PseudoSessions, stat="pooled_corr", n_null=200, n_twin=100, plant_r=(0.05, 0.1, 0.15, 0.2, 0.3),
          plant_frac=0.2, n_plant=40, alpha=0.05, seed=0, label="label"):
    """Full report for one claim "these neurons encode y".

    p_shuffle             the common (invalid for slow labels) trial-shuffle test
    p_calibrated          pseudo-session null drawn from the task
    drift_only_rate       how often the SHUFFLE test fires on twins of these neurons (no code at all): the false-positive
                          rate the common test has on this recording
    calibrated_rate_...   same for the calibrated test (should be ~alpha)
    detection_curve       detection rate of the calibrated test when a code of correlation r is planted in the real neurons
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, float); y = np.asarray(y, float)
    T, N = X.shape
    f = STATS[stat]
    nulls = null.matrix(T, n_null, rng)
    shuf = np.column_stack([rng.permutation(y) for _ in range(n_null)])
    v = f(X, np.column_stack([y, shuf, nulls]))
    value, vs, vn = v[0], v[1:n_null + 1], v[n_null + 1:]
    p_sh, p_cal = _p(value, vs), _p(value, vn)
    from .tests import p_linear_shift
    p_ls = p_linear_shift(X, y, stat=f)          # Harris 2020; valid without a task model (TWIN benchmark, Nov 2)

    sh_hits, cal_hits = [], []
    small = max(50, n_null // 2)
    for _ in range(n_twin):
        Xt = phase_twin(X, rng)
        yt = null.draw(T, rng)
        Y = np.column_stack([yt, np.column_stack([rng.permutation(yt) for _ in range(small)]), null.matrix(T, small, rng)])
        w = f(Xt, Y)
        sh_hits.append(_p(w[0], w[1:small + 1]) < alpha); cal_hits.append(_p(w[0], w[small + 1:]) < alpha)

    curve = {}
    for r in plant_r:
        hits = []
        for _ in range(n_plant):
            yp = null.draw(T, rng)
            Xp = plant(X, yp, r, plant_frac, rng)
            Y = np.column_stack([yp, null.matrix(T, small, rng)])
            w = f(Xp, Y); hits.append(_p(w[0], w[1:]) < alpha)
        curve[float(r)] = float(np.mean(hits))
    mdr = next((r for r, h in sorted(curve.items()) if h >= 0.8), None)

    dfp = float(np.mean(sh_hits))
    if p_cal < alpha:
        verdict = "SUPPORTED: survives the calibrated null" + ("" if p_ls < alpha else " (but not the linear-shift test)")
    elif p_sh < alpha:
        verdict = f"NOT SUPPORTED: significant only under the shuffle test, which fires on {dfp:.0%} of drift-only twins"
    else:
        verdict = "NOT SUPPORTED: not significant under either test"
    if p_cal >= alpha and mdr is not None:
        verdict += f"; a code with per-neuron r >= {mdr} in {plant_frac:.0%} of neurons would have been detected (80%)"
    elif p_cal >= alpha:
        verdict += "; this recording cannot detect even the largest planted code tested (underpowered)"
    return Report(label=label, stat=stat, T=T, N=N, value=float(value), p_shuffle=p_sh, p_calibrated=p_cal,
                  drift_only_rate=dfp, calibrated_rate_on_twins=float(np.mean(cal_hits)), detection_curve=curve,
                  min_detectable_r=mdr, drift=DriftModel.fit(X).dict(), verdict=verdict,
                  n_null=n_null, n_twin=n_twin, drift_only_ci=wilson(int(np.sum(sh_hits)), n_twin),
                  calibrated_ci=wilson(int(np.sum(cal_hits)), n_twin), p_linear_shift=p_ls)


def identifiability(X, null_task_draw, hypotheses: dict, stat="pooled_corr", r=0.2, frac=0.2, n=60, n_null=100,
                    alpha=0.05, seed=0, indicator_tau=0.0):
    """Which explanations can these data tell apart?

    X: one recording (T x N) or a list of recordings (sessions; evidence is Fisher-combined across them, which is what
    makes the matrix reliable for slow labels).
    hypotheses: {name: function(session_dict) -> label}. For each TRUE hypothesis, its code is planted in twins of the
    recording(s); every hypothesis's label is then tested against its own pseudo-session null.
    Returns {"rate": M[true][tested], "ci": 95% intervals, "separable": {true: [tested names it is separable from]},
    "verdict": text}. A true hypothesis is separable from another when the lower bound of its own detection rate is
    above the upper bound of the other's."""
    rng = np.random.default_rng(seed)
    Xs = [np.asarray(x, float) for x in (X if isinstance(X, (list, tuple)) else [X])]
    f = STATS[stat]
    names = list(hypotheses)
    # bank of simulated sessions per session length: planted labels and null labels are drawn from it
    n_bank = max(2 * n_null, 200)
    bank = {}
    for T in sorted({len(x) for x in Xs}):
        ds = [null_task_draw(T, rng) for _ in range(n_bank)]
        bank[T] = {b: np.column_stack([np.asarray(hypotheses[b](d), float) for d in ds]) for b in names}
    hits = {a: {b: 0 for b in names} for a in names}
    for a in names:
        for _ in range(n):
            ps = {b: [] for b in names}
            for Xi in Xs:
                T = len(Xi); B = bank[T]
                j = rng.integers(n_bank)
                Xp = plant(phase_twin(Xi, rng), B[a][:, j], r, frac, rng, indicator_tau)
                others = rng.choice(np.delete(np.arange(n_bank), j), n_null, replace=False)
                for b in names:
                    w = f(Xp, np.column_stack([B[b][:, j], B[b][:, others]])); ps[b].append(_p(w[0], w[1:]))
            for b in names:
                hits[a][b] += _fisher(ps[b]) < alpha
    rate = {a: {b: hits[a][b] / n for b in names} for a in names}
    ci = {a: {b: wilson(hits[a][b], n) for b in names} for a in names}
    sep = {a: [b for b in names if b != a and ci[a][a][0] > ci[a][b][1]] for a in names}
    lines = []
    for a in names:
        others = [b for b in names if b != a]
        if ci[a][a][1] < 0.5:
            lines.append(f"{a}: too weak to detect at r={r} with {len(Xs)} session(s); add sessions.")
        elif set(sep[a]) == set(others):
            lines.append(f"{a}: separable from all alternatives.")
        else:
            lines.append(f"{a}: NOT separable from {', '.join(b for b in others if b not in sep[a])}.")
    return dict(rate=rate, ci=ci, separable=sep, verdict=lines, n=n, sessions=len(Xs))


def plan(drift: DriftModel, null: PseudoSessions, T_grid=(300, 600, 1200), N_grid=(10, 30, 100), S_grid=(1, 5, 10, 20),
         stat="pooled_corr", r=0.1, frac=0.2, n=30, n_null=100, alpha=0.05, seed=0, indicator_tau=0.0):
    """Before the experiment: for each (trials per session, neurons per session, number of sessions), the
    false-positive rate of the common shuffle test, the calibrated test's rate, and the calibrated test's power for
    a code of per-neuron correlation r in a fraction frac of neurons. Sessions are combined with Fisher's method."""
    rng = np.random.default_rng(seed)
    f = STATS[stat]
    rows = []
    for T in T_grid:
        for N in N_grid:
            # per-session p-values, simulated once per (T, N) and reused across S
            P = []
            for _ in range(n * max(S_grid)):
                from .calcium import smear
                X = smear(drift.simulate(T, N, rng), indicator_tau); y = null.draw(T, rng)
                Ysh = np.column_stack([rng.permutation(y) for _ in range(n_null)]); Yn = null.matrix(T, n_null, rng)
                w = f(X, np.column_stack([y, Ysh, Yn]))
                w2 = f(plant(X, y, r, frac, rng, indicator_tau), np.column_stack([y, Yn]))
                P.append((_p(w[0], w[1:n_null + 1]), _p(w[0], w[n_null + 1:]), _p(w2[0], w2[1:])))
            P = np.array(P)
            for S in S_grid:
                G = P[: n * S].reshape(n, S, 3)
                comb = np.array([[_fisher(G[i, :, j]) for j in range(3)] for i in range(n)])
                k = (comb < alpha).sum(0)
                rows.append(dict(T=T, N=N, sessions=S, fp_shuffle=float(k[0] / n), fp_calibrated=float(k[1] / n),
                                 power_calibrated=float(k[2] / n), power_ci=wilson(int(k[2]), n),
                                 fp_calibrated_ci=wilson(int(k[1]), n)))
    return rows


def advice(rows, power_target=0.8):
    ok = [r for r in rows if r.get("power_ci", (r["power_calibrated"],))[0] >= power_target - 0.1]
    worst = max(rows, key=lambda r: r["fp_shuffle"])
    lines = [f"With the common shuffle test, drift alone gives up to {worst['fp_shuffle']:.0%} false positives "
             f"(at {worst['T']} trials x {worst['N']} neurons). Use the calibrated test."]
    if ok:
        best = min(ok, key=lambda r: r["T"] * r["N"] * r.get("sessions", 1))
        lines.append(f"Smallest design reaching {power_target:.0%} power with the calibrated test: {best.get('sessions', 1)} "
                     f"sessions of {best['T']} trials x {best['N']} neurons.")
    else:
        lines.append("No design in the grid reaches the target power: add sessions, trials or neurons, or randomize "
                     "the block structure so the label is less slow.")
    return lines


def _fisher(ps):
    from scipy.stats import chi2
    ps = np.clip(np.asarray(ps, float), 1e-12, 1)
    return float(chi2.sf(-2 * np.log(ps).sum(), 2 * len(ps)))


def check_sessions(sessions, null: PseudoSessions, stat="pooled_corr", n_null=200, n_twin=40, plant_r=(0.05, 0.1, 0.2),
                   plant_frac=0.2, alpha=0.05, seed=0, label="label"):
    """Several sessions (list of (X, y)) of the same claim: per-session calibrated p-values combined with Fisher's
    method, the combined false-positive rate on twins (shuffle vs calibrated), and combined power for planted codes."""
    rng = np.random.default_rng(seed)
    f = STATS[stat]
    p_sh, p_cal = [], []
    for X, y in sessions:
        T = len(y)
        v = f(X, np.column_stack([y, np.column_stack([rng.permutation(y) for _ in range(n_null)]), null.matrix(T, n_null, rng)]))
        p_sh.append(_p(v[0], v[1:n_null + 1])); p_cal.append(_p(v[0], v[n_null + 1:]))
    small = max(50, n_null // 2)

    def combined(gen):
        out_sh, out_cal = [], []
        for _ in range(n_twin):
            ps_sh, ps_cal = [], []
            for X, _y in sessions:
                T = len(X); Xt, yt = gen(X, T)
                Y = np.column_stack([yt, np.column_stack([rng.permutation(yt) for _ in range(small)]), null.matrix(T, small, rng)])
                w = f(Xt, Y); ps_sh.append(_p(w[0], w[1:small + 1])); ps_cal.append(_p(w[0], w[small + 1:]))
            out_sh.append(_fisher(ps_sh) < alpha); out_cal.append(_fisher(ps_cal) < alpha)
        return float(np.mean(out_sh)), float(np.mean(out_cal))

    fp_sh, fp_cal = combined(lambda X, T: (phase_twin(X, rng), null.draw(T, rng)))
    curve = {}
    for r in plant_r:
        def g(X, T, r=r):
            y = null.draw(T, rng); return plant(X, y, r, plant_frac, rng), y
        curve[float(r)] = combined(g)[1]
    pc = _fisher(p_cal)
    return dict(label=label, sessions=len(sessions), n_null=n_null, n_twin=n_twin, p_shuffle_combined=_fisher(p_sh), p_calibrated_combined=pc,
                fp_shuffle_on_twins=fp_sh, fp_calibrated_on_twins=fp_cal, power_curve=curve,
                verdict=("SUPPORTED across sessions" if pc < alpha else "NOT SUPPORTED across sessions"))
