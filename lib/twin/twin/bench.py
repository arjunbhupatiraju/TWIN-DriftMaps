"""Benchmark: every test of "does this population encode y?" on drift-only data (false positives) and on data with a
planted code (power), for labels with and without a known task model.  python3 -m twin.bench [reps]"""
from __future__ import annotations

import sys, time
import numpy as np

from .drift import DriftModel
from .task import BlockTask, PseudoSessions
from .check import plant
from .tests import p_shuffle, p_pseudo, p_linear_shift, p_twin, p_session_perm


class Generator:
    """Label generator with .draw(T, rng) and .matrix(T, n, rng), so it can serve as an (oracle) pseudo-session null."""
    def __init__(self, f):
        self.f = f
    def draw(self, T, rng):
        return self.f(T, rng)
    def matrix(self, T, n, rng):
        return np.column_stack([self.f(T, rng) for _ in range(n)])


def latent_state(T, rng, dwell=40):
    """Engagement-like two-state chain (mean dwell `dwell` trials): no task model exists in real data."""
    s = np.zeros(T); s[0] = rng.integers(2)
    for t in range(1, T):
        s[t] = s[t - 1] if rng.random() > 1 / dwell else 1 - s[t - 1]
    return s


def learning(T, rng):
    """Within-session learning / satiety curve: noisy sigmoid with random midpoint and sign."""
    t = np.arange(T) / T
    m = rng.uniform(0.2, 0.8); k = rng.uniform(5, 20)
    return rng.choice([-1, 1]) / (1 + np.exp(-k * (t - m))) + 0.3 * rng.standard_normal(T)


LABELS = {
    "block prior (task model known)": PseudoSessions(BlockTask(), "prior"),
    "latent state (no task model)": Generator(latent_state),
    "learning curve (no task model)": Generator(learning),
}
DRIFTS = {
    "IBL-like drift": DriftModel(phis=list(np.exp(-1 / np.array([3., 12, 75]))), a=[0.05, 0.1, 0.2], sh=[0.2, 0.4, 0.5], inst_corr=0.03),
    "drift + session-long trend": "trend",
}


def neural(kind, T, N, rng):
    if kind == "trend":
        X = DRIFTS["IBL-like drift"].simulate(T, N, rng)
        t = np.linspace(-1, 1, T)[:, None]
        w = rng.standard_normal(N) * 0.6 + 0.3                 # most neurons share the direction (e.g. satiety)
        return X + t * w
    return kind.simulate(T, N, rng)


TESTS = ("shuffle", "pseudo (oracle)", "linear shift", "twin")


def one(Xk, lab, T, N, rng, r=0.0, n=150):
    X = neural(Xk, T, N, rng)
    y = lab.draw(T, rng)
    if r > 0:
        X = plant(X, y, r, 0.2, rng)
    return dict(zip(TESTS, (p_shuffle(X, y, rng, n), p_pseudo(X, y, rng, lab, n), p_linear_shift(X, y), p_twin(X, y, rng, n)))), (X, y)


def run(reps=60, T=600, N=30, sessions=10, seed=0):
    from scipy.stats import chi2
    fisher = lambda ps: float(chi2.sf(-2 * np.log(np.clip(ps, 1e-12, 1)).sum(), 2 * len(ps)))
    rng = np.random.default_rng(seed); t0 = time.time()
    out = []
    for dname, dk in DRIFTS.items():
        for lname, lab in LABELS.items():
            for r in (0.0, 0.15):
                single = [one(dk, lab, T, N, rng, r)[0] for _ in range(reps)]
                rates = {k: np.mean([s[k] < 0.05 for s in single]) for k in TESTS}
                # multi-session: Fisher over sessions for each single-session test; native session permutation
                multi = {k: [] for k in TESTS + ("session perm",)}
                for _ in range(max(10, reps // 3)):
                    res = [one(dk, lab, T, N, rng, r) for _ in range(sessions)]
                    for k in TESTS:
                        multi[k].append(fisher([q[0][k] for q in res]) < 0.05)
                    multi["session perm"].append(p_session_perm([q[1] for q in res], rng) < 0.05)
                mr = {k: float(np.mean(v)) for k, v in multi.items()}
                out.append(dict(drift=dname, label=lname, r=r, **{f"1s {k}": v for k, v in rates.items()},
                                **{f"{sessions}s {k}": v for k, v in mr.items()}))
                print(f"{dname:26s} | {lname:32s} | {'FALSE POS' if r == 0 else f'POWER r={r}':10s} | 1 session: "
                      + "  ".join(f"{k} {v:.2f}" for k, v in rates.items()) + f" | {sessions} sessions: "
                      + "  ".join(f"{k} {v:.2f}" for k, v in mr.items()) + f"  ({time.time() - t0:.0f}s)", flush=True)
    return out


if __name__ == "__main__":
    import pandas as pd
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    pd.DataFrame(run(reps)).to_csv(sys.argv[2] if len(sys.argv) > 2 else "bench_results.csv", index=False)
