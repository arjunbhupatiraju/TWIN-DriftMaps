"""Quickstart: false positives and power on simulated sessions with slow neural drift.

Run:  python examples/quickstart.py
"""
import numpy as np

from driftaware import drift_test, shuffle_labels, simulate_session
from driftaware.simulate import block_labels

T, N, SESSIONS, NULLS = 600, 30, 100, 100


def rates(beta):
    res = {"drift-aware (pseudo null)": [], "standard (pseudo null)": [], "standard (shuffle null)": []}
    for i in range(SESSIONS):
        X, y = simulate_session(T=T, N=N, beta=beta, seed=i)
        pseudo = lambda rng: block_labels(T, rng)
        res["drift-aware (pseudo null)"].append(drift_test(X, y, pseudo, NULLS, seed=i).p < 0.05)
        res["standard (pseudo null)"].append(drift_test(X, y, pseudo, NULLS, seed=i, method="plain").p < 0.05)
        res["standard (shuffle null)"].append(drift_test(X, y, shuffle_labels(y), NULLS, seed=i, method="plain").p < 0.05)
    return {k: np.mean(v) for k, v in res.items()}


if __name__ == "__main__":
    print(f"{SESSIONS} simulated sessions, {T} trials, {N} neurons, slow shared drift\n")
    fp = rates(0.0)
    pw = rates(0.06)
    print(f"{'method':28s} {'false positives':>16s} {'power':>8s}")
    for k in fp:
        print(f"{k:28s} {fp[k]:16.2f} {pw[k]:8.2f}")
    print("\nTarget false-positive rate: 0.05. The shuffle null is invalid under drift.")
