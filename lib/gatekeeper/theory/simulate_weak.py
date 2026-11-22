"""Weak-drift regime: each neuron = sqrt(w) * slow AR(1) drift + sqrt(1-w) * independent noise, so rho_x(k) = w * a^k."""
import sys, numpy as np
sys.path.insert(0, '.')
from theory.drift_fp import fp_theory, V_from_acf, acf, decode, ar1
from gatekeeper.multiverse import pseudo_pl
rng = np.random.default_rng(1)
T, NP, REPS, KMAX, a = 500, 200, 150, 200, 0.98
out = []
for label in ("prior", "stim"):
    for w in (0.01, 0.03, 0.1):
        for N in (5, 10, 20, 40, 80):
            hits, Vs = 0, []
            for _ in range(REPS):
                pl = pseudo_pl(T, rng)
                y = (pl < 0.5).astype(float) if label == "prior" else (rng.random(T) >= pl).astype(float)
                X = np.sqrt(w) * ar1(T, N, a, rng) + np.sqrt(1 - w) * rng.standard_normal((T, N))
                Y = np.column_stack([y] + [rng.permutation(y) for _ in range(NP)])
                s = decode(X, Y)
                hits += (1 + (s[1:] >= s[0]).sum()) / (NP + 1) < 0.05
                Vs.append(V_from_acf(np.r_[1.0, w * a ** np.arange(1, KMAX)], acf(y, KMAX)))
            V = float(np.mean(Vs)); sim = hits / REPS; th = float(fp_theory(N, V))
            out.append((label, w, N, V, sim, th))
            print(f"{label:5s} w={w:.2f} N={N:3d}  V={V:5.2f}  simulated FP={sim:.2f}  theory={th:.2f}", flush=True)
o = np.array([(r[4], r[5]) for r in out])
print("correlation simulated vs theory:", round(np.corrcoef(o[:, 0], o[:, 1])[0, 1], 3), " mean abs error:", round(np.mean(np.abs(o[:, 0] - o[:, 1])), 3))
