"""Check the formula against simulated sessions decoded exactly as in MULTIVERSE (ridge LOO, shuffle null)."""
import sys, numpy as np
sys.path.insert(0, '.')
from theory.drift_fp import fp_theory, V_from_acf, acf, decode, ar1
from gatekeeper.multiverse import pseudo_pl
rng = np.random.default_rng(0)
T, NP, REPS, KMAX = 500, 200, 120, 150
rows = []
for label in ("prior", "stim"):
    for a in (0.9, 0.97, 0.99):
        for N in (5, 10, 20, 40, 80):
            hits, Vs = 0, []
            for _ in range(REPS):
                pl = pseudo_pl(T, rng)
                y = (pl < 0.5).astype(float) if label == "prior" else (rng.random(T) >= pl).astype(float)
                X = ar1(T, N, a, rng) + 0.0
                Y = np.column_stack([y] + [rng.permutation(y) for _ in range(NP)])
                s = decode(X, Y)
                hits += (1 + (s[1:] >= s[0]).sum()) / (NP + 1) < 0.05
                Vs.append(V_from_acf(np.array([a ** k for k in range(KMAX)]), acf(y, KMAX)))
            V = float(np.mean(Vs))
            rows.append((label, a, N, V, hits / REPS, float(fp_theory(N, V))))
            print(f"{label:5s} a={a:.2f} N={N:3d}  V={V:6.2f}  simulated FP={hits/REPS:.2f}  theory={fp_theory(N, V):.2f}", flush=True)
np.save('theory/sim_rows.npy', np.array(rows, dtype=object))
