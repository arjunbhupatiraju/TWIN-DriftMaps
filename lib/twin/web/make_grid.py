"""Precompute the web planner's grid: per-session p-values for every drift x trials x neurons cell.
The page combines sessions (Fisher) in the browser.  Usage: python3 web/make_grid.py OUT.json [part n_parts]"""
import json, sys
import numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from twin import BlockTask, PseudoSessions, DriftModel, pooled_corr, plant

SLOW = [0.1, 0.2, 0.3, 0.45, 0.6]
TAU = [10, 30, 100]
SHARED = [0.0, 0.3, 0.6]
T_GRID = [300, 600, 1000, 1500]
N_GRID = [10, 30, 100]
R_GRID = [0.05, 0.1, 0.2]
REPS, N_NULL, FRAC = 60, 100, 0.2

def p(real, null):
    return (1 + (null >= real).sum()) / (len(null) + 1)

def main():
    out = sys.argv[1]; part = int(sys.argv[2]) if len(sys.argv) > 2 else 0; nparts = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    null = PseudoSessions(BlockTask(), "prior")
    rng = np.random.default_rng(1000 + part)
    bank = {T: null.matrix(T, 800, rng) for T in T_GRID}
    combos = [(s, t, h) for s in SLOW for t in TAU for h in SHARED]
    cells = {}
    for ci, (s, t, h) in enumerate(combos):
        if ci % nparts != part:
            continue
        dm = DriftModel(slow_frac=s, phi=float(np.exp(-1 / t)), shared=h, inst_corr=0.03)
        for T in T_GRID:
            for N in N_GRID:
                P = []
                for _ in range(REPS):
                    X = dm.simulate(T, N, rng)
                    cols = rng.choice(800, N_NULL + 1 + len(R_GRID), replace=False)
                    y = bank[T][:, cols[0]]; Yn = bank[T][:, cols[1:N_NULL + 1]]
                    Ysh = np.column_stack([rng.permutation(y) for _ in range(N_NULL)])
                    w = pooled_corr(X, np.column_stack([y, Ysh, Yn]))
                    row = [p(w[0], w[1:N_NULL + 1]), p(w[0], w[N_NULL + 1:])]
                    for r in R_GRID:
                        w2 = pooled_corr(plant(X, y, r, FRAC, rng), np.column_stack([y, Yn]))
                        row.append(p(w2[0], w2[1:]))
                    P.append(row)
                cells[f"{s}|{t}|{h}|{T}|{N}"] = np.round(np.array(P).T, 3).tolist()
        print(f"part {part}: drift {ci + 1}/{len(combos)} done", flush=True)
    json.dump(dict(slow=SLOW, tau=TAU, shared=SHARED, T=T_GRID, N=N_GRID, r=R_GRID, reps=REPS, n_null=N_NULL, frac=FRAC,
                   rows=["shuffle_null", "calibrated_null"] + [f"calibrated_r{r}" for r in R_GRID], cells=cells), open(out, "w"))

if __name__ == "__main__":
    main()
