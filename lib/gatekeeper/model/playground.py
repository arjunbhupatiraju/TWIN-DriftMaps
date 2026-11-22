"""MODEL PLAYGROUND - your workbench for improving the new model.

Run:   python3 model/playground.py
It runs in about 5-10 minutes on a laptop, with no IBL data needed. It prints, for each method:
    FP     false-positive rate (no real effect; must stay near 0.05)
    POWER  share of real effects detected (higher is better)
The theoretical ceiling for these settings is roughly 0.55-0.60, and the current model gets about 0.29.

HOW TO WORK: change ONE knob in MY_VARIANT below, run, and write the result in model/NOTEBOOK.md.
A variant only counts as an improvement if POWER goes up AND FP stays <= about 0.07 in every setting.
"""
import sys, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scipy.signal import lfilter
from theory.sim_corrected import label
from model.proto import neurons, p_test
from model.gls import ar_bic, gls_stat

# ------------------------------------------------------------------ YOUR KNOBS
MY_VARIANT = dict(
    n_shared=1,        # how many shared slow components to remove (0, 1, 2, 3 ...)
    smooth=50,         # moving-average window (trials) used to find the shared slow components
    ar_max=10,         # max AR order for each neuron's drift model
    ar_shrink=1.0,     # multiply AR coefficients by this (1 = full whitening, 0 = none, 0.5 = halfway)
    pool_drift=False,  # True: fit ONE AR model to the average neuron and use it for all neurons
    hybrid=True,      # True: two-stage - whiten all neurons by the pooled model, then a small per-neuron AR on what's left
    hybrid_p=10,       # max AR order of the per-neuron second stage
)
REPS = 200             # raise to 200 for a final check (slower)
SETTINGS = [           # (label type, drift type, drift strength, shared fraction)
    ('block', 'ar98', 0.6, 0.2),
    ('block', 'mix', 0.3, 0.2),
    ('block', 'ar98', 0.3, 0.5),
    ('ar', 'mix', 0.6, 0.2),
    ('iid', 'ar98', 0.6, 0.2),
]
# ------------------------------------------------------------------

def my_test(X, y, nulls, n_shared, smooth, ar_max, ar_shrink, pool_drift, hybrid=True, hybrid_p=3):
    T, N = X.shape
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Y = np.column_stack([y] + nulls).astype(float)
    if n_shared:
        Zs = np.apply_along_axis(lambda c: np.convolve(c, np.ones(smooth) / smooth, mode='same'), 0, Z)
        U = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)[0][:, :n_shared]
        Z = Z - U @ (U.T @ Z); Y = Y - U @ (U.T @ Y)
    if hybrid:
        phi = ar_bic(Z.mean(1), ar_max) * ar_shrink
        a = np.r_[1.0, -phi]
        Z = lfilter(a, [1.0], Z, axis=0)[len(phi):]; Y = lfilter(a, [1.0], Y - Y.mean(0), axis=0)[len(phi):]
        phis = [ar_bic(Z[:, i], hybrid_p) for i in range(N)]
    elif pool_drift:
        phi = ar_bic(Z.mean(1), ar_max) * ar_shrink; phis = [phi] * N
    else:
        phis = [ar_bic(Z[:, i], ar_max) * ar_shrink for i in range(N)]
    s = gls_stat(Z, Y, phis=phis)
    return (1 + (s[1:] >= s[0]).sum()) / len(s)

if __name__ == '__main__':
    rng = np.random.default_rng(2026); T, N, t0 = 600, 40, time.time()
    print(f"variant: {MY_VARIANT}\n")
    print(f"{'setting':32s} {'pseudo FP':>9s} {'pseudo POWER':>12s} {'YOURS FP':>9s} {'YOURS POWER':>12s}")
    for lab, dk, w, q in SETTINGS:
        r = {k: 0 for k in ('pf', 'pp', 'yf', 'yp')}
        for _ in range(REPS):
            for beta in (0.0, 0.1):
                y = label(T, lab, rng); X = neurons(T, N, w, q, dk, rng)
                if beta: X[:, :8] += beta * (y - y.mean())[:, None] / y.std()
                nulls = [label(T, lab, rng) for _ in range(199)]
                a = p_test(X, y, nulls, 0) < .05; b = my_test(X, y, nulls, **MY_VARIANT) < .05
                if beta: r['pp'] += a; r['yp'] += b
                else: r['pf'] += a; r['yf'] += b
        print(f"{lab+' '+dk+f' w={w} q={q}':32s} {r['pf']/REPS:9.3f} {r['pp']/REPS:12.3f} {r['yf']/REPS:9.3f} {r['yp']/REPS:12.3f}", flush=True)
    print(f"\n({(time.time()-t0)/60:.1f} min)  Log this run in model/NOTEBOOK.md")
