import sys, numpy as np
sys.path.insert(0, '.')
from theory.sim_corrected import label
from model.proto import neurons, p_test
from model.gls import p_gls
rng = np.random.default_rng(5); T = 600; nrep = int(sys.argv[1]) if len(sys.argv) > 1 else 150
for lab in ('block', 'ar', 'iid'):
    for dk in ('ar98', 'mix'):
        for w in (0.0, 0.3, 0.6):
            for beta in (0.0, 0.1):
                h = dict(pseudo=0, trunc8=0, gls=0)
                for _ in range(nrep):
                    N = 40; y = label(T, lab, rng); X = neurons(T, N, w, 0.2, dk, rng)
                    if beta: X[:, :8] += beta * (y - y.mean())[:, None] / y.std()
                    nulls = [label(T, lab, rng) for _ in range(199)]
                    h['pseudo'] += p_test(X, y, nulls, 0) < .05
                    h['trunc8'] += p_test(X, y, nulls, 8) < .05
                    h['gls'] += p_gls(X, y, nulls) < .05
                print(f"label={lab:5s} drift={dk:4s} w={w} beta={beta}: " + "  ".join(f"{k}:{v/nrep:.3f}" for k, v in h.items()), flush=True)
