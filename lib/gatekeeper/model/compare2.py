import sys, numpy as np
sys.path.insert(0, '.')
from theory.sim_corrected import label
from model.proto import neurons, p_test
from model.gls import p_gls, p_gls_shared
rng = np.random.default_rng(11); T = 600; nrep = int(sys.argv[1])
for lab, dk, w, q in [('block','ar98',.3,.2),('block','ar98',.6,.2),('block','mix',.3,.2),('block','ar98',.3,0.),('block','ar98',.3,.5),('ar','mix',.6,.2),('iid','ar98',.6,.2)]:
    for beta in (0.0, 0.1):
        h = dict(pseudo=0, gls=0, gls_pc1=0, gls_pc2=0)
        for _ in range(nrep):
            y = label(T, lab, rng); X = neurons(T, 40, w, q, dk, rng)
            if beta: X[:, :8] += beta * (y - y.mean())[:, None] / y.std()
            nulls = [label(T, lab, rng) for _ in range(199)]
            h['pseudo'] += p_test(X, y, nulls, 0) < .05; h['gls'] += p_gls(X, y, nulls) < .05
            h['gls_pc1'] += p_gls_shared(X, y, nulls, 1) < .05; h['gls_pc2'] += p_gls_shared(X, y, nulls, 2) < .05
        print(f"{lab:5s} {dk:4s} w={w} q={q} beta={beta}: " + "  ".join(f"{k}:{v/nrep:.3f}" for k, v in h.items()), flush=True)
