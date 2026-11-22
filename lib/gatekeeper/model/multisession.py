"""Sessions needed: region-level power when k sessions are Fisher-combined (block prior-like label, drifting
neurons). pseudo-session test vs GLS-imposter test. Also the false-positive rate (beta=0)."""
import sys, numpy as np, json
sys.path.insert(0, '.')
from scipy.stats import chi2
from theory.sim_corrected import label
from model.proto import neurons, p_test
from model.gls import p_gls, p_gls_shared
from joblib import Parallel, delayed
T, N = 600, 20
def one(seed, k, w, beta):
    rng = np.random.default_rng(seed); pp, pg = [], []
    for _ in range(k):
        y = label(T, 'block', rng); X = neurons(T, N, w, 0.2, 'mix', rng)
        if beta: X[:, :4] += beta * (y - y.mean())[:, None] / y.std()
        nulls = [label(T, 'block', rng) for _ in range(199)]
        pp.append(p_test(X, y, nulls, 0)); pg.append(p_gls_shared(X, y, nulls, 1))
    f = lambda p: chi2.sf(-2 * np.log(p).sum(), 2 * len(p))
    return f(np.array(pp)) < .05, f(np.array(pg)) < .05
out = []
for w in (0.3, 0.6):
    for beta in (0.0, 0.1, 0.15):
        for k in (1, 3, 5, 10, 20):
            r = np.array(Parallel(n_jobs=-1)(delayed(one)(1000 * k + i + int(beta * 1e4) + int(w * 1e5), k, w, beta) for i in range(100)))
            row = dict(w=w, beta=beta, k=k, pseudo=float(r[:, 0].mean()), gls=float(r[:, 1].mean())); out.append(row); print(row, flush=True)
json.dump(out, open('model/multisession.json', 'w'), indent=1)
