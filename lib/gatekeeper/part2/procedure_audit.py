"""Study A: how often do common published PROCEDURES declare that a region encodes a variable that cannot be
encoded (IBL fake variables)? Explore mice, window a, good units.

A 'study' of region R = k randomly drawn sessions that recorded R. Procedures:
  pooled_parametric  pool all neurons from the k sessions; per-neuron parametric test (Mann-Whitney / Spearman,
                     |z|>1.96); binomial test of the count against 5%   (the most common literature procedure:
                     64% of coded claims pool neurons, 51% use no null)
  shuffle_fisher     per-session decoding vs trial-shuffle null, Fisher-combined over the k sessions
  circ_fisher        same with circular-shift null
  pseudo_fisher      same with pseudo-session / imposter null (IBL standard)
Output: false-positive rate per procedure x variable x k (share of studies with p < 0.05)."""
import sys, json, numpy as np, pandas as pd
from scipy.stats import binom, chi2
f = sys.argv[1] if len(sys.argv) > 1 else 'results/multiverse_explore/session_stats.csv.gz'
d = pd.read_csv(f); d = d[(d.window == 'a') & (d.units == 'good')]
rng = np.random.default_rng(7)
KS = (1, 3, 5, 10, 20); NREP = 200
def fisher(p): return chi2.sf(-2 * np.log(np.clip(p, 1e-12, 1)).sum(), 2 * len(p))
out = []
for var in sorted(d['var'].unique()):
    g = d[d['var'] == var]
    for k in KS:
        res = {m: [] for m in ('pooled_parametric', 'shuffle_fisher', 'circ_fisher', 'pseudo_fisher')}
        for reg, h in g.groupby('region'):
            if len(h) < k: continue
            for _ in range(max(1, NREP // 20)):
                s = h.iloc[rng.choice(len(h), k, replace=False)]
                n, x = s.n_units.sum(), s.nsig.sum()
                res['pooled_parametric'].append(binom.sf(x - 1, n, 0.05) < .05)
                res['shuffle_fisher'].append(fisher(s.p_decode_shuffle.values) < .05)
                res['circ_fisher'].append(fisher(s.p_decode_circ.values) < .05)
                res['pseudo_fisher'].append(fisher(s.p_decode_pseudo.values) < .05)
        if res['pseudo_fisher']:
            out.append(dict(var=var, k=k, n_regions=int(g.groupby('region').size().ge(k).sum()), **{m: float(np.mean(v)) for m, v in res.items()}))
o = pd.DataFrame(out)
print(o[o['var'].str.startswith('ctl_')].round(3).to_string(index=False))
print('\nREAL variables (share of regions declared encoding):')
print(o[~o['var'].str.startswith('ctl_')].round(3).to_string(index=False))
tag = sys.argv[2] if len(sys.argv) > 2 else 'explore'; o.to_csv(f'part2/procedure_audit_{tag}.csv', index=False)
