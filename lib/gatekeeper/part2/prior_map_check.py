"""Study B: the brain-wide prior map. Using ALL explore sessions per region (IBL-style: per-session decoding,
pseudo-session null, Fisher across sessions, BH-FDR over regions), how many regions are declared to encode
the prior, and how many does the SAME pipeline declare for a fake prior? Also with the common literature
procedure (pooled neurons, parametric) and the circular-shift null."""
import sys, numpy as np, pandas as pd
from scipy.stats import binom, chi2
sys.path.insert(0, '.')
from gatekeeper.multiverse import _bh
f = sys.argv[1] if len(sys.argv) > 1 else 'results/multiverse_explore/session_stats.csv.gz'
d = pd.read_csv(f); d = d[(d.window == 'a') & (d.units == 'good')]
def fisher(p): return chi2.sf(-2 * np.log(np.clip(p, 1e-12, 1)).sum(), 2 * len(p))
rows = []
for var in ('prior', 'ctl_prior', 'sprior', 'ctl_sprior', 'choice', 'ctl_choice', 'stim_all', 'ctl_stim_all'):
    g = d[d['var'] == var].groupby('region').filter(lambda h: len(h) >= 2)
    P = {m: [] for m in ('pooled_parametric', 'shuffle', 'circ', 'pseudo')}
    for reg, h in g.groupby('region'):
        P['pooled_parametric'].append(binom.sf(h.nsig.sum() - 1, h.n_units.sum(), .05))
        for m in ('shuffle', 'circ', 'pseudo'): P[m].append(fisher(h[f'p_decode_{m}'].values))
    R = g.region.nunique()
    rows.append(dict(var=var, regions=R, **{m: int(_bh(np.array(v), .05).sum()) for m, v in P.items()}))
o = pd.DataFrame(rows); print(o.to_string(index=False))
for real in ('prior', 'sprior', 'choice', 'stim_all'):
    a = o.set_index('var').loc[real]; b = o.set_index('var').loc['ctl_' + real]
    print(f"{real:8s} regions declared (real / fake):  " + "  ".join(f"{m} {a[m]}/{b[m]}" for m in ('pooled_parametric', 'shuffle', 'circ', 'pseudo')))
tag = sys.argv[2] if len(sys.argv) > 2 else 'explore'; o.to_csv(f'part2/prior_map_check_{tag}.csv', index=False)
