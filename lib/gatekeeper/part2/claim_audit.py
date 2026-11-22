"""Study C: for every coded literature claim (literature/claims.csv), the false-positive rate of THAT claim's
own procedure, measured on IBL fake variables (part2/procedure_audit.csv).
Mapping (fixed before looking at results):
  variable_class: stimulus->ctl_stim_all, choice->ctl_choice, outcome->ctl_feedback, prior->ctl_prior
  procedure: null pseudo -> pseudo_fisher; circular -> circ_fisher;
             null none or shuffle with method single OR level pooled -> pooled_parametric;
             null none or shuffle with method decoding -> shuffle_fisher; anything unclear -> 'unclear' (reported)
  sessions k: sessions_per_region, else n_mice, else 5; rounded DOWN to the grid 1,3,5,10,20
Usage: python3 part2/claim_audit.py [claims.csv] [procedure_audit.csv]"""
import sys, numpy as np, pandas as pd
cf = sys.argv[1] if len(sys.argv) > 1 else 'literature/claims.csv'
af = sys.argv[2] if len(sys.argv) > 2 else 'part2/procedure_audit_explore.csv'
c = pd.read_csv(cf, dtype=str); a = pd.read_csv(af)
VAR = {'stimulus': 'ctl_stim_all', 'choice': 'ctl_choice', 'outcome': 'ctl_feedback', 'prior': 'ctl_prior'}
GRID = np.array([1, 3, 5, 10, 20])
def num(x):
    try: return float(x)
    except (TypeError, ValueError): return np.nan
def proc(r):
    n, m, l = str(r['null']).strip().lower(), str(r['method']).strip().lower(), str(r['level']).strip().lower()
    if n == 'pseudo': return 'pseudo_fisher'
    if n == 'circular': return 'circ_fisher'
    if n in ('none', 'shuffle'):
        if m == 'single' or l == 'pooled': return 'pooled_parametric'
        if m == 'decoding': return 'shuffle_fisher'
    return 'unclear'
rows = []
for _, r in c.iterrows():
    v = VAR.get(str(r['variable_class']).strip().lower()); p = proc(r)
    k = num(r['sessions_per_region']); k = k if np.isfinite(k) else num(r['n_mice']); k = k if np.isfinite(k) else 5
    kg = int(GRID[GRID <= max(k, 1)].max())
    fp = np.nan
    if v and p != 'unclear':
        hit = a[(a['var'] == v) & (a.k == kg)]
        if len(hit): fp = float(hit[p].iloc[0])
    rows.append(dict(claim_id=r['claim_id'], variable_class=r['variable_class'], procedure=p, k=kg, procedure_fp=fp))
o = pd.DataFrame(rows); o.to_csv('part2/claim_audit.csv', index=False)
s = o.dropna(subset=['procedure_fp'])
print(f"{len(o)} claims; {len(s)} with a mappable procedure; {(o.procedure == 'unclear').sum()} unclear")
print(f"median procedure false-positive rate: {s.procedure_fp.median():.2f}")
for t in (0.2, 0.5, 0.9):
    print(f"  claims whose own procedure flags a NONEXISTENT variable >{int(t*100)}% of the time: {(s.procedure_fp > t).mean():.0%} ({(s.procedure_fp > t).sum()}/{len(s)})")
print(s.groupby('variable_class').procedure_fp.agg(['size', 'median', lambda x: (x > .5).mean()]).round(2).rename(columns={'<lambda_0>': 'share>50%'}))
print(o.procedure.value_counts().to_string())
