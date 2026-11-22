import sys, numpy as np, pandas as pd
from scipy.stats import chi2, norm
sys.path.insert(0,'.')
from gatekeeper.multiverse import _bh, Universes, default_key
# A. PC formula check
p=np.array([[0.001,0.5,0.6]]); a=np.ones(3)
s=np.sort(p,1)[:, ::-1]; top=s[0,:2]
print("A. PC2 p for [0.001,.5,.6] =",chi2.sf(-2*np.log(top).sum(),4), "(should use 0.5,0.6 only; large) ok")
# C. reproduce standard from sens p200 vs original session_stats
d=pd.read_csv('results/repmap_explore/sens_stats.csv.gz')
o=pd.read_csv('results/multiverse_explore/session_stats.csv.gz')
o=o[(o.window=='a')&(o.units=='good')]
for var in ("stim","choice","sprior","feedback"):
    a1=d[d['var']==var].set_index(['sid','region']).p200
    a2=o[o['var']==var].set_index(['sid','region']).p_decode_pseudo
    j=a1.to_frame('n').join(a2.to_frame('o'),how='inner')
    print(f"C. {var}: rows sens {len(a1)} orig {len(a2)} matched {len(j)}; corr of -log p {np.corrcoef(-np.log(j.n),-np.log(j.o))[0,1]:.2f}; sig agree {np.mean((j.n<.05)==(j.o<.05)):.2f}")
