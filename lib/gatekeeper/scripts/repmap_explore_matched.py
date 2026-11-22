import numpy as np, json, sys
import pandas as pd
from gatekeeper import repmap as R
from gatekeeper.multiverse import Universes
df=pd.read_csv('results/multiverse_explore/session_stats.csv.gz')
df=df[df['var'].isin(R.VARS)&(df.window=='a')&(df.units=='good')]
U=Universes(df)
cache={v:R.mouse_p(U,v) for v in R.VARS}
refmode=sys.argv[1]
def ref(var,b):
    if refmode=='fisher': return R.reference(U,var,b)
    return R.claims(U,var,'pc_u2',b,0.05,cache)
rng=np.random.default_rng(1); M=len(U.mice)
grids={'stim':(0.5,1,2,4),'sprior':(0.5,1,2,4),'choice':(3,6,12,20),'feedback':(5,10,20)}
out={}
for var in ('stim','sprior','choice'):
  for k in (4,8,12,16):
    acc={(m,q):np.zeros(3) for m in ('fisher_session','pc_u2','pc_f30') for q in R.QS}
    for _ in range(150):
        p=rng.permutation(M); a=np.zeros(M); a[p[:k]]=1
        rf=ref(var,1-a)
        for (m,q),r in acc.items():
            c=R.claims(U,var,m,a,q,cache); ok=(c>=0)&(rf>=0)
            r[0]+=((c==1)&(rf==1)&ok).sum(); r[1]+=((c==1)&ok).sum(); r[2]+=1
    row={}
    for m in ('fisher_session','pc_u2','pc_f30'):
        cl=np.array([acc[(m,q)][1]/acc[(m,q)][2] for q in R.QS]); hi=np.array([acc[(m,q)][0]/acc[(m,q)][2] for q in R.QS])
        # precision at claim level g: interpolate hits vs claims
        o=np.argsort(cl); cl,hi=cl[o],hi[o]
        row[m]=[None if g>cl.max() or g<cl.min() else round(float(np.interp(g,cl,hi)/g),2) for g in grids[var]]
    out[f'{var}|{k}']=row
    print(var,k,'claims grid',grids[var],row,flush=True)
json.dump(out,open(f'/tmp/claude-0/rm_{refmode}.json','w'))
