import sys, numpy as np, pandas as pd
from scipy.stats import chi2
sys.path.insert(0,'.')
from gatekeeper.multiverse import _bh
exec(open('/tmp/claude-0/audit2.py').read().split("seed=int")[0].split("from gatekeeper.multiverse import _bh")[1])
def pv2(X,meth,a):
    C,S,pm=X; sel=a>0; c=C[:,sel]; N=c.sum(1); nm=(c>0).sum(1)
    fam,m=meth.split('_'); m=int(m)
    if fam=='F':
        p=chi2.sf(S[:,sel].sum(1),2*np.maximum(N,1)); return np.where((N>=2)&(nm>=m),p,np.nan)
    s=np.sort(np.where(np.isnan(pm[:,sel]),-1,pm[:,sel]),1)[:,::-1]; out=np.full(len(nm),np.nan)
    for i in np.nonzero(nm>=max(m,2))[0]:
        top=s[i,:nm[i]-1]; out[i]=chi2.sf(-2*np.log(np.clip(top,1e-300,1)).sum(),2*len(top))
    return out
METH=('F_1','F_2','F_3','F_4','F_5','PC_2','PC_3','PC_4'); REFS=('F_1','PC_2')
seed=int(sys.argv[1])
for var,G in (('stim',(0.5,1,2)),('sprior',(0.5,1,2)),('choice',(3,6,12)),('feedback',(5,10,20))):
    X=mats(var); M=X[0].shape[1]; tot={m:[] for m in METH}; cov={m:[] for m in METH}
    for k in (8,12,16):
        rng=np.random.default_rng(seed); acc={(r,m,q):np.zeros(2) for r in REFS for m in METH for q in QS}; tested={m:0 for m in METH}
        for _ in range(150):
            p=rng.permutation(M); a=np.zeros(M); a[p[:k]]=1
            refs={r:ver(pv2(X,r,1-a),0.05) for r in REFS}
            for m in METH:
                pp=pv2(X,m,a); tested[m]+=np.sum(~np.isnan(pp))
                for q in QS:
                    c=ver(pp,q)
                    for r in REFS:
                        ok=(c>=0)&(refs[r]>=0); acc[(r,m,q)]+=[((c==1)&(refs[r]==1)&ok).sum(),((c==1)&ok).sum()]
        for m in METH:
            tot[m].append(np.nanmean([[prec([[acc[(r,m,q)][1]/150,acc[(r,m,q)][0]/150] for q in QS],g) for g in G] for r in REFS]))
            cov[m].append(tested[m]/150)
    print(f"seed {seed} {var:8s} "+"  ".join(f"{m}:{np.nanmean(tot[m]):.3f}({np.mean(cov[m]):.0f}r)" for m in METH),flush=True)
