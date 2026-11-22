import sys, numpy as np, pandas as pd
from scipy.stats import chi2, norm
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from gatekeeper.multiverse import _bh
d=pd.read_csv('results/repmap_explore/sens_stats.csv.gz')
QS=(1e-6,1e-5,1e-4,5e-4,1e-3,5e-3,0.01,0.02,0.05,0.1,0.2,0.3,0.5)
def mats(var):
    g=d[d['var']==var]; mice=sorted(d.subject.unique()); regs=sorted(g.region.unique())
    ri={r:i for i,r in enumerate(regs)}; mi={m:j for j,m in enumerate(mice)}
    rr,mm=g.region.map(ri).to_numpy(),g.subject.map(mi).to_numpy(); p=np.clip(g.p200.to_numpy(),1e-12,1)
    C=np.zeros((len(regs),len(mice))); np.add.at(C,(rr,mm),1); S=np.zeros_like(C); np.add.at(S,(rr,mm),-2*np.log(p))
    with np.errstate(all='ignore'): pm=np.where(C>0,chi2.sf(S,2*np.maximum(C,1)),np.nan)
    return C,S,pm
def pv(X,meth,a):
    C,S,pm=X; sel=a>0; c=C[:,sel]; N=c.sum(1); nm=(c>0).sum(1)
    fis=chi2.sf(S[:,sel].sum(1),2*np.maximum(N,1))
    if meth=='F': return np.where(N>=2,fis,np.nan)
    if meth=='F_2m': return np.where((N>=2)&(nm>=2),fis,np.nan)
    if meth=='F_3m': return np.where((N>=2)&(nm>=3),fis,np.nan)
    s=np.sort(np.where(np.isnan(pm[:,sel]),-1,pm[:,sel]),1)[:,::-1]; out=np.full(len(nm),np.nan)
    for i in np.nonzero(nm>=2)[0]:
        top=s[i,:nm[i]-1]; out[i]=chi2.sf(-2*np.log(np.clip(top,1e-300,1)).sum(),2*len(top))
    return out
def ver(p,q):
    o=np.full(len(p),-1,np.int8); t=~np.isnan(p)
    if t.any(): o[t]=_bh(p[t],q).astype(np.int8)
    return o
def prec(curve,g):
    c=np.array(curve); o=np.argsort(c[:,0]); cl,hi=c[o,0],c[o,1]
    return np.nan if g>cl.max() or g<cl.min() else np.interp(g,cl,hi)/g
seed=int(sys.argv[1]); METH=('F','F_2m','F_3m','PC2'); REFS=('F','PC2')
for var,G in (('stim',(0.5,1,2)),('sprior',(0.5,1,2)),('choice',(3,6,12))):
    X=mats(var); M=X[0].shape[1]; tot={m:[] for m in METH}
    for k in (8,12,16):
        rng=np.random.default_rng(seed); acc={(r,m,q):np.zeros(2) for r in REFS for m in METH for q in QS}
        for _ in range(150):
            p=rng.permutation(M); a=np.zeros(M); a[p[:k]]=1
            refs={r:ver(pv(X,r,1-a),0.05) for r in REFS}
            for m in METH:
                pp=pv(X,m,a)
                for q in QS:
                    c=ver(pp,q)
                    for r in REFS:
                        ok=(c>=0)&(refs[r]>=0); acc[(r,m,q)]+=[((c==1)&(refs[r]==1)&ok).sum(),((c==1)&ok).sum()]
        for m in METH:
            v=np.nanmean([[prec([[acc[(r,m,q)][1]/150,acc[(r,m,q)][0]/150] for q in QS],g) for g in G] for r in REFS])
            tot[m].append(v)
    print(f"seed {seed} {var:7s} mean precision over k, levels, refs: "+"  ".join(f"{m}:{np.mean(tot[m]):.3f}" for m in METH),flush=True)
