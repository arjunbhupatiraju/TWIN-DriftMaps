"""Figure: shuffle-null false positives in real IBL fake variables vs number of neurons, with the fitted
v2 theory (one shared neural-correlation structure, one drift factor per variable) and the out-of-sample
prediction for window b."""
import sys, numpy as np, pandas as pd, warnings
sys.path.insert(0, '.'); warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from theory.drift_fp import fp_general
import theory.fit_real_v2_oos as F
fa = F.fit(F.ag); rho0, q, dl = F.unpack(fa.x)
names = sorted(F.ag)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
cols = plt.cm.tab10(np.arange(6))
Ng = np.geomspace(5, 400, 100)
for j, (title, data) in enumerate([("Window a (fitted)", F.ag), ("Window b (predicted, no refit)", F.bg)]):
    for v, de, c in zip(names, dl, cols):
        N, n, k = data[v]; ax[j].plot(N, k / n, 'o', color=c, label=v.replace('ctl_', 'fake ') if j == 0 else None)
        ax[j].plot(Ng, fp_general(Ng, de, rho0, q), '-', color=c, lw=1.5)
    ax[j].axhline(0.05, color='k', ls=':'); ax[j].set_xscale('log'); ax[j].set_xlabel('neurons in region'); ax[j].set_title(title)
ax[0].set_ylabel('shuffle-null false-positive rate'); ax[0].legend(fontsize=8)
fig.suptitle(f"Shared neuron correlation rho0={rho0:.2f}, shared drift q={q:.2f}; per-variable V = " + ", ".join(f"{1+x:.2f}" for x in dl), fontsize=9)
fig.tight_layout(); fig.savefig('theory/figure_theory.png', dpi=140); print('saved')
