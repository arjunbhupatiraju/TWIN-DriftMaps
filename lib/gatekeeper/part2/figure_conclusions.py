"""Figure: (a) false-positive rate of each procedure on fake variables vs sessions per region;
(b) regions declared to encode the real vs the fake variable, brain-wide; (c) literature claims' own-procedure FP."""
import numpy as np, pandas as pd, os
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
a = pd.read_csv('part2/procedure_audit_explore.csv'); b = pd.read_csv('part2/prior_map_check_explore.csv').set_index('var')
M = ['pooled_parametric', 'shuffle_fisher', 'circ_fisher', 'pseudo_fisher']
L = ['pooled neurons, parametric (most common)', 'decoding, shuffle null', 'decoding, circular shift', 'decoding, pseudo-session']
has_c = os.path.exists('part2/claim_audit.csv')
fig, ax = plt.subplots(1, 3 if has_c else 2, figsize=(17 if has_c else 12, 4.6))
for v, ls in (('ctl_prior', '-'), ('ctl_choice', '--'), ('ctl_stim_all', ':')):
    g = a[a['var'] == v]
    for m, lab, c in zip(M, L, plt.cm.tab10(range(4))):
        ax[0].plot(g.k, g[m], ls, color=c, marker='o', label=lab if v == 'ctl_prior' else None)
ax[0].axhline(.05, color='k', lw=.8); ax[0].set_xscale('log'); ax[0].set_xticks([1, 3, 5, 10, 20]); ax[0].set_xticklabels([1, 3, 5, 10, 20])
ax[0].set_xlabel('sessions per region'); ax[0].set_ylabel('regions declared to encode a FAKE variable')
ax[0].set_title('Fake prior (solid), choice (dashed), stimulus (dotted)'); ax[0].legend(fontsize=7)
vars_ = ['prior', 'sprior', 'choice', 'stim_all']; x = np.arange(len(vars_)); w = 0.2
for i, (m, c) in enumerate(zip(['pooled_parametric', 'shuffle', 'circ', 'pseudo'], plt.cm.tab10(range(4)))):
    ax[1].bar(x + (i - 1.5) * w, [b.loc[v, m] for v in vars_], w, color=c, alpha=.9)
    ax[1].bar(x + (i - 1.5) * w, [b.loc['ctl_' + v, m] for v in vars_], w, color='none', edgecolor='k', hatch='///')
ax[1].set_xticks(x); ax[1].set_xticklabels(['block prior', 'subjective prior', 'choice', 'stimulus'])
ax[1].set_ylabel(f"regions declared (of {int(b.regions.iloc[0])})"); ax[1].set_title('Brain-wide map: real (filled) vs fake (hatched)')
if has_c:
    c = pd.read_csv('part2/claim_audit.csv').dropna(subset=['procedure_fp'])
    ax[2].hist(c.procedure_fp, bins=np.linspace(0, 1, 21), color='gray'); ax[2].axvline(.05, color='k', ls=':')
    ax[2].set_xlabel("false-positive rate of the claim's own procedure"); ax[2].set_ylabel('published claims')
    ax[2].set_title(f'{(c.procedure_fp > .5).mean():.0%} of claims: procedure flags a fake variable >50% of the time')
fig.tight_layout(); fig.savefig('part2/figure_conclusions.png', dpi=140); print('saved part2/figure_conclusions.png')
