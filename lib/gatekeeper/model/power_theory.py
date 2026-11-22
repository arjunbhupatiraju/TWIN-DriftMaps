"""Information limit (Whittle): with neural noise spectrum f_n(w) (variance-normalised) and label spectrum f_y(w)
(variance-normalised), a neuron with encoding weight beta gives noncentrality
    GLS (optimal):      lam = T beta^2 * mean_w[ f_y / f_n ]
    plain + valid null: lam = T beta^2 / mean_w[ f_y f_n ]           (signal diluted by Bartlett inflation V)
Compare with simulated power."""
import sys, numpy as np
sys.path.insert(0, '.')
from scipy.stats import ncx2, chi2
from theory.sim_corrected import label
T = 600; rng = np.random.default_rng(0)
w_ = np.fft.rfftfreq(T) * 2 * np.pi
def fy(kind, n=400):
    P = np.zeros(len(w_))
    for _ in range(n):
        y = label(T, kind, rng); y = (y - y.mean()) / y.std(); P += np.abs(np.fft.rfft(y)) ** 2 / T
    return P / n
def fn(w, mix):
    out = (1 - w) * np.ones(len(w_))
    comps = [(0.98, 1.0)] if not mix else [(0.9, .25), (0.98, .25), (0.995, .25), (0.999, .25)]
    for a, wt in comps:
        out = out + w * wt * (1 - a * a) / np.abs(1 - a * np.exp(-1j * w_)) ** 2
    return out / out.mean()
def power(lam_per, n_sig=8, N=40, alpha=.05):
    return ncx2.sf(chi2.isf(alpha, N), N, n_sig * lam_per)
beta = 0.1
for kind in ('block', 'ar', 'iid'):
    f_y = fy(kind)
    for mix in (False, True):
        for w in (0.0, 0.3, 0.6):
            f = fn(w, mix)
            lg = T * beta**2 * np.mean(f_y / f) / (1 + beta**2)
            lp = T * beta**2 / np.mean(f_y * f) / (1 + beta**2)
            print(f"{kind:5s} {'mix ' if mix else 'ar98'} w={w}: predicted power GLS {power(lg):.3f}   plain {power(lp):.3f}")
