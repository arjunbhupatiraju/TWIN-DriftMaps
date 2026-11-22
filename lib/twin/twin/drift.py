"""Neural twins: recordings with the same drift and correlation structure as the real one, but no task alignment.

Two generators:
  phase_twin(X)        nonparametric: multivariate phase randomization of the real recording (same random phases for
                       every neuron, so each neuron's autocorrelation AND the cross-neuron correlations are kept).
  DriftModel.simulate  parametric: fitted per-neuron slow fraction s, slow AR(1) coefficient phi, and shared fraction q.
                       Used by the planner, where no recording exists yet.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


def phase_twin(X, rng):
    X = np.asarray(X, float)
    T = len(X)
    mu = X.mean(0)
    F = np.fft.rfft(X - mu, axis=0)
    ph = rng.uniform(0, 2 * np.pi, F.shape[0])
    ph[0] = 0.0
    if T % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(F * np.exp(1j * ph)[:, None], n=T, axis=0) + mu


TAUS = (2.0, 5.0, 12.0, 30.0, 75.0, 190.0)


def _ar(e, p):
    """Stationary unit-variance AR(1) along axis 0 from white noise e (vectorized filter)."""
    from scipy.signal import lfilter
    q = np.sqrt(1 - p ** 2)
    e = e.copy(); e[0] = e[0] / max(q, 1e-12)
    return lfilter([q], [1, -p], e, axis=0)      # timescale ladder (trials) for the multi-timescale model


@dataclass
class DriftModel:
    """Neural drift as a sum of slow components on a ladder of timescales plus fast noise.

    Each component j (AR(1) with coefficient phis[j]) carries a share a[j] of each neuron's variance, of which a
    fraction shared[j] is common to all neurons. The rest (1 - sum a) is fast noise with instantaneous correlation
    inst_corr. The classic one-timescale model is DriftModel(slow_frac, phi, shared, inst_corr)."""
    slow_frac: float = 0.3
    phi: float = 0.97
    shared: float = 0.35
    inst_corr: float = 0.05
    phis: Optional[list] = None
    a: Optional[list] = None
    sh: Optional[list] = None
    het: Optional[list] = None       # per-neuron drift-strength multipliers (quantiles, mean 1); None = all neurons alike
    fac_pi: Optional[list] = None    # v4: shared slow factors, per-neuron variance share [factor][ladder component]
    fac_rows: Optional[list] = None  # v4: loading rows (neurons x factors, each column mean square 1), bootstrapped in simulation

    def components(self):
        if self.phis is None:
            return [self.phi], [self.slow_frac], [self.shared]
        return list(self.phis), list(self.a), list(self.sh)

    # ------------------------------------------------------------------ fitting
    @staticmethod
    def _curves(X, K):
        X = np.asarray(X, float)
        X = (X - X.mean(0)) / (X.std(0) + 1e-12)
        T, N = X.shape
        F = np.fft.rfft(X, n=2 * T, axis=0)
        ac = np.fft.irfft(F * np.conj(F), axis=0)[: K + 1] / T
        rho = np.mean(ac[1:] / ac[:1], axis=1)
        g = X.sum(1)
        G = np.fft.rfft(g - g.mean(), n=2 * T)
        acg = np.fft.irfft(G * np.conj(G))[: K + 1] / T
        cross = (acg - ac.sum(1)) / max(N * (N - 1), 1)
        return rho, cross

    @classmethod
    def _fit_multi(cls, X, K, taus=None, delta=None):
        """delta = (d_rho, d_cross): estimation bias of the curves, added before fitting (bias correction)."""
        from scipy.optimize import nnls
        T, N = np.asarray(X).shape
        K = int(min(K, T // 4))
        rho, cross = cls._curves(X, K)
        if delta is not None:
            rho, cross = rho + delta[0], cross + delta[1]
        k = np.arange(1, K + 1)
        phis = np.exp(-1.0 / np.array(taus or TAUS))
        B = np.column_stack([p ** k for p in phis])
        a, _ = nnls(B, rho)
        tot = a.sum()
        if tot > 0.99:
            a = a * 0.99 / tot
        c, _ = nnls(B, np.clip(cross[1:], 0, None)) if N > 1 else (np.zeros(len(phis)), 0)
        sh = np.where(a > 1e-6, np.clip(c / np.maximum(a, 1e-9), 0, 1), 0.0)
        inst = float(np.clip((cross[0] - (a * sh).sum()) / max(1 - a.sum(), 1e-6), 0, 0.9)) if N > 1 else 0.0
        # heterogeneity: each neuron's own autocorrelation scaled onto the shared ladder shape
        shape = B @ a
        Xz = np.asarray(X, float); Xz = (Xz - Xz.mean(0)) / (Xz.std(0) + 1e-12)
        F = np.fft.rfft(Xz, n=2 * T, axis=0)
        aci = np.fft.irfft(F * np.conj(F), axis=0)[1: K + 1] / T
        sc = np.clip((shape @ aci) / max(shape @ shape, 1e-12), 0, None)
        het = (sc / sc.mean()).tolist() if sc.mean() > 0 else None
        return dict(phis=phis.tolist(), a=a.tolist(), sh=sh.tolist(), inst_corr=inst, het=het)

    @staticmethod
    def _lagcov(Xz, L):
        T = len(Xz)
        C = np.zeros((Xz.shape[1],) * 2)
        for k in range(1, L + 1):
            c = Xz[:-k].T @ Xz[k:] / (T - k)
            C += (c + c.T) / 2
        return C / L

    @classmethod
    def _fit_factors(cls, X, K, taus=None, F=None, rng=None, max_f=8, L=10, delta=None):
        """Shared slow drift as F orthogonal factors (sign-invariant, any dimensionality).
        Lag-averaged cross-covariance (lags 1..L, so fast noise drops out), principal-axis factoring with the diagonal
        re-estimated, factor count by parallel analysis against independently phase-randomized neurons. Each factor's
        timescale mix comes from a ladder fit to its own off-diagonal lagged covariance curve."""
        from scipy.optimize import nnls
        X = np.asarray(X, float)
        T, N = X.shape
        if N < 3:
            return None
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
        L = int(min(L, T // 10))

        def paf(C, nf):
            d = np.zeros(N)
            for _ in range(6):
                M = C.copy(); np.fill_diagonal(M, d)
                w, V = np.linalg.eigh(M); w, V = w[::-1], V[:, ::-1]
                d = (V[:, :nf] ** 2 * np.clip(w[:nf], 0, None)).sum(1) if nf else np.zeros(N)
            return w, V

        C = cls._lagcov(Xz, L)
        if F is None:
            rng = rng or np.random.default_rng(0)
            null = []
            for _ in range(4):
                Xn = np.column_stack([phase_twin(Xz[:, [i]], rng)[:, 0] for i in range(N)])
                M = cls._lagcov(Xn, L); np.fill_diagonal(M, 0)
                null.append(np.sort(np.linalg.eigvalsh(M))[::-1][:max_f])
            thr = np.max(null, 0) if len(null) else np.zeros(max_f)
            M = C.copy(); np.fill_diagonal(M, 0)
            w0 = np.sort(np.linalg.eigvalsh(M))[::-1][:max_f]
            F = 0
            while F < max_f and F < N - 1 and w0[F] > thr[F]:
                F += 1
        if F == 0:
            return dict(F=0, pi=[], rows=None)
        w, V = paf(C, F)
        V = V[:, :F]
        Km = int(min(K, T // 4)); k = np.arange(1, Km + 1)
        phis = np.exp(-1.0 / np.array(taus or TAUS)); B = np.column_stack([p ** k for p in phis])
        # off-diagonal lagged covariance projected on each factor direction, lags 1..Km (FFT per factor pair term)
        P = Xz @ V                                            # T x F projections
        Fx = np.fft.rfft(Xz, n=2 * T, axis=0)
        pis, curves = [], []
        for f in range(F):
            Fp = np.fft.rfft(P[:, f], n=2 * T)
            tot = np.fft.irfft(Fp * np.conj(Fp))[1: Km + 1] / T                       # v' C_k v (all terms)
            diag = (np.fft.irfft(Fx * np.conj(Fx), axis=0)[1: Km + 1] / T) @ (V[:, f] ** 2)   # sum_i v_i^2 c_ii(k)
            curve = (tot - diag) / max(1 - (V[:, f] ** 4).sum(), 1e-6)                # undo diagonal removal
            curves.append(curve / N)
            if delta is not None:
                curve = curve + delta[f] * N
            b, _ = nnls(B, np.clip(curve, 0, None))
            pis.append((b / N).tolist())                                              # per-neuron variance share
        rows = (V * np.sqrt(N)).tolist()
        return dict(F=F, pi=pis, rows=rows, curves=curves, B=B)

    @classmethod
    def _moments(cls, X, K=60):
        """One-timescale method of moments (kept for the simple model and for comparison)."""
        X = np.asarray(X, float)
        T, N = X.shape
        K = int(min(K, T // 5))
        rho, cross = cls._curves(X, K)
        k = np.arange(1, K + 1)
        best = (np.inf, 0.9, 0.3)
        for phi in np.linspace(0.5, 0.999, 400):
            b = phi ** k
            sl = float(np.clip((b @ rho) / (b @ b), 0, 0.99))
            err = ((rho - sl * b) ** 2).sum()
            if err < best[0]:
                best = (err, phi, sl)
        _, phi, slow = best
        b = phi ** k
        sh_slow = float((b @ cross[1:]) / (b @ b)) if N > 1 else 0.0
        shared = float(np.clip(sh_slow / max(slow, 1e-6), 0, 1))
        inst = float(np.clip((cross[0] - slow * shared) / max(1 - slow, 1e-6), 0, 0.9)) if N > 1 else 0.0
        return dict(slow_frac=slow, phi=float(phi), shared=shared, inst_corr=inst)

    @classmethod
    def fit(cls, X, K=120, multi=True, bias_correct=True, n_sim=6, seed=0, hetero=True, taus=None, factors=False, bc="curves"):
        """Estimate drift from a recording (T trials x N neurons). multi=True fits the timescale ladder (recommended);
        multi=False fits one timescale. One step of simulation-based bias correction removes the downward bias of
        autocorrelations in short recordings."""
        X = np.asarray(X, float)
        T, N = X.shape
        rng = np.random.default_rng(seed)
        if not multi:
            th = cls._moments(X, min(K, 60))
            if not bias_correct:
                return cls(**th)
            sims = [cls._moments(cls(**th).simulate(T, N, rng), min(K, 60)) for _ in range(n_sim)]
            lim = dict(slow_frac=(0, 0.99), phi=(0.5, 0.999), shared=(0, 1), inst_corr=(0, 0.9))
            return cls(**{key: float(np.clip(2 * v - np.mean([s_[key] for s_ in sims]), *lim[key])) for key, v in th.items()})
        th = cls._fit_multi(X, K, taus)
        het = th.pop("het")
        # the raw per-neuron spread includes estimation noise; shrink it by the spread seen in homogeneous simulations
        m0 = cls(phis=th["phis"], a=th["a"], sh=th["sh"], inst_corr=th["inst_corr"], slow_frac=float(sum(th["a"])))
        if het is not None and hetero:
            noise_sd = np.mean([np.std(cls._fit_multi(m0.simulate(T, N, rng), K, taus)["het"] or [1.0]) for _ in range(3)])
            h = np.asarray(het); sd = h.std()
            keep = np.sqrt(max(sd ** 2 - noise_sd ** 2, 0)) / max(sd, 1e-9)
            het = (1 + keep * (h - 1)).clip(0, None)
            het = np.quantile(het, np.linspace(0.05, 0.95, 19)).tolist()
        else:
            het = None
        if not bias_correct:
            return m0
        if bc == "coef":                                          # v0.3 behaviour, kept for comparison
            sims = [cls._fit_multi(m0.simulate(T, N, rng), K, taus) for _ in range(n_sim)]
            a = np.clip(2 * np.array(th["a"]) - np.mean([s_["a"] for s_ in sims], 0), 0, None)
            if a.sum() > 0.99:
                a *= 0.99 / a.sum()
            sh = np.clip(2 * np.array(th["sh"]) - np.mean([s_["sh"] for s_ in sims], 0), 0, 1)
            inst = float(np.clip(2 * th["inst_corr"] - np.mean([s_["inst_corr"] for s_ in sims]), 0, 0.9))
            return cls(phis=th["phis"], a=a.tolist(), sh=sh.tolist(), inst_corr=inst, slow_frac=float(a.sum()), het=het)
        # bias correction on the curves (stable), not on ladder coefficients (mass moves between neighbouring timescales)
        Kc = int(min(K, T // 4)); k = np.arange(1, Kc + 1)
        B = np.column_stack([p ** k for p in th["phis"]]); a0, sh0 = np.array(th["a"]), np.array(th["sh"])
        rho_th = B @ a0
        cross_th = np.concatenate([[(a0 * sh0).sum() + (1 - a0.sum()) * th["inst_corr"]], B @ (a0 * sh0)])
        cs = [cls._curves(m0.simulate(T, N, rng), Kc) for _ in range(n_sim)]
        d_rho = rho_th - np.mean([c[0] for c in cs], 0)
        d_cross = cross_th - np.mean([c[1] for c in cs], 0)
        thc = cls._fit_multi(X, K, taus, delta=(d_rho, d_cross)); thc.pop("het")
        a = np.array(thc["a"]); sh = np.array(thc["sh"]); inst = float(thc["inst_corr"])
        out = cls(phis=th["phis"], a=a.tolist(), sh=sh.tolist(), inst_corr=inst, slow_frac=float(a.sum()), het=het)
        if factors and N >= 3:
            fa = cls._fit_factors(X, K, taus, rng=rng)
            if fa["F"] > 0:
                pi = np.array(fa["pi"])
                m1 = cls(**{**out.dict(), "fac_pi": pi.tolist(), "fac_rows": fa["rows"]})
                fs = [cls._fit_factors(m1.simulate(T, N, rng), K, taus, F=fa["F"]) for _ in range(max(2, n_sim // 2))]
                d = [fa["B"] @ pi[f] - np.mean([f_["curves"][f] for f_ in fs], 0) for f in range(fa["F"])]
                pi = np.array(cls._fit_factors(X, K, taus, F=fa["F"], delta=d)["pi"])
                sh_tot = pi.sum(0)
                scale = min(1.0, 0.98 * a.sum() / max(sh_tot.sum(), 1e-9))       # shared slow cannot exceed total slow
                out = cls(**{**out.dict(), "fac_pi": (pi * scale).tolist(), "fac_rows": fa["rows"]})
            else:
                out = cls(**{**out.dict(), "sh": [0.0] * len(a)})
        return out

    # ------------------------------------------------------------------ simulation
    def simulate(self, T, N, rng):
        phis, a, sh = self.components()
        tot = float(sum(a))
        m = np.ones(N) if not self.het else rng.choice(np.asarray(self.het, float), N)
        m = np.minimum(m, 0.99 / max(tot, 1e-9))                     # each neuron's slow share stays below 0.99
        X = np.zeros((T, N))
        if self.fac_pi:
            return self._simulate_factors(T, N, rng, m)
        for p, aj, sj in zip(phis, a, sh):
            if aj <= 0:
                continue
            z = _ar(rng.standard_normal((T, N + 1)), p)
            X += np.sqrt(aj * m) * (np.sqrt(sj) * z[:, :1] + np.sqrt(1 - sj) * z[:, 1:])
        rest = np.maximum(1e-6, 1 - tot * m)
        g = rng.standard_normal((T, 1))
        X += np.sqrt(rest) * (np.sqrt(self.inst_corr) * g + np.sqrt(1 - self.inst_corr) * rng.standard_normal((T, N)))
        return X

    def _simulate_factors(self, T, N, rng, m):
        phis, a, _ = self.components()
        pi = np.asarray(self.fac_pi, float)                   # F x J
        rows = np.asarray(self.fac_rows, float)
        W = rows[rng.integers(0, len(rows), N)]               # N x F loading rows, bootstrapped
        W2 = W ** 2
        X = np.zeros((T, N))
        shared_var = np.zeros(N)
        priv = np.clip(np.asarray(a) - pi.sum(0), 0, None)   # mean private slow share per component
        tot_slow = np.zeros(N)
        for j, p in enumerate(phis):
            q = np.sqrt(1 - p ** 2)
            need = int((pi[:, j] > 0).sum()) + (priv[j] > 0)
            if not need:
                continue
            z = _ar(rng.standard_normal((T, len(pi) + N)), p)
            sv = W2 @ pi[:, j]                                   # each neuron's shared variance in this component
            X += z[:, :len(pi)] @ (W * np.sqrt(pi[:, j])).T
            pv = priv[j] * m
            X += np.sqrt(pv) * z[:, len(pi):]
            tot_slow += sv + pv
        cap = 0.99
        over = tot_slow > cap                                     # rescale the rare neuron whose slow share would exceed 1
        if over.any():
            X[:, over] *= np.sqrt(cap / tot_slow[over]); tot_slow[over] = cap
        rest = np.maximum(1e-6, 1 - tot_slow)
        g = rng.standard_normal((T, 1))
        X += np.sqrt(rest) * (np.sqrt(self.inst_corr) * g + np.sqrt(1 - self.inst_corr) * rng.standard_normal((T, N)))
        return X

    def dict(self):
        return asdict(self)

    def summary(self):
        phis, a, sh = self.components()
        return {f"tau{1 / (1 - p):.0f}": round(float(x), 3) for p, x in zip(phis, a) if x > 0.005}
