"""Bernoulli GLM-HMM (Ashwood et al. 2022, Nat Neurosci) fit by EM, numpy/scipy only.

Each hidden state k has its own logistic choice model P(right | x_t) = sigmoid(x_t . w_k).
States follow a Markov chain shared across sessions. No-go trials (choice < 0) contribute
no emission likelihood but still advance the chain.

Posteriors:
  smoothed  P(z_t | all choices)       -> used for switch detection (symmetric in time, no lag bias)
  predicted P(z_t | choices before t)  -> used for forecasting (causal)
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def _emission_loglik(X, y, W):
    """(T, K) log P(y_t | x_t, state k); 0 for missing trials."""
    eta = X @ W.T
    ll = y[:, None] * -np.logaddexp(0, -eta) + (1 - y[:, None]) * -np.logaddexp(0, eta)
    ll[y < 0] = 0.0
    return ll


def _forward_backward(logB, logA, logpi):
    T, K = logB.shape
    la = np.zeros((T, K))
    la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logsumexp(la[t - 1][:, None] + logA, axis=0) + logB[t]
    lb = np.zeros((T, K))
    for t in range(T - 2, -1, -1):
        lb[t] = logsumexp(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
    ll = logsumexp(la[-1])
    gamma = np.exp(la + lb - ll)
    xi = np.zeros((K, K))
    for t in range(T - 1):
        m = la[t][:, None] + logA + (logB[t + 1] + lb[t + 1])[None, :] - ll
        xi += np.exp(m)
    return gamma, xi, ll, la


class GLMHMM:
    def __init__(self, K=2, prior_sigma=2.0, n_restarts=5, max_iter=200, tol=1e-4, seed=0):
        self.K, self.prior_sigma, self.n_restarts = K, prior_sigma, n_restarts
        self.max_iter, self.tol, self.seed = max_iter, tol, seed

    # ---------------- fitting ----------------
    def _mstep_weights(self, Xs, ys, gammas, W0):
        X = np.concatenate(Xs)
        y = np.concatenate(ys).astype(float)
        G = np.concatenate(gammas)
        ok = y >= 0
        X, y, G = X[ok], y[ok], G[ok]
        W = W0.copy()
        for k in range(self.K):
            g = G[:, k]

            def f(w):
                eta = X @ w
                nll = -(g * (y * -np.logaddexp(0, -eta) + (1 - y) * -np.logaddexp(0, eta))).sum()
                grad = -X.T @ (g * (y - expit(eta)))
                return nll + 0.5 * w @ w / self.prior_sigma ** 2, grad + w / self.prior_sigma ** 2

            W[k] = minimize(f, W0[k], jac=True, method="L-BFGS-B").x
        return W

    def _fit_once(self, Xs, ys, rng):
        D = Xs[0].shape[1]
        W = self._init_W(Xs, ys, rng, D)
        A = np.full((self.K, self.K), 0.05 / max(self.K - 1, 1))
        np.fill_diagonal(A, 0.95)
        pi = np.full(self.K, 1.0 / self.K)
        prev = -np.inf
        for it in range(self.max_iter):
            gammas, xis, ll = [], np.zeros((self.K, self.K)), 0.0
            for X, y in zip(Xs, ys):
                g, xi, l, _ = _forward_backward(_emission_loglik(X, y, W), np.log(A), np.log(pi))
                gammas.append(g)
                xis += xi
                ll += l
            lp = ll - 0.5 * (W ** 2).sum() / self.prior_sigma ** 2
            if lp - prev < self.tol * abs(prev if np.isfinite(prev) else 1):
                break
            prev = lp
            A = (xis + 1.0) / (xis + 1.0).sum(1, keepdims=True)    # Dirichlet(2) smoothing
            pi = np.mean([g[0] for g in gammas], 0) + 1e-3
            pi /= pi.sum()
            W = self._mstep_weights(Xs, ys, gammas, W)
        return dict(W=W, A=A, pi=pi, ll=ll, n_iter=it + 1)

    def _init_W(self, Xs, ys, rng, D):
        X = np.concatenate(Xs)
        y = np.concatenate(ys).astype(float)
        ok = y >= 0
        g0 = np.ones((ok.sum(), 1))
        w = self._mstep_weights_single(X[ok], y[ok])
        W = np.tile(w, (self.K, 1)) + rng.normal(0, 0.5, (self.K, D))
        W[:, 0] *= np.linspace(1.5, 0.3, self.K)        # break symmetry on stimulus weight
        return W

    def _mstep_weights_single(self, X, y):
        def f(w):
            eta = X @ w
            return (np.logaddexp(0, eta) - y * eta).sum() + 0.5 * w @ w / self.prior_sigma ** 2, \
                X.T @ (expit(eta) - y) + w / self.prior_sigma ** 2
        return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x

    def fit(self, sessions):
        Xs = [s.design() for s in sessions]
        ys = [s.choice for s in sessions]
        rng = np.random.default_rng(self.seed)
        best = None
        for r in range(self.n_restarts):
            res = self._fit_once(Xs, ys, rng)
            if best is None or res["ll"] > best["ll"]:
                best = res
        order = np.argsort(-np.abs(best["W"][:, 0]))     # state 0 = engaged (largest |stimulus weight|)
        self.W, self.A = best["W"][order], best["A"][np.ix_(order, order)]
        self.pi, self.ll, self.n_iter = best["pi"][order], best["ll"], best["n_iter"]
        return self

    # ---------------- inference ----------------
    def posteriors(self, sess):
        """Returns dict(smoothed=(T,K), predicted=(T,K), loglik)."""
        logB = _emission_loglik(sess.design(), sess.choice, self.W)
        g, _, ll, la = _forward_backward(logB, np.log(self.A), np.log(self.pi))
        filt = np.exp(la - logsumexp(la, axis=1, keepdims=True))
        pred = np.vstack([self.pi, filt[:-1] @ self.A])
        # leave-out posterior: P(z_t | choices EXCEPT trials t and t+1). Trial t+1 is also dropped
        # because its GLM inputs (previous choice, win-stay/lose-switch) contain choice t. Use this
        # whenever trials are SORTED by state and then compared on anything correlated with that
        # trial's own choice (otherwise errors are preferentially labelled 'disengaged').
        logA = np.log(self.A)
        lb = self._backward(logB, logA)
        lo = la - logB
        lo[:-1] += logsumexp(logA[None, :, :] + lb[1:, None, :], axis=2)
        loo = np.exp(lo - logsumexp(lo, axis=1, keepdims=True))
        return dict(smoothed=g, predicted=pred, loo=loo, loglik=ll)

    @staticmethod
    def _backward(logB, logA):
        T, K = logB.shape
        lb = np.zeros((T, K))
        for t in range(T - 2, -1, -1):
            lb[t] = logsumexp(logA + (logB[t + 1] + lb[t + 1])[None, :], axis=1)
        return lb

    def p_engaged(self, sess, kind="smoothed"):
        return self.posteriors(sess)[kind][:, 0]

    def bits_per_trial(self, sessions):
        """Held-out predictive log-likelihood per trial (one-step-ahead), in bits."""
        tot, n = 0.0, 0
        for s in sessions:
            p = self.posteriors(s)["predicted"]
            pr = (p * expit(s.design() @ self.W.T)).sum(1)
            ok = s.choice >= 0
            y = s.choice[ok]
            pr = np.clip(pr[ok], 1e-9, 1 - 1e-9)
            tot += (y * np.log2(pr) + (1 - y) * np.log2(1 - pr)).sum()
            n += ok.sum()
        return tot / max(n, 1)
