"""Task specifications: how a session's trial labels are generated.

A TaskSpec draws whole pseudo-sessions from the task's generative process. Pseudo-sessions are the correct null for
slow or block-structured labels: they keep the label's structure but break its alignment with the recorded neurons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class BlockTask:
    """Blocked two-alternative task (IBL-like by default).

    probs:       P(stimulus on side 1) in each block type; blocks alternate between them
    mean_len, min_len, max_len: block lengths ~ truncated exponential (IBL: 60, 20, 100)
    n_unbiased:  opening trials at P = 0.5 (IBL: 90)
    contrasts:   stimulus magnitudes drawn uniformly
    """
    probs: tuple = (0.2, 0.8)
    mean_len: float = 60.0
    min_len: int = 20
    max_len: int = 100
    n_unbiased: int = 90
    contrasts: tuple = (0.0, 0.0625, 0.125, 0.25, 1.0)
    hazard: Optional[float] = None

    def sample(self, T, rng):
        p = np.full(T, 0.5)
        t = self.n_unbiased
        cur = rng.integers(len(self.probs))
        while t < T:
            L = int(np.clip(rng.exponential(self.mean_len), self.min_len, self.max_len))
            p[t:t + L] = self.probs[cur]
            cur = (cur + 1) % len(self.probs)
            t += L
        side = (rng.random(T) < p).astype(float)
        mag = rng.choice(self.contrasts, T)
        return dict(p_side=p, side=side, contrast=np.where(side == 1, mag, -mag),
                    prior=self.bayes_prior(side))

    def bayes_prior(self, side):
        """P(next stimulus on side 1 | past sides): forward filter over block identity."""
        hz = self.hazard or 1.0 / self.mean_len
        lo, hi = min(self.probs), max(self.probs)
        T = len(side); out = np.full(T, 0.5); b = 0.5
        for t in range(T):
            if t >= self.n_unbiased:
                out[t] = b * hi + (1 - b) * lo
                s = side[t]
                if np.isfinite(s):
                    lr, ll = (hi, lo) if s == 1 else (1 - hi, 1 - lo)
                    b = b * lr / (b * lr + (1 - b) * ll)
                b = b * (1 - hz) + (1 - b) * hz
        return out


@dataclass
class ChoiceModel:
    """Logistic choice model so pseudo-sessions get realistic, history-dependent choices.
    weights: intercept, stimulus (signed), prior (centred), choice lags 1..k, rewarded-choice lags 1..k."""
    w: np.ndarray = field(default_factory=lambda: np.r_[0.0, 4.0, 2.5, 0.1, 0.05, 0.03, 0.02, 0.47, 0.37, 0.25, 0.26])
    k: int = 4

    @classmethod
    def fit(cls, contrast, choice, feedback, prior, k=4, lam=1.0):
        """choice in {+1,-1} (0 = no choice), feedback > 0 rewarded."""
        c = np.asarray(choice, float); rc = c * (np.asarray(feedback) > 0)
        X = np.column_stack([np.ones(len(c)), np.tanh(5 * np.asarray(contrast, float)), np.asarray(prior) - 0.5,
                             _lags(c, k), _lags(rc, k)])
        ok = c != 0; ok[:k] = False
        y = (c[ok] > 0).astype(float); Xo = X[ok]
        w = np.zeros(X.shape[1]); P = lam * np.eye(len(w)); P[0, 0] = 0
        for _ in range(60):
            p = 1 / (1 + np.exp(-np.clip(Xo @ w, -30, 30)))
            step = np.linalg.solve((Xo * (p * (1 - p))[:, None]).T @ Xo + P, Xo.T @ (y - p) - P @ w)
            w += step
            if np.abs(step).max() < 1e-8:
                break
        return cls(w=w, k=k)

    def simulate(self, d, rng):
        T = len(d["side"]); k = self.k
        c = np.zeros(T); rc = np.zeros(T)
        for t in range(T):
            hc = [c[t - j] if t - j >= 0 else 0.0 for j in range(1, k + 1)]
            hr = [rc[t - j] if t - j >= 0 else 0.0 for j in range(1, k + 1)]
            x = np.r_[1.0, np.tanh(5 * d["contrast"][t]), d["prior"][t] - 0.5, hc, hr]
            c[t] = 1.0 if rng.random() < 1 / (1 + np.exp(-np.clip(x @ self.w, -30, 30))) else -1.0
            correct = (c[t] > 0) == (d["side"][t] > 0) if d["contrast"][t] != 0 else rng.random() < 0.5
            rc[t] = c[t] * correct
        d = dict(d); d.update(choice=c, rewarded_choice=rc, feedback=np.where(rc != 0, 1.0, -1.0))
        return d


def _lags(x, k):
    T = len(x); M = np.zeros((T, k))
    for j in range(1, k + 1):
        M[j:, j - 1] = x[:-j]
    return M


# ---------------------------------------------------------------- standard labels
LABELS: dict[str, Callable] = {
    "prior": lambda d: d["prior"],
    "block": lambda d: d["p_side"],
    "side": lambda d: d["side"],
    "contrast": lambda d: d["contrast"],
    "choice": lambda d: d["choice"],
    "rewarded_choice_lag1": lambda d: _lags(d["rewarded_choice"], 1)[:, 0],
    "choice_lag1": lambda d: _lags(d["choice"], 1)[:, 0],
}


class PseudoSessions:
    """Generator of null labels: the same label function applied to sessions drawn from the task."""

    def __init__(self, task, label, choice_model: Optional[ChoiceModel] = None):
        self.task = task
        self.label = LABELS[label] if isinstance(label, str) else label
        self.choice_model = choice_model

    def draw(self, T, rng):
        d = self.task.sample(T, rng)
        if self.choice_model is not None:
            d = self.choice_model.simulate(d, rng)
        return np.asarray(self.label(d), float)

    def matrix(self, T, n, rng):
        return np.column_stack([self.draw(T, rng) for _ in range(n)])


# ---------------------------------------------------------------- other task families
class _QAgent:
    """Simple Q-learning agent with a choice-perseveration term, used to generate realistic choices in reward-learning
    tasks. Its internal variables (q_diff, choice kernel) are also the typical labels people decode."""

    def __init__(self, alpha=0.3, beta=4.0, persev=0.5, decay=0.0):
        self.alpha, self.beta, self.persev, self.decay = alpha, beta, persev, decay

    def run(self, p_reward, rng, baiting=False):
        T = len(p_reward)
        q = np.zeros(2); ck = 0.0; bait = np.zeros(2, bool)
        choice = np.zeros(T); reward = np.zeros(T); qd = np.zeros(T); qs = np.zeros(T)
        for t in range(T):
            qd[t] = q[1] - q[0]; qs[t] = q.sum()
            pr = 1 / (1 + np.exp(-(self.beta * qd[t] + self.persev * ck)))
            c = int(rng.random() < pr)
            if baiting:
                bait |= rng.random(2) < p_reward[t]
                r = float(bait[c]); bait[c] = False
            else:
                r = float(rng.random() < p_reward[t, c])
            q[c] += self.alpha * (r - q[c]); q[1 - c] *= (1 - self.decay)
            ck = 0.7 * ck + 0.3 * (2 * c - 1)
            choice[t] = 2 * c - 1; reward[t] = r
        return choice, reward, qd, qs


@dataclass
class ReversalTask:
    """Two-option probabilistic reversal learning: reward probabilities (p_hi, p_lo) swap between options at block
    ends (block lengths uniform in [min_len, max_len])."""
    p_hi: float = 0.8
    p_lo: float = 0.2
    min_len: int = 30
    max_len: int = 60
    agent: _QAgent = field(default_factory=_QAgent)

    def sample(self, T, rng):
        P = np.zeros((T, 2)); t = 0; good = int(rng.integers(2))
        while t < T:
            L = int(rng.integers(self.min_len, self.max_len + 1))
            P[t:t + L, good] = self.p_hi; P[t:t + L, 1 - good] = self.p_lo
            good = 1 - good; t += L
        choice, reward, qd, qs = self.agent.run(P, rng)
        rc = choice * reward
        return dict(block=P[:, 1] - P[:, 0], choice=choice, reward=reward, rewarded_choice=rc, q_diff=qd, q_sum=qs,
                    side=(choice > 0).astype(float), contrast=np.zeros(T), prior=0.5 + 0.5 * (P[:, 1] - P[:, 0]))


@dataclass
class ForagingTask:
    """Dynamic foraging (two ports, baited by default): block-wise reward probability pairs drawn from `pairs`,
    block lengths uniform in [min_len, max_len]. Labels include q_diff, q_sum (total value) and local reward rate."""
    pairs: tuple = ((0.45, 0.05), (0.05, 0.45), (0.4, 0.1), (0.1, 0.4), (0.25, 0.25))
    min_len: int = 40
    max_len: int = 80
    baiting: bool = True
    agent: _QAgent = field(default_factory=lambda: _QAgent(alpha=0.4, beta=5.0, persev=0.3, decay=0.2))

    def sample(self, T, rng):
        P = np.zeros((T, 2)); t = 0; last = -1
        while t < T:
            k = int(rng.integers(len(self.pairs)))
            if k == last:
                continue
            L = int(rng.integers(self.min_len, self.max_len + 1))
            P[t:t + L] = self.pairs[k]; last = k; t += L
        choice, reward, qd, qs = self.agent.run(P, rng, baiting=self.baiting)
        rate = np.convolve(reward, np.ones(10) / 10)[:T]
        rate = np.r_[0.0, rate[:-1]]
        return dict(block=P[:, 1] - P[:, 0], choice=choice, reward=reward, rewarded_choice=choice * reward, q_diff=qd,
                    q_sum=qs, reward_rate=rate, side=(choice > 0).astype(float), contrast=np.zeros(T),
                    prior=0.5 + 0.5 * (P[:, 1] - P[:, 0]))


LABELS.update({
    "q_diff": lambda d: d["q_diff"],
    "q_sum": lambda d: d["q_sum"],
    "reward_rate": lambda d: d["reward_rate"],
    "reward": lambda d: d["reward"],
    "reward_lag1": lambda d: _lags(d["reward"], 1)[:, 0],
})
