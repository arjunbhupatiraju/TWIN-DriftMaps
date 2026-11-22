"""TWIN-Net: instant reliability predictions.

Given a recording (or a planned design) and the label you want to test, TWIN-Net predicts in milliseconds what the
simulation engine would take minutes to estimate:
  fp_shuffle, fp_pseudo, fp_shift    false-positive rate of the trial shuffle, pseudo-session and linear-shift tests
  pow_pseudo, pow_shift              power of the valid tests for a code of per-neuron correlation r in 20% of neurons

It is trained on simulated recordings spanning drift strength, timescales, sharing, neuron heterogeneity, trial and
neuron counts, and label types (block prior, latent state, learning curve, autocorrelated value, fast label).
Inputs are summaries a user can compute from their own data (see `features`).

  python3 -m twin.net make  OUT.npz  N_EXAMPLES [seed]     # simulate training examples (parallel-safe by seed)
  python3 -m twin.net train OUT_MODEL.pkl DATA1.npz [DATA2.npz ...]
"""
from __future__ import annotations

import pickle, sys, time
from pathlib import Path

import numpy as np

from .drift import DriftModel, TAUS
from .stats import pooled_corr
from .check import plant
from .tests import _p, p_linear_shift

LAGS = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89])
TARGETS = ("fp_shuffle", "fp_pseudo", "fp_shift", "pow_pseudo", "pow_shift")


# ------------------------------------------------------------------ features
def _acf(x, K):
    x = np.asarray(x, float)
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    T = len(x)
    F = np.fft.rfft(x, n=2 * T, axis=0)
    ac = np.fft.irfft(F * np.conj(F), axis=0)[: K + 1] / T
    return ac


def features(X, label_samples, r=0.1):
    """X: trials x neurons (a pilot or the real recording). label_samples: labels drawn from the task (or the observed
    label, if that is all you have), list of 1-D arrays. Returns one feature vector."""
    X = np.asarray(X, float); X = X[:, X.std(0) > 0]
    T, N = X.shape
    K = int(LAGS.max())
    ac = _acf(X, K)
    rho = ac[LAGS] / ac[0]                              # per neuron
    # mean pairwise cross-correlation at each lag, from the population sum (cheap)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    Fz = np.fft.rfft(Xz, n=2 * T, axis=0)
    auto = np.fft.irfft(Fz * np.conj(Fz), axis=0)[: K + 1] / T
    G = np.fft.rfft(Xz.sum(1), n=2 * T)
    tot = np.fft.irfft(G * np.conj(G))[: K + 1] / T
    cross = (tot - auto.sum(1)) / max(N * (N - 1), 1)
    het = rho[:5].mean(0)                              # each neuron's short-lag autocorrelation
    lab = np.mean([(_acf(np.asarray(y, float)[:, None], K)[LAGS, 0] / max(_acf(np.asarray(y, float)[:, None], K)[0, 0], 1e-12))
                   for y in label_samples], axis=0)
    return np.r_[rho.mean(1), np.median(rho, 1), cross[0], cross[LAGS], het.std(), np.percentile(het, [10, 90]),
                 np.log(T), np.log(N), lab, r]


# ------------------------------------------------------------------ training data
def _labels():
    from .task import BlockTask, PseudoSessions
    from .bench import Generator, latent_state, learning
    rng0 = np.random.default_rng()

    def ar_label(tau):
        def f(T, rng):
            from .drift import _ar
            return _ar(rng.standard_normal((T, 1)), np.exp(-1 / tau))[:, 0]
        return f
    return {
        "block prior": lambda rng: PseudoSessions(BlockTask(), "prior"),
        "latent state": lambda rng: Generator(lambda T, r, d=float(np.exp(rng.uniform(np.log(5), np.log(200)))): latent_state(T, r, int(d))),
        "learning curve": lambda rng: Generator(learning),
        "autocorrelated value": lambda rng: Generator(ar_label(float(np.exp(rng.uniform(np.log(2), np.log(100)))))),
        "fast label": lambda rng: Generator(lambda T, r: r.standard_normal(T)),
    }


def random_drift(rng):
    J = len(TAUS)
    on = rng.random(J) < 0.5
    if not on.any():
        on[rng.integers(J)] = True
    a = np.where(on, rng.dirichlet(np.ones(J)), 0.0)
    a = a / a.sum() * rng.uniform(0.0, 0.8)
    sh = np.where(on, rng.uniform(0, 0.9, J), 0.0)
    sd = rng.uniform(0, 0.7)
    het = np.quantile(np.exp(rng.normal(-sd * sd / 2, sd, 4000)), np.linspace(0.05, 0.95, 19)).tolist() if sd > 0.05 else None
    return DriftModel(phis=list(np.exp(-1 / np.array(TAUS))), a=a.tolist(), sh=sh.tolist(),
                      inst_corr=float(rng.uniform(0, 0.15)), slow_frac=float(a.sum()), het=het)


def example(rng, sims=32, n_null=80):
    T = int(np.exp(rng.uniform(np.log(200), np.log(1500))))
    N = int(np.exp(rng.uniform(np.log(5), np.log(150))))
    dm = random_drift(rng)
    kinds = _labels(); kind = list(kinds)[rng.integers(len(kinds))]
    gen = kinds[kind](rng)
    r = float(rng.uniform(0.05, 0.3))
    if rng.random() < 0.25:                                           # some recordings also trend over the session
        trend = rng.uniform(0.2, 1.0)
    else:
        trend = 0.0

    def rec():
        X = dm.simulate(T, N, rng)
        if trend:
            X = X + np.linspace(-1, 1, T)[:, None] * trend * (rng.standard_normal(N) * 0.6 + 0.3)
        return X
    hits = np.zeros(5)
    for _ in range(sims):
        X = rec(); y = gen.draw(T, rng)
        v = pooled_corr(X, np.column_stack([y] + [rng.permutation(y) for _ in range(n_null)] + [gen.matrix(T, n_null, rng)]))
        hits[0] += _p(v[0], v[1:n_null + 1]) < 0.05
        hits[1] += _p(v[0], v[n_null + 1:]) < 0.05
        hits[2] += p_linear_shift(X, y) < 0.05
        Xp = plant(rec(), y2 := gen.draw(T, rng), r, 0.2, rng)
        w = pooled_corr(Xp, np.column_stack([y2, gen.matrix(T, n_null, rng)]))
        hits[3] += _p(w[0], w[1:]) < 0.05
        hits[4] += p_linear_shift(Xp, y2) < 0.05
    x = features(rec(), [gen.draw(T, rng) for _ in range(5)], r)
    return x, hits / sims, dict(T=T, N=N, label=kind, trend=trend, r=r)


def cmd_make(out, n, seed=0):
    rng = np.random.default_rng(seed)
    Xs, Ys, M = [], [], []
    t0 = time.time()
    for i in range(n):
        x, y, m = example(rng)
        Xs.append(x); Ys.append(y); M.append(m)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{n} examples ({time.time() - t0:.0f}s)", flush=True)
            np.savez(out, X=np.array(Xs), Y=np.array(Ys), meta=np.array([str(m) for m in M]))
    np.savez(out, X=np.array(Xs), Y=np.array(Ys), meta=np.array([str(m) for m in M]))


# ------------------------------------------------------------------ model
class TwinNet:
    def __init__(self, models, scaler):
        self.models, self.scaler = models, scaler

    @classmethod
    def train(cls, X, Y, seed=0):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X)
        Z = sc.transform(X)
        logit = np.log((np.clip(Y, 0.01, 0.99)) / (1 - np.clip(Y, 0.01, 0.99)))
        models = [MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=3000, early_stopping=True,
                               random_state=seed + k).fit(Z, logit) for k in range(5)]   # small ensemble
        return cls(models, sc)

    def predict_raw(self, F):
        Z = self.scaler.transform(np.atleast_2d(F))
        L = np.mean([m.predict(Z) for m in self.models], axis=0)
        return 1 / (1 + np.exp(-L))

    def predict(self, X, label_samples, r=0.1):
        return dict(zip(TARGETS, map(float, self.predict_raw(features(X, label_samples, r))[0])))

    def save(self, path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path):
        return pickle.loads(Path(path).read_bytes())

    @classmethod
    def default(cls):
        """The shipped model; retrained from the shipped training set if this machine's scikit-learn cannot read it."""
        d = Path(__file__).resolve().parent / "models"
        try:
            return cls.load(d / "twinnet_v0.pkl")
        except Exception:
            z = np.load(d / "train_v0.npz")
            net = cls.train(z["X"], z["Y"])
            try:
                net.save(d / "twinnet_v0.pkl")
            except Exception:
                pass
            return net


def cmd_train(out, files):
    X = np.concatenate([np.load(f)["X"] for f in files]); Y = np.concatenate([np.load(f)["Y"] for f in files])
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); k = len(X) // 5
    te, tr = idx[:k], idx[k:]
    net = TwinNet.train(X[tr], Y[tr])
    P = net.predict_raw(X[te])
    print(f"held-out examples: {len(te)} (trained on {len(tr)})")
    for j, t in enumerate(TARGETS):
        r = np.corrcoef(P[:, j], Y[te, j])[0, 1]
        print(f"  {t:11s} r = {r:.3f}   MAE = {np.abs(P[:, j] - Y[te, j]).mean():.3f}")
    TwinNet.train(X, Y).save(out)
    print("saved", out)


if __name__ == "__main__":
    if sys.argv[1] == "make":
        cmd_make(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    else:
        cmd_train(sys.argv[2], sys.argv[3:])
