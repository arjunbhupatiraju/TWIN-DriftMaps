"""PART 3 step 3a-2: the behavioural choice kernel with the mouse's SLOWLY DRIFTING BIAS modelled explicitly.

Why: a slowly drifting bias (no memory at all) makes past choices predict the current one at EVERY lag, so a
plain logistic regression shows a long, flat "history kernel". Our neural result (choice memory lasts 1 trial)
is only a dissociation if the behavioural kernel at lags 2-6 is real memory, not drift.

Model (per session, like PsyTrack, Roy et al. 2021):
    logit P(right_t) = w . x_t + b_t,     b_t = b_{t-1} + noise(sigma)
    x_t = [stimulus terms, past choices lags 1..6, past rewarded choices lags 1..6]
MAP fit of w and the whole bias path b; sigma chosen by held-out likelihood (every 5th trial left out).

    python3 part3/behaviour_rw.py sim     (checks: pure drift -> kernel ~0; real memory -> kernel kept)
    python3 part3/behaviour_rw.py         (all sessions)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solveh_banded

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "next", HERE):
    sys.path.insert(0, str(p))
from history_map import lags  # noqa: E402

K = 6
SIGMAS = (0.01, 0.03, 0.1, 0.3)


def fit_rw(X, y, use, sigma, iters=40, l2=1e-2):
    """MAP of w (p) and random-walk bias b (T). use: mask of trials whose likelihood counts."""
    T, p = X.shape
    w = np.zeros(p); b = np.zeros(T)
    lam = 1.0 / sigma ** 2
    for _ in range(iters):
        eta = X @ w + b
        mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        r = np.where(use, y - mu, 0.0)
        Wd = np.where(use, mu * (1 - mu), 0.0) + 1e-9
        # gradients
        db_prior = np.zeros(T); db_prior[1:] += lam * (b[1:] - b[:-1]); db_prior[:-1] -= lam * (b[1:] - b[:-1])
        gb = r - db_prior - 1e-3 * b
        gw = X.T @ r - l2 * w
        # Hessian blocks: A (banded, T), B = X^T W (p x T), C = X^T W X
        main = Wd + 1e-3 + lam * np.r_[1, 2 * np.ones(T - 2), 1]
        off = -lam * np.ones(T - 1)
        ab = np.vstack([np.r_[0, off], main])                       # upper banded form
        Bt = (X * Wd[:, None])                                       # (T, p) = (X^T W)^T
        C = X.T @ Bt + l2 * np.eye(p)
        AinvBt = solveh_banded(ab, Bt)                              # (T, p)
        Ainvgb = solveh_banded(ab, gb)
        S = C - Bt.T @ AinvBt
        dw = np.linalg.solve(S, gw - Bt.T @ Ainvgb)
        db = Ainvgb - AinvBt @ dw
        w += dw; b += db
        if max(np.abs(dw).max(), np.abs(db).max()) < 1e-6:
            break
    # approximate standard errors of w from the Schur complement
    se = np.sqrt(np.clip(np.diag(np.linalg.inv(S)), 1e-12, None))
    return w, b, se


def heldout_ll(X, y, v, sigma):
    T = len(y)
    test = v & (np.arange(T) % 5 == 0)
    w, b, _ = fit_rw(X, y, v & ~test, sigma)
    mu = 1 / (1 + np.exp(-np.clip(X @ w + b, -30, 30)))
    return float(np.sum(y[test] * np.log(mu[test] + 1e-12) + (1 - y[test]) * np.log(1 - mu[test] + 1e-12)))


def design(contrast, choice, feedback):
    c = np.asarray(contrast, float)
    ch = np.where(np.asarray(choice) >= 0, 2.0 * np.asarray(choice) - 1, 0.0)
    fb = np.asarray(feedback, float)
    return np.column_stack([np.tanh(5 * c) / np.tanh(5), c, lags(ch, K), lags(ch * (fb > 0), K)])


def session_fit(contrast, choice, feedback, v):
    X = design(contrast, choice, feedback)
    y = (np.asarray(choice) == 1).astype(float)
    lls = [heldout_ll(X, y, v, s) for s in SIGMAS]
    sig = SIGMAS[int(np.argmax(lls))]
    w, b, se = fit_rw(X, y, v, sig)
    # also the no-drift fit (sigma tiny) for comparison
    w0, _, _ = fit_rw(X, y, v, 1e-4)
    out = dict(sigma=sig, bias_sd=float(np.std(b[v])))
    for k in range(K):
        out[f"rw_choice{k + 1}"] = w[2 + k]; out[f"rw_choice{k + 1}_z"] = w[2 + k] / se[2 + k]
        out[f"rw_win{k + 1}"] = w[2 + K + k]
        out[f"fixed_choice{k + 1}"] = w0[2 + k]
    return out


def sim(n=40):
    rng0 = np.random.default_rng(0)
    for mode in ("drift", "memory", "both"):
        rows = []
        for i in range(n):
            rng = np.random.default_rng(i)
            T = 700
            c = rng.choice([-1, -.25, -.125, -.0625, 0, .0625, .125, .25, 1], T)
            ch = np.zeros(T, int); drift = 0.0; mem = 0.0
            for t in range(T):
                drift = 0.99 * drift + 0.15 * rng.standard_normal()
                x = 4 * c[t] + (1.5 * drift if mode in ("drift", "both") else 0) + (mem if mode in ("memory", "both") else 0)
                ch[t] = rng.random() < 1 / (1 + np.exp(-x))
                mem = 0.75 * mem + 0.4 * (2 * ch[t] - 1)
            fb = np.where((ch == 1) == (c > 0), 1.0, -1.0)
            v = np.ones(T, bool); v[:10] = False
            rows.append(session_fit(c, ch, fb, v))
        f = lambda key: [round(float(np.median([r[f"{key}{k}"] for r in rows])), 3) for k in range(1, K + 1)]
        print(f"{mode:7s} fixed-bias kernel {f('fixed_choice')}   drifting-bias kernel {f('rw_choice')}", flush=True)
    print("true memory kernel (logit units): ", [round(0.4 * 0.75 ** (k - 1), 3) for k in range(1, K + 1)])


def main(cache="data/mv_cache", out_dir="results/behaviour_kernel"):
    import pandas as pd
    from scipy.stats import wilcoxon
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    rows = []
    for p in sorted(Path(cache).expanduser().glob("*.npz")):
        s = MVSession(p)
        v = cm.mask_of(s)
        if v is None or v.sum() < 200:
            continue
        try:
            rows.append(dict(sid=s.sid, subject=s.subject, **session_fit(s.contrast, s.choice, s.feedback, v)))
        except Exception as e:
            print("FAILED", p.stem, e)
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "kernel_rw_sessions.csv", index=False)
    L = [f"# Choice kernel with a drifting bias modelled ({len(df)} sessions, {df.subject.nunique()} mice)\n",
         f"Chosen bias drift per trial (sigma): " + ", ".join(f"{s}: {(df.sigma == s).mean():.2f}" for s in SIGMAS)
         + f". Median SD of the fitted bias path: {df.bias_sd.median():.2f}.\n",
         "| Lag | Fixed bias (plain) | Drifting bias modelled | Fraction > 0 | Wilcoxon p | Frac |z|>1.96 | Win weight (drifting bias) |",
         "|---|---|---|---|---|---|---|"]
    for k in range(1, K + 1):
        w = df[f"rw_choice{k}"]
        L.append(f"| {k} | {df[f'fixed_choice{k}'].median():+.3f} | {w.median():+.3f} | {(w > 0).mean():.2f} | "
                 f"{wilcoxon(w).pvalue:.1e} | {(df[f'rw_choice{k}_z'].abs() > 1.96).mean():.2f} | {df[f'rw_win{k}'].median():+.3f} |")
    text = "\n".join(L)
    (Path(out_dir) / "BEHAVIOUR_KERNEL_RW.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sim":
        sim()
    else:
        main()
