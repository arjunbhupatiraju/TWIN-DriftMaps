"""PART 3 step 3a (behaviour only, fast): how many trials back do the mice's CHOICES actually depend on?

Neural result so far: outcome memory lasts 2-3 trials; choice memory only 1 trial, everywhere.
Before interpreting that, measure the behaviour in the SAME sessions and trials:
   choice(t) ~ stimulus + past choices (lags 1..6) + past outcomes (lags 1..6) [+ Bayes prior]
fitted per session by logistic regression; also win-stay / lose-switch rates.

    python3 part3/behaviour_kernel.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "next", ROOT / "driftaware" / "src", HERE):
    sys.path.insert(0, str(p))
from history_map import lags  # noqa: E402

K = 6


def logit_fit(X, y, l2=1e-3, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        g = X.T @ (y - p) - l2 * b
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-8:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


def session_kernel(s):
    import compare_methods as cm
    from gatekeeper.multiverse import _sides
    v = cm.mask_of(s)
    if v is None or v.sum() < 200:
        return None
    side = _sides(s)
    c = np.asarray(s.contrast, float)
    ch = np.where(s.choice >= 0, 2.0 * s.choice - 1, 0.0)                 # +1 right, -1 left, 0 miss
    fb = np.asarray(s.feedback, float)
    stim = np.column_stack([np.tanh(5 * c) / np.tanh(5), c])
    Cl = lags(ch, K)
    Wl = lags(ch * (fb > 0), K)                                           # rewarded choice (win) history
    prior = cm.bayes_prior(side) - 0.5
    y = (s.choice == 1).astype(float)
    out = {}
    for tag, D in (("no_prior", np.column_stack([np.ones(len(y)), stim, Cl, Wl])),
                   ("with_prior", np.column_stack([np.ones(len(y)), stim, Cl, Wl, prior]))):
        b, se = logit_fit(D[v], y[v])
        for k in range(K):
            out[f"{tag}_choice{k + 1}"] = b[3 + k]; out[f"{tag}_choice{k + 1}_z"] = b[3 + k] / se[3 + k]
            out[f"{tag}_win{k + 1}"] = b[3 + K + k]; out[f"{tag}_win{k + 1}_z"] = b[3 + K + k] / se[3 + K + k]
        if tag == "with_prior":
            out["prior"] = b[-1]; out["prior_z"] = b[-1] / se[-1]
    prev = np.r_[np.nan, s.choice[:-1].astype(float)]
    pfb = np.r_[np.nan, fb[:-1]]
    hard = v & (np.abs(c) <= 0.0625) & (prev >= 0)
    stay = (s.choice == prev)
    out["win_stay_hard"] = float(stay[hard & (pfb > 0)].mean()) if (hard & (pfb > 0)).sum() > 10 else np.nan
    out["lose_stay_hard"] = float(stay[hard & (pfb < 0)].mean()) if (hard & (pfb < 0)).sum() > 10 else np.nan
    return out


def main(cache="data/mv_cache", out_dir="results/behaviour_kernel"):
    import pandas as pd
    from gatekeeper.mv_data import MVSession
    rows = []
    for p in sorted(Path(cache).expanduser().glob("*.npz")):
        s = MVSession(p)
        try:
            r = session_kernel(s)
        except Exception as e:
            print("FAILED", p.stem, e); r = None
        if r:
            rows.append(dict(sid=s.sid, subject=s.subject, **r))
    df = pd.DataFrame(rows)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out_dir) / "kernel_sessions.csv", index=False)
    L = [f"# How far back do the mice's choices depend on the past? ({len(df)} sessions, {df.subject.nunique()} mice)\n",
         "Logistic regression per session: choice ~ stimulus + past choices (lags 1-6) + past rewarded choices (lags 1-6) "
         "[+ Bayes prior]. Positive weight = tendency to repeat. Fraction significant = |z| > 1.96.\n",
         "| Lag | Choice weight (no prior) | Frac sig | Choice weight (with prior) | Frac sig | Win weight (no prior) | Frac sig | Win weight (with prior) | Frac sig |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k in range(1, K + 1):
        cells = []
        for var in ("choice", "win"):
            for tag in ("no_prior", "with_prior"):
                w = df[f"{tag}_{var}{k}"]; z = df[f"{tag}_{var}{k}_z"]
                cells.append(f"{w.median():+.3f} | {(z.abs() > 1.96).mean():.2f}")
        L.append(f"| {k} | " + " | ".join([cells[0], cells[1], cells[2], cells[3]]) + " |")
    L += [f"\nPrior weight (with prior): median {df.prior.median():+.2f}, fraction significant {(df.prior_z.abs() > 1.96).mean():.2f}.",
          f"Hard trials (|contrast| <= 6.25%): win-stay {df.win_stay_hard.median():.2f}, lose-stay {df.lose_stay_hard.median():.2f}."]
    text = "\n".join(L)
    (Path(out_dir) / "BEHAVIOUR_KERNEL.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
