"""Lapse forecasting: does pre-stimulus activity predict errors on EASY trials?

Primary target (pre-registered): a lapse h = 5 trials AHEAD (a forecast); h = 0 (same trial)
is reported as secondary.

Lapse = wrong choice (or no-go) on a trial with |contrast| >= 0.25, where engaged mice are
nearly always correct. Features at trial t use only information available before the
stimulus of trial t:
  behavior baseline : GLM-HMM predicted P(engaged_t | choices < t) (causal), error rate over
                      the previous 10 trials, previous-trial correct
  region r          : pre-stimulus activity of trial t projected on a lapse axis that is
                      learned by within-session cross-fitting (units differ between
                      sessions, so the axis is fit on one half of the session and applied
                      to the other half, never to the trials it was fit on).
Evaluation: leave-one-session-out logistic regression, AUC and log-loss of
(behavior + region) vs (behavior) and vs (behavior + other regions).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


def lapse_labels(sess, min_contrast=0.25):
    easy = np.abs(sess.contrast) >= min_contrast
    lapse = (sess.choice < 0) | (sess.correct == 0)
    return easy, lapse.astype(int)


def behavior_features(sess, p_pred):
    corr = sess.correct.astype(float)
    err = np.where(corr == 0, 1.0, 0.0)
    err[corr < 0] = np.nan
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # empty windows -> NaN -> 0 below
        roll = np.array([np.nanmean(err[max(0, t - 10):t]) if t > 0 else 0.0 for t in range(sess.T)])
    roll = np.nan_to_num(roll)
    prev = np.r_[0.0, np.where(corr[:-1] == 1, 1.0, 0.0)]
    return np.c_[p_pred, roll, prev]


def crossfit_axis(X, y, mask):
    """1-D lapse axis per trial, fit on the other parity of trials (no trial used for its own score)."""
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    out = np.zeros(len(Z))
    par = (np.arange(len(Z)) >= len(Z) // 2).astype(int)   # contiguous halves: avoids slow-drift leakage
    for k in (0, 1):
        tr = mask & (par != k)
        if y[tr].sum() < 3 or (1 - y[tr]).sum() < 3:
            continue
        w = Z[tr & (y == 1)].mean(0) - Z[tr & (y == 0)].mean(0)
        out[par == k] = Z[par == k] @ (w / (np.linalg.norm(w) + 1e-9))
    return out


def build_table(sessions, model, regions, horizon=0):
    """horizon h: features from trial t (pre-stimulus) predict a lapse on easy trial t + h."""
    rows = []
    for s in sessions:
        p_pred = model.posteriors(s)["predicted"][:, 0]
        easy, y = lapse_labels(s)
        B = behavior_features(s, p_pred)
        X = {r: s.signals[r] for r in regions if r in s.signals}
        if horizon:
            easy, y = easy[horizon:], y[horizon:]
            B = B[:-horizon]
            X = {r: v[:-horizon] for r, v in X.items()}
        R = {r: crossfit_axis(v, y, easy) for r, v in X.items()}
        rows.append(dict(sid=s.sid, easy=easy, y=y, B=B, R=R))
    return rows


def evaluate(rows, regions, feature_sets=None, C=1.0):
    """Leave-one-session-out predictions; returns metric per feature set."""
    if feature_sets is None:
        feature_sets = {"behavior": []}
        for r in regions:
            feature_sets[f"behavior+{r}"] = [r]
        feature_sets["behavior+all"] = list(regions)
        for r in regions:
            feature_sets[f"behavior+all-{r}"] = [q for q in regions if q != r]
    rows = [r for r in rows if all(q in r["R"] for q in regions)]

    def feats(row, regs):
        F = [row["B"]] + [row["R"][q][:, None] for q in regs]
        return np.hstack(F)[row["easy"]], row["y"][row["easy"]]

    res = {}
    for name, regs in feature_sets.items():
        ys, ps = [], []
        for i in range(len(rows)):
            Xtr = np.vstack([feats(rows[j], regs)[0] for j in range(len(rows)) if j != i])
            ytr = np.concatenate([feats(rows[j], regs)[1] for j in range(len(rows)) if j != i])
            Xte, yte = feats(rows[i], regs)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(C=C, max_iter=1000).fit((Xtr - mu) / sd, ytr)
            ps.append(clf.predict_proba((Xte - mu) / sd)[:, 1])
            ys.append(yte)
        y, p = np.concatenate(ys), np.concatenate(ps)
        res[name] = dict(auc=float(roc_auc_score(y, p)), log_loss=float(log_loss(y, p)), n=int(len(y)),
                         per_session=[(yy, pp) for yy, pp in zip(ys, ps)])
    return res


def session_bootstrap_delta(res, a, b, metric="auc", n_boot=2000, seed=0):
    """95% CI of metric[a] - metric[b], resampling sessions."""
    rng = np.random.default_rng(seed)
    A, Bp = res[a]["per_session"], res[b]["per_session"]
    n = len(A)

    def m(pairs, idx):
        y = np.concatenate([pairs[i][0] for i in idx])
        p = np.concatenate([pairs[i][1] for i in idx])
        if y.min() == y.max():
            return np.nan
        return roc_auc_score(y, p) if metric == "auc" else -log_loss(y, p, labels=[0, 1])

    d0 = m(A, range(n)) - m(Bp, range(n))
    ds = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ds.append(m(A, idx) - m(Bp, idx))
    ds = np.array(ds)
    ds = ds[np.isfinite(ds)]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    p_pos = (1 + np.sum(ds <= 0)) / (1 + len(ds))      # one-sided bootstrap p for a positive gain
    return dict(est=float(d0), lo=float(lo), hi=float(hi), p_pos=float(p_pos))
