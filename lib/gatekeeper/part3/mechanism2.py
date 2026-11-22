"""PART 3 step 4b: is the brain-wide slow bias signal (M1 pooled: real 17.6% vs fake 4.3%, 420 sessions) REAL MEMORY,
not movement, and does it replicate and localize?

Checks (all with the same built-in controls: fake target ~0.05, planted half-strength target detected):
  A  TARGET   the mouse's wandering bias b_t  vs  a reward-integrated value V_t (does the brain-wide signal carry the
              REWARD memory, or only the bias?)
  B  MOVEMENT signed wheel position change and signed wheel velocity in the same pre-stimulus window, plus licks,
              unsigned wheel speed, pupil and time since feedback, removed from activity and from every target
  C  WHERE    the same pooled test within each brain division (Cosmos), Fisher across sessions, BH across divisions
  D  STEINMETZ the same pooled test with face motion, signed wheel, licks and pupil removed

    python3 part3/mechanism2.py fetch --limit 3     (signed wheel from IBL; needs internet; resumable)
    python3 part3/mechanism2.py fetch
    python3 part3/mechanism2.py ibl
    python3 part3/mechanism2.py steinmetz --data <folder with steinmetz_part*.npz>
"""
from __future__ import annotations

import sys
import time
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "next", ROOT / "driftaware" / "src", HERE):
    sys.path.insert(0, str(p))
import mechanism as m1  # noqa: E402

AUX2 = "data/part3_aux2"
AUX3 = "data/part3_aux3"
WIN = (-0.60, -0.10)


# ------------------------------------------------------------------------------------------- fetch signed wheel
def fetch(cache, aux_dir, limit):
    from one.api import ONE
    from gatekeeper.mv_data import MVSession
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international", silent=True)
    Path(aux_dir).mkdir(parents=True, exist_ok=True)
    paths = sorted(Path(cache).expanduser().glob("*.npz"))[: limit or None]
    t0 = time.time()
    for i, p in enumerate(paths):
        out = Path(aux_dir) / f"{p.stem}.npz"
        if out.exists():
            continue
        try:
            s = MVSession(p)
            tr = one.load_object(s.sid, "trials")
            on = np.asarray(tr["stimOn_times"], float)
            on = on[np.isfinite(on)]
            if len(on) != len(s.contrast):
                np.savez(out, ok=False); msg = "trial mismatch"
            else:
                wh = one.load_object(s.sid, "wheel", collection="alf")
                pos, ts = np.asarray(wh["position"], float), np.asarray(wh["timestamps"], float)
                p0 = np.interp(on + WIN[0], ts, pos); p1 = np.interp(on + WIN[1], ts, pos)
                vel = np.r_[0, np.diff(pos) / np.maximum(np.diff(ts), 1e-4)]
                lo = np.searchsorted(ts, on + WIN[0]); hi = np.searchsorted(ts, on + WIN[1])
                vmean = np.array([vel[a:b].mean() if b > a else 0.0 for a, b in zip(lo, hi)])
                # the wheel's absolute position drifts over a session; use its change and its recent level
                plev = p0 - np.interp(on - 5.0, ts, pos)
                np.savez(out, ok=True, wheel_disp=p1 - p0, wheel_vel=vmean, wheel_level=plev); msg = "ok"
        except Exception as e:
            msg = f"FAILED ({type(e).__name__}: {e})"
        print(f"[{i + 1}/{len(paths)}] {p.stem}  {msg}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


def ibl_confounds(sid, T):
    from history_confounds import confounds
    H, names = confounds(AUX2, sid, T)
    cols = [] if H is None else [H]
    f = Path(AUX3) / f"{sid}.npz"
    if f.exists():
        z = np.load(f)
        if bool(z["ok"]):
            for k in ("wheel_disp", "wheel_vel", "wheel_level"):
                if len(z[k]) == T:
                    x = np.nan_to_num(z[k]); cols.append(x[:, None]); names = names + [k]
    return (np.hstack(cols) if cols else None), names


# ------------------------------------------------------------------------------------------- pooled test with controls
def pooled_checks(X, real_t, null_ts, D, own_real, null_owns, fake_t, fake_owns, c1, lam):
    out = {}
    out["p_real"], out["r_real"], out["rnull"] = m1._test_lam(X, np.column_stack([real_t] + null_ts), D, [own_real] + null_owns, lam)
    out["p_fake"], *_ = m1._test_lam(X, fake_t, D, fake_owns, lam)
    breal = m1.resid(real_t[:, None], D)[:, 0]; breal = breal / (breal.std() + 1e-12)
    sd = X.std(0); Z = (X - X.mean(0)) / sd
    beta = (c1 @ Z) / (c1 @ c1)
    Xp = X + 0.5 * np.outer(breal, beta * sd) * 0.5
    out["p_planted_half"], *_ = m1._test_lam(Xp, np.column_stack([real_t] + null_ts), D, [own_real] + null_owns, lam)
    # dataset-independent plants: the residual target added to every neuron with random weights of a FIXED size
    # (fraction of each neuron's SD). The same sizes are used in IBL and Steinmetz, so their power can be compared.
    rng = np.random.default_rng(X.shape[0] * 7919 + X.shape[1])
    wts = rng.standard_normal(X.shape[1])
    for amp in (0.01, 0.02, 0.04):
        Xp = X + amp * np.outer(breal, wts * sd)
        out[f"p_fixed{int(amp * 100):02d}"], *_ = m1._test_lam(Xp, np.column_stack([real_t] + null_ts), D, [own_real] + null_owns, lam)
    return out


def ibl_session(path, bank, seed=0, by_division=True):
    import compare_methods as cm
    from gatekeeper.mv_data import MVSession
    from history_map import lags
    import slowmode as sm
    s = MVSession(path)
    if s.sid not in bank:
        return []
    v = cm.mask_of(s)
    d = bank[s.sid]; T = d["T"]
    rng = np.random.default_rng(seed + 23 + zlib.crc32(s.sid.encode()) % 100000)
    others = [e for s2, e in bank.items() if e["subject"] != s.subject and e["T"] >= T]
    if len(others) < 31:
        return []
    idx = rng.choice(len(others), min(m1.N_NULL + 1, len(others)), replace=False)
    H, names = ibl_confounds(s.sid, len(s.contrast))
    D0 = d["D"][v]
    D = D0 if H is None else np.column_stack([D0, H[v]])
    own_real = d["D"][v]
    null_owns = [others[j]["D"][:T][v] for j in idx[1:]]
    fake_owns = [others[j]["D"][:T][v] for j in idx]
    c1 = lags(np.where(s.choice >= 0, s.choice - 0.5, 0.0), 1)[v, 0]; c1 = c1 - c1.mean()
    blocks = {}
    for r in s.regions:
        X = s.X(r, "prior_a")
        if X is None:
            continue
        X = X[v].astype(float); X = X[:, X.std(0) > 0]
        if X.shape[1]:
            blocks[r] = X
    if not blocks:
        return []
    groups = {"ALL": list(blocks)}
    if by_division:
        div = sm.division_of(sorted(blocks))
        for r in blocks:
            groups.setdefault(div.get(r, r), []).append(r)
    rows = []
    for g, regs in groups.items():
        X = np.hstack([blocks[r] for r in regs])
        if X.shape[1] < 10:
            continue
        lam = max(10.0, X.shape[1] / 5)
        row = dict(sid=s.sid, subject=s.subject, group=g, n_units=X.shape[1], confounds="+".join(names))
        for tname, key in (("bias", "b"), ("value", "V")):
            res = pooled_checks(X, d[key][v], [others[j][key][:T][v] for j in idx[1:]], D, own_real, null_owns,
                                np.column_stack([others[j][key][:T] for j in idx])[v], fake_owns, c1, lam)
            row.update({f"{tname}_{k}": val for k, val in res.items()})
        rows.append(row)
    return rows


def report(df, out_dir, title):
    from scipy.stats import chi2, binomtest
    from driftaware import bh
    fisher = lambda p: float(chi2.sf(-2 * np.log(np.clip(p, 1e-300, 1)).sum(), 2 * len(p))) if len(p) else np.nan
    allp = df[df.group == "ALL"]
    L = [f"# Brain-wide slow signal: memory or movement? {title} ({allp.sid.nunique()} sessions, {allp.subject.nunique()} mice)\n",
         "All neurons pooled per session. Removed from activity and every target: last 3 trials, current choice, "
         "block prior (IBL), and movement/arousal: " + ", ".join(sorted(set("+".join(allp.confounds.dropna()).split("+")) - {""})) + ".\n",
         "| Target | Condition | Sessions | Fraction p < 0.05 | Binomial p vs 0.05 | Fisher p |", "|---|---|---|---|---|---|"]
    for t in ("bias", "value"):
        for c in ("fake", "real", "planted_half", "fixed01", "fixed02", "fixed04"):
            col = f"{t}_p_{c}"
            k = int((allp[col] < .05).sum())
            L.append(f"| {t} | {c} | {len(allp)} | {k / len(allp):.3f} | {binomtest(k, len(allp), .05, 'greater').pvalue:.1e} | "
                     f"{fisher(allp[col].values):.1e} |")
    L.append(f"\nMedian decoding r (bias): real {allp.bias_r_real.median():+.3f}, null {allp.bias_rnull.median():+.3f}.")
    div = df[df.group != "ALL"]
    if len(div):
        L += ["\n## Where: pooled within each brain division (bias target)\n",
              "| Division | Sessions | Fraction p < 0.05 (real) | Fraction p < 0.05 (fake) | Fisher p (real) | FDR 5% |", "|---|---|---|---|---|---|"]
        rows = []
        for g, h in div.groupby("group"):
            if h.sid.nunique() < 5:
                continue
            rows.append((g, h.sid.nunique(), (h.bias_p_real < .05).mean(), (h.bias_p_fake < .05).mean(), fisher(h.bias_p_real.values)))
        ps = np.array([r[4] for r in rows])
        sigs = bh(ps, 0.05) if len(ps) else []
        for (g, n, fr, ff, fp), sg in sorted(zip(rows, sigs), key=lambda x: x[0][4]):
            L.append(f"| {g} | {n} | {fr:.3f} | {ff:.3f} | {fp:.1e} | {'yes' if sg else ''} |")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "MECHANISM2.md").write_text(text + "\n")
    df.to_csv(Path(out_dir) / "rows.csv", index=False)
    print(text)


def run_ibl(cache, out_dir, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    bank = m1.ibl_bank(cache)
    paths = [p for p in sorted(Path(cache).expanduser().glob("*.npz")) if p.stem in bank]
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(ibl_session)(p, bank) for p in paths)
    report(pd.DataFrame([r for rows in res for r in rows]), out_dir, "IBL")


# ------------------------------------------------------------------------------------------- Steinmetz
def stein_session(i, d, bank, seed=0):
    import steinmetz_replication as sr
    from history_map import lags
    mouse = str(d["mouse_name"])
    contrast, choice, fb, side = sr.task(d)
    lens = sorted([e["T"] for e in bank if e["subject"] != mouse], reverse=True)
    if len(lens) < 20:
        return []
    T = min(len(choice), lens[19])
    me = [e for e in bank if e["i"] == i][0]
    others = [e for e in bank if e["subject"] != mouse and e["T"] >= T]
    rng = np.random.default_rng(seed + 31 + i)
    idx = rng.choice(len(others), min(m1.N_NULL + 1, len(others)), replace=False)
    v = choice[:T] >= 0; v[:10] = False
    Hc = sr.confounds(d, len(choice))
    cols = [] if Hc is None else [Hc[:T]]
    if "wheel" in d:                                                    # signed wheel in the pre-stimulus window
        w = np.asarray(d["wheel"], float)
        w = w[0] if w.ndim == 3 else w
        if w.shape[0] >= T:
            cols.append(w[:T, sr.PRE[0]:sr.PRE[1]].sum(-1)[:, None])
    D0 = me["D"][:T][v]
    D = D0 if not cols else np.column_stack([D0, np.hstack(cols)[v]])
    areas = np.asarray(d["brain_area"]).astype(str)
    sel = ~np.isin(areas, list(sr.SKIP))
    X = d["spks"][sel][:, :T, sr.PRE[0]:sr.PRE[1]].sum(-1).T.astype(float)[v]
    X = X[:, X.std(0) > 0]
    if X.shape[1] < 10:
        return []
    c1 = lags(np.where(choice >= 0, choice - 0.5, 0.0), 1)[:T][v, 0]; c1 = c1 - c1.mean()
    row = dict(sid=f"s{i}", subject=mouse, group="ALL", n_units=X.shape[1],
               confounds="face+wheel+licks+pupil+signed_wheel" if cols else "")
    for tname, key in (("bias", "b"), ("value", "V")):
        res = pooled_checks(X, me[key][:T][v], [others[j][key][:T][v] for j in idx[1:]], D, me["D"][:T][v],
                            [others[j]["D"][:T][v] for j in idx[1:]], np.column_stack([others[j][key][:T] for j in idx])[v],
                            [others[j]["D"][:T][v] for j in idx], c1, max(10.0, X.shape[1] / 5))
        row.update({f"{tname}_{k}": val for k, val in res.items()})
    return [row]


def run_steinmetz(data_dir, out_dir, jobs):
    import pandas as pd
    from joblib import Parallel, delayed
    import steinmetz_replication as sr
    from history_behaviour import history_design
    sessions = sr.load_sessions(data_dir)
    bank = []
    for i, d in enumerate(sessions):
        contrast, choice, fb, side = sr.task(d)
        v = choice >= 0; v[:10] = False
        b, _ = m1.bias_path(contrast, choice, fb, v)
        H = history_design(side, choice, fb, np.full(len(choice), 0.5))[:, :-1]
        cur = np.where(choice >= 0, choice - 0.5, 0.0)
        bank.append(dict(i=i, subject=str(d["mouse_name"]), b=b, V=m1.value_trace(choice, fb), T=len(b),
                         D=np.column_stack([H, cur])))
    res = Parallel(n_jobs=jobs)(delayed(stein_session)(i, d, bank) for i, d in enumerate(sessions))
    report(pd.DataFrame([r for rows in res for r in rows]), out_dir, "Steinmetz 2019")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "ibl", "steinmetz"])
    ap.add_argument("--cache", default="data/mv_cache")
    ap.add_argument("--data", default="~/Downloads/MULTIVERSE/steinmetz_data")
    ap.add_argument("--out", default="results/mechanism2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    if a.step == "fetch":
        fetch(a.cache, AUX3, a.limit)
    elif a.step == "ibl":
        run_ibl(a.cache, a.out + "/ibl", a.jobs)
    else:
        run_steinmetz(a.data, a.out + "/steinmetz", a.jobs)
