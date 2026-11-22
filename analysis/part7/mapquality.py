"""MAP QUALITY (pre-registered 2026-11-09): how much of a standard brain-wide encoding map can be trusted?

Uses only existing outputs of the Reliability Atlas (part6/atlas.py): per session-region tests (atlas_rows*.csv:
variable, region, p_shuffle, p_valid, and a session/subject column) and per region-variable power (atlas.csv:
variable, region, sessions_needed_r0.1, sessions_available). Nothing is recomputed from neural data.

Each region x variable cell of the map is classified:
  standard claim  : Fisher-combined p_shuffle over its sessions, BH q < 0.05 across all cells (what a standard map reports)
  valid claim     : the same with p_valid
  answerable      : sessions_needed_r0.1 <= sessions_available (80% power for r = 0.1 with the valid test)
Primary outcomes (registered):
  X = share of standard-claim cells that are also valid claims (survive drift)
  Y = share of ALL cells that are settled = valid claim, or answerable with no valid claim
  N = median sessions needed (r = 0.1) for unsettled cells
Uncertainty: subject bootstrap (500), cells and BH recomputed each time; power columns held fixed.

    cd ~/Downloads/MODEL && python3 part7/mapquality.py run
    python3 part7/mapquality.py test        (synthetic data check)
Output: part7/results_mapquality/{cells.csv, MAPQUALITY.md}
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "results_mapquality"


def bh(p, q=0.05):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    k = np.nonzero(ok)[0].max() + 1 if ok.any() else 0
    out = np.zeros(n, bool); out[o[:k]] = True
    return out


def fisher(p):
    from scipy.stats import chi2
    p = np.clip(np.asarray(p, float), 1e-300, 1)
    return float(chi2.sf(-2 * np.log(p).sum(), 2 * len(p)))


def find(name):
    hits = [p for p in ROOT.rglob(name) if "release" not in p.parts]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def load(root=None):
    import pandas as pd
    global ROOT
    if root:
        ROOT = Path(root)
    parts = [find(n) for n in ("atlas_rows.csv", "atlas_rows_new.csv", "atlas_rows_set3.csv")]
    parts = [p for p in parts if p is not None]
    A = find("atlas.csv")
    if not parts or A is None:
        sys.exit("atlas_rows*.csv or atlas.csv not found under MODEL (they come from part6/atlas.py run / summarize)")
    R = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    R = R.dropna(subset=["p_shuffle", "p_valid"])
    unit = next((c for c in ("subject", "mouse", "subject_nickname", "session", "eid", "pid") if c in R.columns), None)
    if unit is None:
        sys.exit("no subject/session column in atlas_rows")
    P = pd.read_csv(A)[["variable", "region", "sessions_needed_r0.1", "sessions_available"]]
    print(f"rows: {', '.join(p.name for p in parts)} ({len(R)} session-region tests); power: {A.name}; bootstrap unit: {unit}")
    return R, P, unit


def cells(R, P):
    import pandas as pd
    C = R.groupby(["variable", "region"]).agg(p_std=("p_shuffle", fisher), p_val=("p_valid", fisher), n=("p_valid", "size")).reset_index()
    C["std_claim"] = bh(C.p_std.values); C["valid_claim"] = bh(C.p_val.values)
    C = C.merge(P, on=["variable", "region"], how="left")
    need = C["sessions_needed_r0.1"].astype(float).replace(np.inf, np.nan)
    C["answerable"] = (need <= C.sessions_available.astype(float)).fillna(False)
    C["settled"] = C.valid_claim | C.answerable
    C["cat"] = np.select([C.valid_claim, C.std_claim & ~C.valid_claim & C.answerable, C.std_claim & ~C.valid_claim,
                          ~C.std_claim & C.answerable], ["survives", "drift artefact (answerable)", "unresolved claim",
                                                         "settled negative"], "unresolved, no claim")
    return C


def stats(C):
    s = C[C.std_claim]
    X = float(s.valid_claim.mean()) if len(s) else np.nan
    Y = float(C.settled.mean())
    need = C.loc[~C.settled, "sessions_needed_r0.1"].astype(float)
    N = float(np.nanmedian(need.replace(np.inf, np.nan))) if len(need) else np.nan
    inf_share = float(np.isinf(need).mean()) if len(need) else np.nan
    return X, Y, N, inf_share


def cmd_run(a):
    import pandas as pd
    R, P, unit = load(a.root)
    C = cells(R, P); X, Y, N, inf = stats(C)
    rng = np.random.default_rng(0); units = R[unit].unique(); g = {u: d for u, d in R.groupby(unit)}
    bs = []
    for _ in range(a.boot):
        RR = pd.concat([g[u] for u in units[rng.integers(0, len(units), len(units))]], ignore_index=True)
        bs.append(stats(cells(RR, P))[:2])
    bs = np.array(bs); ci = lambda k: np.nanpercentile(bs[:, k], [2.5, 97.5])
    OUT.mkdir(exist_ok=True); C.to_csv(OUT / "cells.csv", index=False)
    L = []; Pr = lambda *x: (print(*x), L.append(" ".join(str(i) for i in x)))
    Pr(f"# MAP QUALITY: {len(C)} region x variable cells, {C.variable.nunique()} variables, {R[unit].nunique()} {unit}s\n")
    Pr(f"standard map reports {C.std_claim.mean():.0%} of cells; valid test {C.valid_claim.mean():.0%}")
    Pr(f"X  share of standard claims that survive drift: {X:.1%} (95% CI {ci(0)[0]:.1%} to {ci(0)[1]:.1%})")
    Pr(f"Y  share of all cells settled (valid claim, or powered negative): {Y:.1%} (95% CI {ci(1)[0]:.1%} to {ci(1)[1]:.1%})")
    Pr(f"N  median sessions needed for unsettled cells (r = 0.1): {N:.0f}; {inf:.0%} of unsettled cells not reachable in the grid")
    Pr("\nby variable (cells; standard claims; X; Y):")
    for v, d in C.groupby("variable"):
        x, y, _, _ = stats(d)
        Pr(f"  {v}: {len(d)}; {d.std_claim.mean():.0%}; {x:.0%}; {y:.0%}")
    Pr("\ncategories: " + ", ".join(f"{k} {v:.0%}" for k, v in C.cat.value_counts(normalize=True).items()))
    lo_x, hi_x = ci(0); lo_y, hi_y = ci(1)
    reframe = hi_x < 0.5 and hi_y < 0.5
    Pr(f"\nPRE-REGISTERED DECISION: {'REFRAME (both upper bounds < 50%)' if reframe else 'KEEP the current framing'}"
       f" (rule: reframe around map reliability only if the upper 95% bounds of X and of Y are both below 50%)")
    (OUT / "MAPQUALITY.md").write_text("\n".join(L) + "\n"); print(f"\nWrote {OUT / 'MAPQUALITY.md'}")


def cmd_test(a):
    import pandas as pd, tempfile
    rng = np.random.default_rng(1); d = Path(tempfile.mkdtemp())
    rows, pw = [], []
    for v, true_frac in (("block", 0.1), ("choice", 0.7)):
        for r in range(60):
            real = rng.random() < true_frac
            for s in range(rng.integers(2, 12)):
                ps = rng.random() ** 8                                     # standard test: drift makes everything small
                pv = rng.random() ** (6 if real else 1)                    # valid: small only for real codes
                rows.append(dict(variable=v, region=f"R{r}", subject=f"m{s % 7}", p_shuffle=ps, p_valid=pv))
            pw.append(dict(variable=v, region=f"R{r}", **{"sessions_needed_r0.1": rng.choice([3, 8, 30, np.inf])},
                           sessions_available=rng.integers(2, 12)))
    pd.DataFrame(rows).to_csv(d / "atlas_rows.csv", index=False); pd.DataFrame(pw).to_csv(d / "atlas.csv", index=False)
    a.root = str(d); a.boot = 50
    global OUT
    OUT = d / "out"; cmd_run(a)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--boot", type=int, default=500); r.add_argument("--root", default=None)
    t = sub.add_parser("test"); t.add_argument("--root", default=None); t.add_argument("--boot", type=int, default=50)
    a = ap.parse_args(); {"run": cmd_run, "test": cmd_test}[a.cmd](a)
