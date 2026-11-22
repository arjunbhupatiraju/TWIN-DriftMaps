"""RELIABILITY ATLAS of the IBL Brain-Wide Map (parts A and C of the plan; claude/idea-replicability-prediction-nov3.md)

For every session x region x variable it records:
  observed tests   p of the trial shuffle (common), the valid test (pseudo-sessions for block and stimulus side, which
                   follow the block structure; Harris's linear shift for choice and feedback, which have no task model)
                   and the linear shift for every variable
  drift risk       how often the trial shuffle fires on twins of these neurons with a task-consistent fake label
  power samples    valid-test p-values after planting a code (per-neuron r = 0.05, 0.1, 0.2 in 20% of neurons) into the
                   real neurons; these feed the sample-size map
Then `summarize` builds, per region x variable:
  A  answerability: drift false-positive risk, single-session power, observed significance rates
  C  sample-size map: sessions needed for 80% power (Fisher-combined) at each effect size, vs sessions available

Run from ~/Downloads/MODEL (needs ~/Downloads/MODEL/twin and the GATEKEEPER repo + IBL cache):
  python3 part6/atlas.py run          # parallel, resumable; --every 1 uses every cached session
  python3 part6/atlas.py summarize
"""
from __future__ import annotations

import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results_atlas"
sys.path.insert(0, str(HERE.parent / "twin"))
R_GRID = (0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3)   # same grid as replication.py
SUMMARY_R = (0.06, 0.1, 0.2)
VARIABLES = {           # name: (window, how the label is made, valid null)
    "block":    ("prior_a", "probabilityLeft", "block"),     # pseudo-sessions of the block structure
    "stimulus": ("stim_a", "stimulus side", "side"),         # pseudo-sessions: stimulus side follows the blocks too
    "choice":   ("choice_a", "choice", "shift"),             # no task model for the mouse's choices: Harris linear shift
    "feedback": ("feedback_a", "feedback", "shift"),
}
# NEW variables for the TWIN v2 confirmation (never used to build either model). No task model: linear shift.
NEW_VARIABLES = {
    "prev_choice":   ("prior_a", "previous trial's choice", "shift"),
    "prev_feedback": ("prior_a", "previous trial's reward", "shift"),
    "contrast":      ("stim_a", "stimulus contrast (absolute)", "shift"),
    "rt":            ("stim_a", "log reaction time", "shift"),
}
# SET 3: fresh history variables for the calibration confirmation (never used before)
SET3_VARIABLES = {
    "prev_stimulus":  ("prior_a", "previous trial's stimulus side", "shift"),
    "choice_2back":   ("prior_a", "choice two trials back", "shift"),
    "feedback_2back": ("prior_a", "reward two trials back", "shift"),
}
VARIABLE_SETS = {"main": VARIABLES, "new": NEW_VARIABLES, "set3": SET3_VARIABLES}
ROW_FILES = {"main": "atlas_rows.csv", "new": "atlas_rows_new.csv", "set3": "atlas_rows_set3.csv"}


def _gk(root):
    root = Path(root or os.environ.get("GATEKEEPER_ROOT", "~/Downloads/GATEKEEPER/gatekeeper")).expanduser()
    for p in (root, root / "next"):
        sys.path.insert(0, str(p))


def _labels(s):
    pl = np.asarray(s.meta.get("probabilityLeft", []), float)
    return {"block": pl if len(pl) == len(s.choice) else None,
            "stimulus": np.sign(np.asarray(s.contrast, float)),
            "choice": np.asarray(s.choice, float),
            "feedback": np.asarray(s.feedback, float),
            "prev_choice": np.r_[np.nan, np.where(np.asarray(s.choice) >= 0, np.asarray(s.choice, float), np.nan)[:-1]],
            "prev_feedback": np.r_[np.nan, np.asarray(s.feedback, float)[:-1]],
            "contrast": np.abs(np.asarray(s.contrast, float)),
            "prev_stimulus": np.r_[np.nan, np.where(np.asarray(s.contrast, float) != 0, np.sign(np.asarray(s.contrast, float)), np.nan)[:-1]],
            "choice_2back": np.r_[np.nan, np.nan, np.where(np.asarray(s.choice) >= 0, np.asarray(s.choice, float), np.nan)[:-2]],
            "feedback_2back": np.r_[np.nan, np.nan, np.asarray(s.feedback, float)[:-2]],
            "rt": np.log(np.clip(np.asarray(getattr(s, "rt", np.full(len(s.choice), np.nan)), float), 0.05, 30))}


def _session(path, gk_root, min_units, max_units, n_null, n_twin, n_plant, burn, seed, var_set="main"):
    _gk(gk_root)
    from gatekeeper.mv_data import MVSession
    from twin.task import BlockTask, PseudoSessions
    from twin.drift import phase_twin
    from twin.stats import pooled_corr
    from twin.check import plant
    from twin.tests import _p, p_linear_shift
    rng = np.random.default_rng(seed)
    s = MVSession(Path(path))
    labs = _labels(s)
    base = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]
    base[:burn] = False
    Tfull = len(base)
    task_nulls = {"block": PseudoSessions(BlockTask(), "block"), "side": PseudoSessions(BlockTask(), "side")}
    rows = []
    for var, (window, _, nulltype) in VARIABLE_SETS[var_set].items():
        y_full = labs[var]
        if y_full is None:
            continue
        v = base & np.isfinite(y_full)
        if var == "stimulus":
            v &= y_full != 0
        if var in ("choice", "feedback") and window != "prior_a":
            ev = {"choice_a": "firstMovement", "feedback_a": "feedback"}.get(window, "stimOn")
            v &= s.valid[ev]
        if var == "stimulus":
            v &= s.valid["stimOn"]
        if var == "rt":
            v &= s.valid["firstMovement"]
        T = int(v.sum())
        if T < 150:
            continue
        y = y_full[v]
        if np.std(y) == 0:
            continue

        pseudo = nulltype in task_nulls
        bank = (np.column_stack([np.asarray(task_nulls[nulltype].draw(Tfull, rng)) for _ in range(400)])[v]
                if pseudo else None)                      # pseudo-session bank, drawn once per session x variable

        def valid_nulls(n):                               # task structure redrawn for the whole session, same mask
            return bank[:, rng.integers(0, bank.shape[1], n)]

        def valid_p(Xa, ya):
            if pseudo:
                ww = pooled_corr(Xa, np.column_stack([ya, valid_nulls(60)]))
                return _p(ww[0], ww[1:])
            return p_linear_shift(Xa, ya)

        for r in s.regions:
            X = s.X(r, window)
            if X is None:
                continue
            X = X[v]; X = X[:, X.std(0) > 0]
            if X.shape[1] < min_units:
                continue
            if X.shape[1] > max_units:
                X = X[:, rng.choice(X.shape[1], max_units, replace=False)]
            N = X.shape[1]
            Ysh = np.column_stack([rng.permutation(y) for _ in range(n_null)])
            w = pooled_corr(X, np.column_stack([y, Ysh]))
            p_shuf = _p(w[0], w[1:])
            p_shift = p_linear_shift(X, y)
            if pseudo:
                wv = pooled_corr(X, np.column_stack([y, valid_nulls(n_null)]))
                p_valid = _p(wv[0], wv[1:])
                excess = float(wv[0] - np.mean(wv[1:]))
            else:
                p_valid = p_shift
                S = max(10, T // 10); L = T - 2 * S
                ws = pooled_corr(X[S:S + L], np.column_stack([y[S + k:S + k + L] for k in range(-S, S + 1)]))
                excess = float((ws[S] - np.mean(np.delete(ws, S))) * T / L)
            r_hat = float(np.sqrt(max(excess, 0.0) / (T * N * 0.2)))
            # drift risk: shuffle test on twins with a task-consistent label
            hits = 0
            for _ in range(n_twin):
                Xt = phase_twin(X, rng)
                yt = valid_nulls(1)[:, 0] if pseudo else y
                wt = pooled_corr(Xt, np.column_stack([yt] + [rng.permutation(yt) for _ in range(50)]))
                hits += _p(wt[0], wt[1:]) < 0.05
            # power samples: plant into the real neurons, test against the valid null
            power_p = {}
            for rr in R_GRID:
                ps = []
                for _ in range(n_plant):
                    # a NEW code for a label with this session's structure (never the real label, which the
                    # neurons may already encode): circularly shifted for variables without a task model
                    yp = valid_nulls(1)[:, 0] if pseudo else np.roll(y, int(rng.integers(T // 4, 3 * T // 4)))
                    ps.append(round(valid_p(plant(X, yp, rr, 0.2, rng), yp), 4))
                power_p[str(rr)] = ps
            rows.append(dict(session=s.sid, subject=s.subject, lab=s.lab, region=r, variable=var, T=T, N=N,
                             p_shuffle=p_shuf, p_valid=p_valid, p_shift=p_shift, drift_fp_shuffle=hits / n_twin,
                             r_hat=r_hat, power_curve=json.dumps(power_p)))
    return rows


def cmd_run(a):
    import pandas as pd
    OUT.mkdir(exist_ok=True)
    var_set = "new" if a.new_variables else a.set
    f = OUT / ROW_FILES[var_set]
    done = set(pd.read_csv(f).session) if f.exists() else set()
    paths = [p for p in sorted(Path(a.cache).expanduser().glob("*.npz"))[::a.every] if p.stem not in done]
    print(f"{len(paths)} sessions to run ({len(done)} done) on {a.workers} workers", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        futs = {ex.submit(_session, str(p), a.gatekeeper, a.min_units, a.max_units, a.null, a.twins, a.plant, 90, i, var_set): p
                for i, p in enumerate(paths)}
        for k, fu in enumerate(as_completed(futs)):
            try:
                rows = fu.result()
            except Exception as e:                        # one bad session must not stop the atlas
                print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: ERROR {e}", flush=True)
                continue
            if rows:
                pd.DataFrame(rows).to_csv(f, mode="a", header=not f.exists(), index=False)
            print(f"[{k + 1}/{len(paths)}] {futs[fu].stem}: {len(rows)} region-variables", flush=True)


def _fisher(ps):
    from scipy.stats import chi2
    ps = np.clip(np.asarray(ps, float), 1e-12, 1)
    return chi2.sf(-2 * np.log(ps).sum(axis=-1), 2 * ps.shape[-1])


def sessions_needed(pvals, target=0.8, s_max=60, B=400, rng=None):
    """Smallest number of sessions whose Fisher-combined valid-test p is < 0.05 with probability >= target,
    resampling this region's per-session planted p-values."""
    rng = rng or np.random.default_rng(0)
    pvals = np.asarray(pvals, float)
    if len(pvals) == 0:
        return np.nan
    for S in list(range(1, 11)) + list(range(12, s_max + 1, 2)):
        draws = pvals[rng.integers(0, len(pvals), (B, S))]
        if np.mean(_fisher(draws) < 0.05) >= target:
            return S
    return np.inf


def cmd_summarize(a):
    import pandas as pd
    D = pd.read_csv(OUT / "atlas_rows.csv")
    rng = np.random.default_rng(0)
    out = []
    for (reg, var), g in D.groupby(["region", "variable"]):
        pp = [json.loads(x) for x in g.power_curve]
        row = dict(region=reg, variable=var, sessions_available=g.session.nunique(), mice=g.subject.nunique(),
                   labs=g.lab.nunique(), median_N=float(g.N.median()),
                   drift_fp_shuffle=float(g.drift_fp_shuffle.mean()),
                   sig_shuffle=float((g.p_shuffle < 0.05).mean()), sig_valid=float((g.p_valid < 0.05).mean()),
                   sig_shift=float((g.p_shift < 0.05).mean()))
        for rr in SUMMARY_R:
            allp = np.concatenate([d[str(rr)] for d in pp])
            row[f"power1_r{rr}"] = float(np.mean(allp < 0.05))
            row[f"sessions_needed_r{rr}"] = sessions_needed(allp, rng=rng)
        out.append(row)
    A = pd.DataFrame(out)
    A.to_csv(OUT / "atlas.csv", index=False)
    lines = [f"# Reliability Atlas (IBL): {len(A)} region x variable cells, {D.session.nunique()} sessions, "
             f"{D.subject.nunique()} mice, {D.lab.nunique()} labs\n"]
    for var in VARIABLES:
        V = A[A.variable == var]
        if not len(V):
            continue
        lines.append(f"## {var} ({len(V)} regions)")
        lines.append(f"  shuffle-test false-positive risk from drift: median {V.drift_fp_shuffle.median():.2f}")
        lines.append(f"  significant: shuffle {V.sig_shuffle.mean():.2f}, valid test {V.sig_valid.mean():.2f}, "
                     f"linear shift {V.sig_shift.mean():.2f} (mean fraction of sessions)")
        for rr in SUMMARY_R:
            need = V[f"sessions_needed_r{rr}"]
            ok = (need <= V.sessions_available).mean()
            lines.append(f"  r = {rr}: single-session power median {V[f'power1_r{rr}'].median():.2f}; sessions needed "
                         f"median {np.nanmedian(need.replace(np.inf, np.nan)):.0f} "
                         f"({np.isinf(need).mean():.0%} need > 60); answerable with the sessions available: {ok:.0%}")
        lines.append("")
    text = "\n".join(lines)
    (OUT / "ATLAS_SUMMARY.md").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--cache", default="~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")
    r.add_argument("--gatekeeper"); r.add_argument("--every", type=int, default=1); r.add_argument("--min-units", type=int, default=10)
    r.add_argument("--max-units", type=int, default=60); r.add_argument("--null", type=int, default=100)
    r.add_argument("--twins", type=int, default=20); r.add_argument("--plant", type=int, default=24)
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--new-variables", action="store_true", help="same as --set new")
    r.add_argument("--set", default="main", choices=list(VARIABLE_SETS))
    sub.add_parser("summarize")
    a = ap.parse_args()
    dict(run=cmd_run, summarize=cmd_summarize)[a.cmd](a)


if __name__ == "__main__":
    main()
