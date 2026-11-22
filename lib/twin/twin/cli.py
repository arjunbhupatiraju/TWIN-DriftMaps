"""Command line:  python3 -m twin <command>

  demo                         synthetic demonstration: drift-only recording vs a recording with a real code
  plan   [--slow ... | --calibrate FILE.npz]   design planner (trials x neurons x sessions)
  check  FILE.npz --label NAME [--stat pooled_corr|decode_r2]   claim check on your data
         (FILE.npz must contain X: trials x neurons, and the label array NAME, one value per trial)
  comm   FILE.npz [--a A --b B]  drift-safe communication check between two regions (arrays A and B: trials x neurons)
  validate-ibl --cache DIR     real-data validation of the twin against IBL sessions (needs the GATEKEEPER repo)
  validate-comm --cache DIR    communication check on IBL region pairs, with cross-session negative controls
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

from .task import BlockTask, PseudoSessions, ReversalTask, ForagingTask
from .drift import DriftModel
from .check import check, plan, advice, plant
from .report import save, methods_text

TASKS = {"ibl": BlockTask, "reversal": ReversalTask, "foraging": ForagingTask}


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2f}"
    if isinstance(v, tuple):
        return f"{v[0]:.2f}-{v[1]:.2f}"
    return str(v)


def _table(rows):
    keys = list(rows[0])
    print(" | ".join(f"{k:>16s}" for k in keys))
    for r in rows:
        print(" | ".join(f"{_fmt(v):>16s}" for v in r.values()))


def cmd_demo(a):
    rng = np.random.default_rng(a.seed)
    task = BlockTask(); null = PseudoSessions(task, "prior")
    dm = DriftModel(slow_frac=0.4, phi=0.97, shared=0.4, inst_corr=0.05)
    y = null.draw(a.T, rng)
    for name, X in (("prior, drift only (no real code)", dm.simulate(a.T, a.N, rng)),
                    ("side of stimulus, real code r=0.15", None)):
        if X is None:
            y = task.sample(a.T, rng)["side"]
            X = plant(dm.simulate(a.T, a.N, rng), y, 0.15, 0.3, rng)
            nl = PseudoSessions(task, "side")
        else:
            nl = null
        rep = check(X, y, nl, n_null=a.null, n_twin=a.twins, n_plant=20, label=name, seed=a.seed).dict()
        print(f"\n== {name}\n" + methods_text(rep))
        out = save(rep, a.out, name.split(",")[0].replace(" ", "_") + ("_nocode" if "no real" in name else "_code"))
    print(f"\nReports and figures saved in {out}")


def cmd_plan(a):
    if a.calibrate:
        z = np.load(a.calibrate); dm = DriftModel.fit(z["X"])
        print(f"calibrated drift: slow share by timescale {dm.summary()}, shared slow factors: {len(dm.fac_pi or [])}, "
              f"fast instantaneous correlation {dm.inst_corr:.3f}")
    else:
        dm = DriftModel(slow_frac=a.slow, phi=a.phi, shared=a.shared, inst_corr=a.inst)
    null = PseudoSessions(TASKS[a.task](), a.label)
    rows = plan(dm, null, T_grid=a.trials, N_grid=a.neurons, S_grid=a.sessions, r=a.r, n=a.reps, n_null=a.null)
    _table(rows); print("\n" + "\n".join(advice(rows)))


def cmd_check(a):
    z = np.load(a.file)
    X, y = z["X"], z[a.label]
    null = PseudoSessions(TASKS[a.task](), a.null_label or a.label)
    rep = check(X, y, null, stat=a.stat, n_null=a.null, n_twin=a.twins, label=a.label).dict()
    print(methods_text(rep)); print("saved in", save(rep, a.out, f"check_{a.label}"))


def cmd_validate(a):
    from .validate import validate_ibl
    validate_ibl(a.cache, a.out, every=a.every, min_units=a.min_units, n_fake=a.fake, gatekeeper_root=a.gatekeeper)


def cmd_validate_comm(a):
    from .validate import validate_comm
    validate_comm(a.cache, a.out, every=a.every, min_units=a.min_units, n_null=a.null, gatekeeper_root=a.gatekeeper)


def cmd_comm(a):
    from .comm import comm_check
    z = np.load(a.file)
    res = comm_check(z[a.a], z[a.b], n_null=a.null, slow_trials=a.slow_trials)
    for k, v in res.items():
        print(f"{k:>18s}: {v:.3f}" if isinstance(v, float) else f"{k:>18s}: {v}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "comm_check.json").write_text(json.dumps(res, indent=2))


def cmd_validate_net(a):
    from .validate import validate_net
    validate_net(a.cache, a.out, every=a.every, min_units=a.min_units, n_fake=a.fake, gatekeeper_root=a.gatekeeper)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo"); d.add_argument("--T", type=int, default=600); d.add_argument("--N", type=int, default=30)
    d.add_argument("--null", type=int, default=200); d.add_argument("--twins", type=int, default=60)
    d.add_argument("--seed", type=int, default=0); d.add_argument("--out", default="twin_results")
    p = sub.add_parser("plan")
    p.add_argument("--task", default="ibl"); p.add_argument("--label", default="prior")
    p.add_argument("--calibrate"); p.add_argument("--slow", type=float, default=0.3); p.add_argument("--phi", type=float, default=0.97)
    p.add_argument("--shared", type=float, default=0.35); p.add_argument("--inst", type=float, default=0.05)
    p.add_argument("--trials", type=int, nargs="+", default=[300, 600, 1200]); p.add_argument("--neurons", type=int, nargs="+", default=[10, 30, 100])
    p.add_argument("--sessions", type=int, nargs="+", default=[1, 5, 10, 20]); p.add_argument("--r", type=float, default=0.1)
    p.add_argument("--reps", type=int, default=30); p.add_argument("--null", type=int, default=100)
    c = sub.add_parser("check"); c.add_argument("file"); c.add_argument("--label", required=True)
    c.add_argument("--null-label"); c.add_argument("--task", default="ibl"); c.add_argument("--stat", default="pooled_corr")
    c.add_argument("--null", type=int, default=200); c.add_argument("--twins", type=int, default=100); c.add_argument("--out", default="twin_results")
    v = sub.add_parser("validate-ibl"); v.add_argument("--cache", required=True); v.add_argument("--out", default="twin_results/validate_ibl")
    v.add_argument("--every", type=int, default=4); v.add_argument("--min-units", type=int, default=10)
    v.add_argument("--fake", type=int, default=40); v.add_argument("--gatekeeper")
    m = sub.add_parser("comm"); m.add_argument("file"); m.add_argument("--a", default="A"); m.add_argument("--b", default="B")
    m.add_argument("--null", type=int, default=100); m.add_argument("--slow-trials", type=float, default=5.0); m.add_argument("--out", default="twin_results")
    vc = sub.add_parser("validate-comm"); vc.add_argument("--cache", required=True); vc.add_argument("--out", default="twin_results/validate_comm")
    vc.add_argument("--every", type=int, default=4); vc.add_argument("--min-units", type=int, default=10)
    vc.add_argument("--null", type=int, default=40); vc.add_argument("--gatekeeper")
    vn = sub.add_parser("validate-net"); vn.add_argument("--cache", required=True); vn.add_argument("--out", default="twin_results/validate_net")
    vn.add_argument("--every", type=int, default=4); vn.add_argument("--min-units", type=int, default=10)
    vn.add_argument("--fake", type=int, default=40); vn.add_argument("--gatekeeper")
    a = ap.parse_args(argv)
    {"demo": cmd_demo, "plan": cmd_plan, "check": cmd_check, "validate-ibl": cmd_validate, "comm": cmd_comm,
     "validate-comm": cmd_validate_comm, "validate-net": cmd_validate_net}[a.cmd](a)


if __name__ == "__main__":
    main()
