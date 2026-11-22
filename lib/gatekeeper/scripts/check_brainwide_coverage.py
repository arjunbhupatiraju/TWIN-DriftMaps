"""Brain-wide coverage: how many good units does every brain region have in the IBL Brain-Wide Map?

    python scripts/check_brainwide_coverage.py            # full run (resumable; ~20-60 min)
    python scripts/check_brainwide_coverage.py --limit 20 # quick smoke test

Reads only region labels and quality labels per probe (no spikes). Regions are reported in
IBL's 'Beryl' parcellation (the one used in the Brain-Wide Map papers). Progress is appended
to results/brainwide_coverage/insertions.jsonl, so if the run stops, re-running continues
where it left off.

PRE-SET definition (fixed before looking):
  a region is ANALYZABLE if >= 5 sessions each have >= 5 good units in it.
  the brain-wide project is GO if >= 30 regions are analyzable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_ibl_coverage import connect  # noqa: E402
from gatekeeper.ibl import region_labels  # noqa: E402

GOOD = 1.0
BWM_PROJECT = "ibl_neuropixel_brainwide_01"
MIN_UNITS, MIN_SESSIONS, MIN_REGIONS = 5, 5, 30


def bwm_insertions(one):
    for attempt in (lambda: one.search_insertions(project=BWM_PROJECT, query_type="remote"),
                    lambda: [r["id"] for r in one.alyx.rest("insertions", "list", project=BWM_PROJECT)],
                    lambda: [r["id"] for r in one.alyx.rest("insertions", "list")]):
        try:
            p = [str(x) for x in attempt()]
            if p:
                return p
        except Exception:
            continue
    raise RuntimeError("could not list insertions")


def one_insertion(one, pid, beryl):
    eid, pname = one.pid2eid(pid)
    eid = str(eid)
    colls = sorted([c for c in one.list_collections(eid) if c.startswith(f"alf/{pname}")], key=len, reverse=True)
    for coll in colls:
        rl = region_labels(one, eid, coll)
        if rl is None:
            continue
        acr, label = rl
        reg = np.asarray(beryl(acr)).astype(str)
        out = {}
        for r in np.unique(reg):
            m = reg == r
            out[r] = [int(m.sum()), int((m & (label == GOOD)).sum())]
        return dict(pid=pid, eid=eid, probe=pname, regions=out, labels=bool(np.isfinite(label).any()))
    raise RuntimeError(f"no region labels in {colls}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/brainwide_coverage")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "insertions.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                d = json.loads(line)
                done[d["pid"]] = d
            except Exception:
                pass
    t0 = time.time()
    one = connect()
    from iblatlas.regions import BrainRegions
    br = BrainRegions()

    def beryl(a):
        return br.acronym2acronym(np.asarray(a), mapping="Beryl")

    pids = bwm_insertions(one)
    if args.limit:
        pids = pids[:args.limit]
    todo = [p for p in pids if p not in done or "error" in done[p]]
    print(f"{len(pids)} Brain-Wide Map insertions; {len(pids) - len(todo)} already done; {len(todo)} to check", flush=True)
    with open(log, "a") as f, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(one_insertion, one, p, beryl): p for p in todo}
        for i, fut in enumerate(as_completed(futs)):
            p = futs[fut]
            try:
                d = fut.result()
            except Exception as e:
                d = dict(pid=p, error=repr(e)[:300])
            done[p] = d
            f.write(json.dumps(d) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print(f"  {i+1}/{len(todo)} checked ({(time.time()-t0)/60:.1f} min)", flush=True)

    import pandas as pd
    rows = []
    for d in done.values():
        if "error" in d or d["pid"] not in pids:
            continue
        for r, (n, g) in d["regions"].items():
            rows.append(dict(pid=d["pid"], eid=d["eid"], region=r, units=n, good_units=g))
    df = pd.DataFrame(rows)
    df.to_csv(out / "units_by_insertion.csv", index=False)
    per_sess = df.groupby(["region", "eid"])[["units", "good_units"]].sum().reset_index()
    reg = per_sess.groupby("region").agg(
        units=("units", "sum"), good_units=("good_units", "sum"),
        sessions=("eid", "nunique"),
        sessions_ge5_good=("good_units", lambda x: int((x >= MIN_UNITS).sum()))).reset_index()
    reg = reg[~reg.region.isin(["void", "root"])].sort_values("sessions_ge5_good", ascending=False)
    reg["analyzable"] = reg.sessions_ge5_good >= MIN_SESSIONS
    reg.to_csv(out / "regions.csv", index=False)
    n_an = int(reg.analyzable.sum())
    errors = sum(1 for d in done.values() if "error" in d)
    summary = dict(insertions_total=len(pids), insertions_failed=errors,
                   sessions=int(df.eid.nunique()) if len(df) else 0,
                   regions_with_any_units=int(len(reg)), regions_analyzable=n_an,
                   verdict="GO" if n_an >= MIN_REGIONS else "NO-GO",
                   top_regions=[f"{r} ({g} good units, {k} sessions>=5)" for r, g, k in
                                reg.head(40)[["region", "good_units", "sessions_ge5_good"]].values.tolist()],
                   runtime_min=round((time.time() - t0) / 60, 1))
    focus = ["NI", "LC", "MS", "NDB", "DR", "CS", "PPN", "VTA", "SNc", "LDT", "PB", "MRN", "SCm", "CA1", "DG", "PO", "LP", "MOs", "ACAd", "PL", "ILA", "ORBl"]
    summary["focus_regions"] = {r: "{} good units, {} sessions>=5".format(
        *((reg[reg.region == r][["good_units", "sessions_ge5_good"]].values.tolist() or [[0, 0]])[0])) for r in focus}
    text = "Brain-wide coverage\n" + json.dumps(summary, indent=1) + \
        f"\n\nAnalyzable = >= {MIN_SESSIONS} sessions with >= {MIN_UNITS} good units. GO if >= {MIN_REGIONS} analyzable regions.\n"
    (out / "SUMMARY.txt").write_text(text)
    print(text)
    if errors:
        print(f"{errors} insertions failed; re-run the same command to retry them. Example error: "
              f"{next(d['error'] for d in done.values() if 'error' in d)}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nFailed before finishing. Re-running continues from where it stopped; if it fails again, send Claude the error.")
        sys.exit(1)
