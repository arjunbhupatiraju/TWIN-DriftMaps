"""Module 0: does the IBL public data contain enough nucleus incertus (NI) neurons?

    pip install ONE-api iblatlas pandas pyarrow
    python scripts/check_ibl_coverage.py [--out results/coverage]

Uses IBL's public server (no account needed). It:
  1. finds every probe insertion passing through NI, LC (locus coeruleus) and SLC
     (subceruleus) in the Allen atlas,
  2. counts sorted units in each region on those probes (all units and 'good' units,
     label == 1 in IBL's quality metrics),
  3. for every session with NI units, lists what else was recorded at the same time
     (hippocampus, other probes) and whether pupil / video data exist.

Writes coverage_units.csv, coverage_sessions.csv and SUMMARY.txt with a verdict
against the thresholds below. Expect it to take 5-30 minutes (it downloads small
cluster tables only, never raw data).

PRE-SET THRESHOLDS (decided before looking):
  GO          >= 50 good NI units across >= 5 sessions
  BORDERLINE  >= 15 good NI units across >= 3 sessions  (characterization only; weak for module 2)
  NO-GO       anything less -> switch to the human 7T fMRI fallback
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper.ibl import region_labels  # noqa: E402

TARGETS = {"NI": "nucleus incertus", "LC": "locus coeruleus", "SLC": "subceruleus nucleus"}
HIPPOCAMPUS = ("CA1", "CA2", "CA3", "DG", "DG-mo", "DG-po", "DG-sg", "SUB", "ProS", "POST", "PRE")
GOOD_LABEL = 1.0


def connect():
    from one.api import ONE
    return ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab",
               password="international", silent=True)


def insertions_via_channels(one, acronym):
    """Second route: insertions with ALIGNED channels in the region (Alyx channels table)."""
    rid = int(_regions().acronym2id([acronym])[0])
    pids = set()
    try:
        chans = one.alyx.rest("channels", "list", brain_region=rid)
    except Exception as e:
        print(f"  (channel search for {acronym} unavailable: {e!r:.120})")
        return pids
    trajs = {c.get("trajectory_estimate") for c in chans if c.get("trajectory_estimate")}
    for t in trajs:
        try:
            pi = one.alyx.rest("trajectories", "read", id=t).get("probe_insertion")
            if pi:
                pids.add(str(pi))
        except Exception:
            pass
    return pids


def insertions_through(one, acronym):
    """Probe insertion IDs whose trajectory passes through `acronym` (server-side atlas filter)."""
    errors = []
    for attempt in ("search_insertions", "rest"):
        try:
            if attempt == "search_insertions":
                pids = one.search_insertions(atlas_acronym=acronym, query_type="remote")
            else:
                pids = [r["id"] for r in one.alyx.rest("insertions", "list", atlas_acronym=acronym)]
            return [str(p) for p in pids]
        except Exception as e:  # API differences across ONE versions
            errors.append(f"{attempt}: {e!r}")
    raise RuntimeError(f"could not query insertions for {acronym}: {errors}")


def _regions():
    from iblatlas.regions import BrainRegions
    return BrainRegions()


def cluster_table(one, pid):
    """Per-cluster brain region acronym and quality label for one probe insertion.

    IBL stores regions in several ways depending on release/alignment; try in order:
      clusters.brainLocationAcronyms_ccf_2017 -> clusters.brainLocationIds_ccf_2017 (Allen ids)
      -> clusters.channels + channels.brainLocationIds_ccf_2017
    """
    eid, pname = one.pid2eid(pid)
    eid = str(eid)
    colls = [c for c in one.list_collections(eid) if c.startswith(f"alf/{pname}")]
    if not colls:
        raise RuntimeError(f"no alf/{pname} collection")
    colls = sorted(colls, key=len, reverse=True)   # most specific (spike-sorter sub-collection) first
    for coll in colls:
        rl = region_labels(one, eid, coll)
        if rl is not None:
            return eid, pname, coll, rl[0], rl[1]
    avail = []
    for coll in colls:
        try:
            avail += [str(d) for d in one.list_datasets(eid, collection=coll) if "cluster" in str(d) or "channel" in str(d)]
        except Exception:
            pass
    raise RuntimeError(f"no region labels for {pid} in {colls}; cluster/channel datasets present: {avail[:30]}")


def alignment_resolved(one, pid):
    """IBL flag: is the probe's ephys-histology alignment marked resolved (reviewed and agreed)? None if unknown"""
    try:
        js = one.alyx.rest("insertions", "read", id=pid).get("json") or {}
        return bool((js.get("extended_qc") or {}).get("alignment_resolved", False))
    except Exception:
        return None


def has_pupil_video(one, eid):
    try:
        names = [str(d) for d in one.list_datasets(eid)]
    except Exception:
        return None, None
    video = any("Camera" in n and n.endswith((".mp4", ".times.npy")) for n in names)
    pupil = any(("Camera.features" in n) or ("Camera.dlc" in n) or ("lightningPose" in n) for n in names)
    return video, pupil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/coverage")
    ap.add_argument("--max-insertions", type=int, default=0, help="debug: cap insertions per target")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    one = connect()
    print("connected to IBL public server")

    import pandas as pd
    unit_rows, failures = [], []
    for acr, name in TARGETS.items():
        pids_a = set(insertions_through(one, acr))
        pids_b = insertions_via_channels(one, acr)
        pids = sorted(pids_a | pids_b)
        print(f"{acr}: {len(pids_a)} insertions by trajectory search, {len(pids_b)} by aligned-channel search, "
              f"{len(pids)} combined")
        if args.max_insertions:
            pids = pids[:args.max_insertions]
        print(f"{acr} ({name}): {len(pids)} insertions pass through it")
        for i, pid in enumerate(pids):
            try:
                eid, pname, coll, acrs, label = cluster_table(one, pid)
                m = acrs == acr
                unit_rows.append(dict(target=acr, pid=pid, eid=eid, probe=pname, collection=coll,
                                      units=int(m.sum()), good_units=int(((label == GOOD_LABEL) & m).sum()),
                                      labels_available=bool(~(label != label).all()),
                                      alignment_resolved=alignment_resolved(one, pid) if m.any() else None))
            except Exception as e:
                failures.append(dict(target=acr, pid=pid, error=repr(e)[:300]))
            if (i + 1) % 10 == 0:
                print(f"  {acr}: {i+1}/{len(pids)} insertions checked")
    units = pd.DataFrame(unit_rows)
    units.to_csv(out / "coverage_units.csv", index=False)

    # sessions with NI units: what else was recorded simultaneously?
    sess_rows = []
    ni = units[(units.target == "NI") & (units.units > 0)] if len(units) else units
    for eid in sorted(set(ni.eid)) if len(ni) else []:
        row = dict(eid=eid, ni_units=int(ni[ni.eid == eid].units.sum()), ni_good=int(ni[ni.eid == eid].good_units.sum()))
        regions = Counter()
        try:
            for ins in one.alyx.rest("insertions", "list", session=eid):
                try:
                    _, _, _, acrs, label = cluster_table(one, str(ins["id"]))
                    good = label == GOOD_LABEL
                    use = good if good.any() else slice(None)
                    regions.update(acrs[use].tolist())
                except Exception:
                    pass
        except Exception as e:
            row["error"] = repr(e)[:200]
        row["hippocampal_units"] = int(sum(v for k, v in regions.items() if k in HIPPOCAMPUS))
        row["lc_units"] = int(regions.get("LC", 0))
        row["n_regions"] = len(regions)
        row["video"], row["pupil"] = has_pupil_video(one, eid)
        sess_rows.append(row)
    sessions = pd.DataFrame(sess_rows)
    sessions.to_csv(out / "coverage_sessions.csv", index=False)

    def tot(t, col):
        return int(units[units.target == t][col].sum()) if len(units) else 0

    def nsess(t, col, k=1):
        if not len(units):
            return 0
        g = units[units.target == t].groupby("eid")[col].sum()
        return int((g >= k).sum())

    ni_good, ni_good_sess = tot("NI", "good_units"), nsess("NI", "good_units")
    labels_ok = bool(len(units) and units.labels_available.any())
    ni_metric = ni_good if labels_ok else tot("NI", "units")
    verdict = ("GO" if ni_metric >= 50 and ni_good_sess >= 5 else
               "BORDERLINE" if ni_metric >= 15 and ni_good_sess >= 3 else "NO-GO")
    if not labels_ok:
        verdict += " (quality labels unavailable: verdict uses ALL units, check manually)"
    summary = {
        "NI": dict(insertions=int((units.target == "NI").sum()) if len(units) else 0, units=tot("NI", "units"),
                   good_units=ni_good, sessions_with_units=nsess("NI", "units"), sessions_with_good_units=ni_good_sess),
        "LC": dict(units=tot("LC", "units"), good_units=tot("LC", "good_units"), sessions_with_units=nsess("LC", "units")),
        "SLC": dict(units=tot("SLC", "units"), good_units=tot("SLC", "good_units")),
        "NI_good_units_on_alignment_resolved_probes": int(units[(units.target == "NI") & (units.alignment_resolved == True)].good_units.sum()) if len(units) else 0,  # noqa: E712
        "NI_sessions_with_hippocampus": int((sessions.hippocampal_units > 0).sum()) if len(sessions) else 0,
        "NI_sessions_with_pupil": int(sessions.pupil.fillna(False).astype(bool).sum()) if len(sessions) else 0,
        "failed_insertions": len(failures), "verdict": verdict, "runtime_min": round((time.time() - t0) / 60, 1),
    }
    (out / "failures.json").write_text(json.dumps(failures, indent=1))
    text = "IBL coverage check (module 0)\n" + json.dumps(summary, indent=2) + (
        "\n\nThresholds: GO >= 50 good NI units in >= 5 sessions; BORDERLINE >= 15 in >= 3; else NO-GO.\n")
    (out / "SUMMARY.txt").write_text(text)
    print("\n" + text)
    print(f"wrote {out}/coverage_units.csv, coverage_sessions.csv, SUMMARY.txt, failures.json")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("\nThe script failed before finishing. Send the error above to Claude.")
        sys.exit(1)
