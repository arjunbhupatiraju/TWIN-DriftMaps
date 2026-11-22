"""Summarise the Steinmetz run: false positives, power, region maps and cross-mouse replication."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "driftaware" / "src"))
from driftaware import region_map  # noqa: E402

METHODS = {"p_naive": "standard + shuffle (common)", "p_plain": "standard + imposter", "p_new": "NEW drift-aware + imposter"}


def summarise(df):
    out = {"n_sessions": int(df.session.nunique()), "n_mice": int(df.mouse.nunique()), "n_regions": int(df.region.nunique())}
    real = [v for v in df["var"].unique() if not v.startswith("ctl_")]
    rows = []
    for v in sorted(real):
        for var in (v, f"ctl_{v}"):
            sub = df[df["var"] == var]
            if sub.empty:
                continue
            for m, name in METHODS.items():
                regions = region_map(sub, m)
                rows.append(dict(variable=var, method=name, session_region_rate=round(float((sub[m] < 0.05).mean()), 3),
                                 regions_significant=int(regions.significant.sum()), regions_tested=len(regions)))
    out["table"] = rows
    # replication across two disjoint halves of the mice
    mice = sorted(df.mouse.unique())
    halves = (set(mice[0::2]), set(mice[1::2]))
    rep = []
    for v in sorted(df["var"].unique()):
        sub = df[df["var"] == v]
        for m, name in METHODS.items():
            sig = []
            for h in halves:
                rm = region_map(sub[sub.mouse.isin(h)], m)
                sig.append(set(rm.index[rm.significant]))
            both = sig[0] & sig[1]
            common = set(region_map(sub[sub.mouse.isin(halves[0])], m).index) & set(region_map(sub[sub.mouse.isin(halves[1])], m).index)
            rep.append(dict(variable=v, method=name, replicated=len(both), regions_in_both_halves=len(common),
                            regions=",".join(sorted(both))))
    out["replication"] = rep
    return out


def main(csv, out_dir):
    df = pd.read_csv(csv)
    s = summarise(df)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "steinmetz_summary.json").write_text(json.dumps(s, indent=1))
    t = pd.DataFrame(s["table"])
    r = pd.DataFrame(s["replication"])
    lines = [f"# Steinmetz 2019 results\n", f"{s['n_sessions']} sessions, {s['n_mice']} mice, {s['n_regions']} regions\n",
             "Fake variables (ctl_) should give rate about 0.05 and 0 significant regions.\n"]
    try:
        lines += [t.to_markdown(index=False), "\n## Replication across two halves of the mice\n", r.drop(columns="regions").to_markdown(index=False)]
    except ImportError:
        lines += [t.to_string(index=False), "\n## Replication across two halves of the mice\n", r.drop(columns="regions").to_string(index=False)]
    (out_dir / "STEINMETZ_SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
