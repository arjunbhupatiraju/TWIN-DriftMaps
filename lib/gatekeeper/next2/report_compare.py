"""Summarise the head-to-head: false positives on fake priors, power on real priors, region maps, overlap."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "driftaware" / "src"))
from driftaware import region_map  # noqa: E402

METHODS = {"p_findling_shuf": "decoding + shuffle null (common)", "p_findling": "Findling 2025 (Lasso R2, pseudo)",
           "p_zhang": "Zhang 2025-style (trial smoothing, pseudo)", "p_plain": "standard single-neuron (pseudo)",
           "p_new": "NEW drift-aware (pseudo)"}


def main(csv, out_dir):
    df = pd.read_csv(csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, maps = [], {}
    for var in ("bprior", "ctl_bprior", "prior", "ctl_prior"):
        sub = df[df["var"] == var]
        if sub.empty:
            continue
        for m, name in METHODS.items():
            rm5 = region_map(sub, m, q=0.05)
            rm1 = region_map(sub, m, q=0.01)
            maps[(var, m)] = set(rm5.index[rm5.significant])
            rows.append(dict(variable=var, method=name, session_region_rate=round(float((sub[m] < 0.05).mean()), 3),
                             regions_FDR5=int(rm5.significant.sum()), regions_FDR1=int(rm1.significant.sum()),
                             regions_tested=len(rm5)))
    t = pd.DataFrame(rows)
    ov = []
    for var in ("bprior", "prior"):
        if (var, "p_new") in maps:
            a, b = maps[(var, "p_new")], maps[(var, "p_findling")]
            ov.append(dict(variable=var, new_only=len(a - b), both=len(a & b), findling_only=len(b - a),
                           both_regions=",".join(sorted(a & b)), findling_only_regions=",".join(sorted(b - a))))
    o = pd.DataFrame(ov)
    meta = dict(sessions=int(df.sid.nunique()), mice=int(df.subject.nunique()), regions=int(df.region.nunique()))
    (out_dir / "compare_summary.json").write_text(json.dumps(dict(meta=meta, table=rows, overlap=ov), indent=1))
    fmt = (lambda d: d.to_markdown(index=False)) if _has_tabulate() else (lambda d: d.to_string(index=False))
    text = "\n".join([f"# Head-to-head on IBL ({meta['sessions']} sessions, {meta['mice']} mice, {meta['regions']} regions)\n",
                      "Fake priors (ctl_) should give rate about 0.05 and 0 regions.\n", fmt(t),
                      "\n## Overlap of significant regions (FDR 5%): new test vs Findling-style decoding\n",
                      fmt(o.drop(columns=["both_regions", "findling_only_regions"])) if len(o) else "(none)"])
    (out_dir / "COMPARE_SUMMARY.md").write_text(text)
    print(text)


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
