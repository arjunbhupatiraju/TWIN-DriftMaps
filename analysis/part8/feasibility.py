"""FEASIBILITY (final idea v2, 2026-11-11). Read-only checks, no analysis, nothing that could bias pre-registration.

A. Causal validation: is the Zatka-Haas et al. 2021 inactivation dataset (figshare 13008038) and the Steinmetz 2019
   Neuropixels dataset (figshare 9598406, same task) downloadable, and which files do they hold?
B. Reproducibility validation: which IBL repeated-site insertions are on the public server, from how many labs?

    cd ~/Downloads/MODEL && python3 part8/feasibility.py
Output: printed summary + part8/FEASIBILITY.md
"""
from __future__ import annotations

import json, urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
L = []


def P(*x):
    s = " ".join(str(i) for i in x); print(s); L.append(s)


def figshare(aid, name):
    P(f"\n## {name} (figshare {aid})")
    try:
        with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{aid}", timeout=30) as r:
            d = json.load(r)
        files = d.get("files", [])
        P(f"title: {d.get('title')}; {len(files)} files, {sum(f['size'] for f in files) / 1e9:.2f} GB total")
        for f in sorted(files, key=lambda f: -f["size"])[:25]:
            P(f"  {f['name']}  ({f['size'] / 1e6:.1f} MB)")
        return True
    except Exception as e:
        P(f"  NOT REACHABLE: {e!r}")
        return False


def repeated_site():
    P("\n## IBL repeated-site insertions (public server)")
    try:
        from one.api import ONE
    except ImportError:
        P("  ONE not installed (pip install ONE-api)"); return False
    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
    tags = ["2024_Q2_IBL_et_al_RepeatedSite", "2022_Q2_IBL_et_al_RepeatedSite", "2022_Q4_IBL_et_al_RepeatedSite",
            "2023_Q1_IBL_et_al_RepeatedSite", "RepeatedSite"]
    best = None
    for t in tags:
        try:
            ins = one.alyx.rest("insertions", "list", django=f"datasets__tags__name,{t}")
            n = len({i["id"] for i in ins})
            P(f"  tag {t}: {n} insertions")
            if n and (best is None or n > len(best[1])):
                best = (t, list({i['id']: i for i in ins}.values()))
        except Exception as e:
            P(f"  tag {t}: error {e!r}")
    if not best:
        P("  no repeated-site tag found; try one.alyx.rest('tags', 'list') by hand"); return False
    t, ins = best
    labs = Counter(i.get("session_info", {}).get("lab") or "?" for i in ins)
    subj = {i.get("session_info", {}).get("subject") for i in ins}
    P(f"  using {t}: {len(ins)} insertions, {len(subj)} mice, {len(labs)} labs: " +
      ", ".join(f"{k} {v}" for k, v in labs.most_common()))
    return True


if __name__ == "__main__":
    P("# FEASIBILITY (final idea v2)")
    a1 = figshare(13008038, "Zatka-Haas et al. 2021: widefield, optogenetic inactivation, behaviour")
    a2 = figshare(9598406, "Steinmetz et al. 2019: Neuropixels in the same task")
    b = repeated_site()
    P(f"\nSUMMARY: A inactivation data {'OK' if a1 else 'MISSING'}; A recordings {'OK' if a2 else 'MISSING'}; "
      f"B repeated-site {'OK' if b else 'MISSING'}")
    (HERE / "FEASIBILITY.md").write_text("\n".join(L) + "\n"); print(f"\nWrote {HERE / 'FEASIBILITY.md'}")
