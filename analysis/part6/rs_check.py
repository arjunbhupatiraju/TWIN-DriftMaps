"""Which IBL repeated-site recordings are in the cache?  The repeated site is one probe through VISa/VISam, CA1, DG, LP
and PO. Lists sessions that contain those regions, by lab.   python3 part6/rs_check.py"""
import sys
from collections import Counter
from pathlib import Path
root = Path("~/Downloads/GATEKEEPER/gatekeeper").expanduser()
for p in (root, root / "next"):
    sys.path.insert(0, str(p))
from gatekeeper.mv_data import MVSession

RS = {"VISa", "VISam", "CA1", "DG", "LP", "PO"}
cache = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache").expanduser()
rows = []
for p in sorted(cache.glob("*.npz")):
    s = MVSession(p)
    hit = RS & set(s.regions)
    if {"CA1", "DG"} <= hit and ({"LP", "PO"} & hit) and ({"VISa", "VISam"} & hit):
        rows.append((s.lab, s.subject, s.sid, sorted(hit)))
print(f"{len(list(cache.glob('*.npz')))} cached sessions; {len(rows)} look like repeated-site recordings")
for lab, n in Counter(r[0] for r in rows).most_common():
    print(f"  {lab:20s} {n} sessions, {len({r[1] for r in rows if r[0] == lab})} mice")
for reg in sorted(RS):
    print(f"  {reg:6s} present in {sum(reg in r[3] for r in rows)} of them")
