"""Inspect the MATLAB-table file Inactivation_52Coord.mat (v7.3, table stored as an MCOS object).
Prints the column names hidden in the MCOS blob and a summary of every per-trial column. No statistics are computed.

    cd ~/Downloads/MODEL && python3 part8/inspect_zh.py ~/Downloads/MULTIVERSE/Zatka-Hass/OptogeneticInactivation/Inactivation_52Coord.mat
"""
import re, sys
import h5py, numpy as np

h = h5py.File(sys.argv[1], "r")
mc = h["#subsystem#/MCOS"]
names = set()
for r in mc[()].ravel():
    try:
        o = h[r]
    except Exception:
        continue
    if isinstance(o, h5py.Dataset) and o.dtype.kind in "ui" and o.size > 50:
        b = o[()].astype(np.uint8).tobytes()
        names |= {s.decode() for s in re.findall(rb"[A-Za-z_][A-Za-z0-9_]{2,40}", b)}
print("NAMES IN MCOS BLOB:", sorted(names))

print("\nPER-TRIAL COLUMNS (#refs# datasets with a 47002 dimension):")
for k in sorted(h["#refs#"].keys()):
    o = h["#refs#/" + k]
    if not isinstance(o, h5py.Dataset) or 47002 not in o.shape:
        continue
    v = o[()]
    v2 = v.reshape(-1, 47002) if v.shape[-1] == 47002 else v.T.reshape(-1, 47002)
    for j, row in enumerate(v2[:3]):
        row = row.astype(float)
        u = np.unique(row[~np.isnan(row)])
        print(f"  {k}[{j}] shape {o.shape} dtype {o.dtype}: nan {np.isnan(row).mean():.2f}, "
              f"min {np.nanmin(row):.3g}, max {np.nanmax(row):.3g}, n unique {len(u)}, first unique {u[:8]}")
    if v2.shape[0] > 3:
        print(f"  {k}: {v2.shape[0]} rows (probably a time series, e.g. wheel)")
