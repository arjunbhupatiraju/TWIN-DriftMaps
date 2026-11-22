"""Look inside the Zatka-Haas et al. 2021 optogenetic inactivation tables (figshare 13008038) before the causal test.

The dataset page does not say which columns the trial tables have or which file format they use, so this prints
every file's structure (names, sizes, column names, first rows). Paste the output back.

    python3 part3/causal_inspect.py <folder where the figshare download was unzipped>
"""
import sys
from pathlib import Path

import numpy as np


def show_array(name, a, ind="    "):
    a = np.asarray(a)
    print(f"{ind}{name}: shape {a.shape} dtype {a.dtype}", end="")
    if a.dtype.names:
        print(f"  fields {a.dtype.names}")
        for f in a.dtype.names[:40]:
            try:
                v = a[f].ravel()[0] if a.size else None
                show_array(f, v, ind + "    ")
            except Exception as e:
                print(f"{ind}    {f}: ({e})")
    else:
        flat = a.ravel()
        print(f"  first {flat[:6].tolist() if flat.dtype != object else [type(x).__name__ for x in flat[:3]]}")


def inspect_mat(p):
    try:
        from scipy.io import loadmat
        z = loadmat(p, squeeze_me=False, struct_as_record=True)
        for k, v in z.items():
            if not k.startswith("__"):
                show_array(k, v)
        return
    except NotImplementedError:
        pass
    except Exception as e:
        print(f"    scipy could not read it: {type(e).__name__}: {e}")
    try:
        import h5py
        with h5py.File(p, "r") as f:
            f.visititems(lambda n, o: print(f"    {n}: {getattr(o, 'shape', '')} {getattr(o, 'dtype', '')}")
                         if hasattr(o, "shape") else None)
    except Exception as e:
        print(f"    h5py could not read it either: {type(e).__name__}: {e}")


def main(root):
    root = Path(root).expanduser()
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"{len(files)} files under {root}")
    for p in files:
        print(f"{p.relative_to(root)}  ({p.stat().st_size / 1e6:.1f} MB)")
    for p in files:
        rel = str(p.relative_to(root)).lower()
        if "widefield" in rel or "stanmodel" in rel:
            continue                                         # not needed for the causal test
        print(f"\n=== {p.relative_to(root)}")
        suf = p.suffix.lower()
        if suf in (".csv", ".txt", ".tsv"):
            with open(p) as f:
                for i, line in zip(range(4), f):
                    print("    " + line.rstrip()[:400])
        elif suf == ".mat":
            inspect_mat(p)
        elif suf in (".npy", ".npz"):
            z = np.load(p, allow_pickle=False)
            for k in (z.files if hasattr(z, "files") else ["array"]):
                show_array(k, z[k] if hasattr(z, "files") else z)
        elif suf in (".parquet", ".pq", ".feather", ".pkl", ".h5", ".hdf5"):
            import pandas as pd
            try:
                df = {".parquet": pd.read_parquet, ".pq": pd.read_parquet, ".feather": pd.read_feather,
                      ".pkl": pd.read_pickle}.get(suf, pd.read_hdf)(p)
                print(f"    {df.shape}\n    columns {list(df.columns)}\n{df.head(3).to_string()}")
            except Exception as e:
                print(f"    could not read: {e}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/Downloads/MULTIVERSE/zatkahaas")
