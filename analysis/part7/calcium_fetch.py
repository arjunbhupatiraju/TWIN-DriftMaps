"""STEP 4, data: download the Stringer, Pachitariu et al. (Science 2019) two-photon recordings of about 10,000 neurons
in mouse visual cortex during spontaneous behaviour, with face-video motion SVDs, running and pupil.
figshare article 6163622 (public). Only the two-photon 'spont_*.mat' files are downloaded (resumable).

    python3 part7/calcium_fetch.py --list       # show the files and sizes first
    python3 part7/calcium_fetch.py
"""
from __future__ import annotations

import argparse, json, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "results_calcium" / "raw"
API = "https://api.figshare.com/v2/articles/6163622/files?page_size=100"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    req = urllib.request.Request(API, headers={"User-Agent": "twin-reanalysis"})
    files = json.load(urllib.request.urlopen(req, timeout=60))
    spont = [f for f in files if f["name"].startswith("spont_") and f["name"].endswith(".mat")]
    for f in files:
        print(f"  {f['name']}  {f['size'] / 1e9:.2f} GB" + ("   <- used" if f in spont else ""))
    print(f"{len(spont)} two-photon spontaneous recordings, {sum(f['size'] for f in spont) / 1e9:.1f} GB in total")
    if a.list:
        return
    RAW.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(spont):
        out = RAW / f["name"]
        if out.exists() and out.stat().st_size == f["size"]:
            print(f"[{i + 1}/{len(spont)}] {f['name']}: skip"); continue
        tmp = out.with_suffix(".part")
        print(f"[{i + 1}/{len(spont)}] {f['name']}: downloading {f['size'] / 1e9:.2f} GB ...", flush=True)
        r = urllib.request.urlopen(urllib.request.Request(f["download_url"], headers={"User-Agent": "twin-reanalysis"}), timeout=120)
        with open(tmp, "wb") as fh:
            while True:
                b = r.read(1 << 22)
                if not b:
                    break
                fh.write(b)
        tmp.rename(out)
        print(f"   done", flush=True)


if __name__ == "__main__":
    main()
