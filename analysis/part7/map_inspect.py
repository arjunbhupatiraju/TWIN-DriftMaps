"""STEP 3 (prep): look inside the Druckmann/Svoboda MAP dataset (DANDI:000363) without downloading it.

The files are huge (raw voltage), so this streams only the small parts we need to see: the list of sessions, and for
one session the units (neurons), brain areas, trials and the behaviour/tracking data. Nothing is analysed.

    pip3 install dandi pynwb remfile h5py
    python3 part7/map_inspect.py
"""
from __future__ import annotations

import sys

DANDISET, VERSION = "000363", "0.231012.2129"


def main():
    from dandi.dandiapi import DandiAPIClient
    import h5py, pynwb, remfile
    with DandiAPIClient() as c:
        ds = c.get_dandiset(DANDISET, VERSION)
        assets = sorted(ds.get_assets(), key=lambda a: a.size)
    print(f"{len(assets)} files; sizes {assets[0].size / 1e9:.1f} to {assets[-1].size / 1e9:.1f} GB")
    for a in assets[:5]:
        print(f"  {a.path}  {a.size / 1e9:.1f} GB")
    a = assets[int(sys.argv[1]) if len(sys.argv) > 1 else 0]
    print(f"\nopening (streamed) {a.path}")
    url = a.get_content_url(follow_redirects=1, strip_query=True)
    h = h5py.File(remfile.File(url), "r")
    nwb = pynwb.NWBHDF5IO(file=h, load_namespaces=True).read()
    print("session:", nwb.session_description, "|", nwb.session_start_time)
    print("acquisition:", list(nwb.acquisition))
    for name, mod in nwb.processing.items():
        print(f"processing/{name}:")
        for k, v in mod.data_interfaces.items():
            sub = list(getattr(v, "time_series", {}) or getattr(v, "spatial_series", {}) or [])
            print(f"   {k} ({type(v).__name__}) {sub[:30]}")
    if nwb.units is not None:
        print(f"\nunits: {len(nwb.units)}; columns: {nwb.units.colnames}")
    if nwb.electrodes is not None:
        E = nwb.electrodes.to_dataframe()
        print(f"electrodes columns: {list(E.columns)}")
        for col in ("location", "brain_regions", "group_name"):
            if col in E:
                print(f"  {col}: {E[col].astype(str).value_counts().head(12).to_dict()}")
    if nwb.trials is not None:
        T = nwb.trials.to_dataframe()
        print(f"\ntrials: {len(T)}; columns: {list(T.columns)}")
        print(T.head(3).to_string())
    for k in ("BehavioralTimeSeries", "BehavioralEvents"):
        for name, mod in nwb.processing.items():
            if k in mod.data_interfaces:
                for ts_name, ts in mod.data_interfaces[k].time_series.items():
                    try:
                        print(f"  {k}/{ts_name}: shape {ts.data.shape}, rate {ts.rate}, unit {ts.unit}")
                    except Exception as e:
                        print(f"  {k}/{ts_name}: ({e})")


if __name__ == "__main__":
    main()
