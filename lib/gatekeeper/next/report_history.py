"""How many prior regions survive controlling for the last k trials?  Same region rule as verify_numbers.py.

    python3 next/report_history.py results/history
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_numbers import region_p, sig  # noqa: E402


def main(d):
    d = Path(d)
    parts = [pd.read_csv(f) for f in sorted(d.glob("history_rows_*.csv"))]
    if not parts:
        parts = [pd.read_csv(f) for f in d.glob("sessions_*/*.csv") if f.stat().st_size > 1]
    df = pd.concat(parts, ignore_index=True)
    cols = [c for c in ["p_new_k0", "p_new_k1", "p_new_k3", "p_new_k5", "p_findling_k3"] if c in df.columns]
    names = {"p_new_k0": "New test, no history control", "p_new_k1": "New test, controlling last 1 trial",
             "p_new_k3": "New test, controlling last 3 trials", "p_new_k5": "New test, controlling last 5 trials",
             "p_findling_k3": "Findling decoding, controlling last 3 trials"}
    L = [f"# Prior vs recent trials ({df.sid.nunique()} sessions, {df.subject.nunique()} mice)\n",
         "Region rule: >= 2 sessions; Fisher; BH. Fake prior (ctl_bprior) must stay near 0.05 and 0 regions.\n",
         "| Method | Bayes prior: rate | Bayes prior: regions FDR 5% | Fake prior: rate | Fake prior: regions FDR 5% | Replicated (explore & confirm) |",
         "|---|---|---|---|---|---|"]
    k0 = None
    for c in cols:
        r, f = df[df["var"] == "bprior"], df[df["var"] == "ctl_bprior"]
        pr, pf = region_p(r, c), region_p(f, c)
        sr = sig(pr)
        if c == "p_new_k0":
            k0 = sr
        rep = "-"
        if "split" in df and df.split.nunique() == 2:
            a = region_p(r[r.split == "explore"], c); b = region_p(r[r.split == "confirm"], c)
            com = a.index.intersection(b.index)
            rep = f"{len(sig(a[com]) & sig(b[com]))} / {len(com)}"
        L.append(f"| {names[c]} | {(r[c] < .05).mean():.3f} | {len(sr)} / {len(pr)} | {(f[c] < .05).mean():.3f} | {len(sig(pf))} / {len(pf)} | {rep} |")
        if k0 is not None and c != "p_new_k0":
            L.append(f"|   of which also significant without control | | {len(sr & k0)} | | | |")
    text = "\n".join(L)
    (d / "HISTORY_SUMMARY.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/history")
