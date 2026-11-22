"""Summarise the Part 2 real-brain run: false positives (fake variables), power (real variables),
and the parameter-free theory check. Usage: python3 scripts/part2_report.py [rows csv]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
f = sys.argv[1] if len(sys.argv) > 1 else "results/part2_real/rows_explore.csv.gz"
d = pd.read_csv(f); out = Path(f).parent
def md(t):
    try: return t.to_markdown()
    except ImportError: return "```\n" + t.to_string() + "\n```"
M = ["p_naive", "p_shuffle", "p_circ", "p_pseudo", "p_dct_hac", "p_dct_long", "p_dct_pw"]
d["fake"] = d["var"].str.startswith("ctl_"); d["base"] = d["var"].str.replace("ctl_", "", regex=False)
d["Nbin"] = pd.cut(d.n_units, [5, 10, 20, 40, 80, 160, 10000], right=False)
R = {}
lines = [f"# Part 2 real-brain validation ({f})", "", f"{d.sid.nunique()} sessions, {d.subject.nunique()} mice, {len(d)} rows", ""]
for u in sorted(d.units.unique()):
    g = d[d.units == u]
    fp = g[g.fake].groupby("base")[M].apply(lambda x: (x < .05).mean())
    pw = g[~g.fake].groupby("base")[M].apply(lambda x: (x < .05).mean())
    R[f"fp_{u}"] = fp.round(4).to_dict(); R[f"power_{u}"] = pw.round(4).to_dict()
    lines += [f"## Units: {u}", "", "False-positive rate on fake variables (should be 0.05):", "", md(fp.round(3)), "",
              "Share significant on real variables:", "", md(pw.round(3)), ""]
    byN = g[g.fake].groupby("Nbin", observed=True)[M].apply(lambda x: (x < .05).mean())
    lines += ["False-positive rate by number of neurons (all fake variables pooled):", "", md(byN.round(3)), ""]
# parameter-free theory check: predicted vs observed shuffle FP, fake variables
c = d[d.fake & d.pred_fp.notna()].copy()
c["pbin"] = pd.qcut(c.pred_fp, 10, duplicates="drop")
cal = c.groupby("pbin", observed=True).agg(pred=("pred_fp", "mean"), obs=("p_shuffle", lambda x: (x < .05).mean()), n=("pred_fp", "size"))
r = np.corrcoef(cal.pred, cal.obs)[0, 1]
R["theory_check"] = dict(r=float(r), mae=float((cal.pred - cal.obs).abs().mean()), table=cal.reset_index(drop=True).round(4).to_dict("list"))
lines += ["## Parameter-free theory check (no fitting)", "",
          "Each session's shuffle false-positive probability predicted from its measured drift (Sigma) and neuron correlations (R0):", "",
          md(cal.round(3)), "", f"r = {r:.3f}, mean abs error = {(cal.pred - cal.obs).abs().mean():.3f}", ""]
vh = d[d.fake].groupby(["units", "base"]).V_hat.median().unstack(0).round(3)
lines += ["Median measured Bartlett inflation V_hat by fake variable:", "", md(vh), ""]
R["V_hat_median"] = vh.to_dict()
(out / "part2_summary.json").write_text(json.dumps(R, indent=1, default=str))
(out / "PART2_SUMMARY.md").write_text("\n".join(lines))
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
g = d[(d.units == "good") & d.fake]
for m in ["p_shuffle", "p_circ", "p_pseudo", "p_dct_hac", "p_dct_long"]:
    s = g.groupby("Nbin", observed=True).apply(lambda x: (x[m] < .05).mean())
    ax[0].plot([iv.left for iv in s.index], s.values, "o-", label=m[2:])
ax[0].axhline(.05, c="k", ls=":"); ax[0].set_xscale("log"); ax[0].set_xlabel("neurons"); ax[0].set_ylabel("false-positive rate"); ax[0].legend(); ax[0].set_title("Fake variables, good units")
ax[1].plot(cal.pred, cal.obs, "o"); ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set_xlabel("predicted (no fitting)"); ax[1].set_ylabel("observed shuffle FP"); ax[1].set_title(f"Theory check r={r:.2f}")
pw = d[(d.units == "good") & ~d.fake].groupby("base")[["p_pseudo", "p_dct_hac", "p_dct_long", "p_circ"]].apply(lambda x: (x < .05).mean())
pw.plot.bar(ax=ax[2]); ax[2].set_title("Real variables: share significant"); ax[2].set_ylabel("share")
fig.tight_layout(); fig.savefig(out / "part2_figure.png", dpi=130)
print("\n".join(lines)); print(f"\nwrote {out/'PART2_SUMMARY.md'}, part2_summary.json, part2_figure.png")
