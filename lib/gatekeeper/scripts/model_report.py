"""Summary of the new model on real data. Usage: python3 scripts/model_report.py [rows csv ...]"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper.multiverse import _bh
files = sys.argv[1:] or ["results/model_real/rows_explore.csv.gz"]
M = ["p_S_pseudo", "p_NEW_pseudo", "p_NEW_imposter", "p_FINAL_pseudo", "p_FINAL_imposter"]
NAMES = {"p_S_pseudo": "standard + pseudo-session (IBL)", "p_S_imposter": "standard + imposter",
         "p_G_imposter": "drift-whitened + imposter", "p_NEW_pseudo": "NEW model + pseudo-session",
         "p_NEW_imposter": "NEW model (imposter, no task model)",
         "p_FINAL_pseudo": "FINAL pooled + pseudo-session", "p_FINAL_imposter": "FINAL pooled (imposter)"}
def md(t):
    try: return t.to_markdown()
    except ImportError: return "```\n" + t.to_string() + "\n```"
def fisher(p): return chi2.sf(-2 * np.log(np.clip(p, 1e-12, 1)).sum(), 2 * len(p))
out_all = {}
for f in files:
    d = pd.read_csv(f); out = Path(f).parent; tag = Path(f).stem.replace("rows_", "").replace(".csv", "")
    d["fake"] = d["var"].str.startswith("ctl_"); d["base"] = d["var"].str.replace("ctl_", "", regex=False)
    L = [f"# New model on real data ({tag})", "", f"{d.sid.nunique()} sessions, {d.subject.nunique()} mice, {len(d)} rows", ""]
    fp = d[d.fake].groupby("base")[M].apply(lambda x: (x < .05).mean()).rename(columns=NAMES)
    pw = d[~d.fake].groupby("base")[M].apply(lambda x: (x < .05).mean()).rename(columns=NAMES)
    L += ["## Session level", "", "False-positive rate, fake variables (target 0.05):", "", md(fp.round(3)), "",
          "Share significant, real variables (power):", "", md(pw.round(3)), ""]
    # region maps
    rows = []
    for var in sorted(d["var"].unique()):
        g = d[d["var"] == var].groupby("region").filter(lambda h: len(h) >= 2)
        row = dict(var=var, regions=g.region.nunique())
        for m in M:
            p = np.array([fisher(h[m].values) for _, h in g.groupby("region")])
            row[NAMES[m]] = int(_bh(p, .05).sum()) if len(p) else 0
        rows.append(row)
    rm = pd.DataFrame(rows).set_index("var")
    L += ["## Brain-wide maps (Fisher across sessions, BH-FDR over regions)", "", md(rm), ""]
    # sessions needed: subsample k sessions per region, share of regions declared, real vs fake
    rng = np.random.default_rng(1); sn = []
    for var in ("prior", "sprior", "ctl_prior", "ctl_sprior", "stim_all", "ctl_stim_all"):
        g = d[d["var"] == var]
        for k in (1, 3, 5, 10, 20):
            res = {m: [] for m in M}
            for reg, h in g.groupby("region"):
                if len(h) < k: continue
                for _ in range(10):
                    s = h.iloc[rng.choice(len(h), k, replace=False)]
                    for m in M: res[m].append(fisher(s[m].values) < .05)
            if res[M[0]]: sn.append(dict(var=var, k=k, **{NAMES[m]: float(np.mean(v)) for m, v in res.items()}))
    sn = pd.DataFrame(sn)
    L += ["## Region detection vs sessions per region", "", md(sn.set_index(["var", "k"]).round(3)), ""]
    (out / f"MODEL_SUMMARY_{tag}.md").write_text("\n".join(L)); print("\n".join(L))
    out_all[tag] = dict(fp=fp.to_dict(), power=pw.to_dict(), maps=rm.to_dict(), sessions=sn.to_dict("list"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    fp.plot.bar(ax=ax[0]); ax[0].axhline(.05, c="k", ls=":"); ax[0].set_title("False positives (fake variables)"); ax[0].legend(fontsize=7)
    pw.plot.bar(ax=ax[1], legend=False); ax[1].set_title("Power (real variables)")
    for v, ls in (("prior", "-"), ("ctl_prior", ":")):
        s = sn[sn["var"] == v]
        for m, c in zip(M, plt.cm.tab10(range(5))): ax[2].plot(s.k, s[NAMES[m]], ls, color=c, marker="o")
    ax[2].set_xlabel("sessions per region"); ax[2].set_title("Block prior: real (solid) vs fake (dotted)")
    fig.tight_layout(); fig.savefig(out / f"model_figure_{tag}.png", dpi=130)
Path(files[0]).parent.joinpath("model_summary.json").write_text(json.dumps(out_all, indent=1, default=str))
