"""Figure 1 (overview): TWIN results to date. Every number is a reported, pre-registered result (see the project docs
named in SOURCES); nothing is estimated for the figure."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.labelcolor": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "legend.fontsize": 6.5})

fig = plt.figure(figsize=(7.2, 6.2))
gt = fig.add_gridspec(1, 3, wspace=0.55, left=0.08, right=0.985, top=0.93, bottom=0.6)
gb = fig.add_gridspec(1, 3, wspace=0.6, left=0.15, right=0.985, top=0.43, bottom=0.14)
gs = {(0, i): gt[0, i] for i in range(3)}; gs.update({(1, i): gb[0, i] for i in range(3)})


def letter(ax, s, title):
    ax.text(-0.32, 1.13, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    ax.set_title(title, fontsize=7.5, loc="left", pad=8, color=INK)


# a: drift fools the common tests
ax = fig.add_subplot(gs[0, 0]); letter(ax, "a", "Drift fools the common tests")
groups = ["Encoding", "Commun-\nication", "Prior\ndecoding"]
common, valid = [93, 91, 16.8], [4.9, 7, 3.3]
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w / 2 - 0.01, common, w, color=ORANGE, label="common test", edgecolor="white", linewidth=0.8)
b2 = ax.bar(x + w / 2 + 0.01, valid, w, color=BLUE, label="valid test / TWIN", edgecolor="white", linewidth=0.8)
for bars, vals in ((b1, common), (b2, valid)):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:g}%", ha="center", va="bottom", fontsize=6, color=INK)
ax.axhline(5, ls="--", lw=0.6, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=6); ax.set_ylim(0, 128); ax.set_ylabel("false positives (%)")
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03), columnspacing=0.8, handlelength=1)

# b: risk is predictable before recording
ax = fig.add_subplot(gs[0, 1]); letter(ax, "b", "False-positive risk is predictable")
names = ["Planner", "Twin", "TWIN-Net"]
rs = [0.814, 0.80, 0.78]
y = np.arange(3)[::-1]
ax.barh(y, rs, 0.55, color=BLUE, edgecolor="white", linewidth=0.8)
for yi, v in zip(y, rs):
    ax.text(v + 0.012, yi, f"{v:.2f}", va="center", fontsize=6, color=INK)
ax.axvline(0.896, ls="--", lw=0.6, color=INK2); ax.text(0.886, 2.48, "noise ceiling 0.90", fontsize=5.5, color=INK2, ha="right")
ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xlim(0, 1); ax.set_ylim(-0.6, 2.75)
ax.set_xlabel("r with real IBL false-positive rate")

# c: answerability atlas
ax = fig.add_subplot(gs[0, 2]); letter(ax, "c", "What IBL data can answer")
vars_ = ["Block\nprior", "Stimulus\nside", "Choice", "Feedback"]
shuf, val, ans = [85, 49, 57, 87], [2, 13, 57, 87], [2, 44, 73, 80]
x = np.arange(4); w = 0.26
for off, vals, c, lab in ((-w - 0.01, shuf, ORANGE, "significant, shuffle test"),
                          (0, val, BLUE, "significant, valid test"),
                          (w + 0.01, ans, AQUA, "answerable at r = 0.1")):
    ax.bar(x + off, vals, w, color=c, label=lab, edgecolor="white", linewidth=0.8)
ax.text(0 - w - 0.01, 87, "85", ha="center", fontsize=5.5); ax.text(0, 4, "2", ha="center", fontsize=5.5)
ax.text(w + 0.01, 4, "2", ha="center", fontsize=5.5)
ax.set_xticks(x); ax.set_xticklabels(vars_, fontsize=6); ax.set_ylim(0, 125); ax.set_ylabel("regions (%)")
ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(0, 1.03))

# d: replication prediction across six pre-registered tests
ax = fig.add_subplot(gs[1, 0]); letter(ax, "d", "Predicting replication across labs")
tests = ["Repeated site", "Brain-Wide Map", "New variables", "3 datasets", "IBL → Allen", "Prior map"]
twin = [0.771, 0.748, 0.774, 0.811, 0.897, 0.684]
ci = [(0.713, 0.834), (0.732, 0.765), (0.763, 0.782), (0.774, 0.838), None, (0.670, 0.700)]
pv = [0.645, 0.644, 0.615, 0.490, None, 0.577]
nv = [0.711, 0.639, 0.636, 0.650, None, 0.579]
ds = [None, 0.831, 0.663, 0.750, 0.599, None]
y = np.arange(len(tests))[::-1]
for yi in y:
    ax.axhline(yi, color=GRID, lw=0.5, zorder=0)
ax.axvline(0.5, ls="--", lw=0.6, color=INK2)
for vals, c, m, lab, s in ((pv, GRAY, "s", "original p-value", 16), (nv, YELLOW, "^", "textbook power", 18),
                           (ds, AQUA, "D", "design rule (trained on outcomes)", 14)):
    yy = [yi for yi, v in zip(y, vals) if v is not None]; vv = [v for v in vals if v is not None]
    ax.scatter(vv, yy, s=s, marker=m, color=c, edgecolor="white", linewidth=0.5, label=lab, zorder=3)
for yi, v, c in zip(y, twin, ci):
    if c:
        ax.plot(c, [yi, yi], color=BLUE, lw=1.2, zorder=3)
ax.scatter(twin, y, s=30, color=BLUE, edgecolor="white", linewidth=0.8, label="TWIN (zero-shot)", zorder=4)
ax.set_yticks(y); ax.set_yticklabels(tests, fontsize=6); ax.set_xlim(0.45, 0.95); ax.set_xlabel("AUC")
ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.2), ncol=2, columnspacing=0.8, handletextpad=0.2)

# e: the hidden prior
ax = fig.add_subplot(gs[1, 1]); letter(ax, "e", "The block prior is mostly hidden")
ax.bar([0], [7.1], 0.55, color="#184f95", edgecolor="white")
ax.bar([1], [51.5], 0.55, color="#6da7ec", edgecolor="white")
ax.errorbar([1], [51.5], yerr=[[51.5 - 36.2], [68.1 - 51.5]], color=INK, lw=0.8, capsize=3)
ax.text(0, 9.5, "7%", ha="center", fontsize=6.5); ax.text(1.32, 51.5, "52%\n(36–68)", va="center", fontsize=6)
ax.set_xticks([0, 1]); ax.set_xticklabels(["detected", "estimated\n(TWIN)"], fontsize=6)
ax.set_ylabel("regions (%)"); ax.set_ylim(0, 80); ax.set_xlim(-0.5, 1.75)
ax.text(0.6, 74, "226 regions, 436 sessions", ha="center", fontsize=5.5, color=INK2)

# f: where it hides
ax = fig.add_subplot(gs[1, 2]); letter(ax, "f", "Where the hidden prior lives")
sysn = {"HB": "Hindbrain", "TH": "Thalamus", "Isocortex": "Cortex", "MB": "Midbrain", "CNU": "Striatum/pallidum",
        "CB": "Cerebellum", "HY": "Hypothalamus", "CTXsp": "Cortical subplate", "OLF": "Olfactory", "HPF": "Hippocampal form."}
T = {"CTXsp": (0, 5, 2, 0), "OLF": (0, 3, 4, 2), "HY": (0, 5, 5, 1), "HPF": (0, 2, 5, 5), "CB": (0, 7, 10, 3),
     "CNU": (2, 11, 4, 3), "MB": (6, 11, 8, 0), "TH": (3, 19, 8, 2), "Isocortex": (5, 12, 22, 4), "HB": (0, 21, 23, 3)}
order = sorted(T, key=lambda k: T[k][0] + T[k][1])
cats = [("detected", "#184f95"), ("hidden (likely)", "#6da7ec"), ("unanswerable", "#c7c6c1"), ("likely absent", "#ebe9e4")]
y = np.arange(len(order)); left = np.zeros(len(order))
for j, (lab, c) in enumerate(cats):
    v = np.array([T[k][j] for k in order])
    ax.barh(y, v, 0.7, left=left, color=c, label=lab, edgecolor="white", linewidth=0.8)
    left += v
ax.set_yticks(y); ax.set_yticklabels([sysn[k] for k in order], fontsize=6); ax.set_xlabel("regions")
ax.legend(loc="lower right", fontsize=6, handlelength=1, borderaxespad=0.1)

for ext in ("png", "pdf"):
    fig.savefig(f"/home/claude/out/TWIN_Figure1.{ext}", dpi=300, facecolor="white")
