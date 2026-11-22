"""TWIN paper: six main figures at journal depth, built from the pre-registered result files and the IBL cache.

    cd ~/Downloads/MODEL && python3 paper_figs/make_figures.py
    python3 paper_figs/make_figures.py --only 1 6          # selected figures

Every panel is computed from data on disk (result CSVs, and the GATEKEEPER cache for the example recordings).
Panels marked "summary" plot numbers copied from the pre-registered result logs (SUMMARY). A panel whose input is
missing is drawn as an empty box naming the file and the command that makes it; nothing is ever invented.
"""
from __future__ import annotations

import argparse, json, os, re, sys, traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "twin")); sys.path.insert(0, str(ROOT / "part6"))
CACHE = Path(os.environ.get("TWIN_CACHE", "~/Downloads/GATEKEEPER/gatekeeper/data/mv_cache")).expanduser()
GK = Path(os.environ.get("GATEKEEPER_ROOT", "~/Downloads/GATEKEEPER/gatekeeper")).expanduser()

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8984"
VIOLET, MAGENTA, GREEN = "#6250d6", "#e87ba4", "#008300"
DBLUE, LBLUE, LGRAY, XLGRAY, SAND = "#184f95", "#6da7ec", "#c7c6c1", "#ebe9e4", "#d9c7bb"
INK, INK2, GRID, PANEL = "#0b0b0b", "#52514e", "#e6e5e1", "#f4f3f0"
PRED = {"twin3": ("TWIN v3", BLUE), "twin2": ("TWIN v2", BLUE), "twin": ("TWIN v1", BLUE), "design": ("design rule", AQUA),
        "naive": ("textbook power", YELLOW), "p_orig": ("original p", GRAY)}
VARC = {"block": VIOLET, "stimulus": MAGENTA, "choice": GREEN, "feedback": YELLOW}
VARN = {"block": "Block prior", "stimulus": "Stimulus side", "choice": "Choice", "feedback": "Feedback"}
SYS = {"HB": "Hindbrain", "TH": "Thalamus", "Isocortex": "Cortex", "MB": "Midbrain", "CNU": "Striatum/pallidum",
       "CB": "Cerebellum", "HY": "Hypothalamus", "CTXsp": "Cortical subplate", "OLF": "Olfactory", "HPF": "Hippocampal form."}
GRID_R = (0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3)
SUMMARY = {   # copied from the pre-registered result logs (project docs claude/preregistration-*.md and *-results-*.md)
    "auc_tests": [("Repeated site", 566, 0.771, (0.713, 0.834), 0.645, 0.711, None),
                  ("Brain-Wide Map", 9414, 0.748, (0.732, 0.765), 0.644, 0.639, 0.831),
                  ("Unseen variables", 11707, 0.774, (0.763, 0.782), 0.615, 0.636, 0.663),
                  ("IBL + Steinmetz + Allen", 2561, 0.811, (0.774, 0.838), 0.490, 0.650, 0.750),
                  ("IBL → Allen", 448, 0.897, None, None, None, 0.599),
                  ("Prior map", 5336, 0.684, (0.670, 0.700), 0.577, 0.579, None)],
    "datasets": [("IBL Brain-Wide Map", "12 labs, Neuropixels 1.0", 425, 135, "visual decision task with blocks"),
                 ("Steinmetz et al. 2019", "1 lab, Neuropixels 1.0", 21, 8, "two-alternative contrast task"),
                 ("Allen Visual Behavior", "Allen Institute, Neuropixels", 102, 54, "visual change detection")],
    "pooled_brier_gain": [("design rule", 0.022, 0.006, 0.039), ("textbook power", 0.046, 0.028, 0.068),
                          ("original p", 0.030, 0.011, 0.049)],
}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 5.8, "axes.titlesize": 6.8,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
MISSING = []
RNG = np.random.default_rng(7)


# ================================================================ data access
def find(name):
    hits = [p for p in ROOT.rglob(name) if "release" not in p.parts and "paper_figs" not in p.parts]
    if (HERE / "data" / name).exists():
        hits.append(HERE / "data" / name)
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None


def csv(name, how):
    import pandas as pd
    p = find(name)
    if p is None:
        raise FileNotFoundError(f"{name}\n(make it with: {how})")
    return pd.read_csv(p)


def text_file(name, how):
    p = find(name)
    if p is None:
        raise FileNotFoundError(f"{name}\n(make it with: {how})")
    return p.read_text()


def session(sid):
    for p in (GK, GK / "next"):
        sys.path.insert(0, str(p))
    from gatekeeper.mv_data import MVSession
    f = CACHE / f"{sid}.npz"
    if not f.exists():
        raise FileNotFoundError(f"cache session {sid}.npz\n(set TWIN_CACHE to the mv_cache folder)")
    return MVSession(f)


def example(sid, region, window="prior_a", burn=90):
    """Valid trials of one real session-region: X (trials x units), Bayes prior, block probability."""
    from twin.task import BlockTask
    s = session(sid)
    c = np.asarray(s.contrast, float)
    prior = BlockTask().bayes_prior(np.where(c != 0, (c > 0).astype(float), np.nan))
    v = (np.asarray(s.choice) >= 0) & s.valid["stimOn"]; v[:burn] = False
    X = np.asarray(s.X(region, window), float)[v]; X = X[:, X.std(0) > 0]
    pl = np.asarray(s.meta.get("probabilityLeft", np.full(len(c), np.nan)), float)
    return X, prior[v], pl[v] if len(pl) == len(c) else None


# ================================================================ drawing helpers
def label(ax, s, title=None, dx=-0.22, dy=1.13):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top", ha="left")
    if title:
        ax.set_title(title, loc="left", pad=6, color=INK)


def safe(ax, letter, title, fn, dx=-0.22):
    label(ax, letter, title, dx=dx)
    try:
        fn(ax)
    except Exception as e:
        for a in list(ax.figure.axes):
            if getattr(a, "_twin_parent", None) is ax:
                a.remove()
        ax.cla(); label(ax, letter, title, dx=dx)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_on()
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
        ax.text(0.5, 0.5, "missing input:\n" + "\n".join(str(e).split("\n")[:3]), ha="center", va="center", fontsize=5.5,
                color=INK2, transform=ax.transAxes)
        MISSING.append(f"{title}: {e}" if isinstance(e, FileNotFoundError) else f"{title}: {traceback.format_exc(limit=3)}")


def inset(ax, rect):
    a = ax.inset_axes(rect); a._twin_parent = ax
    return a


def note(ax, s, x=0.98, y=0.02, ha="right", va="bottom", **kw):
    ax.text(x, y, s, transform=ax.transAxes, ha=ha, va=va, fontsize=5.3, color=INK2, **kw)


def box(ax, x, y, w, h, text, fc=PANEL, ec=INK2, fs=5.8, color=INK, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02", fc=fc, ec=ec, lw=0.6,
                                transform=ax.transAxes))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, transform=ax.transAxes,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1, color=INK2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", lw=0.7, color=color, shrinkA=0, shrinkB=0, mutation_scale=6))


def blank(ax):
    ax.set_axis_off()


def raincloud(ax, data, colors, names, s=3):
    from scipy.stats import gaussian_kde
    for i, (d, c) in enumerate(zip(data, colors)):
        d = np.asarray(d, float); d = d[np.isfinite(d)]
        if len(d) > 3 and d.std() > 0:
            ys = np.linspace(d.min(), d.max(), 120); w = gaussian_kde(d)(ys); w = w / w.max() * 0.38
            ax.fill_betweenx(ys, i - w - 0.04, i - 0.04, color=c, alpha=0.35, lw=0)
        jit = RNG.uniform(0.04, 0.36, len(d))
        ax.scatter(i + jit, d, s=s, color=c, alpha=0.45, edgecolor="none", rasterized=True)
        if len(d):
            q = np.percentile(d, [25, 50, 75])
            ax.plot([i - 0.04, i - 0.04], [q[0], q[2]], color=INK, lw=1.4, solid_capstyle="butt")
            ax.scatter([i - 0.04], [q[1]], s=10, color="white", edgecolor=INK, lw=0.7, zorder=5)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_xlim(-0.6, len(names) - 0.4)


def reg_scatter(ax, x, y, c=BLUE, xl="", yl="", ident=True, ceiling=None):
    from scipy.stats import pearsonr
    x = np.asarray(x, float); y = np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    ax.scatter(x, y, s=4, color=c, alpha=0.35, edgecolor="none", rasterized=True)
    if x.std() > 0:
        xs = np.linspace(x.min(), x.max(), 50); B = []
        for _ in range(300):
            i = RNG.integers(0, len(x), len(x))
            if x[i].std() > 0:
                B.append(np.polyval(np.polyfit(x[i], y[i], 1), xs))
        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0); ax.plot(xs, np.polyval(np.polyfit(x, y, 1), xs), color=c, lw=1.2)
    if ident:
        m = [min(x.min(), y.min()), max(x.max(), y.max())]; ax.plot(m, m, color=INK2, lw=0.5, ls="--")
    r, p = pearsonr(x, y) if x.std() > 0 and y.std() > 0 else (np.nan, np.nan)
    ax.text(0.04, 0.97, f"r = {r:.2f}, P = {p:.1g}" + (f"\nnoise ceiling {ceiling:.2f}" if ceiling else "") + f"\nn = {len(x)}",
            transform=ax.transAxes, va="top", fontsize=5.6, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel(xl); ax.set_ylabel(yl)


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    return float(roc_auc_score(y, np.asarray(s, float))) if 0 < y.mean() < 1 else np.nan


def pd_concat(x):
    import pandas as pd
    return pd.concat(x)


def roc_band(ax, A, keys, cluster="orig_lab", B=200):
    from sklearn.metrics import roc_curve
    grid = np.linspace(0, 1, 101); groups = [g for _, g in A.groupby(cluster)]
    for k in keys:
        if k not in A:
            continue
        lab, c = PRED[k]; curves, aucs = [], []
        for _ in range(B):
            d = pd_concat([groups[i] for i in RNG.integers(0, len(groups), len(groups))])
            if d.replicated.nunique() < 2:
                continue
            fp, tp, _ = roc_curve(d.replicated.astype(int), d[k]); curves.append(np.interp(grid, fp, tp)); aucs.append(auc(d.replicated, d[k]))
        fp, tp, _ = roc_curve(A.replicated.astype(int), A[k])
        main = k.startswith("twin")
        if curves:
            lo, hi = np.percentile(curves, [2.5, 97.5], axis=0); ax.fill_between(grid, lo, hi, color=c, alpha=0.15 if main else 0.07, lw=0)
        a_lo, a_hi = (np.percentile(aucs, [2.5, 97.5]) if aucs else (np.nan, np.nan))
        ax.plot(fp, tp, color=c, lw=1.6 if main else 1.0, ls="-" if main or k == "design" else "--",
                label=f"{lab}  {auc(A.replicated, A[k]):.2f} [{a_lo:.2f}, {a_hi:.2f}]")
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.7)
    ax.set_xlabel("false-positive rate"); ax.set_ylabel("true-positive rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.legend(loc="lower right", title=f"AUC [95% CI, {cluster.replace('orig_lab', 'lab')} bootstrap]", title_fontsize=5.3,
              handlelength=1.3, fontsize=5.3)


def reliability(ax, A, keys, bins=8):
    edges = np.linspace(0, 1, bins + 1)
    for k in keys:
        lab, c = PRED[k]; idx = np.clip(np.digitize(A[k], edges) - 1, 0, bins - 1)
        px, py, lo, hi = [], [], [], []
        for b in range(bins):
            m = idx == b
            if m.sum() < 15:
                continue
            ys = A.replicated[m].values.astype(float); bs = [RNG.choice(ys, len(ys)).mean() for _ in range(200)]
            px.append(A[k][m].mean()); py.append(ys.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
        ax.fill_between(px, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(px, py, "o-", color=c, lw=1.3, ms=3.5, mec="white", mew=0.5, label=lab)
    ax.plot([0, 1], [0, 1], color=INK2, lw=0.5, ls="--")
    ax.set_xlabel("predicted P(replicate)"); ax.set_ylabel("observed replication rate"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper left")
    h = inset(ax, [0.0, 0.0, 1, 0.14]); h.axis("off"); h.set_xlim(0, 1)
    for k in keys:
        h.hist(A[k], bins=edges, color=PRED[k][1], alpha=0.3)


def add_porig(A):
    from sklearn.linear_model import LogisticRegression
    y = A.replicated.values.astype(float); po = np.zeros(len(A))
    for lab in A.orig_lab.unique():
        tr, te = (A.orig_lab != lab).values, (A.orig_lab == lab).values
        yt = y[tr]
        po[te] = (LogisticRegression().fit(A.loc[tr, ["orig_logp"]].values, yt).predict_proba(A.loc[te, ["orig_logp"]].values)[:, 1]
                  if 0 < yt.mean() < 1 else yt.mean())
    return A.assign(p_orig=po)


def add_design(A):
    import replication as RP
    return A.assign(design=RP._design_predict(A, "PRIMARY"))


def swanson(ax, acr, vals, cmap, vmin, vmax, cbar_label, ticks=None, ticklabels=None):
    from iblatlas.plots import plot_swanson_vector
    from iblatlas.regions import BrainRegions
    plot_swanson_vector(np.asarray(acr), np.asarray(vals, float), ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, br=BrainRegions(),
                        empty_color="#f0efec", linewidth=0.15)
    ax.set_axis_off(); ax.set_aspect("equal")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cb = ax.figure.colorbar(sm, ax=ax, fraction=0.03, pad=0.01, shrink=0.75); cb.set_label(cbar_label, fontsize=5.6); cb.ax.tick_params(labelsize=5.3)
    cb.outline.set_linewidth(0.4)
    if ticks is not None:
        cb.set_ticks(ticks); cb.set_ticklabels(ticklabels)


def cv_predict(X, y, k=5):
    """Cross-validated ridge decoding (as in the prior-map pipeline), returning the held-out predictions."""
    X = np.asarray(X, float); T = len(X); pred = np.zeros(T); edges = np.linspace(0, T, k + 1).astype(int)
    for a, b in zip(edges[:-1], edges[1:]):
        tr = np.r_[0:a, b:T]; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[a:b] - mu) / sd; ym = y[tr].mean(); yt = y[tr] - ym
        U, s, Vt = np.linalg.svd(Xtr, full_matrices=False); Uy = U.T @ yt; n = len(tr); best, bg = None, np.inf
        for al in np.logspace(-3, 2, 9) * n:
            f = s ** 2 / (s ** 2 + al); rss = (yt ** 2).sum() - (Uy ** 2).sum() + (((1 - f) ** 2) * Uy ** 2).sum()
            g = n * rss / (n - f.sum()) ** 2
            if g < bg:
                bg, best = g, Xte @ Vt.T @ ((s / (s ** 2 + al)) * Uy) + ym
        pred[a:b] = best
    return pred


def newfig(h=8.4, rows=3, cols=6, hr=None, hspace=0.8, wspace=1.2):
    f = plt.figure(figsize=(7.2, h))
    gs = f.add_gridspec(rows, cols, height_ratios=hr, hspace=hspace, wspace=wspace, left=0.075, right=0.955, top=0.9, bottom=0.07)
    return f, gs


def save(f, name, title):
    f.text(0.01, 0.985, title, ha="left", va="top", fontsize=8.5, fontweight="bold")
    out = ROOT / "figures"; out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        f.savefig(out / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(f); print(f"wrote {out / name}.pdf", flush=True)


def pick_drift_example():
    V = csv("validate_ibl.csv", "python3 -m twin validate-ibl")
    V = V[(V.N >= 15)] if (V.N >= 15).any() else V
    r = V.sort_values(["empirical_fp", "drift_slow"], ascending=False).iloc[0]
    return r.sid, r.region


# ================================================================ Figure 1
def figure1():
    from twin import BlockTask, PseudoSessions
    from twin.stats import pooled_corr
    f, gs = newfig(8.6, hr=[1, 1, 0.95])
    null = PseudoSessions(BlockTask(), "prior")

    def a(ax):
        blank(ax); rng = np.random.default_rng(2); T = 300
        y = null.draw(T, rng)
        rows = [("Observed label", y, VIOLET), ("Trial shuffle\n(common practice)", rng.permutation(y), ORANGE),
                ("Pseudo-session\n(valid)", null.draw(T, rng), BLUE)]
        for i, (name, v, c) in enumerate(rows):
            a_ = inset(ax, [0.3, 0.72 - i * 0.27, 0.68, 0.19]); a_.plot(v, color=c, lw=0.7); a_.axis("off")
            ax.text(0.0, 0.81 - i * 0.27, name, transform=ax.transAxes, fontsize=5.9, va="center", color=INK)
        ax.text(0.0, 0.02, "Neurons drift slowly. A shuffled label has no slow structure, so its null\n"
                           "is far too narrow; a regenerated label keeps the task's slow structure.", transform=ax.transAxes,
                fontsize=5.4, color=INK2, va="bottom")
    safe(f.add_subplot(gs[0, 0:3]), "a", "Two ways to build a null", a, dx=-0.05)

    def b(ax):
        sid, reg = pick_drift_example(); X, prior, _ = example(sid, reg)
        Z = (X - X.mean(0)) / (X.std(0) + 1e-9); k = np.ones(9) / 9
        S = np.column_stack([np.convolve(z, k, "same") for z in Z.T])
        u, s_, vt = np.linalg.svd(S - S.mean(0), full_matrices=False); order = np.argsort(vt[0])
        im = ax.imshow(S[:, order].T, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2, interpolation="nearest",
                       extent=[0, len(S), 0, S.shape[1]])
        ax.set_xlabel("trial"); ax.set_ylabel("unit (sorted by slow mode)")
        top = inset(ax, [0, 1.02, 1, 0.2]); pc = u[:, 0] * s_[0]
        top.plot(np.arange(len(pc)), (pc - pc.mean()) / pc.std(), color=ORANGE, lw=0.8, label="slow population mode")
        top.plot(np.arange(len(prior)), (prior - prior.mean()) / (prior.std() + 1e-9), color=VIOLET, lw=0.8, label="block prior")
        top.set_xlim(0, len(pc)); top.axis("off")
        top.text(0, 1.02, "slow population mode", color=ORANGE, fontsize=5.4, transform=top.transAxes, va="bottom")
        top.text(0.35, 1.02, "block prior", color=VIOLET, fontsize=5.4, transform=top.transAxes, va="bottom")
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.ax.set_title("z", fontsize=5.5, pad=2); cb.ax.tick_params(labelsize=5)
        note(ax, f"IBL session, {reg}, {X.shape[1]} units, smoothed over 9 trials", y=-0.3, x=0, ha="left")
    safe(f.add_subplot(gs[0, 3:6]), "b", "", b)

    def c(ax):
        sid, reg = pick_drift_example(); X, _, _ = example(sid, reg); T = len(X); rng = np.random.default_rng(5)
        y = null.draw(T, rng)
        Ysh = np.column_stack([rng.permutation(y) for _ in range(1000)]); Yn = null.matrix(T, 1000, rng)
        w = pooled_corr(X, np.column_stack([y, Ysh, Yn]))
        p_sh = (1 + (w[1:1001] >= w[0]).sum()) / 1001; p_ps = (1 + (w[1001:] >= w[0]).sum()) / 1001
        bins = np.linspace(min(w.min(), 0), w.max(), 60)
        ax.hist(w[1:1001], bins, color=ORANGE, alpha=0.75, label=f"shuffle null, P = {p_sh:.3f}")
        ax.hist(w[1001:], bins, color=BLUE, alpha=0.5, label=f"pseudo-session null, P = {p_ps:.2f}")
        ax.axvline(w[0], color=INK, lw=1); ax.text(w[0], ax.get_ylim()[1] * 0.97, " fake label", fontsize=5.5, va="top")
        ax.set_xlabel("pooled correlation statistic"); ax.set_ylabel("count"); ax.legend(loc="upper right", bbox_to_anchor=(1.02, 0.85))
        note(ax, "same recording; label = random task draw,\nindependent of the neurons", y=-0.38, x=0, ha="left")
    safe(f.add_subplot(gs[1, 0:2]), "c", "The shuffle test is fooled", c)

    def d(ax):
        V = csv("validate_ibl.csv", "python3 -m twin validate-ibl")
        data, names, cols = [V.empirical_fp * 100], ["trial\nshuffle"], [ORANGE]
        try:
            N = csv("validate_net.csv", "python3 -m twin validate-net")
            if "shift_fp" in N:
                data.append(N.shift_fp * 100); names.append("linear\nshift"); cols.append(YELLOW)
        except FileNotFoundError:
            pass
        data.append(V.calibrated_fp * 100); names.append("pseudo-\nsession"); cols.append(BLUE)
        names = [f"{n}\n({np.median(dd):.1f}%)" for n, dd in zip(names, data)]
        raincloud(ax, data, cols, names); ax.axhline(5, color=INK2, lw=0.5, ls="--")
        ax.set_ylabel("false positives per session-region (%)"); ax.set_ylim(-3, 103)
        note(ax, f"{len(V)} session-regions, {V.sid.nunique()} sessions;\nfake block-prior labels; (median)", y=-0.45, x=0, ha="left")
    safe(f.add_subplot(gs[1, 2:4]), "d", "Across the brain-wide map", d)

    def e(ax):
        V = csv("validate_ibl.csv", "python3 -m twin validate-ibl")
        reg_scatter(ax, V.drift_slow, V.empirical_fp * 100, c=ORANGE, xl="slow-drift share of variance (fitted)",
                    yl="shuffle-test false positives (%)", ident=False)
        ax.set_ylim(-2, 104)
    safe(f.add_subplot(gs[1, 4:6]), "e", "Risk scales with drift", e)

    def f_(ax):
        C = csv("validate_comm.csv", "python3 -m twin validate-comm"); o = C[C.kind == "other_session"]
        ps = (o.p_common, o.p_shuffle, o.p_twin)
        nm = [f"{n}\n{(p < 0.05).mean():.0%} coupled" for n, p in zip(["random-fold CV\n+ shuffle", "blocked CV\n+ shuffle", "TWIN\n"], ps)]
        raincloud(ax, [-np.log10(np.clip(p, 1e-4, 1)) for p in ps], [ORANGE, YELLOW, BLUE], nm)
        ax.axhline(-np.log10(0.05), color=INK2, lw=0.5, ls="--"); ax.set_ylabel("−log10 P (coupling)")
        note(ax, f"{len(o)} pairs of regions recorded in different sessions (they cannot communicate)", y=-0.4, x=0, ha="left")
    safe(f.add_subplot(gs[2, 0:3]), "f", "Communication between unrelated regions", f_, dx=-0.13)

    def g(ax):
        C = csv("validate_comm.csv", "python3 -m twin validate-comm"); s = C[C.kind == "same_session"]
        ax.hist(s.slow_share.clip(0, 1), np.linspace(0, 1, 21), color=LBLUE, edgecolor="white", lw=0.5)
        med = s.slow_share.median(); ax.axvline(med, color=DBLUE, lw=1)
        ax.text(med, ax.get_ylim()[1] * 0.97, f"median {med:.0%} ", fontsize=5.6, va="top", ha="right" if med > 0.6 else "left")
        ax.set_xlabel("share of excess predictability on slow timescales"); ax.set_ylabel("same-session region pairs")
        ins = inset(ax, [0.24, 0.5, 0.24, 0.45])
        cats = [("common test", (s.p_common < .05).mean(), ORANGE), ("TWIN", (s.p_twin < .05).mean(), BLUE),
                ("slow only", ((s.p_twin < .05) & (s.p_slow < .05) & (s.p_fast >= .05)).mean(), LBLUE),
                ("fast", (s.p_fast < .05).mean(), DBLUE)]
        ins.barh(range(4)[::-1], [c[1] * 100 for c in cats], color=[c[2] for c in cats], height=0.7)
        ins.set_yticks(range(4)[::-1]); ins.set_yticklabels([c[0] for c in cats], fontsize=5); ins.tick_params(labelsize=5)
        ins.set_xlabel("pairs coupled (%)", fontsize=5); ins.set_xlim(0, 100)
    safe(f.add_subplot(gs[2, 3:6]), "g", "Real coupling is mostly slow", g)
    save(f, "Figure1", "Fig. 1 | Slow drift creates false discoveries in standard analyses of neural recordings")


# ================================================================ Figure 2
def figure2():
    from twin import DriftModel
    from twin.drift import phase_twin
    f, gs = newfig(8.6, hr=[0.9, 1, 1])

    def a(ax):
        blank(ax)
        box(ax, 0.0, 0.55, 0.17, 0.32, "Recording\ntrials × neurons")
        box(ax, 0.22, 0.55, 0.2, 0.32, "Fit drift model\ntimescale ladder,\nshared + private")
        box(ax, 0.47, 0.55, 0.19, 0.32, "Digital twins\nsame drift and\ncorrelations, no code", fc="#e3eefb", ec=BLUE)
        for x0, x1 in ((0.17, 0.22), (0.42, 0.47)):
            arrow(ax, x0, 0.71, x1, 0.71)
        outs = ["false-positive risk\nof any test", "power and\nsessions needed", "P(another lab\nreplicates)"]
        for i, t in enumerate(outs):
            box(ax, 0.73, 0.74 - i * 0.27, 0.25, 0.21, t, fc="white", ec=BLUE, color=DBLUE)
            arrow(ax, 0.66, 0.71, 0.73, 0.845 - i * 0.27, color=BLUE)
        ax.text(0.0, 0.42, "Planted codes: a new label is planted into a random\n20% of the real neurons at a chosen per-neuron\n"
                          "correlation r, so detection is measured on the\nrecording itself.",
                transform=ax.transAxes, fontsize=5.4, color=INK2, va="top")
    safe(f.add_subplot(gs[0, 0:4]), "a", "How TWIN works", a, dx=-0.03)

    def b(ax):
        sid, reg = pick_drift_example(); X, _, _ = example(sid, reg); rng = np.random.default_rng(3)
        def acf(M, L=60):
            Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
            return np.array([np.nanmean([np.corrcoef(z[:-l], z[l:])[0, 1] for z in Z.T]) if l else 1.0 for l in range(L)])
        lags = np.arange(60)
        ax.plot(lags, acf(X), color=INK, lw=1.4, label="real neurons")
        tw = np.array([acf(phase_twin(X, rng)) for _ in range(5)])
        ax.fill_between(lags, tw.min(0), tw.max(0), color=BLUE, alpha=0.25, lw=0); ax.plot(lags, tw.mean(0), color=BLUE, lw=1, label="phase twins")
        try:
            dm = DriftModel.fit(X); sim = np.array([acf(dm.simulate(len(X), X.shape[1], rng)) for _ in range(5)])
            ax.plot(lags, sim.mean(0), color=AQUA, lw=1, ls="--", label="fitted drift model")
        except Exception:
            pass
        ax.set_xlabel("lag (trials)"); ax.set_ylabel("autocorrelation"); ax.legend(loc="upper right"); ax.set_xlim(1, 59)
        note(ax, f"{reg}, {X.shape[1]} units", y=0.55)
    safe(f.add_subplot(gs[0, 4:6]), "b", "Twins keep the slow structure", b)

    def scat(col, src, how):
        def fn(ax):
            V = csv(src, how)
            ceil = np.corrcoef(V.empirical_fp, V.empirical_fp_repeat)[0, 1] if "empirical_fp_repeat" in V else None
            reg_scatter(ax, V[col] * 100, V.empirical_fp * 100, c=BLUE, xl="predicted false positives (%)",
                        yl="observed false positives (%)", ceiling=ceil)
            ax.set_xlim(-2, 104); ax.set_ylim(-2, 104)
        return fn
    safe(f.add_subplot(gs[1, 0:2]), "c", "Simulation planner", scat("planner_v3b_fp", "validate_ibl.csv", "python3 -m twin validate-ibl"))
    safe(f.add_subplot(gs[1, 2:4]), "d", "Nonparametric twins", scat("twin_fp", "validate_ibl.csv", "python3 -m twin validate-ibl"))
    safe(f.add_subplot(gs[1, 4:6]), "e", "TWIN-Net (6 ms per region)", scat("net_fp_shuffle", "validate_net.csv", "python3 -m twin validate-net"))

    def bench_heat(ax, cols, names, rsel, foot, cmap):
        B = csv("bench_results.csv", "python3 -m twin.bench 60"); Bs = B[(B.r > 0) == rsel].reset_index(drop=True)
        M = Bs[cols].values.T * 100
        ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.2, color="white" if M[i, j] > 55 else INK)
        sc = [("prior" if "block" in l else "latent" if "latent" in l else "learning") + ("\n+trend" if "trend" in d else "")
              for d, l in zip(Bs.drift, Bs.label)]
        ax.set_xticks(range(len(sc))); ax.set_xticklabels(sc, fontsize=5.2); ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        note(ax, foot, y=-0.36, x=0, ha="left")

    safe(f.add_subplot(gs[2, 0:2]), "f", "Benchmark: false positives (%)",
         lambda ax: bench_heat(ax, ["1s shuffle", "1s pseudo (oracle)", "1s linear shift", "1s twin"],
                               ["shuffle", "pseudo", "shift", "twin"], False, "synthetic, 1 session, no code", "Oranges"))
    safe(f.add_subplot(gs[2, 2:4]), "g", "Benchmark: power (%)",
         lambda ax: bench_heat(ax, ["10s pseudo (oracle)", "10s linear shift", "10s twin", "10s session perm"],
                               ["pseudo", "shift", "twin", "sess. perm."], True, "synthetic, 10 sessions, r = 0.15", "Blues"))

    def h(ax):
        from twin import BlockTask, PseudoSessions, plan
        Ts, Ss = (300, 600, 1000), (1, 3, 5, 10, 20)
        rows = plan(DriftModel(slow_frac=0.35, phi=0.97, shared=0.35, inst_corr=0.03), PseudoSessions(BlockTask(), "prior"),
                    T_grid=Ts, N_grid=(30,), S_grid=Ss, r=0.1, n=20, n_null=60, seed=1)
        M = np.array([[next(r["power_calibrated"] for r in rows if r["T"] == T and r["sessions"] == S) for S in Ss] for T in Ts])
        FP = np.array([[next(r["fp_shuffle"] for r in rows if r["T"] == T and r["sessions"] == S) for S in Ss] for T in Ts])
        ax.imshow(M * 100, origin="lower", cmap="Blues", vmin=0, vmax=100, aspect="auto")
        for i in range(len(Ts)):
            for j in range(len(Ss)):
                ax.text(j, i, f"{M[i, j]:.0%}\n({FP[i, j]:.0%})", ha="center", va="center", fontsize=4.8, color="white" if M[i, j] > 0.55 else INK)
        if M.max() >= 0.8 > M.min():
            ax.contour(M, levels=[0.8], colors=[DBLUE], linewidths=1)
        ax.set_xticks(range(len(Ss))); ax.set_xticklabels(Ss); ax.set_yticks(range(len(Ts))); ax.set_yticklabels(Ts)
        ax.set_xlabel("sessions"); ax.set_ylabel("trials per session"); ax.tick_params(length=0)
        note(ax, "power (shuffle false positives); prior, r = 0.1, 30 units", y=-0.36, x=0, ha="left")
    safe(f.add_subplot(gs[2, 4:6]), "h", "Planning an experiment", h)
    save(f, "Figure2", "Fig. 2 | Digital twins predict false-positive risk and power before data are collected")


# ================================================================ Figure 3
def figure3():
    f, gs = newfig(8.4, hr=[0.8, 0.95, 1.05])

    def atlas():
        return csv("atlas.csv", "python3 part6/atlas.py summarize")

    def flat(var):
        def fn(ax):
            A = atlas(); A = A[A.variable == var]
            need = A["sessions_needed_r0.1"].replace(np.inf, 80).clip(1, 80)
            try:
                swanson(ax, A.region.values, np.log10(need.values), "magma_r", 0, np.log10(80), "sessions needed (80% power, r = 0.1)",
                        ticks=np.log10([1, 3, 10, 30, 80]), ticklabels=["1", "3", "10", "30", ">60"])
            except Exception as e:
                raise FileNotFoundError(f"Swanson flatmap failed ({type(e).__name__})\n(pip3 install iblatlas; needs internet once)")
            note(ax, f"answerable with the sessions recorded: {(A['sessions_needed_r0.1'] <= A.sessions_available).mean():.0%} of {len(A)} regions",
                 y=-0.12, x=0, ha="left")
        return fn
    safe(f.add_subplot(gs[0, 0:3]), "a", "Block prior: sessions needed per region", flat("block"), dx=-0.05)
    safe(f.add_subplot(gs[0, 3:6]), "b", "Choice: sessions needed per region", flat("choice"), dx=-0.05)

    def c(ax):
        blank(ax); D = csv("atlas_rows.csv", "python3 part6/atlas.py run"); vs = [v for v in VARN if v in set(D.variable)]
        for i, v in enumerate(vs):
            a_ = inset(ax, [i * 0.255, 0, 0.215, 0.86]); g = D[D.variable == v]; per_reg = {}
            for reg, gg in g.groupby("region"):
                cv = [json.loads(x) for x in gg.power_curve]; rs = sorted(float(k) for k in cv[0])
                per_reg[reg] = [np.mean([np.mean(np.asarray(x[str(r)]) < 0.05) for x in cv]) for r in rs]
            for pw in list(per_reg.values())[:200]:
                a_.plot(rs, pw, color=VARC[v], lw=0.3, alpha=0.25)
            a_.plot(rs, np.median(list(per_reg.values()), 0), color=VARC[v], lw=1.6)
            a_.axhline(0.8, color=INK2, lw=0.5, ls="--"); a_.set_ylim(0, 1); a_.set_xlim(0, 0.3)
            a_.set_title(f"{VARN[v]} ({len(per_reg)} regions)", fontsize=5.8, color=INK, pad=2)
            a_.set_xticks([0, 0.1, 0.2, 0.3]); a_.tick_params(labelsize=5)
            if i == 0:
                a_.set_ylabel("single-session power", fontsize=5.8)
            else:
                a_.set_yticklabels([])
            a_.set_xlabel("planted r", fontsize=5.5)
    safe(f.add_subplot(gs[1, 0:6]), "c", "Detection power of every region, from codes planted in its own neurons (thin: regions; thick: median)", c, dx=-0.03)

    def d(ax):
        A = atlas()
        ax.fill_between([0.8, 40], [0.8, 40], [100, 100], color=XLGRAY, alpha=0.6, lw=0, zorder=0)
        for v in [v for v in VARN if v in set(A.variable)]:
            g = A[A.variable == v]; need = g["sessions_needed_r0.1"].replace(np.inf, 90)
            ax.scatter(g.sessions_available * np.exp(RNG.uniform(-0.06, 0.06, len(g))), need, s=6, color=VARC[v], alpha=0.6,
                       edgecolor="none", label=f"{VARN[v]} ({(g['sessions_needed_r0.1'] <= g.sessions_available).mean():.0%})")
        ax.plot([0.8, 90], [0.8, 90], color=INK2, lw=0.5, ls="--"); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(0.8, 40); ax.set_ylim(0.8, 100); ax.text(0.85, 93, ">60", fontsize=5, color=INK2, va="bottom")
        ax.set_xlabel("sessions recorded"); ax.set_ylabel("sessions needed (80% power, r = 0.1)")
        ans = ", ".join(f"{VARN[v].split()[0].lower()} {(A[A.variable == v]['sessions_needed_r0.1'] <= A[A.variable == v].sessions_available).mean():.0%}"
                        for v in VARN if v in set(A.variable))
        note(ax, f"answerable: {ans}", y=-0.3, x=0, ha="left", fontsize=4.9) if False else ax.text(0, -0.3, f"answerable: {ans}", transform=ax.transAxes, fontsize=4.9, color=INK2)
        ax.text(0.03, 0.62, "unanswerable\n(needed > recorded)", fontsize=5.2, color=INK2, transform=ax.transAxes, zorder=6,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    safe(f.add_subplot(gs[2, 0:2]), "d", "Recorded vs needed", d)

    def e(ax):
        A = atlas()
        for v in [v for v in VARN if v in set(A.variable)]:
            g = A[A.variable == v]
            ax.scatter(g.sig_shuffle * 100 + RNG.uniform(-1, 1, len(g)), g.sig_valid * 100 + RNG.uniform(-1, 1, len(g)), s=6,
                       color=VARC[v], alpha=0.6, edgecolor="none", label=VARN[v])
        ax.plot([0, 100], [0, 100], color=INK2, lw=0.5, ls="--"); ax.set_xlim(-3, 103); ax.set_ylim(-3, 103)
        ax.set_xlabel("sessions significant, shuffle test (%)"); ax.set_ylabel("sessions significant, valid test (%)")
        ax.legend(loc="upper left", fontsize=5.2)
        ax.annotate("drift-driven\n'discoveries'", xy=(88, 4), xytext=(8, 55), fontsize=5.5, color=INK2,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2))
    safe(f.add_subplot(gs[2, 2:4]), "e", "Each point is a region", e)

    def f_(ax):
        import pandas as pd
        parts = [pd.read_csv(p, usecols=["variable", "p_shuffle", "p_valid", "drift_fp_shuffle"]) for p in
                 (find(n) for n in ("atlas_rows.csv", "atlas_rows_new.csv", "atlas_rows_set3.csv")) if p is not None]
        if not parts:
            raise FileNotFoundError("atlas_rows*.csv\n(python3 part6/atlas.py run)")
        D = pd.concat(parts)
        S = D.groupby("variable").agg(sh=("p_shuffle", lambda x: (x < .05).mean() * 100), va=("p_valid", lambda x: (x < .05).mean() * 100),
                                      risk=("drift_fp_shuffle", "mean")).sort_values("va")
        y = np.arange(len(S))
        ax.hlines(y, S.va, S.sh, color=LGRAY, lw=1.6)
        ax.scatter(S.sh, y, s=16, color=ORANGE, zorder=3, label="shuffle test", edgecolor="white", lw=0.4)
        ax.scatter(S.va, y, s=16, color=BLUE, zorder=3, label="valid test", edgecolor="white", lw=0.4)
        ax.set_yticks(y); ax.set_yticklabels([s.replace("_", " ") for s in S.index], fontsize=5.6)
        ax.set_xlabel("sessions significant (%)"); ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=5.2); ax.set_xlim(0, 112)
        for yi, r in zip(y, S.risk):
            ax.text(112, yi, f"{r:.0%}", fontsize=4.8, color=INK2, va="center", ha="right")
        ax.text(112, len(S) - 0.3, "drift\nrisk", fontsize=4.8, color=INK2, ha="right", va="bottom")
    axf = f.add_subplot(gs[2, 4:6]); pos = axf.get_position(); axf.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])
    safe(axf, "f", "All 11 task variables", f_, dx=-0.6)
    save(f, "Figure3", "Fig. 3 | An answerability atlas: which questions the IBL Brain-Wide Map can and cannot settle")


# ================================================================ Figure 4
def figure4():
    f, gs = newfig(8.6, hr=[0.95, 1, 1])

    def a(ax):
        blank(ax)
        D = csv("atlas_rows.csv", "python3 part6/atlas.py run"); D = D[D.variable == "choice"]
        reg = D.groupby("region").lab.nunique().idxmax(); g = D[D.region == reg]
        labs = g.lab.value_counts().index[:2]; A_, B_ = g[g.lab == labs[0]].iloc[0], g[g.lab == labs[1]].iloc[0]
        ca, cb = json.loads(A_.power_curve), json.loads(B_.power_curve)
        rs = [float(k) for k in sorted(ca, key=float)]
        u = np.log10(max(A_.p_valid, 1e-4)); kern = lambda z: np.mean(np.exp(-0.5 * ((u - np.log10(np.clip(z, 1e-4, 1))) / 0.35) ** 2)) + 1e-6
        lik = np.array([kern(RNG.random(400))] + [kern(np.asarray(ca[str(r)])) for r in rs])
        post = lik / lik.sum(); powb = np.array([0.05] + [np.mean(np.asarray(cb[str(r)]) < 0.05) for r in rs])
        grid0 = [0.0] + rs; pr = float(np.sum(post * powb))
        def p1(q):
            q.plot(rs, [np.mean(np.asarray(ca[str(r)]) < .05) for r in rs], "o-", color=DBLUE, ms=2, lw=1); q.set_ylim(0, 1); q.set_ylabel("detected", fontsize=5)
        def p2(q):
            q.bar(range(len(post)), post, color=LBLUE); q.set_xticks([0, len(post) - 1]); q.set_xticklabels([0, rs[-1]]); q.set_ylabel("posterior", fontsize=5)
        def p3(q):
            q.plot(grid0, powb, "o-", color=AQUA, ms=2, lw=1); q.set_ylim(0, 1); q.set_ylabel("power", fontsize=5)
        panels = [("Lab A: detection curve\n(codes planted in its neurons)", p1),
                  (f"Posterior over code size\n(Lab A observed P = {A_.p_valid:.2g})", p2),
                  ("Lab B: power at each r\n(its own sessions)", p3),
                  ("P(Lab B replicates)\n= Σ posterior × power", None)]
        for i, (t, draw) in enumerate(panels):
            x0 = i * 0.26
            ax.text(x0 + 0.1, 1.02, t, transform=ax.transAxes, ha="center", va="top", fontsize=5.6)
            if draw:
                q = inset(ax, [x0 + 0.03, 0.2, 0.16, 0.45]); draw(q); q.tick_params(labelsize=4.8); q.set_xlabel("code size r", fontsize=5)
            else:
                box(ax, x0 + 0.01, 0.2, 0.2, 0.4, f"{pr:.0%}", fc="#e3eefb", ec=BLUE, fs=11, color=DBLUE, bold=True)
            if i < 3:
                arrow(ax, x0 + 0.205, 0.4, x0 + 0.245, 0.4)
        note(ax, f"worked example, two real labs ({labs[0]} → {labs[1]}), region {reg}, choice", y=-0.12, x=0, ha="left")
    safe(f.add_subplot(gs[0, 0:6]), "a", "How TWIN predicts replication without ever seeing one", a, dx=-0.03)

    def b(ax):
        A = csv("confirm_attempts.csv", "python3 part6/replication.py confirm"); A = add_design(add_porig(A[A.orig_sig].reset_index(drop=True)))
        roc_band(ax, A, ["twin", "design", "naive", "p_orig"]); note(ax, f"{len(A)} attempts", y=0.47)
    safe(f.add_subplot(gs[1, 0:3]), "b", "Brain-Wide Map: confirmation", b, dx=-0.13)

    def c(ax):
        A = csv("v2_attempts_new_variables.csv", "python3 part6/replication.py confirm2")
        A = add_design(add_porig(A[A.orig_sig].reset_index(drop=True)))
        roc_band(ax, A, ["twin2", "design", "naive", "p_orig"]); note(ax, f"{len(A)} attempts, 4 unseen variables", y=0.47)
    safe(f.add_subplot(gs[1, 3:6]), "c", "Unseen kinds of findings (zero-shot)", c, dx=-0.13)

    def d(ax):
        A = csv("v2_attempts_new_variables.csv", "python3 part6/replication.py confirm2")
        A = add_design(A[A.orig_sig].reset_index(drop=True)); reliability(ax, A, ["twin2", "design"])
    safe(f.add_subplot(gs[2, 0:2]), "d", "Calibration (unseen variables)", d)

    def e(ax):
        import pandas as pd
        parts = []
        for name, key in (("confirm_attempts.csv", "twin"), ("v2_attempts_new_variables.csv", "twin2")):
            p = find(name)
            if p is not None:
                A = pd.read_csv(p); A = add_porig(A[A.orig_sig].reset_index(drop=True))
                for v, g in A.groupby("variable"):
                    if g.replicated.nunique() == 2 and len(g) >= 30:
                        parts.append((v, auc(g.replicated, g[key]), auc(g.replicated, g.p_orig), auc(g.replicated, g.naive)))
        if not parts:
            raise FileNotFoundError("confirm_attempts.csv\n(python3 part6/replication.py confirm)")
        parts.sort(key=lambda t: t[1]); y = np.arange(len(parts))
        ax.hlines(y, [min(p[1:]) for p in parts], [max(p[1:]) for p in parts], color=LGRAY, lw=1.2)
        ax.scatter([p[2] for p in parts], y, s=12, marker="s", color=GRAY, label="original p", zorder=3)
        ax.scatter([p[3] for p in parts], y, s=12, marker="^", color=YELLOW, label="textbook power", zorder=3)
        ax.scatter([p[1] for p in parts], y, s=20, color=BLUE, label="TWIN", zorder=4, edgecolor="white", lw=0.5)
        ax.axvline(0.5, color=INK2, lw=0.5, ls="--"); ax.set_yticks(y); ax.set_yticklabels([p[0].replace("_", " ") for p in parts], fontsize=5.6)
        ax.set_xlabel("AUC within the variable"); ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3, fontsize=5, columnspacing=0.5, handletextpad=0.1)
    safe(f.add_subplot(gs[2, 2:4]), "e", "Within each finding type", e, dx=-0.4)

    def f_(ax):
        rows = SUMMARY["auc_tests"]; y = np.arange(len(rows))[::-1]
        for yi in y:
            ax.axhline(yi, color=GRID, lw=0.4, zorder=0)
        ax.axvline(0.5, ls="--", lw=0.5, color=INK2)
        for j, (c, m, lab) in enumerate(((GRAY, "s", "original p"), (YELLOW, "^", "textbook power"), (AQUA, "D", "design rule"))):
            pts = [(r[4 + j], yi) for r, yi in zip(rows, y) if r[4 + j] is not None]
            ax.scatter(*zip(*pts), s=12, marker=m, color=c, edgecolor="white", lw=0.4, label=lab, zorder=3)
        for r, yi in zip(rows, y):
            if r[3]:
                ax.plot(r[3], [yi, yi], color=BLUE, lw=1.2, zorder=3)
        ax.scatter([r[2] for r in rows], y, s=22, color=BLUE, edgecolor="white", lw=0.6, label="TWIN", zorder=4)
        ax.set_yticks(y); ax.set_yticklabels([f"{r[0]}\n(n = {r[1]:,})" for r in rows], fontsize=4.9); ax.set_xlim(0.45, 0.95)
        ax.set_xlabel("AUC"); ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.22), ncol=2, fontsize=5.2, columnspacing=0.5)
    safe(f.add_subplot(gs[2, 4:6]), "f", "Six pre-registered tests (summary)", f_, dx=-0.62)
    save(f, "Figure4", "Fig. 4 | TWIN predicts which findings another lab will reproduce, without training on any replication")


# ================================================================ Figure 5
def figure5():
    f, gs = newfig(8.6, hr=[0.9, 1, 1])
    EXT = {"steinmetz2019": "Steinmetz", "allen_vbn": "Allen"}
    cache = {}

    def pooled():
        if "A" not in cache:
            A = csv("pooled_attempts.csv", "python3 part6/replication.py confirm4")
            cache["A"] = add_design(add_porig(A[A.orig_sig].reset_index(drop=True)))
        return cache["A"]

    def a(ax):
        blank(ax)
        for i, (name, rig, s, m, task) in enumerate(SUMMARY["datasets"]):
            x0 = i * 0.34
            box(ax, x0, 0.12, 0.3, 0.78, "", fc=PANEL)
            ax.text(x0 + 0.15, 0.8, name, transform=ax.transAxes, ha="center", fontsize=6.2, fontweight="bold")
            ax.text(x0 + 0.15, 0.66, rig, transform=ax.transAxes, ha="center", fontsize=5.4, color=INK2)
            ax.text(x0 + 0.15, 0.42, f"{s} sessions\n{m} mice", transform=ax.transAxes, ha="center", fontsize=7, color=DBLUE)
            ax.text(x0 + 0.15, 0.2, task, transform=ax.transAxes, ha="center", fontsize=5.2, color=INK2)
        ax.text(0.5, 0.0, "different labs, rigs, spike sorting and tasks; same TWIN model, never refitted", transform=ax.transAxes,
                ha="center", fontsize=5.4, color=INK2)
    safe(f.add_subplot(gs[0, 0:4]), "a", "Three institutions", a, dx=-0.03)

    def b(ax):
        A = pooled(); grp = lambda lab: EXT.get(lab, "IBL")
        A = A.assign(o=A.orig_lab.map(grp), r=A.rep_lab.map(grp)); names = ["IBL", "Steinmetz", "Allen"]
        M = np.full((3, 3), np.nan); Dm = np.full((3, 3), np.nan)
        for i, o in enumerate(names):
            for j, r in enumerate(names):
                g = A[(A.o == o) & (A.r == r)]
                if len(g) >= 20 and g.replicated.nunique() == 2:
                    M[i, j] = auc(g.replicated, g.twin2); Dm[i, j] = auc(g.replicated, g.design)
        ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1, aspect="auto")
        for i in range(3):
            for j in range(3):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}\n({Dm[i, j]:.2f})", ha="center", va="center", fontsize=5.3, color="white" if M[i, j] > 0.8 else INK)
        ax.set_xticks(range(3)); ax.set_xticklabels(names); ax.set_yticks(range(3)); ax.set_yticklabels(names)
        ax.set_xlabel("replicating dataset"); ax.set_ylabel("original dataset"); ax.tick_params(length=0)
        note(ax, "TWIN v2 AUC (design rule); blank = too few attempts", y=-0.38, x=0, ha="left")
    safe(f.add_subplot(gs[0, 4:6]), "b", "Every direction", b)

    safe(f.add_subplot(gs[1, 0:3]), "c", "Pooled cross-dataset test", lambda ax: roc_band(ax, pooled(), ["twin2", "design", "naive", "p_orig"], cluster="region"), dx=-0.13)
    safe(f.add_subplot(gs[1, 3:6]), "d", "Calibration across institutions", lambda ax: reliability(ax, pooled(), ["twin2", "design"]), dx=-0.13)

    def e(ax):
        A = pooled(); R = A.groupby("region").agg(pred=("twin2", "mean"), obs=("replicated", "mean"), n=("replicated", "size"))
        R = R[R.n >= 5]
        ax.scatter(R.pred, R.obs, s=np.clip(R.n, 5, 80) * 0.5, color=BLUE, alpha=0.5, edgecolor="white", lw=0.3)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.5, ls="--"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        r = np.corrcoef(R.pred, R.obs)[0, 1]; ax.text(0.04, 0.96, f"r = {r:.2f}\n{len(R)} regions (size = attempts)", transform=ax.transAxes, va="top", fontsize=5.6)
        ax.set_xlabel("mean predicted P(replicate)"); ax.set_ylabel("observed replication rate")
    safe(f.add_subplot(gs[2, 0:2]), "e", "Region by region", e)

    def f_(ax):
        G = SUMMARY["pooled_brier_gain"]; y = np.arange(len(G))[::-1]
        ax.errorbar([g[1] for g in G], y, xerr=[[g[1] - g[2] for g in G], [g[3] - g[1] for g in G]], fmt="o", color=BLUE, ms=4,
                    capsize=2, lw=1, mec="white")
        ax.axvline(0, color=INK2, lw=0.5, ls="--"); ax.set_yticks(y); ax.set_yticklabels(["vs " + g[0] for g in G])
        ax.set_xlabel("Brier improvement of TWIN v2"); ax.set_ylim(-0.6, len(G) - 0.4)
        note(ax, "summary; 95% CI,\nregion bootstrap", y=0.02)
    safe(f.add_subplot(gs[2, 2:4]), "f", "Better than every alternative", f_, dx=-0.45)

    def g(ax):
        A = pooled(); vs = [v for v in ("stimulus", "choice", "feedback", "prev_choice") if v in set(A.variable)]
        y = np.arange(len(vs))
        tw = [auc(A[A.variable == v].replicated, A[A.variable == v].twin2) for v in vs]
        ds = [auc(A[A.variable == v].replicated, A[A.variable == v].design) for v in vs]
        ax.hlines(y, ds, tw, color=LGRAY, lw=1.4)
        ax.scatter(ds, y, s=16, marker="D", color=AQUA, label="design rule", zorder=3); ax.scatter(tw, y, s=22, color=BLUE, label="TWIN v2", zorder=4)
        ax.axvline(0.5, color=INK2, lw=0.5, ls="--"); ax.set_yticks(y); ax.set_yticklabels([v.replace("_", " ") for v in vs])
        ax.set_xlabel("AUC"); ax.legend(loc="lower right", fontsize=5.2)
    safe(f.add_subplot(gs[2, 4:6]), "g", "By kind of finding", g, dx=-0.4)
    save(f, "Figure5", "Fig. 5 | Replication prediction transfers across institutions, tasks and pipelines")


# ================================================================ Figure 6
def figure6():
    f, gs = newfig(9.6, rows=4, hr=[0.85, 0.95, 0.9, 1.15], hspace=0.85)

    def rows():
        return csv("prior_rows.csv", "python3 part6/prior_map.py run")

    def table():
        T = csv("prior_map_figure_table.csv", "python3 part6/prior_followup.py figure")
        return T.set_index(T.columns[0])

    def a(ax):
        blank(ax)
        t = inset(ax, [0.02, 0.6, 0.96, 0.22])
        t.axvspan(-0.6, -0.1, color=LBLUE, alpha=0.5); t.axvline(0, color=INK, lw=0.8)
        t.text(-0.35, 0.5, "decoding\nwindow", ha="center", va="center", fontsize=5.3); t.text(0.03, 0.5, "stimulus", fontsize=5.3, va="center")
        t.set_xlim(-0.9, 0.5); t.set_yticks([]); t.set_xlabel("time from stimulus (s)", fontsize=5.3); t.tick_params(labelsize=5)
        for s in ("left", "top", "right"):
            t.spines[s].set_visible(False)
        ax.text(0.0, 0.33, "Target: Bayes-optimal block prior\nDecoder: ridge, 5 contiguous folds\n"
                           "Valid null: 200 pseudo-sessions\nPower: new priors planted in 20% of units", transform=ax.transAxes,
                fontsize=5.4, color=INK2, va="top", linespacing=1.4)
    safe(f.add_subplot(gs[0, 0:2]), "a", "Decoding the prior", a, dx=-0.1)

    def b(ax):
        blank(ax); D = rows(); T = table()
        best = D.sort_values(["p_valid", "score"], ascending=[True, False]).iloc[0]
        ex = [(best.session, best.region, "detected", DBLUE)]
        hid = T[T.category.astype(str).str.startswith("hidden")].sort_values("p_carries", ascending=False)
        if len(hid):
            h = D[D.region == hid.index[0]].sort_values("score", ascending=False).iloc[0]; ex.append((h.session, h.region, "hidden", LBLUE))
        for k, (sid, reg, kind, c) in enumerate(ex):
            X, prior, _ = example(sid, reg); pred = cv_predict(X, prior)
            q = inset(ax, [0, 0.55 - k * 0.55, 1, 0.4]); tt = np.arange(len(prior))
            q.plot(tt, prior, color=VIOLET, lw=0.9, label="Bayes prior")
            sm = np.convolve(pred, np.ones(9) / 9, "same")
            q.plot(tt, (sm - sm.mean()) / (sm.std() + 1e-9) * prior.std() + prior.mean(), color=c, lw=0.9, label="decoded (held out, scaled)")
            r = np.corrcoef(pred, prior)[0, 1]
            q.text(0.01, 1.0, f"{reg} ({kind}): r = {r:.2f}", transform=q.transAxes, fontsize=5.8, va="bottom")
            q.set_xlim(0, len(tt)); q.tick_params(labelsize=5); q.set_ylabel("P(right)", fontsize=5.5)
            if k == 0:
                q.legend(loc="upper right", ncol=2, fontsize=5.2, bbox_to_anchor=(1, 1.22)); q.set_xticklabels([])
            else:
                q.set_xlabel("trial", fontsize=5.8)
    safe(f.add_subplot(gs[0, 2:6]), "b", "", b, dx=-0.05)

    def c(ax):
        D = rows()
        n = D.n_twin.sum(); rate = lambda col: (D[col] * D.n_twin).sum() / n
        raincloud(ax, [D.twin_fp_shuffle * 100, D.twin_fp_valid * 100], [ORANGE, BLUE],
                  [f"shuffle null\n{rate('twin_fp_shuffle'):.1%}", f"pseudo-session\nnull {rate('twin_fp_valid'):.1%}"], s=2)
        ax.axhline(5, color=INK2, lw=0.5, ls="--"); ax.set_ylabel("false positives, drift twins (%)")
        note(ax, f"{len(D)} session-regions, {D.session.nunique()} sessions", y=-0.4, x=0, ha="left")
    safe(f.add_subplot(gs[1, 0:2]), "c", "The decoder test is valid", c, dx=-0.27)

    def d(ax):
        t = text_file("PRIOR_MAP.md", "python3 part6/prior_map.py test")
        det = float(re.search(r"valid null ([0-9.]+) \(", t).group(1))
        est, lo, hi = map(float, re.search(r"fraction with r >= 0.04: ([0-9.]+) \(95% CI ([0-9.]+)-([0-9.]+)\)", t).groups())
        ax.bar([0], [det * 100], 0.55, color=DBLUE); ax.bar([1], [est * 100], 0.55, color=LBLUE)
        ax.errorbar([1], [est * 100], yerr=[[(est - lo) * 100], [(hi - est) * 100]], color=INK, lw=0.8, capsize=2.5)
        ax.text(0, det * 100 + 2, f"{det:.0%}", ha="center", fontsize=6)
        ax.text(1.32, est * 100, f"{est:.0%}\n[{lo:.0%}, {hi:.0%}]", va="center", fontsize=5.6)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["detected\n(valid test)", "estimated\n(TWIN)"]); ax.set_ylabel("regions with the prior (%)")
        ax.set_ylim(0, 100); ax.set_xlim(-0.5, 1.9)
        w = re.search(r"estimated distribution of code size across regions: \{(.*?)\}", t).group(1)
        vals = [float(x) for x in re.findall(r"np\.float64\(([0-9.e-]+)\)", w)] or [float(x.split(":")[1]) for x in w.split(",")]
        q = inset(ax, [0.05, 0.66, 0.42, 0.28]); q.bar(range(len(vals)), vals, color=[GRAY] + [LBLUE] * (len(vals) - 1))
        q.set_xticks([0, len(vals) - 1]); q.set_xticklabels(["0", f"{GRID_R[len(vals) - 1]:g}"], fontsize=4.8); q.set_yticks([])
        q.set_title("fitted code size r", fontsize=5, pad=1); q.spines["left"].set_visible(False)
    safe(f.add_subplot(gs[1, 2:4]), "d", "Most of the prior is undetected", d)

    def e(ax):
        import prior_map as PM
        D = rows(); T = table(); rng = np.random.default_rng(0)
        LL = PM.region_logliks(D, "p_valid", rng); regs = list(LL); w = PM.em_pi([LL[r] for r in regs])
        P = np.array([np.exp(LL[r] + np.log(w + 1e-12) - (LL[r] + np.log(w + 1e-12)).max()) for r in regs]); P /= P.sum(1, keepdims=True)
        order = np.argsort(P[:, 0])[::-1]; P = P[order]
        im = ax.imshow(P.T, aspect="auto", cmap="Blues", vmin=0, vmax=1, origin="lower", interpolation="nearest")
        ax.set_yticks(range(len(GRID_R))[::2]); ax.set_yticklabels([f"{g:g}" for g in GRID_R][::2]); ax.set_ylabel("code size r")
        ax.set_xlabel(f"{len(regs)} regions, sorted (\u25bc detected by the valid test)"); ax.set_xticks([])
        det = T.reindex([regs[i] for i in order]).fdr_valid.fillna(False).astype(bool).values
        ax.scatter(np.nonzero(det)[0], np.full(det.sum(), len(GRID_R) - 0.2), marker="v", s=6, color=DBLUE, clip_on=False)
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.04, pad=0.01); cb.set_label("posterior", fontsize=5.5); cb.ax.tick_params(labelsize=5)
    safe(f.add_subplot(gs[1, 4:6]), "e", "Posterior code size, every region", e)

    def f_(ax):
        T = table()
        try:
            swanson(ax, T.index.values, T.p_carries.values, "viridis", 0, 1, "P(region carries the block prior)")
            ax.set_anchor("W")
        except Exception as e:
            raise FileNotFoundError(f"Swanson flatmap failed ({type(e).__name__})\n(pip3 install iblatlas; needs internet once)")
    safe(f.add_subplot(gs[3, 0:4]), "h", "A hidden prior map: posterior probability that each region carries the block prior", f_, dx=-0.02)

    def g(ax):
        T = table()
        cats = [("detected", DBLUE), ("hidden (likely carries it)", LBLUE), ("unanswerable", LGRAY), ("likely absent", XLGRAY)]
        Tb = T.groupby(["system", "category"]).size().unstack(fill_value=0).reindex(columns=[c for c, _ in cats], fill_value=0)
        Tb = Tb.loc[[s for s in Tb.index if s in SYS]]; Tb = Tb.loc[(Tb.iloc[:, 0] + Tb.iloc[:, 1]).sort_values().index]
        y = np.arange(len(Tb)); left = np.zeros(len(Tb))
        for c, col in cats:
            ax.barh(y, Tb[c], 0.72, left=left, color=col, edgecolor="white", lw=0.6, label=c.split(" (")[0]); left += Tb[c].values
        ax.set_yticks(y); ax.set_yticklabels([SYS[s] for s in Tb.index]); ax.set_xlabel("regions"); ax.legend(loc="lower right", fontsize=5.2)
    safe(f.add_subplot(gs[2, 1:4]), "f", "By brain system", g, dx=-0.42)

    def h(ax):
        A = add_porig(csv("prior_map_attempts.csv", "python3 part6/prior_map.py test"))
        from sklearn.metrics import roc_curve
        for k in ("twin3", "naive", "p_orig"):
            fpr, tpr, _ = roc_curve(A.replicated.astype(int), A[k]); lab, c = PRED[k]
            ax.plot(fpr, tpr, color=c, lw=1.4 if k == "twin3" else 0.9, ls="-" if k == "twin3" else "--",
                    label=f"{'TWIN' if k == 'twin3' else lab} {auc(A.replicated, A[k]):.2f}")
        ax.plot([0, 1], [0, 1], color=GRID, lw=0.6); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(loc="lower right", fontsize=5.2, handlelength=1.2, title="AUC", title_fontsize=5.2)
    safe(f.add_subplot(gs[2, 4:6]), "g", "Which labs reproduce it", h, dx=-0.3)
    save(f, "Figure6", "Fig. 6 | The block prior is carried far more widely than any test can currently detect")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args()
    for i, fn in enumerate([figure1, figure2, figure3, figure4, figure5, figure6], 1):
        if not a.only or i in a.only:
            try:
                fn()
            except Exception:
                print(f"Figure {i} failed:\n{traceback.format_exc()}")
    if MISSING:
        print("\nPanels without input (send me this list):")
        for m in MISSING:
            print("  - " + m.replace("\n", " "))


if __name__ == "__main__":
    main()
