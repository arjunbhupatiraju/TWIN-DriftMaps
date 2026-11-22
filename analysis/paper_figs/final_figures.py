"""Final main figures (5; three tables are in the manuscript) for the paper "What do brain-wide encoding maps really show?".

Every panel is drawn from the result files on disk. A panel whose input is missing is drawn as a labelled placeholder,
never with invented numbers. Numbers that come from earlier reported runs (task false positives, answerability,
replication AUCs) are taken from the project result docs and marked SUMMARY.

Brain maps use the IBL Swanson flatmap (iblatlas). The first run downloads the flatmap outline once (needs internet).

    cd ~/Downloads/MODEL && python3 paper_figs/final_figures.py
    python3 paper_figs/final_figures.py 3          (one figure only)
Output: MODEL/figures/final/Fig1-5 (.png 300 dpi and .pdf)
"""
from __future__ import annotations

import os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

ROOT = Path(os.environ.get("TWIN_MODEL", Path(__file__).resolve().parents[1]))
P6, P7 = ROOT / "part6", ROOT / "part7"
OUTDIR = ROOT / "figures" / "final"
OUTDIR.mkdir(parents=True, exist_ok=True)

ORANGE, BLUE, AQUA, GRAY, NAVY, LIGHT = "#eb6834", "#2a78d6", "#1baf7a", "#8a8984", "#184f95", "#6da7ec"
INK, INK2, GRID, EMPTY = "#0b0b0b", "#52514e", "#e6e5e1", "#ecebe7"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "axes.edgecolor": INK2, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.labelcolor": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 6, "axes.titlesize": 7,
                     "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
CM_R2 = LinearSegmentedColormap.from_list("r2", ["#f7f4ea", "#f6c86b", "#e0602b", "#8f1d3c", "#2b0b3f"])
CM_LOST = LinearSegmentedColormap.from_list("lost", ["#f5f7fa", "#b9d3ee", "#5b8fd1", "#1f4f9a", "#0b2453"])
CM_PRIOR = LinearSegmentedColormap.from_list("prior", ["#f6f6f2", "#c9e6d8", "#5fbf95", "#1b8a63", "#0b4a35"])
SHORT = {"Isocortex": "Ctx", "OLF": "Olf", "HPF": "Hipp", "CTXsp": "Sub", "CNU": "Str/Pal", "TH": "Th", "HY": "Hy",
         "MB": "MB", "HB": "HB", "CB": "CB"}
COSMOS = ["Isocortex", "OLF", "HPF", "CTXsp", "CNU", "TH", "HY", "MB", "HB", "CB"]
SYSN = {"Isocortex": "Cortex", "OLF": "Olfactory", "HPF": "Hippocampal", "CTXsp": "Cortical subplate", "CNU": "Striatum/pallidum",
        "TH": "Thalamus", "HY": "Hypothalamus", "MB": "Midbrain", "HB": "Hindbrain", "CB": "Cerebellum"}

# ---------------------------------------------------------------- SUMMARY numbers from reported runs (project docs)
SUMMARY = dict(
    task_fp=(92.5, 5.0),                                   # prior-map/atlas validation, 454 IBL session-regions
    prior=dict(detected=7.1, before=54.6, after=42.4, after_ci=(28.1, 56.8)),   # prior-map and prior-movement runs
    answer=dict(names=["Block prior", "Stimulus", "Choice", "Feedback"], shuffle=[85, 49, 57, 87], valid=[2, 13, 57, 87],
                answerable=[2, 44, 73, 80]),
    repl=dict(tests=["Repeated site", "Brain-Wide Map", "New variables", "3 datasets", "IBL → Allen", "Prior map"],
              twin=[0.771, 0.748, 0.774, 0.811, 0.897, 0.684],
              ci=[(0.713, 0.834), (0.732, 0.765), (0.763, 0.782), (0.774, 0.838), None, (0.670, 0.700)],
              pval=[0.645, 0.644, 0.615, 0.490, None, 0.577], power=[0.711, 0.639, 0.636, 0.650, None, 0.579],
              rule=[None, 0.831, 0.663, 0.750, 0.599, None]))


# ================================================================ helpers
def read(path, **kw):
    import pandas as pd
    for p in ([path] if not isinstance(path, (list, tuple)) else path):
        if Path(p).exists():
            return pd.read_csv(p, **kw)
    return None


def letter(ax, s, title=None, x=-0.16, y=1.12):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    if title:
        ax.set_title(title, loc="left", pad=6)


def missing(ax, what):
    ax.text(0.5, 0.5, f"input not found:\n{what}", ha="center", va="center", fontsize=6, color=INK2, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([])


def boot_samples(groups, stat, B=500, seed=0):
    import pandas as pd
    rng = np.random.default_rng(seed); keys = list(groups)
    return np.array([stat(pd.concat([groups[keys[i]] for i in rng.integers(0, len(keys), len(keys))])) for _ in range(B)])


def boot(groups, stat, B=500, seed=0):
    return np.nanpercentile(boot_samples(groups, stat, B, seed), [2.5, 97.5])


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig); print(f"wrote {OUTDIR / name}.png/.pdf")


def rate(g, c, n="n_twin"):
    return float((g[c] * g[n]).sum() / max(g[n].sum(), 1))


_BR = None


def BR():
    global _BR
    if _BR is None:
        from iblatlas.regions import BrainRegions
        _BR = BrainRegions()
    return _BR


def system_of(acr):
    """Cosmos brain system for each Allen acronym ('' if unknown)."""
    acr = np.asarray(acr, dtype=object).astype(str)
    out = np.array([""] * len(acr), dtype=object)
    try:
        br = BR(); ok = np.isin(acr, br.acronym)
        if ok.any():
            out[ok] = br.acronym2acronym(acr[ok], mapping="Cosmos")
    except Exception:
        pass
    return out


def atlas_rgb(acr):
    try:
        br = BR(); i = np.nonzero(br.acronym == acr)[0]
        return tuple(br.rgb[i[0]] / 255) if len(i) else (0.6, 0.6, 0.6)
    except Exception:
        return (0.6, 0.6, 0.6)


_NAME2ACR = None


def beryl_of(names):
    """Allen CCF full names (as stored in the MAP files) to coarse Beryl acronyms; unknown names are shortened."""
    global _NAME2ACR
    out = []
    try:
        br = BR()
        if _NAME2ACR is None:
            k = lambda n: "".join(ch for ch in str(n).lower() if ch.isalnum())
            _NAME2ACR = {k(n): a for n, a in zip(br.name, br.acronym)}
            _NAME2ACR["_k"] = k
        k = _NAME2ACR["_k"]
        for n in names:
            a = _NAME2ACR.get(k(n))
            if a is None and str(n) in set(br.acronym):
                a = str(n)
            out.append(str(br.acronym2acronym(np.array([a]), mapping="Beryl")[0]) if a else str(n)[:8])
    except Exception:
        out = [str(n)[:8] for n in names]
    return np.array(out)


def swanson(ax, acr, vals, cmap, vmin, vmax, label, ticks=None, ticklabels=None, extend="neither"):
    """IBL Swanson flatmap of one value per brain region, with a horizontal colour bar underneath."""
    acr = np.asarray(acr).astype(str); vals = np.asarray(vals, float)
    try:
        from iblatlas.plots import plot_swanson_vector
        br = BR(); ok = np.isin(acr, br.acronym) & np.isfinite(vals)
        plot_swanson_vector(acr[ok], vals[ok], ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, br=br, empty_color=EMPTY, linewidth=0.15)
        ax.set_axis_off(); ax.set_aspect("equal")
        label = f"{label} ({ok.sum()} regions)" if label else f"{ok.sum()} regions"
    except Exception as e:
        if os.environ.get("TWIN_FIG_STUB"):              # layout testing only: a box with the flatmap's aspect ratio
            ax.add_patch(plt.Rectangle((0, 0), 2.25, 1, color="#cdd6e0")); ax.set_xlim(0, 2.25); ax.set_ylim(0, 1)
            ax.set_aspect("equal"); ax.set_axis_off()
        else:
          ax.set_axis_off()
          ax.text(0.5, 0.55, f"brain flatmap not drawn\n({type(e).__name__}; needs iblatlas and internet once)",
                  ha="center", va="center", fontsize=5.5, color=INK2, transform=ax.transAxes)
    cax = ax.inset_axes([0.2, -0.07, 0.6, 0.045])
    cb = plt.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), cax=cax, orientation="horizontal", extend=extend)
    cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.4)
    if ticks is not None:
        cb.set_ticks(ticks)
        if ticklabels is not None:
            cb.set_ticklabels(ticklabels)
    cb.set_label(label, fontsize=6, labelpad=1.5)


def half_violin(ax, data, pos, color, width=0.36, side=-1, points=True, seed=0, alpha=0.55, ptsize=2.2):
    """Raincloud: half violin on one side, jittered points on the other, median and IQR as a bar."""
    from scipy.stats import gaussian_kde
    d = np.asarray(data, float); d = d[np.isfinite(d)]
    if len(d) < 3:
        return
    lo, hi = np.percentile(d, [0.5, 99.5]); pad = 0.05 * (hi - lo + 1e-9)
    ys = np.linspace(lo - pad, hi + pad, 160)
    try:
        k = gaussian_kde(d)(ys); k = k / k.max() * width
    except Exception:
        return
    ax.fill_betweenx(ys, pos, pos + side * k, color=color, alpha=alpha, lw=0)
    ax.plot(pos + side * k, ys, color=color, lw=0.6)
    if points:
        rng = np.random.default_rng(seed)
        sub = d if len(d) <= 400 else rng.choice(d, 400, replace=False)
        ax.scatter(pos - side * (0.06 + rng.uniform(0, width * 0.55, len(sub))), sub, s=ptsize, color=color, alpha=0.5, lw=0)
    q1, m, q3 = np.percentile(d, [25, 50, 75])
    ax.plot([pos, pos], [q1, q3], color=INK, lw=1.6, solid_capstyle="butt")
    ax.scatter([pos], [m], s=9, color="white", edgecolor=INK, lw=0.6, zorder=5)


def kde_ridge(ax, samples, y0, color, height=0.8, alpha=0.6):
    from scipy.stats import gaussian_kde
    s = np.asarray(samples, float); s = s[np.isfinite(s)]
    xs = np.linspace(s.min() - 3 * s.std(), s.max() + 3 * s.std(), 200)
    k = gaussian_kde(s)(xs); k = k / k.max() * height
    ax.fill_between(xs, y0, y0 + k, color=color, alpha=alpha, lw=0); ax.plot(xs, y0 + k, color=color, lw=0.7)


def smooth(x, k):
    x = np.asarray(x, float)
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), np.ones(k) / k, mode="valid")


def map_population(f, probe, window, seed=0):
    """Rebuild one MAP population exactly as map_decompose.py does for MA1 (clean trials, task regressors, drift basis,
    good units, processed movement, up to 150 units). Returns None if it does not qualify."""
    for q in (P7, P6, ROOT / "twin", ROOT):
        if str(q) not in sys.path:
            sys.path.insert(0, str(q))
    import map_decompose as MD
    import decompose as DC
    z = np.load(f, allow_pickle=False)
    keep = MD._clean_trials(z); T = int(keep.sum())
    if T < 200:
        return None
    ins = np.where(z["trial__instruction"] == "right", 1.0, -1.0); hit = np.where(z["trial__outcome"] == "hit", 1.0, -1.0)
    Tk = np.column_stack([ins, hit, np.r_[0, ins[:-1]], np.r_[0, hit[:-1]]])[keep]
    D = DC.drift(T); S = 2 * (min(50, T // 6) // 2)
    uq = z["unit__unit_quality"] if "unit__unit_quality" in z.files else np.array(["good"] * len(z["unit__probe"]))
    good = uq == "good"
    if "unit__presence_ratio" in z.files:
        good &= np.nan_to_num(z["unit__presence_ratio"], nan=1.0) >= 0.9
    M, names = MD._movement(z, window, keep)
    C = np.asarray(z[f"counts__{window}"], float)[keep]
    if M is None or not np.isfinite(C).all():
        return None
    idx = np.nonzero((z["unit__probe"].astype(str) == str(probe)) & good)[0]
    X = np.sqrt(np.maximum(C[:, idx], 0)); X = X[:, X.std(0) > 0]
    if X.shape[1] < 10:
        return None
    Yall = (X - X.mean(0)) / X.std(0)
    Y = Yall if Yall.shape[1] <= 150 else Yall[:, np.sort(np.random.default_rng(seed).choice(Yall.shape[1], 150, replace=False))]
    return dict(Y=Y, Yall=Yall, Tk=Tk, M=M, D=D, S=S, names=names, DC=DC)


def phase_twin(Y, rng):
    """Multivariate phase randomisation: same random phases for every column (keeps auto- and cross-correlation)."""
    F = np.fft.rfft(Y - Y.mean(0), axis=0)
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, F.shape[0])); ph[0] = 1
    if Y.shape[0] % 2 == 0:
        ph[-1] = 1
    return np.fft.irfft(F * ph[:, None], n=Y.shape[0], axis=0) + Y.mean(0)


# ---------------------------------------------------------------- shared data loaders
def ibl_region_table(min_sess=3):
    """Per IBL region: mean standard and valid unique movement R², share lost, number of sessions."""
    E = read(P6 / "results_decompose" / "decompose_effects_robust.csv")
    if E is None:
        return None, None
    g = E.groupby("region")
    T = g.agg(std=("dM_std", "mean"), val=("dM", "mean"), s_std=("dM_std", "sum"), s_val=("dM", "sum"), n=("session", "nunique"))
    T = T[T.n >= min_sess]
    T["lost"] = np.where(T.s_std > 0, 1 - T.s_val / T.s_std, np.nan)
    return T, E


def ibl_prior_table():
    import pandas as pd
    R = read([P6 / "results_prior" / "prior_map_regions_amend1.csv", P6 / "results_prior" / "prior_map_regions.csv"], index_col=0)
    M = read(P6 / "results_movement" / "movement_regions.csv", index_col=0)
    F = read([P6 / "results_prior" / "prior_map_figure_table_amend1.csv", P6 / "results_prior" / "prior_map_figure_table.csv"],
             index_col=0)
    if R is None:
        return None, F
    if M is not None:
        R = R.join(M[[c for c in M.columns if c not in R.columns]], how="left")
    return R, F


def corrected_share(ds, B=500):
    """C4 (part7/calibrate.py adjust): share lost after mapping each analysis back to the truth scale with its own
    calibration (estimate = a + b * truth); mouse bootstrap with the calibration refitted. Returns (value %, samples %)."""
    import pandas as pd
    D = read(P7 / "results_calibrate" / "calibrate_rows.csv"); C = read(P7 / "results_calibrate" / "calibration_long.csv")
    if D is None or C is None or not (D.dataset == ds).any():
        return None
    d, c = D[D.dataset == ds], C[C.dataset == ds]

    def sh(d_, c_):
        bs_, as_ = np.polyfit(c_.truth, c_.standard, 1); bv, av = np.polyfit(c_.truth, c_.valid, 1)
        return 100 * (1 - ((d_.valid - av) / bv).sum() / ((d_.standard - as_) / bs_).sum())
    rng = np.random.default_rng(0); ms = d.mouse.unique()
    gd = {m: g for m, g in d.groupby("mouse")}; gc = {m: g for m, g in c.groupby("mouse")}
    out = []
    for _ in range(B):
        pick = ms[rng.integers(0, len(ms), len(ms))]
        out.append(sh(pd.concat([gd[m] for m in pick]), pd.concat([gc[m] for m in pick if m in gc])))
    return sh(d, c), np.array(out)


# ================================================================ Figure 1: the problem, and the valid test
def fig1():
    import pandas as pd
    fig = plt.figure(figsize=(7.1, 7.0))
    G = fig.add_gridspec(3, 3, height_ratios=[1.25, 1, 1], hspace=0.62, wspace=0.55, left=0.075, right=0.985, top=0.95, bottom=0.07)

    # ---- a: recorded neurons and their drift twin (one MAP population, processed exactly as in the MA1 analysis)
    sub = G[0, 0:2].subgridspec(2, 3, height_ratios=[0.32, 1], width_ratios=[1, 0.22, 1], hspace=0.06, wspace=0.05)
    rows = read(P7 / "results_map" / "map_rows.csv"); sdir = P7 / "results_map" / "sessions"
    ex = None
    if rows is not None and sdir.exists():
        rows = rows.assign(gap=rows.dM_std - rows.dM)
        cand = rows[(rows.window == "pre") & (rows["T"] >= 300)].sort_values("gap", ascending=False)
        for _, r in cand.iterrows():
            f = sdir / f"{r.session}.npz"
            if f.exists():
                ex = (r, f); break
    axm = fig.add_subplot(sub[0, 0]); letter(axm, "a", None, x=-0.13, y=1.45)
    pop = None
    if ex is not None:
        try:
            pop = map_population(ex[1], ex[0].probe, ex[0].window)
            if pop is not None:
                pop["r"] = ex[0]
        except Exception as e:
            missing(axm, f"MAP example could not be rebuilt ({type(e).__name__}: {e})")
    if pop is not None:
        r, Yall, M = pop["r"], pop["Yall"], pop["M"]
        mv = M[:, 0]                                                   # the analysis' own processed feature (log, clipped)
        U, Sv, Vt = np.linalg.svd(Yall, full_matrices=False); pc = U[:, 0] * Sv[0]
        if np.corrcoef(pc, mv)[0, 1] < 0:
            pc, Vt = -pc, -Vt
        order = np.argsort(Vt[0])
        Yt = phase_twin(Yall, np.random.default_rng(1)); Yt = (Yt - Yt.mean(0)) / Yt.std(0); pct = Yt @ Vt[0]
        mvz = (mv - mv.mean()) / mv.std(); t = np.arange(len(mv))
        axr = fig.add_subplot(sub[1, 0], sharex=axm); axt = fig.add_subplot(sub[1, 2], sharex=axm)
        axm2 = fig.add_subplot(sub[0, 2], sharex=axm)
        for a_, sig, lab in ((axm, pc, "neural PC1"), (axm2, pct, "twin PC1")):
            ms_ = smooth(mvz, 25)
            a_.plot(t, ms_, color=ORANGE, lw=1.2)
            s_ = smooth((sig - sig.mean()) / sig.std(), 25); s_ = s_ / max(s_.std(), 1e-9) * ms_.std()
            a_.plot(t, s_, color=NAVY, lw=1.2)
            a_.set_yticks([]); a_.spines["left"].set_visible(False); a_.spines["bottom"].set_visible(False)
            a_.tick_params(axis="x", length=0); plt.setp(a_.get_xticklabels(), visible=False)
            a_.text(0.0, 1.02, pop['names'][0].replace('_', ' '), color=ORANGE, transform=a_.transAxes, fontsize=5.6, va="bottom")
            a_.text(0.36, 1.02, lab, color=NAVY, transform=a_.transAxes, fontsize=5.6, va="bottom")
            a_.text(1.0, 1.02, f"r = {np.corrcoef(sig, mv)[0, 1]:+.2f}", transform=a_.transAxes, fontsize=5.6, ha="right", va="bottom")
        for a_, M_, ttl in ((axr, Yall, f"recorded: {Yall.shape[1]} units, MAP {r.region}"), (axt, Yt, "drift twin: same statistics, no movement link")):
            im = a_.imshow(np.apply_along_axis(smooth, 0, M_[:, order], 5).T, aspect="auto", cmap="RdBu_r", vmin=-1.6, vmax=1.6,
                           interpolation="nearest", extent=(0, len(t), 0, M_.shape[1]))
            a_.set_xlabel(f"trial ({r.window} epoch)")
            a_.text(0.02, 0.97, ttl, transform=a_.transAxes, fontsize=5.6, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        axr.set_ylabel("units (sorted by PC1 weight)"); axt.set_yticks([])
        axa = fig.add_subplot(sub[:, 1]); axa.set_axis_off()
        axa.annotate("", xy=(0.95, 0.42), xytext=(0.05, 0.42), xycoords="axes fraction",
                     arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
        axa.text(0.5, 0.5, "phase\nrandomise", ha="center", va="bottom", fontsize=5.8, color=INK2, transform=axa.transAxes)
        cax = axt.inset_axes([1.02, 0.0, 0.03, 0.5]); cb = plt.colorbar(im, cax=cax); cb.outline.set_linewidth(0.3)
        cb.ax.tick_params(labelsize=5, length=1.5); cb.set_label("z", fontsize=5.5, labelpad=1)
    elif ex is None:
        missing(axm, "part7/results_map/sessions (MAP session files)")

    # ---- b: the paper's own two tests on one drift twin of each of many randomly drawn MAP populations (cached)
    ax = fig.add_subplot(G[0, 2]); letter(ax, "b", "Why the standard test fails", x=-0.2)
    if rows is not None and sdir.exists():
        import pandas as pd
        n_pop = int(os.environ.get("TWIN_FIG1_POPS", "120"))
        cache = OUTDIR / f"fig1b_random_populations_{n_pop}.csv"
        if cache.exists():
            P_ = pd.read_csv(cache)
        else:
            avail = rows[[ (sdir / f"{x}.npz").exists() for x in rows.session ]]
            pick = avail.sample(min(n_pop, len(avail)), random_state=0)          # random, seeded; not chosen by outcome
            rng = np.random.default_rng(2); out = []
            print(f"  Fig1b: running both tests on drift twins of {len(pick)} random MAP populations (one-time, several minutes)...",
                  flush=True)
            for k, (_, q) in enumerate(pick.iterrows()):
                try:
                    pp = map_population(sdir / f"{q.session}.npz", q.probe, q.window, seed=k)
                except Exception:
                    pp = None
                if pp is None:
                    continue
                Yt_ = phase_twin(pp["Y"], rng); Yt_ = (Yt_ - Yt_.mean(0)) / Yt_.std(0)
                o = pp["DC"].tests(Yt_, pp["Tk"], pp["M"], pp["D"], pp["S"], 100, rng)
                out.append(dict(session=q.session, probe=q.probe, window=q.window, p_std=o["p_std_M"], p_valid=o["p_valid_M"]))
            P_ = pd.DataFrame(out); P_.to_csv(cache, index=False)
        ps, ph = P_.p_std.values, P_.p_valid.values
        bins = np.linspace(0, 1, 11)
        ax.hist(ps, bins=bins, color=ORANGE, alpha=0.75, density=True, label=f"standard: {np.mean(ps < 0.05):.0%} P < 0.05")
        ax.hist(ph, bins=bins, color=BLUE, alpha=0.55, density=True, label=f"valid: {np.mean(ph < 0.05):.0%} P < 0.05")
        ax.axhline(1, color=INK, lw=0.7, ls="--")
        ax.text(0.98, 1.0, "expected if valid ", ha="right", va="bottom", fontsize=5.3, transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1); ax.set_xlabel("movement-encoding P value"); ax.set_ylabel("density")
        ax.legend(loc="upper right", fontsize=5.4)
        ax.text(0.98, 0.76, f"drift twins of {len(ps)} randomly\ndrawn MAP populations", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.2, color=INK2)
    else:
        missing(ax, "MAP session files")

    # ---- c: cross-validation schematic on the real drift
    ax = fig.add_subplot(G[1, 0]); letter(ax, "c", "Two ways to cross-validate", x=-0.2)
    T0 = 200; tt = np.arange(T0)
    dr = smooth(mvz, 25)[:T0] if pop is not None else np.sin(tt / 30)
    dr = (dr - dr.min()) / (dr.max() - dr.min() + 1e-9)
    ax.plot(tt, 2.35 + 0.8 * dr, color=NAVY, lw=1); ax.text(0, 3.3, "slow drift", color=NAVY, fontsize=5.8)
    for k in range(0, T0, 4):
        ax.add_patch(plt.Rectangle((k, 1.3), 4, 0.55, color=ORANGE if (k // 4) % 5 == 0 else "#f6d3c2", lw=0))
    for k in range(5):
        ax.add_patch(plt.Rectangle((k * 40, 0.1), 40, 0.55, color=BLUE if k == 2 else "#cfe0f5", lw=0.6, ec="white"))
    ax.text(0, 1.95, "standard: interleaved folds", fontsize=5.4, color=ORANGE)
    ax.text(0, 0.75, "valid: drift removed, contiguous folds, shift null", fontsize=5.4, color=BLUE)
    ax.set_xlim(0, T0); ax.set_ylim(0, 3.5); ax.set_yticks([]); ax.spines["left"].set_visible(False); ax.set_xlabel("trial")

    # ---- d: false positives, session by session, every dataset
    ax = fig.add_subplot(G[1, 1:]); letter(ax, "d", "Movement encoding where there is none: per-session false-positive rate", x=-0.1)
    sets = [("IBL\ndrift twins", P6 / "results_decompose" / "decompose_rows.csv", "twin_fp_std", "twin_fp_valid", "n_twin", "session"),
            ("IBL\nother mouse", P7 / "results_robustness" / "robustness_rows.csv", "cross_fp_std", "cross_fp_valid", "n_partner", "session"),
            ("MAP\ndrift twins", P7 / "results_map" / "map_rows.csv", "twin_fp_std", "twin_fp_valid", "n_twin", "session"),
            ("calcium\ndrift twins", P7 / "results_calcium" / "calcium_rows.csv", "twin_fp_std", "twin_fp_valid", "n_twin", "population")]
    labs, x0 = [], 0
    for lab, f, cs, cv, n, grp in sets:
        D = read(f)
        if D is None:
            continue
        if grp not in D:
            grp = "session"
        if grp == "population" and "session" in D:          # population ids repeat across recordings
            D = D.assign(population=D.session.astype(str) + "_" + D.population.astype(str))
        S_ = D.groupby(grp).apply(lambda g: pd.Series(dict(s=100 * rate(g, cs, n), v=100 * rate(g, cv, n))))
        half_violin(ax, S_.s, x0 - 0.04, ORANGE, side=-1, seed=x0)
        half_violin(ax, S_.v, x0 + 0.04, BLUE, side=1, seed=x0 + 1)
        ax.text(x0 - 0.2, 110, f"{100 * rate(D, cs, n):.0f}%", color=ORANGE, ha="center", fontsize=6)
        ax.text(x0 + 0.2, 110, f"{100 * rate(D, cv, n):.1f}%", color=BLUE, ha="center", fontsize=6)
        labs.append((x0, lab + f"\n(n = {len(S_)})")); x0 += 1
    ax.axhline(5, ls="--", lw=0.5, color=INK2)
    ax.set_xticks([p for p, _ in labs]); ax.set_xticklabels([l for _, l in labs], fontsize=5.6)
    ax.set_ylim(-4, 118); ax.set_xlim(-0.6, max(x0 - 0.4, 0.6)); ax.set_ylabel("false positives (%)")
    ax.plot([], [], color=ORANGE, lw=4, label="standard test"); ax.plot([], [], color=BLUE, lw=4, label="valid test")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.26), ncol=2)
    ax.text(1.0, -0.44, f"task variables, fake labels (SUMMARY): standard {SUMMARY['task_fp'][0]}%, valid {SUMMARY['task_fp'][1]}%",
            transform=ax.transAxes, ha="right", fontsize=5.4, color=INK2)

    # ---- e: epoch x dataset tile matrix
    sub = G[2, 0:2].subgridspec(1, 2, wspace=0.12)
    first = True
    for j, (f, nm, order, labs_) in enumerate(((P6 / "results_decompose" / "decompose_rows.csv", "IBL",
                                                ["prior_a", "stim_a", "choice_a", "feedback_a"], ["pre-stim", "stimulus", "choice", "feedback"]),
                                               (P7 / "results_map" / "map_rows.csv", "MAP", ["pre", "sample", "delay", "response"],
                                                ["pre", "sample", "delay", "response"]))):
        ax = fig.add_subplot(sub[0, j])
        if first:
            letter(ax, "e", None, x=-0.22, y=1.3); ax.text(0, 1.22, "False positives by trial epoch", transform=ax.transAxes, fontsize=7); first = False
        D = read(f)
        if D is None:
            missing(ax, f.name); continue
        Mx = np.array([[100 * rate(D[D.window == w], c) for w in order] for c in ("twin_fp_std", "twin_fp_valid")])
        ax.imshow(Mx, cmap=LinearSegmentedColormap.from_list("fp", ["#ffffff", "#f6c3a8", ORANGE, "#8f2a07"]), vmin=0, vmax=100,
                  aspect="auto")
        for (i, k), v in np.ndenumerate(Mx):
            ax.text(k, i, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if v > 60 else INK)
        ax.set_xticks(range(4)); ax.set_xticklabels(labs_, fontsize=5.6, rotation=30, ha="right")
        ax.set_yticks([0, 1]); ax.set_yticklabels(["standard", "valid"] if j == 0 else [], fontsize=6)
        ax.set_title(nm, fontsize=6.3, pad=2, loc="right"); ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    # ---- f: power, per session
    ax = fig.add_subplot(G[2, 2]); letter(ax, "f", "The valid test keeps its power", x=-0.22)
    D = read(P7 / "results_robustness" / "robustness_rows.csv")
    if D is not None:
        rs = [0.05, 0.1, 0.2]
        S_ = D.groupby("session")[[f"power_{r}" for r in rs]].mean() * 100
        for _, rr in S_.sample(min(len(S_), 160), random_state=0).iterrows():
            ax.plot(rs, rr.values, color=BLUE, lw=0.3, alpha=0.18)
        groups = {k: g for k, g in D.groupby("session")}
        m = [100 * D[f"power_{r}"].mean() for r in rs]
        ci = np.array([100 * boot(groups, lambda g, r=r: g[f"power_{r}"].mean(), 300) for r in rs])
        ax.fill_between(rs, ci[:, 0], ci[:, 1], color=NAVY, alpha=0.25, lw=0)
        ax.plot(rs, m, color=NAVY, marker="o", ms=3.5, lw=1.4)
        for r_, v in zip(rs, m):
            ax.text(r_, v + 5, f"{v:.0f}%", ha="center", fontsize=6, color=NAVY)
        ax.set_xticks(rs); ax.set_xlim(0.03, 0.22); ax.set_ylim(0, 105)
        ax.set_xlabel("planted correlation with real movement"); ax.set_ylabel("detected by valid test (%)")
        ax.text(0.98, 0.04, f"thin lines: {len(S_)} sessions", transform=ax.transAxes, ha="right", fontsize=5.4, color=INK2)
    else:
        missing(ax, "results_robustness")
    save(fig, "Fig1")


# ================================================================ Figure 2: brain-wide maps are overstated where strong
def fig2():
    import pandas as pd
    fig = plt.figure(figsize=(7.1, 7.0))
    G = fig.add_gridspec(3, 3, height_ratios=[0.6, 1.2, 0.95], hspace=0.78, wspace=0.42, left=0.07, right=0.92, top=0.95, bottom=0.06)
    T, E = ibl_region_table()
    titles = [("a", "Standard analysis", "std", CM_R2, "unique movement R²"), ("b", "Valid analysis", "val", CM_R2, "unique movement R²"),
              ("c", "Share that does not survive", "lost", CM_LOST, "standard R² lost")]
    vmax = float(np.nanpercentile(T["std"], 95)) if T is not None else 1
    for k, (L, ttl, col, cm, lab) in enumerate(titles):
        ax = fig.add_subplot(G[0, k]); letter(ax, L, None, x=-0.02, y=1.1); ax.text(0.08, 1.1, f"IBL: {ttl.lower()}" if k < 2 else ttl, transform=ax.transAxes, fontsize=7, va="top")
        if T is None:
            missing(ax, "decompose_effects_robust.csv"); continue
        if col == "lost":
            swanson(ax, T.index, T.lost.clip(0, 1), cm, 0, 1, lab, ticks=[0, .5, 1], ticklabels=["0%", "50%", "100%"])
        else:
            swanson(ax, T.index, T[col], cm, 0, vmax, lab, extend="max")

    # d: density of standard vs valid, with marginals
    ax = fig.add_subplot(G[1, 0]); letter(ax, "d", None, x=-0.28, y=1.36); ax.text(0, 1.3, "Every region-epoch, two analyses", transform=ax.transAxes, fontsize=7, va="top")
    if E is not None:
        hi = float(np.nanpercentile(E.dM_std, 99)); lo = -0.1 * hi
        x, y = E.dM.clip(lo, hi), E.dM_std.clip(lo, hi)
        hb = ax.hexbin(x, y, gridsize=38, cmap=LinearSegmentedColormap.from_list("d", ["#eef3fb", LIGHT, NAVY, "#071a3a"]),
                       mincnt=1, bins="log", linewidths=0, extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.7, ls="--"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("valid movement R²"); ax.set_ylabel("standard movement R²")
        ax.text(0.04, 0.96, f"{np.mean(E.dM_std > E.dM):.0%} above\nthe diagonal", transform=ax.transAxes, va="top", fontsize=5.8)
        top = ax.inset_axes([0, 1.02, 1, 0.16]); right = ax.inset_axes([1.02, 0, 0.16, 1])
        bins = np.linspace(lo, hi, 50)
        top.hist(x, bins=bins, color=BLUE, alpha=0.7); top.hist(y, bins=bins, color=ORANGE, alpha=0.45)
        right.hist(y, bins=bins, color=ORANGE, alpha=0.7, orientation="horizontal")
        for a_ in (top, right):
            a_.set_axis_off()
        cax = ax.inset_axes([0.6, 0.14, 0.3, 0.035]); cb = plt.colorbar(hb, cax=cax, orientation="horizontal")
        cb.ax.tick_params(labelsize=4.8, length=1); cb.outline.set_linewidth(0.3); cb.set_label("count", fontsize=5, labelpad=0.5)
    else:
        missing(ax, "decompose_effects_robust.csv")

    # e: bootstrap distributions of the share lost, every dataset
    ax = fig.add_subplot(G[1, 1:].subgridspec(1, 2, width_ratios=[0.42, 1])[0, 1])
    letter(ax, "e", "Share of standard movement encoding that does not survive", x=-0.06)
    entries = []
    if E is not None:
        entries.append(("IBL, uncorrected\n(clipping rule, post hoc)", E, "dM", "dM_std", "region", GRAY))
    M = read(P7 / "results_map" / "map_rows.csv")
    if M is not None:
        M["ins"] = M.session + M.probe.astype(str)
        entries.append(("MAP, 5 markers\n(pre-registered)", M, "dM", "dM_std", "ins", BLUE))
    Rr = read(P7 / "results_map" / "map_rich_rows.csv")
    if Rr is not None:
        Rr["ins"] = Rr.session + Rr.probe.astype(str)
        entries.append(("MAP, 69 features\n(pre-registered)", Rr, "dM_rich", "dM_std_rich", "ins", NAVY))
    Ca = read(P7 / "results_calcium" / "calcium_rows.csv")
    if Ca is not None:
        entries.append(("Calcium imaging\n(pre-registered)", Ca, "R2_valid", "R2_std", "session", AQUA))
    for ds in ("MAP", "IBL"):                               # C4: both analyses corrected for their own calibrated bias
        c4 = corrected_share(ds)
        if c4 is not None:
            entries.insert(0, (f"{ds}, bias-corrected\n(C4, pre-registered)", c4, None, None, None, "#7a3fb0"))
    for i, (lab, D, cv, cs, grp, col) in enumerate(entries[::-1]):
        if cv is None:
            v, bs = D
        else:
            sh = lambda g, cv=cv, cs=cs: 100 * (1 - g[cv].sum() / g[cs].sum())
            bs = boot_samples({k: g for k, g in D.groupby(grp)}, sh, 600); v = sh(D)
        kde_ridge(ax, bs, i, col, 0.78)
        lo_, hi_ = np.percentile(bs, [2.5, 97.5])
        ax.plot([lo_, hi_], [i - 0.06, i - 0.06], color=INK, lw=1.2); ax.scatter([v], [i - 0.06], s=16, color=col, edgecolor=INK, lw=0.6, zorder=4)
        ax.text(hi_ + 1.2, i + 0.05, f"{v:.0f}% [{lo_:.0f}, {hi_:.0f}]", fontsize=6, va="bottom")
    ax.axvline(10, ls="--", lw=0.6, color=INK2); ax.text(10.5, len(entries) - 0.15, "pre-registered\ncriterion", fontsize=5.3, color=INK2, va="top")
    ax.set_yticks(np.arange(len(entries)) + 0.25)
    ax.set_yticklabels([e[0] for e in entries[::-1]], fontsize=5.6)
    for t_, e in zip(ax.get_yticklabels(), entries[::-1]):
        t_.set_color(e[5] if e[5] != GRAY else INK2)
    ax.set_xlim(0, 85); ax.set_ylim(-0.4, len(entries) + 0.05); ax.set_xlabel("standard movement R² that does not survive (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)

    # f: region x epoch matrix of share lost, grouped by brain system
    ax = fig.add_subplot(G[2, :]); letter(ax, "f", None, x=-0.035, y=1.42); ax.text(0, 1.4, "Where the overstatement sits: IBL regions × trial epochs", transform=ax.transAxes, fontsize=7, va="top")
    if E is not None:
        order = ["prior_a", "stim_a", "choice_a", "feedback_a"]
        g = E.groupby(["region", "window"]).agg(s=("dM_std", "sum"), v=("dM", "sum"), n=("session", "nunique")).reset_index()
        g["lost"] = np.where((g.s > 0) & (g.n >= 3), 1 - g.v / g.s, np.nan)
        P = g.pivot(index="region", columns="window", values="lost").reindex(columns=order)
        tot = E.groupby("region").dM_std.sum()
        nmin = E.groupby("region").session.nunique()
        keep = tot[nmin.reindex(tot.index) >= 5].sort_values(ascending=False).index[:60]
        P = P.reindex(keep)
        sysm = pd.Series(system_of(P.index), index=P.index)
        rk = {s: i for i, s in enumerate(COSMOS)}
        P = P.assign(_s=sysm.map(lambda s: rk.get(s, 99)), _m=P.mean(1)).sort_values(["_s", "_m"], ascending=[True, False])
        sysm = sysm.reindex(P.index); P = P.drop(columns=["_s", "_m"])
        im = ax.imshow(P.T.values, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-0.6, vcenter=0, vmax=1), interpolation="nearest")
        ax.set_yticks(range(4)); ax.set_yticklabels(["pre-stim", "stimulus", "choice", "feedback"], fontsize=5.8)
        ax.set_xticks(range(len(P))); ax.set_xticklabels(P.index, rotation=90, fontsize=4.6)
        ax.tick_params(length=0)
        strip = ax.inset_axes([0, 1.02, 1, 0.1], sharex=ax)
        for i, s in enumerate(sysm):
            strip.add_patch(plt.Rectangle((i - 0.5, 0), 1, 1, color=atlas_rgb(s) if s else (0.8, 0.8, 0.8), lw=0))
        strip.set_xlim(-0.5, len(P) - 0.5); strip.set_ylim(0, 1); strip.set_axis_off()
        prev = None
        for i, s in enumerate(list(sysm) + [None]):
            if s != prev and prev is not None:
                st = [j for j, q in enumerate(sysm) if q == prev]
                if len(st) >= 3:
                    nm_ = SYSN.get(prev, prev or "other") if len(st) >= 7 else SHORT.get(prev, prev or "")
                    strip.text(np.mean(st), 1.15, nm_, ha="center", va="bottom", fontsize=5.2)
                if i < len(sysm):
                    ax.axvline(i - 0.5, color="white", lw=1.2)
            prev = s
        cax = ax.inset_axes([1.008, 0.0, 0.012, 1.0])
        cb = plt.colorbar(im, cax=cax); cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=5, length=1.5)
        cb.set_ticks([-0.5, 0, 0.5, 1]); cb.set_ticklabels(["−50%", "0", "50%", "100%"]); cb.set_label("R² lost", fontsize=5.5)
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        missing(ax, "decompose_effects_robust.csv")
    save(fig, "Fig2")


# ================================================================ Figure 3: the MAP reanalysis along the probe
def fig3():
    import pandas as pd
    fig = plt.figure(figsize=(7.1, 7.2))
    G = fig.add_gridspec(3, 3, height_ratios=[1.3, 1, 1], hspace=0.55, wspace=0.75, left=0.08, right=0.93, top=0.95, bottom=0.07)
    Nn = read(P7 / "results_map" / "map_neurons.csv")
    M = read(P7 / "results_map" / "map_rows.csv"); Rr = read(P7 / "results_map" / "map_rich_rows.csv")

    # a: probe scans
    ax = fig.add_subplot(G[0, :]); letter(ax, "a", None, x=-0.045, y=1.1)
    ax.text(0, 1.1, "Unit by unit along the probe: movement EV, standard vs valid (MAP, Wang et al. 2025)", transform=ax.transAxes,
            fontsize=7, va="top")
    if Nn is not None and "posy" in Nn and Nn.posy.notna().any():
        U = Nn.dropna(subset=["posy"]).groupby(["session", "probe", "unit"]).agg(
            posy=("posy", "first"), anno=("anno", "first"), region=("region", "first"), s=("ev_std", "mean"), v=("ev_valid", "mean")).reset_index()
        cnt = U.groupby(["session", "probe"]).size()
        regs = U.groupby(["session", "probe"]).region.first()
        pick = []
        for reg in regs.unique():                                  # the best-sampled insertion of each area
            c = cnt[regs == reg].sort_values(ascending=False)
            if len(c):
                pick.append(c.index[0])
        pick = pick[:7]
        vmax = float(np.nanpercentile(U.s, 97))
        rng = np.random.default_rng(0); span = U.posy.max() - U.posy.min()
        for i, key in enumerate(pick):
            g = U[(U.session == key[0]) & (U.probe == key[1])]
            x0 = i * 3.2
            for j, (c, nm) in enumerate((("s", "std"), ("v", "valid"))):
                xs = x0 + j * 1.15
                ax.add_patch(plt.Rectangle((xs - 0.48, g.posy.min() - 60), 0.96, g.posy.max() - g.posy.min() + 120, color="#f1f0ec", lw=0, zorder=0))
                sc = ax.scatter(xs + rng.uniform(-0.38, 0.38, len(g)), g.posy, c=g[c].clip(0, vmax), cmap=CM_R2, vmin=0, vmax=vmax,
                                s=5, lw=0, zorder=2)
                ax.text(xs, g.posy.max() + 0.03 * span, nm, ha="center", fontsize=5.4, color=ORANGE if j == 0 else BLUE)
            ax.text(x0 + 0.575, g.posy.max() + 0.15 * span, f"{g.region.iloc[0]}", ha="center", fontsize=6.2, fontweight="bold")
            ax.text(x0 + 0.575, g.posy.max() + 0.09 * span, f"{len(g)} units", ha="center", fontsize=5.2, color=INK2)
            if "anno" in g and g.anno.astype(str).str.len().gt(0).any():
                raw = g.anno.astype(str).values
                ok_ = ~np.isin(np.array([str(x).lower() for x in raw]), ["nan", "", "none", "root", "void"])
                lab = beryl_of(raw[ok_]); y_ = g.posy.values[ok_]
                cand = []
                for L_ in np.unique(lab):
                    m_ = lab == L_
                    if m_.sum() >= max(10, 0.08 * len(g)) and L_.lower() not in ("nan", "root", "void"):
                        cand.append((np.median(y_[m_]), L_, m_.sum()))
                last = -1e9
                for ym, L_, n_ in sorted(cand):
                    if ym - last > 0.07 * span:
                        ax.text(x0 - 0.56, ym, L_, ha="right", va="center", fontsize=4.8, color=INK2); last = ym
        ax.set_xlim(-1.6, len(pick) * 3.2 - 1.2); ax.set_xticks([])
        ax.set_ylim(U.posy.min() - 0.03 * span, U.posy.max() + 0.2 * span)
        ax.spines["bottom"].set_visible(False); ax.set_ylabel("position along the probe (µm)")
        cax = ax.inset_axes([1.003, 0.05, 0.008, 0.6]); cb = plt.colorbar(sc, cax=cax, extend="max")
        cb.outline.set_linewidth(0.3); cb.ax.tick_params(labelsize=5, length=1.5); cb.set_label("unique movement EV", fontsize=5.5)
    else:
        missing(ax, "map_neurons.csv with probe positions (posy)")

    # b: area ranking, standard -> valid
    ax = fig.add_subplot(G[1, 0]); letter(ax, "b", "Area ranking changes", x=-0.3)
    if Nn is not None:
        A = Nn.groupby("region")[["ev_std", "ev_valid"]].mean()
        rs, rv = A.ev_std.rank(ascending=False), A.ev_valid.rank(ascending=False)
        cols = plt.get_cmap("tab10")
        for i, a_ in enumerate(A.index):
            c = cols(i % 10)
            ax.plot([0, 1], [rs[a_], rv[a_]], color=c, lw=1.3, marker="o", ms=3.5)
            ax.text(-0.06, rs[a_], f"{a_} {A.ev_std[a_]:.3f}", ha="right", va="center", fontsize=5.2, color=c)
            ax.text(1.06, rv[a_], f"{A.ev_valid[a_]:.3f} {a_}", ha="left", va="center", fontsize=5.2, color=c)
        ax.set_xlim(-0.9, 1.9); ax.set_xticks([0, 1]); ax.set_xticklabels(["standard", "valid"]); ax.invert_yaxis()
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
    else:
        missing(ax, "map_neurons.csv")

    # c: markers vs rich features per area (share lost)
    ax = fig.add_subplot(G[1, 1]); letter(ax, "c", "Richer movement description", x=-0.3)
    if M is not None:
        M["ins"] = M.session + M.probe.astype(str)
        sh = M.groupby("region").apply(lambda g: 100 * (1 - g.dM.sum() / g.dM_std.sum())).sort_values()
        y = np.arange(len(sh))
        ax.scatter(sh.values, y, s=18, color=BLUE, zorder=3, label="5 markers")
        if Rr is not None:
            rich = Rr.groupby("region").apply(lambda g: 100 * (1 - g.dM_rich.sum() / g.dM_std_rich.sum())).reindex(sh.index)
            for yi, a_, b_ in zip(y, sh.values, rich.values):
                ax.plot([a_, b_], [yi, yi], color=GRID, lw=2.2, zorder=1)
            ax.scatter(rich.values, y, s=18, color=NAVY, marker="D", zorder=3, label="69 features")
        n = M.groupby("region").ins.nunique().reindex(sh.index)
        ax.set_yticks(y); ax.set_yticklabels([f"{a_} ({k})" for a_, k in zip(sh.index, n)], fontsize=5.5)
        ax.set_xlabel("standard R² lost (%)"); ax.axvline(0, color=INK2, lw=0.4); ax.legend(loc="lower left", fontsize=5.3)
    else:
        missing(ax, "map_rows.csv")

    # d: electrode drift
    ax = fig.add_subplot(G[1, 2]); letter(ax, "d", "Electrode drift predicts it", x=-0.3)
    if Nn is not None and "drift_metric" in Nn and Nn.drift_metric.notna().any():
        D = Nn.dropna(subset=["drift_metric"]).copy()
        D["dec"] = D.groupby(["session", "probe"]).drift_metric.transform(lambda x: np.floor(10 * (x.rank(method="first") - 1) / len(x)))
        D["loss"] = D.ev_std - D.ev_valid
        per = D.groupby(["session", "probe", "dec"]).loss.mean().unstack()
        for _, r in per.sample(min(len(per), 80), random_state=0).iterrows():
            ax.plot(r.index + 1, r.values, color=NAVY, lw=0.3, alpha=0.15)
        m = per.mean(); se = per.sem()
        ax.fill_between(m.index + 1, m - 1.96 * se, m + 1.96 * se, color=NAVY, alpha=0.3, lw=0)
        ax.plot(m.index + 1, m, color=NAVY, lw=1.4, marker="o", ms=2.5)
        lo, hi = np.nanpercentile(per.values, [5, 95]); ax.set_ylim(lo, hi)
        ax.set_xticks([1, 5, 10]); ax.set_xlabel("Kilosort drift (decile within probe)"); ax.set_ylabel("EV lost per unit\n(standard − valid)")
        ax.text(0.03, 0.97, f"thin lines: insertions", transform=ax.transAxes, va="top", fontsize=5.3, color=INK2)
    else:
        missing(ax, "map_neurons.csv (drift_metric)")

    # e: unit-level density
    ax = fig.add_subplot(G[2, 0]); letter(ax, "e", "All neuron-epochs", x=-0.3)
    if Nn is not None:
        hi = float(np.nanpercentile(Nn.ev_std, 99)); lo = float(np.nanpercentile(Nn.ev_valid, 1))
        ax.hexbin(Nn.ev_valid.clip(lo, hi), Nn.ev_std.clip(lo, hi), gridsize=40, bins="log", mincnt=1, linewidths=0,
                  cmap=LinearSegmentedColormap.from_list("d", ["#fbefe6", "#f2a97f", ORANGE, "#7a2104"]), extent=(lo, hi, lo, hi))
        ax.plot([lo, hi], [lo, hi], ls="--", color=INK2, lw=0.6); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("valid EV"); ax.set_ylabel("standard EV")
        ax.text(0.04, 0.96, f"n = {len(Nn):,}\n{np.mean(Nn.ev_std > Nn.ev_valid):.0%} above", transform=ax.transAxes, va="top", fontsize=5.6)
    else:
        missing(ax, "map_neurons.csv")

    # f: smoothness rainclouds
    ax = fig.add_subplot(G[2, 1:]); letter(ax, "f", "Within-area structure along the probe is mostly real", x=-0.13)
    S = read(P7 / "results_map" / "map_smoothness.csv")
    if S is not None:
        cols = [("std", "real neurons,\nstandard", ORANGE), ("valid", "real neurons,\nvalid", BLUE), ("twin", "drift twin\n(no movement)", AQUA)]
        groups = {k: g for k, g in S.groupby("session")}
        for i, (c, lab, col) in enumerate(cols):
            half_violin(ax, S[c], i, col, width=0.4, side=1, seed=i)
            lo, hi = boot(groups, lambda g, c=c: g[c].mean(), 400); v = S[c].mean()
            ax.errorbar(i - 0.12, v, yerr=[[v - lo], [hi - v]], fmt="D", color=col, mec=INK, mew=0.5, ms=4, capsize=2, lw=1, ecolor=INK)
        ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in cols], fontsize=5.8); ax.set_xlim(-0.6, 2.6)
        ax.set_ylabel("spatial smoothness\n(above shuffled positions)")
        ax.text(0.99, 0.97, f"twin / real standard = {S.twin.mean() / S['std'].mean():.2f}", transform=ax.transAxes, ha="right",
                va="top", fontsize=5.8)
    else:
        missing(ax, "map_smoothness.csv")
    save(fig, "Fig3")


# ================================================================ Figure 4: why it happens, with ground truth
def fig4():
    import pandas as pd
    from scipy.stats import spearmanr
    fig = plt.figure(figsize=(7.1, 5.6))
    G = fig.add_gridspec(2, 3, height_ratios=[1, 1.1], hspace=0.55, wspace=0.5, left=0.08, right=0.97, top=0.93, bottom=0.09)
    C = read(P7 / "results_calibrate" / "calibration_long.csv")
    for k, ds in enumerate(("IBL", "MAP")):
        ax = fig.add_subplot(G[0, k]); letter(ax, "ab"[k], f"Planted codes of known size: {ds}", x=-0.25)
        if C is None or not (C.dataset == ds).any():
            missing(ax, "results_calibrate (run calibrate.py)"); continue
        D = C[C.dataset == ds]; lk = "level" if "level" in D else "truth"; levels = sorted(D[lk].unique())
        for col, c, mk in (("standard", ORANGE, "o"), ("valid", BLUE, "s")):
            W = D.pivot_table(index="mouse", columns=lk, values=col)            # thin lines: one per mouse
            for _, r in W.iterrows():
                ax.plot(levels, r.reindex(levels).values, color=c, lw=0.3, alpha=0.25)
            groups = {m_: g for m_, g in D.groupby("mouse")}
            ci = np.array([boot({m_: g[g[lk] == lv] for m_, g in groups.items()}, lambda d, col=col: d[col].mean(), 300) for lv in levels])
            m = np.array([D[D[lk] == lv][col].mean() for lv in levels])         # same pooled mean as calibrate.py test
            ax.fill_between(levels, ci[:, 0], ci[:, 1], color=c, alpha=0.3, lw=0)
            ax.plot(levels, m, color=c, marker=mk, ms=3.5, lw=1.4, label=col)
        mx = max(levels) * 1.12 + 0.005
        ax.plot([0, mx], [0, mx], ls="--", color=INK, lw=0.7, label="truth"); ax.set_xlim(-0.003, mx)
        ax.set_ylim(-0.03, mx * 1.15)
        ax.text(0.98, 0.12, "mouse lines outside the range are clipped", transform=ax.transAxes, ha="right", fontsize=4.8, color=INK2)
        ax.set_xlabel("true movement R²"); ax.set_ylabel("estimated movement R²"); ax.legend(loc="upper left", fontsize=5.6)
        ax.text(0.98, 0.04, f"thin lines: {D.mouse.nunique()} mice", transform=ax.transAxes, ha="right", fontsize=5.3, color=INK2)

    ax = fig.add_subplot(G[0, 2]); letter(ax, "c", "Estimate when the truth is zero", x=-0.3)
    if C is not None:
        Z = C[C.truth == 0].groupby(["dataset", "mouse"])[["standard", "valid"]].mean().reset_index()
        for i, ds in enumerate(("IBL", "MAP")):
            z = Z[Z.dataset == ds]
            if len(z):
                half_violin(ax, z.standard, 2 * i - 0.12, ORANGE, width=0.5, side=-1, seed=i)
                half_violin(ax, z.valid, 2 * i + 0.12, BLUE, width=0.5, side=1, seed=i + 5)
        ax.axhline(0, color=INK, lw=0.6, ls="--"); ax.set_xticks([0, 2]); ax.set_xticklabels(["IBL", "MAP"]); ax.set_xlim(-0.9, 2.9)
        lo_, hi_ = np.nanpercentile(np.r_[Z.standard, Z.valid], [3, 99.5]); ax.set_ylim(min(lo_, -0.01) * 1.2, hi_ * 1.3)
        ax.set_ylabel("estimated movement R² (per mouse)")
        ax.plot([], [], color=ORANGE, lw=4, label="standard"); ax.plot([], [], color=BLUE, lw=4, label="valid")
        ax.legend(loc="upper center", fontsize=5.6, ncol=2, bbox_to_anchor=(0.5, 1.02))
    else:
        missing(ax, "results_calibrate")

    Dl = read(P7 / "results_calibrate" / "calibrate_rows.csv")
    ax = fig.add_subplot(G[1, 0:2]); letter(ax, "d", None, x=-0.08, y=1.2); ax.text(0, 1.2, "Inflation over the slow-drift plane", transform=ax.transAxes, fontsize=7, va="top")
    if Dl is not None and {"slow_neural", "slow_move"} <= set(Dl.columns):
        qx = np.nanpercentile(Dl.slow_neural, 98); qy = np.nanpercentile(Dl.slow_move, 98)
        bx = np.linspace(0, qx, 15); by = np.linspace(0, qy, 15)
        ix = np.digitize(Dl.slow_neural, bx) - 1; iy = np.digitize(Dl.slow_move, by) - 1
        Mz = np.full((len(by) - 1, len(bx) - 1), np.nan)
        for i in range(len(bx) - 1):
            for j in range(len(by) - 1):
                s = Dl.inflation[(ix == i) & (iy == j)]
                if len(s) >= 5:
                    Mz[j, i] = s.mean()
        vmax = np.nanpercentile(np.abs(Mz), 97) if np.isfinite(Mz).any() else 1
        pm = ax.pcolormesh(bx, by, Mz, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax / 3, vcenter=0, vmax=vmax), shading="flat")
        XX, YY = np.meshgrid(np.linspace(1e-4, qx, 200), np.linspace(1e-4, qy, 200))
        lv = np.nanpercentile(Dl.law, [50, 80, 95])
        cs = ax.contour(XX, YY, XX * YY, levels=np.unique(lv), colors=INK, linewidths=0.6, linestyles="--")
        ax.clabel(cs, fmt=lambda v: f"law {v:.3f}", fontsize=5)
        ax.set_xlabel("slow share of neural variance"); ax.set_ylabel("slow share of movement variance")
        ax.set_aspect("auto"); cb = plt.colorbar(pm, cax=ax.inset_axes([0.55, 1.04, 0.42, 0.035]), orientation="horizontal"); cb.ax.xaxis.set_ticks_position("top")
        cb.outline.set_linewidth(0.3)
        cb.ax.tick_params(labelsize=5); cb.set_label("standard − valid movement R²", fontsize=5.5, labelpad=1); cb.ax.xaxis.set_label_position("top")
        ax.text(0.99, 0.02, "dashed: equal product (law)", transform=ax.transAxes, ha="right", fontsize=5.3, color=INK)
    else:
        missing(ax, "calibrate_rows.csv (run calibrate.py)")

    ax = fig.add_subplot(G[1, 2]); letter(ax, "e", "One number predicts it", x=-0.3)
    if Dl is not None:
        for ds, c in (("IBL", NAVY), ("MAP", ORANGE)):
            g = Dl[Dl.dataset == ds]
            if len(g) < 10:
                continue
            ax.scatter(g.law, g.inflation, s=1.5, color=c, alpha=0.15, lw=0)
            q = pd.qcut(g.law, 8, labels=False, duplicates="drop")
            m = g.groupby(q).agg(x=("law", "median"), y=("inflation", "mean"), se=("inflation", "sem"))
            rho = spearmanr(g.law, g.inflation)[0]
            ax.errorbar(m.x, m.y, yerr=1.96 * m.se, color=c, marker="o", ms=3, lw=1.3, capsize=1.5, label=f"{ds}  ρ = {rho:+.2f}")
        hi = np.nanpercentile(Dl.inflation, 99); lo = np.nanpercentile(Dl.inflation, 1)
        ax.set_ylim(lo, hi); ax.set_xlim(0, np.nanpercentile(Dl.law, 99)); ax.axhline(0, color=INK2, lw=0.5)
        ax.set_xlabel("law = neural × movement slow share"); ax.set_ylabel("standard − valid movement R²")
        ax.legend(loc="upper left", fontsize=5.6)
    else:
        missing(ax, "calibrate_rows.csv")
    save(fig, "Fig4")


# ================================================================ Figure 5: hidden where empty, and replication
def fig5():
    """The hidden block prior: what is detected, what is carried, and what it would take to settle each region."""
    import pandas as pd
    fig = plt.figure(figsize=(7.1, 4.7))
    G = fig.add_gridspec(2, 3, height_ratios=[0.55, 1], hspace=0.62, wspace=0.5, left=0.08, right=0.985, top=0.93, bottom=0.1)
    R, F = ibl_prior_table()
    # a: detected
    ax = fig.add_subplot(G[0, 0]); letter(ax, "a", None, x=-0.02, y=1.22)
    t_ = ax.text(0.08, 1.22, "Prior detected", transform=ax.transAxes, fontsize=7, va="top")
    if R is not None and "fdr_valid" in R:
        v = R["fdr_valid"].astype(float)
        swanson(ax, R.index, v, ListedColormap(["#d9d7d0", NAVY]), 0, 1, "detected", ticks=[0.25, 0.75], ticklabels=["no", "yes"])
        t_.set_text(f"Prior detected ({100 * v.mean():.0f}%)")
    else:
        missing(ax, "prior_map_regions*.csv")
    # b: carried, movement removed
    ax = fig.add_subplot(G[0, 1]); letter(ax, "b", None, x=-0.02, y=1.22)
    t_ = ax.text(0.08, 1.22, "Carries the prior, movement removed", transform=ax.transAxes, fontsize=7, va="top")
    if R is not None and "p_carries_move" in R:
        v = R["p_carries_move"].astype(float)
        swanson(ax, R.index, v, CM_PRIOR, 0, 1, "posterior probability", ticks=[0, 0.5, 1])
        t_.set_text(f"Carried, movement out ({100 * (v > 0.5).mean():.0f}% > 0.5)")
    else:
        missing(ax, "movement_regions.csv (p_carries_move)")
    # c: sessions needed
    ax = fig.add_subplot(G[0, 2]); letter(ax, "c", None, x=-0.02, y=1.22)
    ax.text(0.08, 1.22, "Sessions needed to settle it", transform=ax.transAxes, fontsize=7, va="top")
    if F is not None and "sessions_needed_r0.06" in F:
        v = np.log10(F["sessions_needed_r0.06"].astype(float).clip(1, 300))
        swanson(ax, F.index, v, "magma_r", 0, np.log10(300), "sessions needed",
                ticks=[0, 1, 2], ticklabels=["1", "10", "100"], extend="max")
    else:
        missing(ax, "prior_map_figure_table*.csv")

    ax = fig.add_subplot(G[1, 0]); letter(ax, "d", "Movement explains little of it", x=-0.3)
    if R is not None and "p_carries_move" in R:
        sysm = system_of(R.index)
        sz = 4 + 2.5 * np.sqrt(R["n_sess"]) if "n_sess" in R else 8
        ax.scatter(R.p_carries, R.p_carries_move, s=sz, c=[atlas_rgb(s) if s else (0.6, 0.6, 0.6) for s in sysm], lw=0.3,
                   edgecolor="white", alpha=0.9)
        ax.plot([0, 1], [0, 1], ls="--", color=INK2, lw=0.6); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("P(carries prior)"); ax.set_ylabel("P(carries prior), movement removed")
        for s in [s for s in COSMOS if s in set(sysm)][:6]:
            ax.scatter([], [], s=10, color=atlas_rgb(s), label=SYSN[s])
        ax.legend(fontsize=4.8, loc="upper left", handletextpad=0.1, labelspacing=0.25, borderaxespad=0.2)
    else:
        missing(ax, "movement_regions.csv")

    ax = fig.add_subplot(G[1, 1]); letter(ax, "e", "What would settle each region", x=-0.3)
    if F is not None and "sessions_needed_r0.06" in F:
        f = np.isfinite(F["sessions_needed_r0.06"])
        col = {"detected": NAVY, "hidden (likely carries it)": AQUA, "unanswerable": "#9b9a96", "likely absent": "#d9d6cf"}
        for c, cc in col.items():
            s = f & (F.category == c)
            ax.scatter(F.n_sess[s], F["sessions_needed_r0.06"][s], s=7, color=cc, lw=0, label=c.replace(" (likely carries it)", ""))
        mx = float(max(np.nanmax(F["sessions_needed_r0.06"][f]), F.n_sess.max())) * 1.3
        ax.fill_between([1, mx], [1, mx], [mx, mx], color="#f4f2ec", lw=0, zorder=0)
        ax.plot([1, mx], [1, mx], ls="--", color=INK2, lw=0.6); ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, mx); ax.set_ylim(1, mx)
        ax.text(0.97, 0.05, "settled", transform=ax.transAxes, ha="right", fontsize=5.6, color=INK2)
        ax.text(0.05, 0.93, "needs more data", transform=ax.transAxes, fontsize=5.6, color=INK2)
        ax.set_xlabel("sessions recorded"); ax.set_ylabel("sessions needed (80% power)")
        ax.legend(fontsize=4.8, loc="lower right", bbox_to_anchor=(1.0, 0.1), handletextpad=0.1, labelspacing=0.25, frameon=True, framealpha=0.9, edgecolor="none")
    else:
        missing(ax, "prior_map_figure_table*.csv")

    # f: what the data can answer (SUMMARY)
    ax = fig.add_subplot(G[1, 2]); letter(ax, "f", "What the data can answer", x=-0.42)
    A = SUMMARY["answer"]
    Mx = np.array([A["shuffle"], A["valid"], A["answerable"]], float).T
    ax.imshow(Mx, cmap=LinearSegmentedColormap.from_list("a", ["#ffffff", "#cfe0f5", BLUE, NAVY]), vmin=0, vmax=100, aspect="auto")
    for (i, k), v in np.ndenumerate(Mx):
        ax.text(k, i, f"{v:.0f}%", ha="center", va="center", fontsize=6, color="white" if v > 60 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(["detected,\nstandard", "detected,\nvalid", "answerable\nat r = 0.1"], fontsize=5.3)
    ax.set_yticks(range(4)); ax.set_yticklabels(A["names"], fontsize=6); ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.text(0.5, -0.38, "earlier pre-registered runs (SUMMARY)", transform=ax.transAxes, ha="center", fontsize=5, color=INK2)
    save(fig, "Fig5")


if __name__ == "__main__":
    figs = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5}
    sel = [int(a) for a in sys.argv[1:]] or list(figs)
    for k in sel:
        try:
            figs[k]()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"Fig{k} FAILED: {type(e).__name__}: {e}")
