"""PART 3 mechanism, causal: which cortical regions are needed to READ the reward memory, and which are needed to
TAG a choice so that a later reward can be stored?

Data: Zatka-Haas et al. 2021 eLife (figshare 13008038), data/inactivation/Inactivation_52Coord.mat (MATLAB table D,
one row per trial; repeat trials already removed by the authors). Same lab and task family as Steinmetz 2019.
Unilateral optogenetic silencing (VGAT/PV-ChR2) at one of 52 sites on ~75% of trials, from stimulus onset until the
choice. The laser is OFF during the outcome and the inter-trial interval.

C1 READOUT. Trials with a Left/Right choice:
    logit P(Right) = session bias + stimulus (+ its change under laser, per hemisphere) + laser main effect (per
                     hemisphere) + previous choice + previous rewarded side (rc)  +  gamma1 * laser_area(t) * rc
    gamma1 < 0: silencing that area during the decision weakens the use of the reward memory (needed to read it).
C2 TAGGING (eligibility trace). Same model, but the laser term is on the PREVIOUS trial:
    ... + previous-trial laser main effect (x previous choice) + rc x previous contrast + gamma2 * laser_area(t-1) * rc
    The laser on t-1 ends at the choice, BEFORE the reward. gamma2 < 0 means the area's activity at the moment of
    choice is needed for the reward that follows to be stored (a synaptic eligibility trace).
Null for both: parametric bootstrap from the fitted model without the tested term (all real laser and history\neffects kept). Calibrated in simulation; label permutation was slightly liberal because it erases the laser's real effects.
Positive control (must replicate the paper): VIS / M2 silencing reduces contralateral choices.
Previous-trial variables come from the authors' prev_response / prev_feedback (exact). The previous-trial LASER needs
the row before; it is used only where that row matches prev_response, prev_feedback and prev_stimulus.

    python3 part3/causal_readout.py inspect --mat <path to Inactivation_52Coord.mat>
    python3 part3/causal_readout.py sim
    python3 part3/causal_readout.py run --mat <path>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

N_PERM = 1000
MIN_TRIALS = 300


# ------------------------------------------------------------------------------------------- loading
def _split(df, col, names):
    if col not in df:
        return df
    v = df[col].values
    if v.dtype == object:
        A = np.vstack([np.asarray(x, float).ravel() for x in v])
    else:
        A = np.asarray(v, float).reshape(len(df), -1)
    for j, n in enumerate(names[:A.shape[1]]):
        df[n] = A[:, j]
    return df.drop(columns=[col])


def _decode_table(obj):
    """MATLAB table saved by older MATLAB versions -> DataFrame. Numeric columns as is (matrix columns split into
    name_1, name_2, ...), categorical columns decoded from codes + category names, other column types skipped."""
    import pandas as pd
    P = obj.properties
    data, varnames = P["data"], P["varnames"]
    nrows = int(np.asarray(P["nrows"]).item())
    names = [str(np.asarray(varnames[0, i]).ravel()[0]) for i in range(varnames.shape[1])]
    cols, skipped = {}, []
    for i, name in enumerate(names):
        x = data[0, i]
        if getattr(x, "classname", None) == "categorical":
            cp = x.properties
            cats = [str(np.asarray(c).ravel()[0]) if np.asarray(c).size else "" for c in np.asarray(cp["categoryNames"]).ravel()]
            codes = np.asarray(cp["codes"]).astype(int).ravel()
            cols[name] = np.array([cats[c - 1] if c > 0 else "<undefined>" for c in codes], dtype=object)
            continue
        A = None
        if isinstance(x, np.ndarray) and x.dtype.kind in "biuf" and x.shape[0] == nrows:
            A = x.astype(float).reshape(nrows, -1)
        elif isinstance(x, np.ndarray) and x.dtype == object and x.size == nrows:
            try:                                  # one small numeric array per row
                A = np.vstack([np.asarray(v, float).ravel() for v in x.ravel()])
            except Exception:
                A = None
        if A is not None and A.shape[1] > 4:     # wheel traces etc.: not needed
            A = None
        if A is not None:
            if A.shape[1] == 1:
                cols[name] = A[:, 0]
            else:
                for j in range(A.shape[1]):
                    cols[f"{name}_{j + 1}"] = A[:, j]
            continue
        skipped.append(name)
    if skipped:
        print(f"(columns not needed and skipped: {skipped})")
    df = pd.DataFrame(cols)
    ren = {"stimulus_1": "CL", "stimulus_2": "CR", "prev_stimulus_1": "prevCL", "prev_stimulus_2": "prevCR",
           "laserCoord_1": "ML", "laserCoord_2": "AP"}
    return df.rename(columns={k: v for k, v in ren.items() if k in df})


def load(path):
    import pandas as pd
    path = Path(path).expanduser()
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        import matio
        z = matio.load_from_mat(str(path), variable_names=["D"])
        df = z["D"]
        if not isinstance(df, pd.DataFrame):
            if getattr(df, "classname", None) == "table":
                df = _decode_table(df)          # older MATLAB tables (no version field): decode the columns directly
            else:
                raise TypeError(f"D was read as {type(df)}, not a table; paste this message")
    df = df.copy()
    # matrix-valued columns come as 2-D arrays (or arrays per row): split the ones we need
    for col, names in (("stimulus", ["CL", "CR"]), ("prev_stimulus", ["prevCL", "prevCR"]), ("laserCoord", ["ML", "AP"])):
        df = _split(df, col, names)
    keep = [c for c in df.columns if df[c].dtype != object or c in ("laserRegion",)]
    df = df[keep]
    if "laserRegion" in df:
        df["laserRegion"] = df["laserRegion"].astype(str)
    return df.reset_index(drop=True)


def inspect(path):
    import pandas as pd
    import matio
    path = Path(path).expanduser()
    try:
        z = matio.load_from_mat(str(path), variable_names=["D"])
        raw = z["D"]
        print(f"D read as {type(raw).__name__}")
        if not isinstance(raw, pd.DataFrame) and getattr(raw, "classname", None) == "table":
            raw = _decode_table(raw)
            print("(older MATLAB table format: decoded directly)")
        if isinstance(raw, pd.DataFrame):
            for c in raw.columns:
                v = raw[c].values
                ex = v[0] if len(v) else None
                print(f"  {c}: dtype {raw[c].dtype}  example {np.asarray(ex).ravel()[:4] if ex is not None else None}")
    except Exception as e:
        print(f"could not read: {type(e).__name__}: {e}")
        return
    df = load(path)
    print(f"\n{len(df)} trials, columns after splitting: {list(df.columns)}")
    for c in ("response", "feedbackType", "laserType", "repeatNum", "prev_response", "prev_feedback"):
        if c in df:
            print(f"  {c}: values {dict(zip(*np.unique(df[c].dropna(), return_counts=True)))}")
    if "laserRegion" in df:
        print(f"  laserRegion: {df.laserRegion.value_counts().head(30).to_dict()}")
    if "sessionID" in df:
        print(f"  sessions {df.sessionID.nunique()}, subjects {df.subjectID.nunique() if 'subjectID' in df else '?'}")
    print(df.head(5).to_string())


# ------------------------------------------------------------------------------------------- variables
def area_of(region, ml, ap, sites=False):
    """Area of a laser site from its coordinate (mm from bregma), following the authors' own definitions
    (Zatka-Haas code: loadBehaviouralData.m and inactivation_preprocessing.m). Hemisphere is handled separately."""
    if not np.isfinite(ml):
        return "off"
    a, m = float(ap), abs(float(ml))
    if sites:
        return f"site_ML{m:g}_AP{a:g}"
    if a == 3 and m == 3.5:
        return "FrontOutside"
    if a == -7:
        return "BackOutside"
    if m == 0.5 and a == -2:
        return "RSP"
    if m == 3.5 and a == -0.5:
        return "S1"
    if m == 1 and a == -0.5:
        return "M1"
    if a <= -2 and m > 1:
        return "VIS"
    if a == 0:
        return "S1"
    if 2 <= a <= 3:
        return "M2"
    return "Other"


def prepare(df, sites=False):
    f = lambda x: np.tanh(4 * np.clip(np.nan_to_num(np.asarray(x, float)), 0, None)) / np.tanh(4)
    d = dict(sess=df["sessionID"].values.astype(int), resp=df["response"].values.astype(int),
             fb=np.asarray(df["feedbackType"], float))
    d["CL"], d["CR"] = f(df["CL"]), f(df["CR"])
    lt = np.asarray(df["laserType"], float) if "laserType" in df else np.isfinite(df["ML"]).astype(float)
    ml = np.asarray(df["ML"], float) if "ML" in df else np.full(len(df), np.nan)
    ap = np.asarray(df["AP"], float) if "AP" in df else np.full(len(df), np.nan)
    on = (lt == 1) & np.isfinite(ml) & (ml != 0)                       # unilateral silencing only
    reg = df["laserRegion"].values if "laserRegion" in df else np.full(len(df), "")
    d["area"] = np.array([area_of(r, x, y, sites) if o else "off" for r, x, y, o in zip(reg, ml, ap, on)])
    d["hemi"] = np.where(on, np.sign(ml), 0.0)                          # -1 left, +1 right
    # previous trial (exact, from the authors' columns)
    pr = np.asarray(df["prev_response"], float) if "prev_response" in df else np.r_[np.nan, d["resp"][:-1]]
    pf = np.asarray(df["prev_feedback"], float) if "prev_feedback" in df else np.r_[np.nan, d["fb"][:-1]]
    pc = np.where(pr == 2, 1.0, np.where(pr == 1, -1.0, 0.0))
    d["c_prev"] = pc
    d["rc_prev"] = pc * (pf > 0)
    pcl = np.asarray(df["prevCL"], float) if "prevCL" in df else np.r_[np.nan, df["CL"].values[:-1]]
    pcr = np.asarray(df["prevCR"], float) if "prevCR" in df else np.r_[np.nan, df["CR"].values[:-1]]
    d["prev_diff"] = np.abs(np.nan_to_num(pcr) - np.nan_to_num(pcl))
    # previous-trial laser: the row before, only where it really is the previous trial
    same = np.r_[False, d["sess"][1:] == d["sess"][:-1]]
    match = same & np.r_[False, d["resp"][:-1] == pr[1:]] & np.r_[False, d["fb"][:-1] == pf[1:]]
    if "prevCL" in df:
        match &= np.r_[False, np.isclose(np.nan_to_num(df["CL"].values[:-1]), np.nan_to_num(pcl[1:]))]
    d["prev_ok"] = match
    d["area_prev"] = np.concatenate([np.array(["off"]), d["area"][:-1]])
    d["hemi_prev"] = np.r_[0.0, d["hemi"][:-1]]
    d["prev_valid"] = np.isfinite(pr) & np.isfinite(pf)
    return d


# ------------------------------------------------------------------------------------------- model
def logit_fit(X, y, w0=None, lam=1e-3, iters=30):
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) + lam * w
        H = (X * (p * (1 - p))[:, None]).T @ X + lam * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


def design(d, rows, area, kind, lab_area, lab_hemi):
    """Columns: session dummies, stimulus, laser terms for this area (per hemisphere), history, test term (last)."""
    sess = d["sess"][rows]
    us, si = np.unique(sess, return_inverse=True)
    S = np.zeros((len(rows), len(us))); S[np.arange(len(rows)), si] = 1
    L = (lab_area[rows] == area).astype(float)
    hl, hr = L * (lab_hemi[rows] < 0), L * (lab_hemi[rows] > 0)
    CL, CR = d["CL"][rows], d["CR"][rows]
    rc, cp = d["rc_prev"][rows], d["c_prev"][rows]
    cols = [S, CL, CR, hl, hr, hl * CL, hl * CR, hr * CL, hr * CR, cp, rc]
    if kind == "readout":
        # the current-trial laser for this area is the tested manipulation; any other laser on trial t is a covariate
        other = ((d["area"][rows] != "off") & (d["area"][rows] != area)).astype(float)
        cols += [other]
        test = L * rc
    else:
        cols += [hl * cp, hr * cp, rc * d["prev_diff"][rows]]
        cur = (d["area"][rows] != "off").astype(float)               # laser on the current trial (any area)
        cols += [cur, cur * d["hemi"][rows], cur * d["hemi"][rows] * (CR - CL)]
        # readout effects of the CURRENT trial's laser (any area) on the memory, so tagging is not confused with reading
        cols += [(d["area"][rows] == a).astype(float) * rc for a in np.unique(d["area"]) if a != "off"]
        test = L * rc
    X = np.column_stack(cols + [test])
    j_rc = sum(c.shape[1] if np.ndim(c) == 2 else 1 for c in cols[:10])     # position of rc (the 11th block)
    return X, j_rc


def area_test(d, area, kind, n_perm=N_PERM, seed=0):
    go = (d["resp"] == 1) | (d["resp"] == 2)
    if kind == "readout":
        lab_area, lab_hemi = d["area"], d["hemi"]
        rows = np.flatnonzero(go & d["prev_valid"] & ((d["area"] == "off") | (d["area"] == area)))
    else:
        lab_area, lab_hemi = d["area_prev"], d["hemi_prev"]
        rows = np.flatnonzero(go & d["prev_valid"] & d["prev_ok"] & ((d["area_prev"] == "off") | (d["area_prev"] == area)))
    n_on = int((lab_area[rows] == area).sum())
    if n_on < MIN_TRIALS:
        return None
    y = (d["resp"][rows] == 2).astype(float)
    X, j_rc = design(d, rows, area, kind, lab_area, lab_hemi)
    w = logit_fit(X, y)
    real = w[-1]
    beta_rc = w[j_rc]                                                # coefficient of rc (no laser)
    if kind != "readout":                                            # rc also enters as rc x previous contrast difference
        rcv = d["rc_prev"][rows] != 0
        beta_rc = w[j_rc] + w[j_rc + 3] * d["prev_diff"][rows][rcv].mean()
    rng = np.random.default_rng(seed)
    # null: parametric bootstrap from the fitted model WITHOUT the tested term, so every real effect of the laser
    # (on the stimulus response, on bias) and of history is kept and only the tested interaction is set to zero
    w0 = logit_fit(X[:, :-1], y)
    p0 = 1 / (1 + np.exp(-np.clip(X[:, :-1] @ w0, -30, 30)))
    null = []
    for _ in range(n_perm):
        yb = (rng.random(len(y)) < p0).astype(float)
        null.append(logit_fit(X, yb, w0=np.r_[w0, 0.0], iters=15)[-1])
    null = np.asarray(null)
    # positive control (readout only): stimulus gain on the contralateral side under silencing
    ctrl = np.nan
    if kind == "readout":
        k0 = len(np.unique(d["sess"][rows]))
        ctrl = 0.5 * (w[k0 + 5] - w[k0 + 6])      # left hemi x CR, minus right hemi x CL (contralateral stimulus)
    return dict(area=area, kind=kind, n_laser=n_on, n=len(rows), gamma=float(real), beta_rc=float(beta_rc),
                frac_removed=float(-real / beta_rc) if beta_rc > 0 else np.nan,
                z=float((real - null.mean()) / (null.std() + 1e-12)),
                p_less=float((1 + (null <= real).sum()) / (1 + n_perm)), contra_stim_change=float(ctrl))


def side_test(d, area, n_boot=N_PERM, seed=0):
    """Follow-up fixed in advance (Oct 25, after the first tagging result): is the memory boost from silencing on the
    previous trial larger when the silenced hemisphere is the one that saw the CHOSEN stimulus (contralateral to the
    chosen side) than when it is the other hemisphere? A confidence-weighted write rule predicts contra > ipsi;
    a general effect of the laser (light, distraction, arousal) predicts no difference.
    Null: bootstrap from the model with one common boost term (no hemisphere difference)."""
    go = (d["resp"] == 1) | (d["resp"] == 2)
    lab_area, lab_hemi = d["area_prev"], d["hemi_prev"]
    rows = np.flatnonzero(go & d["prev_valid"] & d["prev_ok"] & ((d["area_prev"] == "off") | (d["area_prev"] == area)))
    L = (lab_area[rows] == area).astype(float)
    if L.sum() < MIN_TRIALS:
        return None
    y = (d["resp"][rows] == 2).astype(float)
    X, j_rc = design(d, rows, area, "tagging", lab_area, lab_hemi)
    base = X[:, :-1]
    rc, cp, hp = d["rc_prev"][rows], d["c_prev"][rows], lab_hemi[rows]
    contra = L * rc * (hp == -cp)
    ipsi = L * rc * (hp == cp)
    Xf = np.column_stack([base, contra, ipsi])
    w = logit_fit(Xf, y)
    stat = w[-2] - w[-1]
    X0 = np.column_stack([base, L * rc])
    w0 = logit_fit(X0, y)
    p0 = 1 / (1 + np.exp(-np.clip(X0 @ w0, -30, 30)))
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_boot):
        yb = (rng.random(len(y)) < p0).astype(float)
        wb = logit_fit(Xf, yb, w0=np.r_[w0[:-1], w0[-1], w0[-1]], iters=15)
        null.append(wb[-2] - wb[-1])
    null = np.asarray(null)
    # does silencing that hemisphere on the previous trial lower accuracy there (it should, for VIS)?
    prev_rows = rows[d["prev_ok"][rows]]
    return dict(area=area, n_contra=int((contra != 0).sum()), n_ipsi=int((ipsi != 0).sum()),
                boost_contra=float(w[-2]), boost_ipsi=float(w[-1]), diff=float(stat),
                z=float((stat - null.mean()) / (null.std() + 1e-12)),
                p_greater=float((1 + (null >= stat).sum()) / (1 + n_boot)))


# ------------------------------------------------------------------------------------------- simulation
def fake_table(rng, n_sess=30, T=400, read_area=None, tag_area=None, effect=0.8, conf_write=0.0, general_boost=0.0):
    """Synthetic table in the Zatka-Haas format. read_area: silencing it removes `effect` of the memory on that trial.
    tag_area: silencing it on trial t-1 removes `effect` of the memory that trial's reward creates."""
    import pandas as pd
    sites = [("LeftVIS", -2.5, -3.5), ("RightVIS", 2.5, -3.5), ("LeftM2", -1, 2), ("RightM2", 1, 2),
             ("LeftS1", -3.5, -0.5), ("RightS1", 3.5, -0.5)]
    rows = []
    levels = [0, .1, .24, .54]
    for s in range(n_sess):
        prev = None
        for t in range(T):
            cl, cr = rng.choice(levels), rng.choice(levels)
            on = rng.random() < .75
            reg, ml, ap = (sites[rng.integers(len(sites))] if on else ("", np.nan, np.nan))
            area = area_of(reg, ml, ap) if on else "off"
            lg = 4 * (np.tanh(4 * cr) - np.tanh(4 * cl)) / np.tanh(4)
            if on and area in ("VIS", "M2"):                          # contralateral stimulus weakened
                lg += (-2.5 * np.tanh(4 * cr) if ml < 0 else 2.5 * np.tanh(4 * cl)) / np.tanh(4)
            if prev is not None:
                rc = prev["rc"] * (1 - effect * (prev["area"] == tag_area))
                rc = rc * (1 + conf_write * (1 - prev["conf"])) * (1 + general_boost * (prev["area"] != "off"))
                lg += 1.0 * rc * (1 - effect * (area == read_area)) + 0.2 * prev["c"]
            pgo = 0.85
            pR = 1 / (1 + np.exp(-lg))
            resp = 3 if rng.random() > pgo else (2 if rng.random() < pR else 1)
            corr = (cr > cl and resp == 2) or (cl > cr and resp == 1) or (cl == cr and cl > 0 and resp < 3 and rng.random() < .5) \
                or (cl == cr == 0 and resp == 3)
            fb = 1 if corr else -1
            row = dict(sessionID=s + 1, subjectID=s // 5 + 1, response=resp, feedbackType=fb, laserType=int(on),
                       laserRegion=reg if on else "", stimulus=np.array([cl, cr]),
                       laserCoord=np.array([ml, ap]) if on else np.array([np.nan, np.nan]))
            if prev is not None:
                row.update(prev_response=prev["resp"], prev_feedback=prev["fb"], prev_stimulus=prev["stim"])
            else:
                row.update(prev_response=np.nan, prev_feedback=np.nan, prev_stimulus=np.array([np.nan, np.nan]))
            rows.append(row)
            c = 1 if resp == 2 else (-1 if resp == 1 else 0)
            prev = dict(resp=resp, fb=fb, stim=np.array([cl, cr]), c=c, rc=c * (fb > 0), area=area,
                        conf=(pR if resp == 2 else 1 - pR))
    df = pd.DataFrame(rows)
    keep = rng.random(len(df)) > 0.03                                 # rows removed (repeats), as in the real table
    return df[keep].reset_index(drop=True)


def _prep_fake(df):
    df = df.copy()
    for col, names in (("stimulus", ["CL", "CR"]), ("prev_stimulus", ["prevCL", "prevCR"]), ("laserCoord", ["ML", "AP"])):
        df = _split(df, col, names)
    return df


def sim(reps=20, seed=0):
    from joblib import Parallel, delayed

    def one(i, read_area, tag_area):
        rng = np.random.default_rng(seed + 31 * i)
        d = prepare(_prep_fake(fake_table(rng, read_area=read_area, tag_area=tag_area)))
        r1 = area_test(d, "M2", "readout", n_perm=100, seed=i)
        r2 = area_test(d, "M2", "tagging", n_perm=100, seed=i)
        return r1["p_less"] < .05, r2["p_less"] < .05, r1["contra_stim_change"]

    for read_area, tag_area, lab in ((None, None, "no effect"), ("M2", None, "M2 needed to READ (80%)"),
                                     (None, "M2", "M2 needed to TAG (80%)")):
        R = np.array(Parallel(n_jobs=-1)(delayed(one)(i, read_area, tag_area) for i in range(reps)))
        print(f"{lab:30s}  readout test rejects {R[:, 0].mean():.2f}   tagging test rejects {R[:, 1].mean():.2f}   "
              f"(positive control: contralateral stimulus change {R[:, 2].mean():.2f}, should be < 0)", flush=True)


def sim_side(reps=60, seed=0):
    from joblib import Parallel, delayed

    def one(i, cw, gb):
        rng = np.random.default_rng(seed + 53 * i)
        d = prepare(_prep_fake(fake_table(rng, conf_write=cw, general_boost=gb)))
        r = side_test(d, "VIS", n_boot=100, seed=i)
        return r["p_greater"] < .05, r["boost_contra"], r["boost_ipsi"]

    for cw, gb, lab in ((0, 0, "no effect"), (0, 0.5, "general laser boost (both hemispheres)"),
                        (1.5, 0, "confidence-weighted writing")):
        R = np.array(Parallel(n_jobs=-1)(delayed(one)(i, cw, gb) for i in range(reps)))
        print(f"{lab:42s} side test rejects {R[:, 0].mean():.3f}   boost contra {R[:, 1].mean():.2f}  ipsi {R[:, 2].mean():.2f}",
              flush=True)


def run_side(mat, out_dir, jobs, n_boot):
    import pandas as pd
    from joblib import Parallel, delayed
    d = prepare(load(mat))
    areas = sorted(str(a) for a in set(d["area"]) if a != "off")
    res = Parallel(n_jobs=jobs)(delayed(side_test)(d, a, n_boot, 11 + i) for i, a in enumerate(areas))
    R = pd.DataFrame([r for r in res if r is not None])
    L = ["# Follow-up: is the memory boost from silencing on the previous trial specific to the hemisphere that saw the "
         "chosen stimulus?\n",
         "Prediction fixed in advance (confidence-weighted writing): VIS boost contra > ipsi. A general laser effect gives "
         "no difference. Null: bootstrap from a model with one common boost.\n",
         "| Area | Rewarded trials, contra | ipsi | Boost contra | Boost ipsi | Difference | z | p (one-sided) |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in R.iterrows():
        L.append(f"| {r.area} | {r.n_contra} | {r.n_ipsi} | {r.boost_contra:.2f} | {r.boost_ipsi:.2f} | {r["diff"]:.2f} | "
                 f"{r.z:.2f} | {r.p_greater:.3f} |")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "CAUSAL_SIDE.md").write_text(text + "\n")
    print(text)


def diagnose(mat, out_dir, n_boot=500, seed=0):
    """Is the reward memory present after laser-OFF trials, or only after laser-ON trials? Two independent views.
    1. Model: same covariates as the tagging model, but the memory term is split by the previous trial's laser:
       rc x [previous laser off], rc x [previous laser on, each area]. Uncertainty: bootstrap over sessions.
    2. Model-free: on current trials where the stimulus does not favour a side (equal contrasts, Left/Right choice),
       how often does the mouse repeat the previous REWARDED choice, after laser-off vs laser-on previous trials?
       Same for the previous UNREWARDED choice (perseveration, for comparison)."""
    import pandas as pd
    d = prepare(load(mat))
    go = (d["resp"] == 1) | (d["resp"] == 2)
    rows = np.flatnonzero(go & d["prev_valid"] & d["prev_ok"])
    sess = d["sess"][rows]
    us, si = np.unique(sess, return_inverse=True)
    S = np.zeros((len(rows), len(us))); S[np.arange(len(rows)), si] = 1
    CL, CR = d["CL"][rows], d["CR"][rows]
    rc, cp = d["rc_prev"][rows], d["c_prev"][rows]
    cur = (d["area"][rows] != "off").astype(float); hc = d["hemi"][rows]
    pa = d["area_prev"][rows]
    areas = [a for a in sorted(set(map(str, pa))) if a != "off"]
    cols = [S, CL, CR, cur, cur * hc, cur * hc * (CR - CL), cp,
            (d["area_prev"][rows] != "off").astype(float) * d["hemi_prev"][rows]]
    names = ["memory after laser-OFF previous trial"] + [f"memory after laser-ON previous trial ({a})" for a in areas]
    test = [rc * (pa == "off")] + [rc * (pa == a) for a in areas]
    X = np.column_stack(cols + test)
    y = (d["resp"][rows] == 2).astype(float)
    w = logit_fit(X, y)
    k = len(test)
    est = w[-k:]
    rng = np.random.default_rng(seed)
    B = []
    for _ in range(n_boot):                                   # resample whole sessions
        pick = rng.choice(len(us), len(us), replace=True)
        ix = np.concatenate([np.flatnonzero(si == j) for j in pick])
        Xb = X[ix]
        keep = Xb.std(0) > 0
        keep[-k:] = True
        wb = np.full(X.shape[1], np.nan)
        wb[keep] = logit_fit(Xb[:, keep], y[ix], iters=15)
        B.append(wb[-k:])
    B = np.asarray(B)
    L = ["# Diagnostic: is the reward memory present after normal (laser-off) trials?\n",
         "Memory = weight of the previous rewarded side on the current choice (log-odds), split by the previous trial's laser. "
         "95% intervals: bootstrap over sessions.\n", "| Previous trial | Rewarded trials | Memory | 95% interval |", "|---|---|---|---|"]
    for j, n in enumerate(names):
        cnt = int(((test[j]) != 0).sum())
        lo, hi = np.nanpercentile(B[:, j], [2.5, 97.5])
        L.append(f"| {n} | {cnt} | {est[j]:.2f} | {lo:.2f} to {hi:.2f} |")
    # model-free
    eq = (np.abs(d["CL"][rows] - d["CR"][rows]) < 1e-9)
    ch = np.where(d["resp"][rows] == 2, 1.0, -1.0)
    L += ["\nModel-free, current stimulus favours neither side (equal contrasts). P(repeat the previous choice):\n",
          "| Previous trial | Previous outcome | Trials | P(repeat) |", "|---|---|---|---|"]
    for lab, m in (("laser OFF", pa == "off"), ("laser ON (any area)", pa != "off")):
        for out, mm in (("rewarded", rc != 0), ("unrewarded", (rc == 0) & (cp != 0))):
            sel = eq & m & mm
            if sel.sum():
                L.append(f"| {lab} | {out} | {int(sel.sum())} | {(ch[sel] == cp[sel]).mean():.3f} |")
    text = "\n".join(L)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "CAUSAL_DIAGNOSE.md").write_text(text + "\n")
    print(text)


# ------------------------------------------------------------------------------------------- run
def run(mat, out_dir, jobs, n_perm, sites=False):
    import pandas as pd
    from joblib import Parallel, delayed
    from statsmodels.stats.multitest import multipletests
    df = load(mat)
    d = prepare(df, sites)
    rt = df["RT"].values if "RT" in df else None
    if rt is not None:
        print(f"median reaction time {np.nanmedian(rt):.2f} s (laser ends at the choice in this experiment)", flush=True)
    areas = sorted(str(a) for a in set(d["area"]) if a != "off")
    print(f"{len(df)} trials, {len(np.unique(d['sess']))} sessions; areas {areas}; previous-trial laser usable on "
          f"{d['prev_ok'].mean():.2%} of trials", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=jobs, verbose=5)(delayed(area_test)(d, a, k, n_perm, 7 + i)
                                          for i, (a, k) in enumerate([(a, k) for a in areas for k in ("readout", "tagging")]))
    R = pd.DataFrame([r for r in res if r is not None])
    for k in ("readout", "tagging"):
        m = R.kind == k
        if m.any():
            R.loc[m, "q"] = multipletests(R.loc[m, "p_less"], method="fdr_bh")[1]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    R.to_csv(Path(out_dir) / ("causal_sites.csv" if sites else "causal.csv"), index=False)
    L = [f"# Causal tests (Zatka-Haas 2021 inactivation, {len(np.unique(d['sess']))} sessions, {len(df)} trials)\n",
         "gamma < 0: silencing weakens the reward memory. Fraction removed = -gamma / weight of the memory without laser. "
         f"Null: parametric bootstrap without the tested term ({n_perm}). q = BH across areas.\n",
         "| Test | Area | Laser trials | Memory weight (no laser) | gamma | Fraction of memory removed | z | p (one-sided) | q | "
         "Contralateral stimulus change (positive control) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in R.sort_values(["kind", "p_less"]).iterrows():
        L.append(f"| {r.kind} | {r.area} | {r.n_laser} | {r.beta_rc:.2f} | {r.gamma:.2f} | {r.frac_removed:.2f} | {r.z:.2f} | "
                 f"{r.p_less:.3f} | {r.q:.3f} | {r.contra_stim_change:.2f} |")
    text = "\n".join(L)
    (Path(out_dir) / ("CAUSAL_SITES.md" if sites else "CAUSAL.md")).write_text(text + "\n")
    print(text)
    print(f"({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["inspect", "sim", "run", "sim_side", "side", "diagnose"])
    ap.add_argument("--mat", default="~/Downloads/MULTIVERSE/zatkahaas/data/inactivation/Inactivation_52Coord.mat")
    ap.add_argument("--out", default="results/causal")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--perm", type=int, default=N_PERM)
    ap.add_argument("--sites", action="store_true", help="test each of the 26 sites (both hemispheres) separately")
    a = ap.parse_args()
    if a.step == "inspect":
        inspect(a.mat)
    elif a.step == "sim":
        sim()
    elif a.step == "sim_side":
        sim_side()
    elif a.step == "diagnose":
        diagnose(a.mat, a.out)
    elif a.step == "side":
        run_side(a.mat, a.out, a.jobs, a.perm)
    else:
        run(a.mat, a.out, a.jobs, a.perm, a.sites)
