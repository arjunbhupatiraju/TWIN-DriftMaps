"""MULTIVERSE literature test: do published brain-map claims hold up in 139 mice, and can we predict which?

Three stages, each run once, in this order, under the literature pre-registration (OSF):

1. frame()    Pull the sampling frame from OpenAlex: every work citing an anchor paper, published
              2017-01-01 .. FRAME_END. Fixed random order: sha256(SEED | key). An inclusive keyword
              pre-filter (registered) marks works worth screening; the rest are kept and audited.
              Writes manifests/lit_frame.csv + lit_frame.meta.json (with sha256).
2. (by hand)  Screen in rank order until N_TARGET eligible papers; code each abstract claim into
              literature/claims.csv (see literature/CODEBOOK.md). Coding is frozen (CODING_LOCK.txt)
              before stage 3 opens any IBL result.
3. analyse()  For each claim: (a) the observed IBL reference verdict (all 459 sessions, calibrated
              default pipeline), (b) the model-predicted probability that a claim made the paper's way
              holds up, estimated inside IBL on disjoint mice. Outputs the support rate, predicted
              rate, AUC / calibration, reporting gaps and the registered secondary splits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED = "MULTIVERSE-LIT-v1"
FRAME_START, FRAME_END = "2017-01-01", "2026-10-13"
N_TARGET = 100
ANCHORS = {
    "jun2017_neuropixels": "10.1038/nature24636",
    "steinmetz2019_distributed": "10.1038/s41586-019-1787-x",
    "ibl2025_brainwide_map": "10.1038/s41586-025-09235-0",
    "ibl2023_brainwide_map_preprint": "10.1101/2023.07.04.547681",
}
# Pre-filter (inclusive on purpose): title+abstract must mention a rodent AND a task variable word.
RODENT = r"\b(mouse|mice|murine|rat|rats|rodent|rodents)\b"
TASKWORD = (r"\b(choice|choices|decision|decisions|stimul\w*|reward\w*|prior|priors|expectation\w*|"
            r"feedback|outcome\w*|movement\w*|lick\w*|task|tasks|behavio\w*|action\w*)\b")

# variable class -> IBL variable (registered mapping)
VAR_OF = {"stimulus": "stim", "choice": "choice", "outcome": "feedback", "prior": "sprior"}
# coded null -> pipeline null; "none" (parametric / no null) -> pooled single-neuron test
NULL_OF = {"shuffle": "shuffle", "circular": "circ", "pseudo": "pseudo", "none": "none", "unclear": "shuffle"}
METHOD_OF = {"decoding": "decode", "single": "single", "other": "euclid", "unclear": "decode"}
LEVEL_OF = {"pooled": "pooled", "session": "session", "mouse": "mouse", "unclear": "pooled"}


# ----------------------------------------------------------------------------------------------- stage 1
def order_hash(key):
    return hashlib.sha256(f"{SEED}|{key}".encode()).hexdigest()


def _abstract(inv):
    if not inv:
        return ""
    pos = sorted((i, w) for w, idx in inv.items() for i in idx)
    return " ".join(w for _, w in pos)


def prefilter(title, abstract):
    t = f"{title or ''} {abstract or ''}".lower()
    return bool(re.search(RODENT, t)) and bool(re.search(TASKWORD, t))


def _get(url, tries=5):
    import requests   # installed with the IBL tools; handles macOS certificates
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)


API = "https://api.openalex.org"


def frame(out_dir=ROOT / "manifests", log=print, get=_get):
    out_dir = Path(out_dir)
    csv_path = out_dir / "lit_frame.csv"
    if csv_path.exists():
        raise SystemExit(f"{csv_path} exists: the frame is pulled once. Delete nothing; analyse it.")
    works, anchor_ids = {}, {}
    for name, doi in ANCHORS.items():
        a = get(f"{API}/works/doi:{doi}")
        anchor_ids[name] = a["id"].rsplit("/", 1)[-1]
        log(f"anchor {name}: {anchor_ids[name]} (cited by {a.get('cited_by_count')})")
    for name, wid in anchor_ids.items():
        cursor, n = "*", 0
        while cursor:
            url = (f"{API}/works?filter=cites:{wid},from_publication_date:{FRAME_START},"
                   f"to_publication_date:{FRAME_END}&per-page=200&cursor={cursor}"
                   "&select=id,doi,title,publication_year,type,primary_location,abstract_inverted_index")
            page = get(url)
            for w in page["results"]:
                wid2 = w["id"].rsplit("/", 1)[-1]
                rec = works.get(wid2)
                if rec is None:
                    src = ((w.get("primary_location") or {}).get("source") or {})
                    doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                    rec = works[wid2] = dict(
                        openalex_id=wid2, doi=doi, title=(w.get("title") or "").replace("\n", " "),
                        year=w.get("publication_year"), type=w.get("type"),
                        journal=src.get("display_name") or "", source_type=src.get("type") or "",
                        prefilter=prefilter(w.get("title"), _abstract(w.get("abstract_inverted_index"))),
                        anchors=set())
                rec["anchors"].add(name)
                n += 1
            cursor = page["meta"].get("next_cursor")
            if not page["results"]:
                break
        log(f"  citing {name}: {n}")
    rows = []
    for r in works.values():
        r["anchors"] = ";".join(sorted(r["anchors"]))
        r["order_hash"] = order_hash(r["doi"] or r["openalex_id"])
        rows.append(r)
    rows.sort(key=lambda r: r["order_hash"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    cols = ["rank", "order_hash", "doi", "openalex_id", "title", "year", "type", "journal", "source_type",
            "prefilter", "anchors"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: r[c] for c in cols})
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    meta = dict(pulled_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), seed=SEED, anchors=ANCHORS,
                anchor_openalex_ids=anchor_ids, window=[FRAME_START, FRAME_END], n_works=len(rows),
                n_prefilter=sum(r["prefilter"] for r in rows), rodent_regex=RODENT, task_regex=TASKWORD,
                csv_sha256=sha)
    (out_dir / "lit_frame.meta.json").write_text(json.dumps(meta, indent=1))
    log(f"frame: {len(rows)} unique works, {meta['n_prefilter']} pass the pre-filter; sha256 {sha}")
    return meta


def screening_sheet(frame_csv=ROOT / "manifests/lit_frame.csv", out=ROOT / "literature/screening.csv", audit_n=100):
    """Screening order = frame rank among pre-filter passes. Also draws the pre-filter audit sample
    (audit_n random pre-filter FAILS, by the same hash order) to estimate missed eligible papers."""
    rows = list(csv.DictReader(open(frame_csv)))
    keep = [r for r in rows if r["prefilter"] == "True"]
    fails = [r for r in rows if r["prefilter"] != "True"][:audit_n]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["screen_order", "rank", "doi", "title", "year", "journal", "eligible", "exclusion_reason", "notes"]
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        for i, r in enumerate(keep):
            wr.writerow(dict(screen_order=i + 1, rank=r["rank"], doi=r["doi"], title=r["title"], year=r["year"],
                             journal=r["journal"], eligible="", exclusion_reason="", notes=""))
    with open(out.with_name("prefilter_audit.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        for i, r in enumerate(fails):
            wr.writerow(dict(screen_order=i + 1, rank=r["rank"], doi=r["doi"], title=r["title"], year=r["year"],
                             journal=r["journal"], eligible="", exclusion_reason="", notes=""))
    return len(keep), len(fails)


KAPPA_FIELDS = ("variable_class", "null", "level", "method", "sessions_per_region", "beryl_regions")


def coder2_papers(screening_csv=ROOT / "literature/screening.csv", frac=0.2):
    rows = [r for r in csv.DictReader(open(screening_csv)) if r["eligible"].strip().upper() == "Y"]
    rows.sort(key=lambda r: hashlib.sha256(f"{SEED}|coder2|{r['doi']}".encode()).hexdigest())
    return rows[: max(1, round(frac * len(rows)))]


def kappa(claims1, claims2):
    """Cohen's kappa per field over claims matched on (paper_doi, claim_id)."""
    from collections import Counter
    a = {(c["paper_doi"], c["claim_id"]): c for c in claims1}
    b = {(c["paper_doi"], c["claim_id"]): c for c in claims2}
    common = sorted(set(a) & set(b))
    out = {}
    for f in KAPPA_FIELDS:
        x = [str(a[k][f]).strip().lower() for k in common]
        y = [str(b[k][f]).strip().lower() for k in common]
        if not x:
            out[f] = None
            continue
        po = np.mean([i == j for i, j in zip(x, y)])
        cx, cy, n = Counter(x), Counter(y), len(x)
        pe = sum(cx[c] * cy[c] for c in set(cx) | set(cy)) / n ** 2
        out[f] = dict(n=n, agreement=float(po), kappa=float((po - pe) / (1 - pe)) if pe < 1 else 1.0)
    return out


# ----------------------------------------------------------------------------------------------- stage 3
CLAIM_COLS = ["paper_doi", "claim_id", "claim_text", "region_reported", "beryl_regions", "variable_class",
              "n_mice", "sessions_per_region", "null", "level", "method", "task_similarity", "journal",
              "data_available", "coder"]


def sbin(n):
    return "2" if n <= 2 else "3-4" if n <= 4 else "5-8" if n <= 8 else "9+"


def _num(x):
    try:
        v = float(x)
        return v if np.isfinite(v) and v > 0 else None
    except (TypeError, ValueError):
        return None


def study_key(var, null, level, method):
    from .multiverse import default_key
    if null == "none" or level == "pooled":
        return (var, "a", "good", "single", "none", "pooled", 2)
    return default_key(var, null=null, level=level, method=method)


def reference(U, var):
    """Observed IBL verdict for every region: default calibrated pipeline, all mice."""
    from .multiverse import default_key
    k = default_key(var)
    v, n = U.verdict(k, np.ones(len(U.mice)))
    regs = U.regions(k)
    return {r: (int(v[i]), int(n[i])) for i, r in enumerate(regs)}


def prediction_table(U, n_rep=300, seed=0, ks=(2, 3, 4, 6, 8, 12, 16, 20, 24)):
    """P(region verdict holds in held-out mice | significant in a k-mouse study made a given way).
    Study A = k random mice analysed with the claim's key; reference = default calibrated verdict on all
    mice NOT in A. Pooled over regions, stratified by sessions-per-region bin of the claim in A.
    Returns {(var, key5, sbin): [hits, claims]}."""
    from .multiverse import default_key
    rng = np.random.default_rng(seed)
    M = len(U.mice)
    tab = {}
    combos = [(n, l, m) for n in ("shuffle", "circ", "pseudo") for l in ("session", "mouse")
              for m in ("decode", "single", "euclid")] + [("none", "pooled", "single")]
    for var in VAR_OF.values():
        dk = default_key(var)
        for (nl, lev, meth) in combos:
            key = study_key(var, nl, lev, meth)
            if key[:5] not in U.mats:
                continue
            ra, rr = U.regions(key), U.regions(dk)
            idx = np.array([rr.index(r) if r in rr else -1 for r in ra])
            for k in ks:
                if 2 * k > M:
                    continue
                for _ in range(n_rep):
                    p = rng.permutation(M)
                    a = np.zeros(M)
                    a[p[:k]] = 1
                    va, na = U.verdict(key, a)
                    vr, _ = U.verdict(dk, 1 - a)
                    for i in np.nonzero(va == 1)[0]:
                        j = idx[i]
                        if j < 0 or vr[j] < 0:
                            continue
                        c = tab.setdefault((var, key[3] + "|" + key[4] + "|" + key[5], sbin(na[i])), [0, 0])
                        c[0] += int(vr[j] == 1)
                        c[1] += 1
    return tab


def predict(tab, var, key, sessions):
    kk = key[3] + "|" + key[4] + "|" + key[5]
    h, n = tab.get((var, kk, sbin(sessions)), [0, 0])
    if n < 20:   # registered fallback: pool the sessions bins for that (var, key)
        h = sum(v[0] for (a, b, _), v in tab.items() if a == var and b == kk)
        n = sum(v[1] for (a, b, _), v in tab.items() if a == var and b == kk)
    return (h + 1) / (n + 2)


def _auc(y, s):
    from scipy.stats import rankdata
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    r = rankdata(s)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _beryl(names):
    try:
        from iblatlas.regions import BrainRegions
        br = BrainRegions()
        return [str(br.acronym2acronym(np.array([n]), mapping="Beryl")[0]) for n in names]
    except Exception:
        return list(names)


def score_claims(claims, U, tab):
    """claims: list of dicts (CLAIM_COLS). Adds observed / predicted / status for each claim."""
    refs = {v: reference(U, v) for v in VAR_OF.values()}
    out = []
    for c in claims:
        c = dict(c)
        var = VAR_OF.get(c["variable_class"].strip().lower())
        regs = [r.strip() for r in c["beryl_regions"].split(";") if r.strip()]
        regs = _beryl(regs)
        if var is None or not regs:
            c.update(status="out_of_scope", observed="", predicted="")
            out.append(c)
            continue
        ref = refs[var]
        tested = [ref[r] for r in regs if r in ref and ref[r][0] >= 0]
        nl = NULL_OF.get(c["null"].strip().lower(), "shuffle")
        lev = LEVEL_OF.get(c["level"].strip().lower(), "pooled")
        meth = METHOD_OF.get(c["method"].strip().lower(), "decode")
        sess = _num(c["sessions_per_region"]) or _num(c["n_mice"]) or 2
        key = study_key(var, nl, lev, meth)
        c["predicted"] = round(predict(tab, var, key, sess), 4)
        c["sessions_used"] = sess
        if not tested:
            c.update(status="untestable", observed="")
        else:
            c.update(status="tested", observed=int(any(t[0] == 1 for t in tested)),
                     observed_strict=int(all(t[0] == 1 for t in tested)),
                     ibl_sessions=";".join(str(t[1]) for t in tested))
        out.append(c)
    return out


def _cluster_boot(rows, f, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    papers = sorted({r["paper_doi"] for r in rows})
    by = {p: [r for r in rows if r["paper_doi"] == p] for p in papers}
    vals = []
    for _ in range(n_boot):
        s = [x for p in rng.choice(papers, len(papers)) for x in by[p]]
        v = f(s)
        if v is not None:
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def summarise(scored, papers_screened=None):
    t = [r for r in scored if r["status"] == "tested"]
    obs = lambda rs: float(np.mean([r["observed"] for r in rs])) if rs else None
    prd = lambda rs: float(np.mean([r["predicted"] for r in rs])) if rs else None
    auc = lambda rs: _auc([r["observed"] for r in rs], [r["predicted"] for r in rs])
    S = dict(n_claims=len(scored), n_papers=len({r["paper_doi"] for r in scored}),
             n_tested=len(t), n_untestable=sum(r["status"] == "untestable" for r in scored),
             n_out_of_scope=sum(r["status"] == "out_of_scope" for r in scored),
             observed_support=obs(t), observed_support_ci=_cluster_boot(t, obs),
             observed_support_strict=float(np.mean([r["observed_strict"] for r in t])) if t else None,
             predicted_support=prd(t), predicted_support_ci=_cluster_boot(t, prd),
             auc=auc(t), auc_ci=_cluster_boot(t, auc))
    t2 = [r for r in t if str(r["variable_class"]).strip().lower() != "outcome"]
    S.update(auc_excluding_outcome=auc(t2), auc_excluding_outcome_ci=_cluster_boot(t2, auc),
             observed_support_excluding_outcome=obs(t2))
    cal = []
    for lo, hi in ((0, .25), (.25, .5), (.5, .75), (.75, 1.01)):
        rs = [r for r in t if lo <= r["predicted"] < hi]
        cal.append(dict(bin=f"{lo:.2f}-{min(hi, 1):.2f}", n=len(rs), predicted=prd(rs), observed=obs(rs)))
    S["calibration"] = cal
    nr = lambda field: float(np.mean([str(r[field]).strip().lower() in ("", "nr", "not reported", "unclear")
                                      for r in scored])) if scored else None
    S["not_reported"] = {f: nr(f) for f in ("sessions_per_region", "n_mice", "null", "level")}
    sec = {}
    for field in ("task_similarity", "null", "variable_class", "journal_tier"):
        groups = {}
        for r in t:
            g = r.get(field) if field != "journal_tier" else tier(r.get("journal", ""))
            groups.setdefault(str(g).strip().lower(), []).append(r)
        sec[field] = {g: dict(n=len(rs), observed=obs(rs), predicted=prd(rs)) for g, rs in sorted(groups.items())}
    groups = {}
    for r in t:
        groups.setdefault(sbin(r["sessions_used"]), []).append(r)
    sec["sessions_bin"] = {g: dict(n=len(rs), observed=obs(rs), predicted=prd(rs)) for g, rs in sorted(groups.items())}
    S["secondary"] = sec
    lo = [r for r in t2 if r["sessions_used"] < 5]
    hi = [r for r in t2 if r["sessions_used"] >= 5]
    S["L3_groups"] = dict(lt5=dict(n=len(lo), observed=obs(lo)), ge5=dict(n=len(hi), observed=obs(hi)))
    S["hypotheses"] = hypotheses(S, len(t2))
    if papers_screened is not None:
        S["screening"] = papers_screened
    return S


def hypotheses(S, n_nonoutcome):
    """Registered pass lines L1-L4 (literature pre-registration, Section 6)."""
    out = []
    auc, ci = S.get("auc_excluding_outcome"), S.get("auc_excluding_outcome_ci")
    if n_nonoutcome < 40 or auc is None:
        out.append(dict(id="L1", verdict="INCONCLUSIVE", value=auc))
    else:
        out.append(dict(id="L1", verdict="PASS" if auc >= 0.65 and ci and ci[0] > 0.5 else "FAIL", value=auc, ci=ci))
    o, p = S.get("observed_support"), S.get("predicted_support")
    gap = None if o is None or p is None else abs(p - o)
    out.append(dict(id="L2", verdict="INCONCLUSIVE" if gap is None else ("PASS" if gap <= 0.10 else "FAIL"), value=gap))
    g = S["L3_groups"]
    if n_nonoutcome < 40 or g["lt5"]["n"] < 15 or g["ge5"]["n"] < 15:
        out.append(dict(id="L3", verdict="INCONCLUSIVE", value=None))
    else:
        d = g["ge5"]["observed"] - g["lt5"]["observed"]
        out.append(dict(id="L3", verdict="PASS" if d >= 0.10 else "FAIL", value=d))
    nr = S["not_reported"]["sessions_per_region"]
    out.append(dict(id="L4", verdict="INCONCLUSIVE" if nr is None else ("PASS" if nr >= 0.30 else "FAIL"), value=nr))
    return out


TOP = ("nature", "science", "cell", "neuron", "nature neuroscience")


def tier(journal):
    j = (journal or "").strip().lower()
    return "top" if j in TOP else "other"


def load_universes(stats_files):
    import pandas as pd
    from .multiverse import Universes
    df = pd.concat([pd.read_csv(f) for f in stats_files], ignore_index=True)
    df = df[df["var"].isin(list(VAR_OF.values())) & (df.window == "a") & (df.units == "good")]
    return Universes(df)


def analyse(claims_csv, stats_files, out_dir, n_rep=300, log=print):
    claims = list(csv.DictReader(open(claims_csv)))
    missing = [c for c in CLAIM_COLS if claims and c not in claims[0]]
    if missing:
        raise SystemExit(f"claims.csv is missing columns: {missing}")
    log(f"{len(claims)} claims from {len({c['paper_doi'] for c in claims})} papers")
    U = load_universes(stats_files)
    log(f"IBL reference: {len(U.mice)} mice; building the prediction table ({n_rep} studies per size) ...")
    tab = prediction_table(U, n_rep=n_rep)
    scored = score_claims(claims, U, tab)
    S = summarise(scored)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(CLAIM_COLS + ["status", "observed", "observed_strict", "ibl_sessions",
                                            "predicted", "sessions_used"]))
    with open(out / "claims_scored.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(scored)
    (out / "prediction_table.json").write_text(json.dumps({"|".join(k): v for k, v in tab.items()}, indent=1))
    (out / "literature_summary.json").write_text(json.dumps(S, indent=1))
    log(json.dumps({k: S[k] for k in ("n_papers", "n_claims", "n_tested", "observed_support",
                                      "observed_support_ci", "predicted_support", "auc", "auc_ci",
                                      "auc_excluding_outcome", "auc_excluding_outcome_ci")}, indent=1))
    log("\nPRE-REGISTERED HYPOTHESES (literature test, run once):")
    for h in S["hypotheses"]:
        v = h["value"]
        log(f"  {h['id']}  {h['verdict']:12s} value={'n/a' if v is None else f'{v:.3f}'}")
    return S
