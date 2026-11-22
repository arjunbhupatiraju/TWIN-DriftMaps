"""Reading packet for screening (helper only; not part of the registered analysis).

    python3 scripts/lit_packet.py

Adds each paper's abstract and link to literature/screening.csv and literature/prefilter_audit.csv
(new columns only: your Y/N, reasons and notes are kept), and writes two pages to read from:
    literature/screening_packet.html      papers in screen_order, with title, journal, year, link, abstract
    literature/audit_packet.html          the 100 pre-filter rejects for the audit
Safe to run again: it only fills in what is missing.
"""
from __future__ import annotations

import csv
import html
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gatekeeper.literature import _abstract  # noqa: E402

LIT = Path("literature")
API = "https://api.openalex.org/works"


def fetch(dois, get):
    out = {}
    for i in range(0, len(dois), 50):
        chunk = [d for d in dois[i:i + 50] if d]
        if not chunk:
            continue
        url = f"{API}?filter=doi:{'|'.join(chunk)}&per-page=50&select=doi,abstract_inverted_index"
        for t in range(5):
            try:
                r = get(url)
                break
            except Exception:
                if t == 4:
                    raise
                time.sleep(2 ** t)
        for w in r["results"]:
            d = (w.get("doi") or "").replace("https://doi.org/", "").lower()
            out[d] = _abstract(w.get("abstract_inverted_index"))
        print(f"  abstracts {min(i + 50, len(dois))}/{len(dois)}", flush=True)
    return out


def _get(url):
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def enrich(path, get=_get):
    rows = list(csv.DictReader(open(path)))
    cols = list(rows[0].keys()) if rows else []
    for c in ("link", "abstract"):
        if c not in cols:
            cols.append(c)
    need = [r["doi"].lower() for r in rows if not r.get("abstract")]
    got = fetch(need, get) if need else {}
    for r in rows:
        r["link"] = f"https://doi.org/{r['doi']}" if r["doi"] else ""
        if not r.get("abstract"):
            r["abstract"] = got.get(r["doi"].lower(), "") or "(no abstract in OpenAlex: open the link)"
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, cols)
        wr.writeheader()
        wr.writerows(rows)
    return rows


CSS = """body{font:16px/1.55 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:860px;margin:0 auto;padding:16px;color:#1d1d1f;background:#fff}
h1{font-size:22px}.p{border:1px solid #ddd;border-radius:10px;padding:14px 16px;margin:14px 0}
.n{font-weight:700;color:#555}.t{font-size:18px;font-weight:600;margin:4px 0}.m{color:#666;font-size:14px}
.a{margin-top:8px;white-space:pre-wrap}.q{background:#f5f5f7;border-radius:8px;padding:10px 14px;font-size:14px}
a{color:#0a58ca}@media(prefers-color-scheme:dark){body{background:#111;color:#eee}.p{border-color:#333}.q{background:#1d1d1f}.m,.n{color:#aaa}a{color:#6ea8fe}}"""

QUESTIONS = """<div class="q"><b>For each paper, stop at the first NO:</b><br>
1. Journal research article (not preprint/review)? &nbsp;no &rarr; <code>not_article</code><br>
2. Spikes recorded in mice or rats (not imaging only)? &nbsp;no &rarr; <code>no_rodent_spikes</code><br>
3. Animals doing a task with trials? &nbsp;no &rarr; <code>no_task</code><br>
4. Abstract says a named region encodes stimulus / choice / outcome / prior? &nbsp;no &rarr; <code>no_region_claim</code><br>
All yes &rarr; <b>Y</b>, then code its claims in claims.csv (CODEBOOK.md).</div>"""


def page(rows, title, out):
    parts = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
             f"<title>{title}</title><style>{CSS}</style><h1>{title}</h1>{QUESTIONS}"]
    for r in rows:
        parts.append(
            f"<div class=p id=s{r['screen_order']}><div class=n>#{r['screen_order']}</div>"
            f"<div class=t>{html.escape(r['title'])}</div>"
            f"<div class=m>{html.escape(r['journal'] or '')} &middot; {r['year']} &middot; "
            f"<a href='{r['link']}' target=_blank>{html.escape(r['doi'])}</a></div>"
            f"<div class=a>{html.escape(r['abstract'])}</div></div>")
    Path(out).write_text("\n".join(parts))


def main():
    for name, title in (("screening", "Screening packet (stop at 100 eligible)"), ("prefilter_audit", "Pre-filter audit packet")):
        p = LIT / f"{name}.csv"
        if not p.exists():
            raise SystemExit(f"{p} missing: run python3 scripts/run_literature.py sheet first")
        print(f"{p}:")
        rows = enrich(p)
        out = LIT / ("screening_packet.html" if name == "screening" else "audit_packet.html")
        page(rows, title, out)
        print(f"  wrote {out} ({len(rows)} papers)")


if __name__ == "__main__":
    main()
