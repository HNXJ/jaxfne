"""OCMA stage 0: inventory every reference row against code, tests, lanes and the dashboard.

Usage: python artifacts/etudes/meta/ocma/tools/inventory.py [--out PATH]  (default data/inventory.json)
Writes a JSON list, one record per row of docs/reference/references.md, with mechanical evidence only
(DOI hits in jaxfne/, tests/, artifacts/etudes/; dashboard citation match). No judgement is made here.
Exit 0 on success, 2 if the reference page cannot be parsed.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
REF = ROOT / "docs" / "reference" / "references.md"
SOURCES = ROOT / "dashboard" / "sources.json"
DOI_RE = re.compile(r"doi\.org/(10\.[^)\s]+)")
SCAN = {"code": ROOT / "jaxfne", "tests": ROOT / "tests", "etudes": ROOT / "artifacts" / "etudes"}
EXTS = {".py", ".md", ".json", ".ipynb"}


def rows():
    section, out = None, []
    for n, line in enumerate(REF.read_text(encoding="utf-8").splitlines(), 1):
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if not line.startswith("| ") or line.startswith("| Reference") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        m = DOI_RE.search(cells[0])
        first = re.match(r"([^,(]+)", cells[0])
        year = re.search(r"\((\d{4})\)", cells[0])
        out.append({
            "line": n, "section": section, "reference": re.sub(r"\s*\[doi:.*", "", cells[0]),
            "doi": m.group(1) if m else None,
            "first_author": first.group(1).strip() if first else None,
            "year": year.group(1) if year else None,
            "result_or_relation": cells[1] if len(cells) > 1 else "",
            "status_or_functions": cells[-1] if len(cells) > 2 else "",
        })
    return out


SELF = Path(__file__).resolve().parents[1]  # the OCMA folder: its own outputs are never evidence


def corpus():
    files = {}
    for key, base in SCAN.items():
        for p in base.rglob("*"):
            if (p.is_file() and p.suffix in EXTS and "_temp" not in p.parts and "__pycache__" not in p.parts
                    and SELF not in p.resolve().parents):
                try:
                    files[p] = (key, p.read_text(encoding="utf-8", errors="ignore"))
                except OSError:
                    pass
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "etudes" / "meta" / "ocma" / "data" / "inventory.json"))
    args = ap.parse_args()
    recs = rows()
    if len(recs) < 50:
        print(f"parsed only {len(recs)} rows")
        return 2
    files = corpus()
    # id, title and citation together: citations often omit the year ("Vogels et al., Science 334(6062)"),
    # while ids and titles carry it ("vogels2011", "Vogels et al. 2011, Fig. 4").
    cites = [" ".join([p.get("id", ""), p.get("title", ""), p.get("citation", "")]).lower()
             for p in json.loads(SOURCES.read_text(encoding="utf-8"))["papers"]]
    for r in recs:
        hits = {k: [] for k in SCAN}
        if r["doi"]:
            needle = r["doi"].lower()
            for p, (key, text) in files.items():
                if needle in text.lower():
                    hits[key].append(p.relative_to(ROOT).as_posix())
        r["hits"] = {k: sorted(v) for k, v in hits.items()}
        sur = (r["first_author"] or "").split()[0].rstrip(",") if r["first_author"] else ""
        named = {k: [] for k in SCAN}
        if sur and r["year"]:
            pat = re.compile(re.escape(sur) + r"[^\n]{0,60}?" + r["year"], re.IGNORECASE)
            for p, (key, text) in files.items():
                if pat.search(text):
                    named[key].append(p.relative_to(ROOT).as_posix())
        r["name_hits"] = {k: sorted(v) for k, v in named.items()}
        fa = sur.lower()
        r["dashboard"] = any(fa and fa in c and (r["year"] or "") in c for c in cites)
    Path(args.out).write_text(json.dumps(recs, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    by = {}
    for r in recs:
        key = (r["section"], r["dashboard"], bool(r["hits"]["code"] or r["name_hits"]["code"]),
               bool(r["hits"]["tests"] or r["name_hits"]["tests"]))
        by[key] = by.get(key, 0) + 1
    print(f"rows {len(recs)}; with DOI {sum(1 for r in recs if r['doi'])}; on dashboard {sum(r['dashboard'] for r in recs)}")
    for (sec, dash, code, tests), n in sorted(by.items(), key=lambda kv: str(kv[0])):
        print(f"{n:4d}  dash={int(dash)} code={int(code)} tests={int(tests)}  {sec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
