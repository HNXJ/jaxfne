"""Read-only agreement check between the manuscript and the publication claim ledger.

Reports four measures:
  (1) CL ids cited in the manuscript that the ledger does not contain;
  (2) ledger claims that the manuscript never cites (with manuscript_destination);
  (3) ledger statuses outside the declared tier set;
  (4) orphan candidates: paragraphs in abstract.md and results.md that cite a
      figure (Fig, Figure) or state a number with a unit, and carry no CL id.

results.md carries its ids in the traceability map: a heading
"### [Pa]–[Pb] Title — `{CL-xx, CL-yy..CL-zz, Qnn}`" gives the ids of the a-th to
b-th "## " sections of results.md, and a paragraph in such a section counts as
citing them. The map is `../results_reconstruction/traceability/traceability_map.md`
beside the manuscript directory when present, or --trace PATH.

Usage: python scripts/check_manuscript_claims.py [--ledger PATH]
       [--manuscript-dir PATH] [--trace PATH] [--check] [--strict] [--json]

Relative paths resolve against the repository root that holds this script.
--check exits 1 when (1) or (3) fails. --strict exits 1 when any of (1) to (4)
fails. Exit 2 when an input is missing or unparseable: the result is unknown,
never reported as clean. Reads only; no network; no repository imports.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = "artifacts/publication/publication_claim_ledger.json"
DEFAULT_MANUSCRIPT_DIR = "artifacts/programme/paper/publication/manuscript"
ORPHAN_FILES = ("abstract.md", "results.md")
# Status kinds named in the ledger's `purpose` field. The ledger test asserts four
# of them and reserves UNSUPPORTED. No code owns this set; it is copied from prose.
TIER_SET = frozenset({"SUPPORTED", "NEGATIVE", "UNRESOLVED", "METHOD_ONLY", "UNSUPPORTED"})
CL_RE = re.compile(r"\bCL-\d+\b")
FIG_RE = re.compile(r"\bFig(?:ure)?s?\b|\bFig\d")
UNIT_RE = re.compile(
    r"\d[\d.]*[\s-]?(?:%|(?:k?Hz|ms|s|mV|mm|[µμ]m|nm|pA|nA|mM)(?![A-Za-z])"
    r"|(?:cells?|neurons?|seeds?|sites?|areas?|steps?|points?|conditions?|cases?"
    r"|trajectories|runs?|arms?|spikes?|edges?)\b)"
)
TRACE_BESIDE = Path("results_reconstruction") / "traceability" / "traceability_map.md"
TRACE_HEAD_RE = re.compile(r"^### \[P(\d+)\](?:\s*[–-]\s*\[P(\d+)\])?.*`\{([^}]*)\}`")
CL_RANGE_RE = re.compile(r"CL-(\d+)\.\.CL-(\d+)")


class InputError(Exception):
    """A required input is missing or unparseable, so the check is unknown."""


def resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def safe(text: object) -> str:
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def load_ledger(path: Path) -> list[dict]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"cannot read ledger {path}: {exc}") from exc
    rows = doc.get("ledger") if isinstance(doc, dict) else None
    if not isinstance(rows, list) or not rows:
        raise InputError(f"ledger {path} has no non-empty 'ledger' list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("claim_id"), str):
            raise InputError(f"ledger row {index} has no string claim_id")
    return rows


def paragraphs(lines: list[str]):
    """Yield (first line number, joined text) for runs of non-blank, non-heading lines."""
    buf: list[str] = []
    start = 0
    for number, raw in enumerate(lines, start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            if buf:
                yield start, " ".join(buf)
                buf = []
            continue
        if not buf:
            start = number
        buf.append(text)
    if buf:
        yield start, " ".join(buf)


def load_trace(path: Path) -> tuple[dict[int, list[str]], list[tuple[str, int]]]:
    """(section number in "## " order of results.md -> its CL ids, [(CL id, map line)])."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError(f"cannot read trace map {path}: {exc}") from exc
    sections: dict[int, list[str]] = {}
    citations: list[tuple[str, int]] = []
    for number, line in enumerate(lines, start=1):
        match = TRACE_HEAD_RE.match(line)
        if not match:
            continue
        first = int(match.group(1))
        last = int(match.group(2) or first)
        tags = match.group(3)
        ids = [f"CL-{n:02d}" for a, b in CL_RANGE_RE.findall(tags) for n in range(int(a), int(b) + 1)]
        ids += CL_RE.findall(CL_RANGE_RE.sub("", tags))
        citations += [(cid, number) for cid in ids]
        for section in range(first, last + 1):
            sections[section] = ids
    if not sections:
        raise InputError(f"trace map {path} labels no section")
    return sections, citations


def section_of(lines: list[str]) -> list[int]:
    """For each line, how many "## " headings precede or start it."""
    out, count = [], 0
    for raw in lines:
        if raw.startswith("## "):
            count += 1
        out.append(count)
    return out


def scan(ledger_path: Path, manuscript_dir: Path, trace_path: Path | None = None) -> dict:
    rows = load_ledger(ledger_path)
    if not manuscript_dir.is_dir():
        raise InputError(f"manuscript dir {manuscript_dir} does not exist")
    md_files = sorted(manuscript_dir.glob("*.md"))
    if not md_files:
        raise InputError(f"no .md files in {manuscript_dir}")
    for name in ORPHAN_FILES:
        if not (manuscript_dir / name).is_file():
            raise InputError(f"orphan-check file {name} missing from {manuscript_dir}")

    if trace_path is None and (manuscript_dir.parent / TRACE_BESIDE).is_file():
        trace_path = manuscript_dir.parent / TRACE_BESIDE
    trace: dict[int, list[str]] = {}
    known = {row["claim_id"] for row in rows}
    cited: dict[str, list[tuple[str, int]]] = {}
    if trace_path is not None:
        trace, citations = load_trace(trace_path)
        results_lines = (manuscript_dir / "results.md").read_text(encoding="utf-8").splitlines()
        n_sections = sum(raw.startswith("## ") for raw in results_lines)
        if max(trace) > n_sections:
            raise InputError(
                f"trace map {trace_path} names P{max(trace)}; results.md has {n_sections} sections"
            )
        for cid, number in citations:
            cited.setdefault(cid, []).append((trace_path.name, number))
    for md in md_files:
        lines = md.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            for match in CL_RE.finditer(line):
                cited.setdefault(match.group(), []).append((md.name, number))

    cited_not_in_ledger = [
        {"id": cid, "file": name, "line": number}
        for cid in sorted(cited)
        if cid not in known
        for name, number in cited[cid]
    ]
    uncited = [
        {"id": row["claim_id"], "manuscript_destination": row.get("manuscript_destination")}
        for row in rows
        if row["claim_id"] not in cited
    ]
    status_outside = [
        {"id": row["claim_id"], "status": row.get("status")}
        for row in rows
        if not (isinstance(row.get("status"), str) and row.get("status") in TIER_SET)
    ]

    orphans = []
    examined = 0
    for name in ORPHAN_FILES:
        lines = (manuscript_dir / name).read_text(encoding="utf-8").splitlines()
        sections = section_of(lines)
        for number, text in paragraphs(lines):
            examined += 1
            reasons = []
            if FIG_RE.search(text):
                reasons.append("figure")
            if UNIT_RE.search(text):
                reasons.append("number_with_unit")
            mapped = name == "results.md" and trace.get(sections[number - 1])
            if reasons and not CL_RE.search(text) and not mapped:
                orphans.append(
                    {"file": name, "line": number, "reasons": reasons, "excerpt": text[:100]}
                )

    observed = Counter(safe(row.get("status")) for row in rows)
    return {
        "ledger": str(ledger_path),
        "manuscript_dir": str(manuscript_dir),
        "trace": str(trace_path) if trace_path is not None else None,
        "examined": {
            "ledger_rows": len(rows),
            "cited_ids_distinct": len(cited),
            "manuscript_files": len(md_files),
            "orphan_paragraphs": examined,
            "trace_sections": len(trace),
        },
        "observed_statuses": dict(observed),
        "tier_set": sorted(TIER_SET),
        "cited_not_in_ledger": cited_not_in_ledger,
        "uncited_ledger_ids": uncited,
        "status_outside_tier": status_outside,
        "orphan_candidates": orphans,
        "counts": {
            "cited_not_in_ledger": len(cited_not_in_ledger),
            "uncited_ledger_ids": len(uncited),
            "status_outside_tier": len(status_outside),
            "orphan_candidates": len(orphans),
        },
    }


def failed_checks(counts: dict, check: bool, strict: bool) -> list[str]:
    failed: list[str] = []
    if check or strict:
        failed += [k for k in ("cited_not_in_ledger", "status_outside_tier") if counts[k]]
    if strict:
        failed += [k for k in ("uncited_ledger_ids", "orphan_candidates") if counts[k]]
    return failed


def print_text(report: dict) -> None:
    ex = report["examined"]
    print(f"ledger: {report['ledger']} ({ex['ledger_rows']} rows)")
    print(
        f"manuscript: {report['manuscript_dir']} ({ex['manuscript_files']} files, "
        f"{ex['cited_ids_distinct']} distinct CL ids cited)"
    )
    print(f"trace map: {report['trace'] or 'none'} ({ex['trace_sections']} results.md sections mapped)")
    observed = ", ".join(f"{k}={v}" for k, v in sorted(report["observed_statuses"].items()))
    print(f"observed statuses: {observed}")
    print(f"tier set (declared): {', '.join(report['tier_set'])}")
    print(f"(1) cited ids not in ledger: {report['counts']['cited_not_in_ledger']}")
    for item in report["cited_not_in_ledger"]:
        print(f"    {item['file']}:{item['line']}  {item['id']}")
    print(f"(2) uncited ledger ids: {report['counts']['uncited_ledger_ids']}")
    for item in report["uncited_ledger_ids"]:
        print(f"    {item['id']}  manuscript_destination: {safe(item['manuscript_destination'])}")
    print(f"(3) status outside tier set: {report['counts']['status_outside_tier']}")
    for item in report["status_outside_tier"]:
        print(f"    {item['id']}  status={safe(item['status'])}")
    print(f"(4) orphan candidates (abstract.md, results.md): {report['counts']['orphan_candidates']}")
    for item in report["orphan_candidates"]:
        reasons = ",".join(item["reasons"])
        print(f"    {item['file']}:{item['line']}  [{reasons}]  {safe(item['excerpt'])}")
    c = report["counts"]
    print(
        f"COUNTS cited_not_in_ledger={c['cited_not_in_ledger']} "
        f"uncited_ledger_ids={c['uncited_ledger_ids']} "
        f"status_outside_tier={c['status_outside_tier']} "
        f"orphan_candidates={c['orphan_candidates']}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check manuscript CL citations against the ledger")
    parser.add_argument("--ledger", default=DEFAULT_LEDGER, help="path to the claim ledger JSON")
    parser.add_argument(
        "--manuscript-dir", default=DEFAULT_MANUSCRIPT_DIR, help="directory of manuscript .md files"
    )
    parser.add_argument(
        "--trace", default=None, help="traceability map for results.md (default: the one beside the manuscript)"
    )
    parser.add_argument("--check", action="store_true", help="exit 1 when (1) or (3) fails")
    parser.add_argument("--strict", action="store_true", help="exit 1 when any of (1) to (4) fails")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    try:
        trace = resolve(args.trace) if args.trace else None
        report = scan(resolve(args.ledger), resolve(args.manuscript_dir), trace)
    except InputError as exc:
        print(f"check_manuscript_claims: {exc}", file=sys.stderr)
        return 2

    mode = "strict" if args.strict else "check" if args.check else "report"
    failed = failed_checks(report["counts"], args.check, args.strict)
    report["mode"] = mode
    report["failed"] = failed
    report["exit_code"] = 1 if failed else 0
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_text(report)
        print(f"mode={mode} failed={','.join(failed) or 'none'} exit={report['exit_code']}")
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
