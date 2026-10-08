"""Ledger/coverage agreement check for artifacts/programme/claim_ledger_055.md (0.5.5 ATLAS, manuscript item 11).

The ledger's State and evidence-path columns are mechanically generated from
artifacts/programme/atlas_coverage.json and must never be hand-drifted. This
script reads both files and exits 0 only when every coverage id appears exactly
once in the ledger and each row's state and evidence path agree with the
coverage row (coverage evidence "" matches ledger `none`; non-empty evidence
matches the ledger cell with backticks stripped). Read-only over both files.

Usage: python scripts/check_claim_ledger_055.py [--ledger PATH] [--coverage PATH]
Exit 0 when valid, 1 otherwise. No network, no jaxfne import.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

DEFAULT_LEDGER = "artifacts/programme/claim_ledger_055.md"
DEFAULT_COVERAGE = "artifacts/programme/atlas_coverage.json"
ROW_RE = re.compile(r"^\|\s*(AT-\d{2}-R\d+)\s*\|")
EMPTY_EVIDENCE = ("", "none", "n/a")


def parse_ledger(ledger_path: pathlib.Path) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Return {id: (state, evidence)} and error strings for table-row defects.

    Evidence "" means the ledger names no path (`none` / `n/a`).
    """
    rows: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    try:
        text = ledger_path.read_text(encoding="utf-8")
    except OSError as exc:
        return {}, [f"cannot load {ledger_path}: {exc!r}"]
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("|"):
            continue
        m = ROW_RE.match(line)
        if m is None:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rid = m.group(1)
        if len(cells) != 7:
            errors.append(f"{rid}: ledger row {lineno} has {len(cells)} cells, want 7")
            continue
        state, evidence_cell = cells[3], cells[4]
        if state == "PLANNED" and evidence_cell not in EMPTY_EVIDENCE:
            errors.append(f"{rid}: PLANNED row names evidence {evidence_cell!r}")
        if state != "PLANNED" and evidence_cell in EMPTY_EVIDENCE:
            errors.append(f"{rid}: non-PLANNED row names no evidence path")
        evidence = "" if evidence_cell in EMPTY_EVIDENCE else evidence_cell.strip("`")
        if rid in rows:
            errors.append(f"{rid}: duplicate ledger row")
        rows[rid] = (state, evidence)
    return rows, errors


def check(ledger_path: pathlib.Path, coverage_path: pathlib.Path) -> list[str]:
    """Return a list of error strings; empty means the ledger agrees."""
    errors: list[str] = []
    try:
        doc = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot load {coverage_path}: {exc!r}"]
    coverage_rows = doc.get("requirements")
    if not isinstance(coverage_rows, list) or not coverage_rows:
        return ["atlas_coverage.json: requirements must be a non-empty list"]

    ledger_rows, ledger_errors = parse_ledger(ledger_path)
    errors.extend(ledger_errors)

    for row in coverage_rows:
        rid = row.get("id", "row[?]")
        if rid not in ledger_rows:
            errors.append(f"{rid}: coverage row missing from ledger")
            continue
        state, evidence = ledger_rows[rid]
        want_state = row.get("state")
        if state != want_state:
            errors.append(f"{rid}: state {state!r} != coverage state {want_state!r}")
        want_evidence = row.get("evidence") or ""
        if evidence != want_evidence:
            errors.append(f"{rid}: evidence {evidence!r} != coverage evidence {want_evidence!r}")

    for rid in ledger_rows:
        if not any(r.get("id") == rid for r in coverage_rows):
            errors.append(f"{rid}: ledger row not present in coverage")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check claim_ledger_055.md against atlas_coverage.json"
    )
    parser.add_argument("--ledger", default=DEFAULT_LEDGER, help="path to the ledger markdown")
    parser.add_argument("--coverage", default=DEFAULT_COVERAGE, help="path to atlas_coverage.json")
    args = parser.parse_args(argv)
    root = pathlib.Path.cwd()
    ledger = pathlib.Path(args.ledger)
    coverage = pathlib.Path(args.coverage)
    if not ledger.is_absolute():
        ledger = root / ledger
    if not coverage.is_absolute():
        coverage = root / coverage

    errors = check(ledger, coverage)
    if errors:
        print(f"claim_ledger_055 INVALID ({len(errors)} mismatches):")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("claim_ledger_055 VALID: every coverage id appears once; states and evidence paths agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())