"""Coverage check for artifacts/programme/atlas_coverage.json (0.5.1 ATLAS item 8b).

Validates: schema tag, unique well-formed IDs, release values, states,
sim/ID consistency; every non-PLANNED row must name an evidence path that
exists (relative to the repo root). Read-only over the JSON: never edits it
(dispatcher-only file).

Usage: PYTHONPATH=. python scripts/check_atlas_coverage.py [--check]
       [--coverage PATH] [--root PATH]
Exit 0 when valid, 1 otherwise. Runs in CI; every later seal uses it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

SCHEMA_TAG = "jaxfne.atlas_coverage.v1"
KNOWN_STATES = ("PLANNED", "SUPPORTED", "VALIDATED", "CANONICAL", "OUT_OF_SCOPE")
REQUIRED_FIELDS = (
    "id",
    "sim",
    "requirement",
    "capability",
    "release",
    "candidate",
    "evidence",
    "state",
)
ID_RE = re.compile(r"AT-\d{2}-R\d+$")
RELEASE_RE = re.compile(r"0\.5\.[1-5](-0\.5\.[1-5])?$")
SIM_RE = re.compile(r"S([1-9]|10)$")


def check(coverage_path: pathlib.Path, root: pathlib.Path) -> list[str]:
    """Return a list of error strings; empty means VALID."""
    errors: list[str] = []
    try:
        doc = json.loads(coverage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot load {coverage_path}: {exc!r}"]

    if doc.get("schema") != SCHEMA_TAG:
        errors.append(f"schema tag must be {SCHEMA_TAG!r}, got {doc.get('schema')!r}")
    if not isinstance(doc.get("states"), dict) or set(doc["states"]) != set(KNOWN_STATES):
        errors.append(f"states table must define exactly {list(KNOWN_STATES)}")
    if not doc.get("seal_rule"):
        errors.append("missing seal_rule")

    rows = doc.get("requirements")
    if not isinstance(rows, list) or not rows:
        return errors + ["requirements must be a non-empty list"]

    seen: set[str] = set()
    for i, row in enumerate(rows):
        where = row.get("id", f"row[{i}]") if isinstance(row, dict) else f"row[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{where}: not an object")
            continue
        missing = [f for f in REQUIRED_FIELDS if f not in row]
        if missing:
            errors.append(f"{where}: missing fields {missing}")
            continue

        rid = row["id"]
        if not isinstance(rid, str) or not ID_RE.fullmatch(rid):
            errors.append(f"{where}: bad id {rid!r} (want AT-0n-R<k>)")
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)

        sim = row["sim"]
        num = rid.split("-")[1] if isinstance(rid, str) and "-" in rid else None
        if num == "00":
            if sim != "all":
                errors.append(f"{rid}: AT-00 row must have sim 'all', got {sim!r}")
        elif isinstance(sim, str) and SIM_RE.fullmatch(sim):
            if num is not None and sim != f"S{int(num)}":
                errors.append(f"{rid}: sim {sim!r} inconsistent with id number {num}")
        else:
            errors.append(f"{rid}: bad sim {sim!r} (want 'all' or S1..S10)")

        if not isinstance(row["release"], str) or not RELEASE_RE.fullmatch(row["release"]):
            errors.append(f"{rid}: bad release {row['release']!r}")
        if row["state"] not in KNOWN_STATES:
            errors.append(f"{rid}: bad state {row['state']!r}")
        if not isinstance(row["candidate"], bool):
            errors.append(f"{rid}: candidate must be bool, got {row['candidate']!r}")
        if not isinstance(row["requirement"], str) or not row["requirement"].strip():
            errors.append(f"{rid}: empty requirement")
        if not isinstance(row["capability"], str) or not row["capability"].strip():
            errors.append(f"{rid}: empty capability")

        ev = row["evidence"]
        if row["state"] == "PLANNED":
            if ev is not None and not (isinstance(ev, str) and (root / ev).exists()):
                errors.append(f"{rid}: PLANNED evidence names a missing path {ev!r}")
        else:
            if not isinstance(ev, str) or not ev:
                errors.append(f"{rid}: non-PLANNED row must name an evidence path")
            elif not (root / ev).exists():
                errors.append(f"{rid}: evidence path does not exist: {ev!r}")
    return errors


def summary(doc: dict) -> str:
    rows = doc.get("requirements", [])
    by_state: dict[str, int] = {}
    by_release: dict[str, int] = {}
    for row in rows:
        by_state[row["state"]] = by_state.get(row["state"], 0) + 1
        by_release[row["release"]] = by_release.get(row["release"], 0) + 1
    lines = [f"rows: {len(rows)}"]
    lines.append("by state: " + ", ".join(f"{k}={by_state[k]}" for k in sorted(by_state)))
    lines.append("by release: " + ", ".join(f"{k}={by_release[k]}" for k in sorted(by_release)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate atlas_coverage.json")
    parser.add_argument(
        "--check", action="store_true", help="CI spelling: validate, exit code only"
    )
    parser.add_argument(
        "--coverage",
        default="artifacts/programme/atlas_coverage.json",
        help="path to atlas_coverage.json (repo-relative or absolute)",
    )
    parser.add_argument("--root", default=".", help="repo root for evidence paths")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root)
    coverage = pathlib.Path(args.coverage)
    if not coverage.is_absolute():
        coverage = root / coverage

    errors = check(coverage, root)
    if errors:
        print(f"atlas_coverage INVALID ({len(errors)} errors):")
        for err in errors:
            print(f"  - {err}")
        return 1
    if not args.check:
        doc = json.loads(coverage.read_text(encoding="utf-8"))
        print("atlas_coverage VALID")
        print(summary(doc))
    else:
        print("atlas_coverage VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
