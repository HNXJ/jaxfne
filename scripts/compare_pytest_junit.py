#!/usr/bin/env python3
"""Compare two pytest --junitxml files by node ID and outcome.

Usage:
    python scripts/compare_pytest_junit.py --rc rc.xml --ci ci.xml [--json]

Each record carries node_id, outcome (passed/skipped/failed/error),
skip/xfail reason, skip type (pytest.skip or pytest.xfail) and the
jaxfne markers that tests/conftest.py records as the "jaxfne_marker"
user property. Differences are classified:

  INTENTIONAL_PLATFORM_DIFFERENCE - skip reason names a platform
    (POSIX, Windows, Linux, macOS, executable bit).
  INTENTIONAL_VERSION_DIFFERENCE - the CI file is Python 3.11 and the
    skip is the declared jnwb contract (pyproject extra installs jnwb
    only for Python >= 3.12). Not applied to any other CI Python.
  ENVIRONMENT_DEFECT - a required capability is missing on one side
    (reportlab/viz; extends to importorskip names when the reason
    names the distribution).
  INTENTIONAL_DATA_GATE - data_gated marker; one leg passed, the other
    skipped (warehouse inputs present on one leg only).
  INTENTIONAL_OPTIONAL_ORACLE - optional_oracle marker; one leg passed,
    the other skipped (an oracle package absent on one leg).
  INTENTIONAL_DIAGNOSTIC - bitwise_diagnostic marker; one leg passed (xpass),
    the other an xfail skip. Informational by marker definition.
  UNKNOWN - everything else.

The three marker classes need the marker on both legs, so a CI file
written before the property existed stays UNKNOWN. A marker never excuses
a failed or error outcome, and it applies only to a passed-versus-skipped
pair (the diagnostic also needs the skipped leg to be pytest.xfail).

Exit codes: 0 when node sets match and every outcome difference is in
PASSING_DIFFERENCES; 1 otherwise (node-set mismatch, any
ENVIRONMENT_DEFECT/UNKNOWN diff, or any failure/error).
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

PLATFORM_MARKERS = ("posix", "windows", "linux", "macos", "darwin", "executable bit")
ENV_MARKERS = ("reportlab", "viz extra", "not installed")
# pyproject.toml: jnwb extra installs jnwb only when python_version >= 3.12.
# These are the skip texts the 3.11 tests emit. They justify a 3.11 CI
# file only; a 3.14 CI skip with the same text stays UNKNOWN.
JNWB_311_SKIP_MARKERS = (
    "no module named 'jnwb'",
    'no module named "jnwb"',
    "installed jnwb has no compute_psd",
)
# Written by tests/conftest.py as a per-testcase JUnit property.
MARKER_PROPERTY = "jaxfne_marker"
# marker -> class; checked after the reason-text classes, so a skip whose
# reason names an environment defect stays ENVIRONMENT_DEFECT.
MARKER_CLASSES = (
    ("data_gated", "INTENTIONAL_DATA_GATE"),
    ("optional_oracle", "INTENTIONAL_OPTIONAL_ORACLE"),
    ("bitwise_diagnostic", "INTENTIONAL_DIAGNOSTIC"),
)
PASSING_DIFFERENCES = frozenset({
    "INTENTIONAL_PLATFORM_DIFFERENCE",
    "INTENTIONAL_VERSION_DIFFERENCE",
    "INTENTIONAL_DATA_GATE",
    "INTENTIONAL_OPTIONAL_ORACLE",
    "INTENTIONAL_DIAGNOSTIC",
})


def _outcome(case: ET.Element) -> tuple[str, str]:
    for tag in ("failure", "error"):
        el = case.find(tag)
        if el is not None:
            return tag, (el.get("message") or "").strip()
    skip = case.find("skipped")
    if skip is not None:
        return "skipped", (skip.get("message") or "").strip()
    return "passed", ""


def _skip_type(case: ET.Element) -> str:
    """pytest.xfail for an xfail-marked skip, pytest.skip for a plain skip."""
    skip = case.find("skipped")
    return (skip.get("type") or "") if skip is not None else ""


def _markers(case: ET.Element) -> list[str]:
    found = {p.get("value") or "" for p in case.findall(
        f"properties/property[@name='{MARKER_PROPERTY}']")}
    return sorted(found - {""})


def load(path: Path) -> dict[str, dict]:
    root = ET.parse(path).getroot()
    out: dict[str, dict] = {}
    for case in root.iter("testcase"):
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        node = f"{classname}::{name}" if classname and name else (classname or name)
        outcome, reason = _outcome(case)
        out[node] = {"outcome": outcome, "reason": reason,
                     "skip_type": _skip_type(case), "markers": _markers(case)}
    return out


def _marker_class(a: dict, b: dict) -> str | None:
    """Class for a passed-versus-skipped pair whose test carries a marker on both legs."""
    if {a["outcome"], b["outcome"]} != {"passed", "skipped"}:
        return None
    skipped = a if a["outcome"] == "skipped" else b
    shared = set(a.get("markers", [])) & set(b.get("markers", []))
    for marker, cls in MARKER_CLASSES:
        if marker not in shared:
            continue
        if cls == "INTENTIONAL_DIAGNOSTIC" and skipped.get("skip_type") != "pytest.xfail":
            continue
        return cls
    return None


def classify(node: str, a: dict, b: dict, ci_python: str | None = None) -> str:
    reason = f"{a.get('reason', '')} {b.get('reason', '')}".lower()
    if any(m in reason for m in PLATFORM_MARKERS):
        return "INTENTIONAL_PLATFORM_DIFFERENCE"
    if ci_python == "3.11" and any(m in reason for m in JNWB_311_SKIP_MARKERS):
        return "INTENTIONAL_VERSION_DIFFERENCE"
    if any(m in reason for m in ENV_MARKERS):
        return "ENVIRONMENT_DEFECT"
    return _marker_class(a, b) or "UNKNOWN"


def compare(rc: dict, ci: dict, ci_python: str | None = None) -> dict:
    rc_ids, ci_ids = set(rc), set(ci)
    diffs = []
    for node in sorted((rc_ids ^ ci_ids) | {
        n for n in rc_ids & ci_ids
        if (rc[n]["outcome"], ci[n]["outcome"]) != (rc[n]["outcome"], rc[n]["outcome"])
        or rc[n]["outcome"] != ci[n]["outcome"]
    }):
        if node not in rc:
            diffs.append({"node_id": node, "rc": None, "ci": ci[node],
                          "classification": "UNKNOWN", "kind": "ci_only"})
        elif node not in ci:
            diffs.append({"node_id": node, "rc": rc[node], "ci": None,
                          "classification": "UNKNOWN", "kind": "rc_only"})
        else:
            diffs.append({"node_id": node, "rc": rc[node], "ci": ci[node],
                          "classification": classify(node, rc[node], ci[node], ci_python),
                          "kind": "outcome"})
    unjustified = [d for d in diffs
                   if d["classification"] not in PASSING_DIFFERENCES]
    failures = [n for n, r in {**rc, **ci}.items()
                if r["outcome"] in ("failure", "error")]
    return {
        "rc_count": len(rc), "ci_count": len(ci),
        "rc_only": sorted(set(rc) - set(ci)),
        "ci_only": sorted(set(ci) - set(rc)),
        "outcome_differences": diffs,
        "unjustified_differences": unjustified,
        "failures_or_errors": sorted(set(failures)),
        "pass": not (set(rc) ^ set(ci)) and not unjustified and not failures,
    }


def load_many(paths: list[Path]) -> dict[str, dict]:
    """Merge several JUnit files (e.g. the RC broad/slow/notebook sweeps)."""
    merged: dict[str, dict] = {}
    for path in paths:
        for node, rec in load(path).items():
            if node in merged and merged[node] != rec:
                raise ValueError(f"duplicate node {node!r} with conflicting records")
            merged[node] = rec
    return merged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rc", required=True, nargs="+",
                    help="RC-side JUnit XML file(s); several are merged")
    ap.add_argument("--ci", required=True, help="CI-side JUnit XML")
    ap.add_argument("--json", action="store_true", help="emit full report as JSON")
    args = ap.parse_args(argv)
    report = compare(load_many([Path(p) for p in args.rc]), load(Path(args.ci)))
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"rc nodes: {report['rc_count']}  ci nodes: {report['ci_count']}")
        print(f"rc_only: {len(report['rc_only'])}  ci_only: {len(report['ci_only'])}")
        for d in report["outcome_differences"]:
            print(f"  {d['classification']:32s} {d['kind']:8s} {d['node_id']}")
        print("PASS" if report["pass"] else "FAIL")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
