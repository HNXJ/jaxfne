#!/usr/bin/env python3
"""Compare two pytest --junitxml files by node ID and outcome.

Usage:
    python scripts/compare_pytest_junit.py --rc rc.xml --ci ci.xml [--json]

Each record carries node_id, outcome (passed/skipped/failed/error),
and skip/xfail reason. Differences are classified:

  INTENTIONAL_PLATFORM_DIFFERENCE - skip reason names a platform
    (POSIX, Windows, Linux, macOS, executable bit).
  ENVIRONMENT_DEFECT - a required capability is missing on one side
    (reportlab/viz; extends to importorskip names when the reason
    names the distribution).
  UNKNOWN - everything else.

Exit codes: 0 when node sets match and every outcome difference is
INTENTIONAL_PLATFORM_DIFFERENCE; 1 otherwise (node-set mismatch,
any ENVIRONMENT_DEFECT/UNKNOWN diff, or any failure/error).
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PLATFORM_MARKERS = ("posix", "windows", "linux", "macos", "darwin", "executable bit")
ENV_MARKERS = ("reportlab", "viz extra", "not installed")


def _outcome(case: ET.Element) -> tuple[str, str]:
    for tag in ("failure", "error"):
        el = case.find(tag)
        if el is not None:
            return tag, (el.get("message") or "").strip()
    skip = case.find("skipped")
    if skip is not None:
        return "skipped", (skip.get("message") or "").strip()
    return "passed", ""


def load(path: Path) -> dict[str, dict]:
    root = ET.parse(path).getroot()
    out: dict[str, dict] = {}
    for case in root.iter("testcase"):
        classname = case.get("classname") or ""
        name = case.get("name") or ""
        node = f"{classname}::{name}" if classname and name else (classname or name)
        outcome, reason = _outcome(case)
        out[node] = {"outcome": outcome, "reason": reason}
    return out


def classify(node: str, a: dict, b: dict) -> str:
    reason = f"{a.get('reason', '')} {b.get('reason', '')}".lower()
    if any(m in reason for m in PLATFORM_MARKERS):
        return "INTENTIONAL_PLATFORM_DIFFERENCE"
    if any(m in reason for m in ENV_MARKERS):
        return "ENVIRONMENT_DEFECT"
    return "UNKNOWN"


def compare(rc: dict, ci: dict) -> dict:
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
                          "classification": classify(node, rc[node], ci[node]),
                          "kind": "outcome"})
    unjustified = [d for d in diffs
                   if d["classification"] != "INTENTIONAL_PLATFORM_DIFFERENCE"]
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
