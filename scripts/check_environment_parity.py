#!/usr/bin/env python3
"""Fail when the local gate environment cannot execute what release CI executes.

The release-candidate gate exists to guarantee ``PRE_RELEASE_GATE >=
RELEASE_CI_GATE``. That guarantee is about *executed tests*, not about check
family names, and it silently breaks when an optional dependency is absent
locally: a module guarded by ``pytest.importorskip`` is skipped at collection,
so its tests produce no node IDs at all. Nothing fails, the family still
reports PASS, and the gate under-covers CI without saying so.

Observed instance (2026-09-06): the ``jaxley`` extra was not installed locally,
so three modules skipped at collection and 16 tests that release CI runs were
never executed by the RC gate -- local collection 3746 against CI's 3762.

Two independent checks:

1. Every distribution required by the extras that the release workflow installs
   is present. The extras are read from the workflow, so this cannot drift from
   what CI actually does.
2. No test module is skipped at collection. This catches the same class of
   failure for any cause, including ones no dependency list would predict.

Usage:
    python scripts/check_environment_parity.py
    python scripts/check_environment_parity.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release_ci.yml"

# `pip install -e ".[dev,jaxley]"` -- capture the extras group list.
_EXTRAS_RE = re.compile(r"""pip\s+install\s+(?:-e\s+)?["']?\.\[([^\]]+)\]""")


def workflow_extras(path: Path = RELEASE_WORKFLOW) -> set[str]:
    """Extras the release workflow installs, read from the workflow itself."""
    if not path.exists():
        raise SystemExit(f"missing release workflow: {path}")
    text = path.read_text(encoding="utf-8")
    extras: set[str] = set()
    for match in _EXTRAS_RE.finditer(text):
        extras.update(part.strip() for part in match.group(1).split(",") if part.strip())
    return extras


def _requirement_name(spec: str) -> str | None:
    """Distribution name from a requirement string, or None to skip it."""
    spec = spec.split(";", 1)[0].strip()            # drop environment marker
    spec = re.split(r"[<>=!~\[]", spec, maxsplit=1)[0]  # drop version pin / extras
    name = spec.strip()
    if not name or name.lower().startswith("jaxfne"):
        return None                                  # self-reference
    return name


def required_distributions(extras: set[str]) -> dict[str, list[str]]:
    """Map distribution name -> extras that require it."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = data["project"].get("optional-dependencies", {})
    out: dict[str, list[str]] = {}
    for extra in sorted(extras):
        for spec in declared.get(extra, []):
            name = _requirement_name(spec)
            if name:
                out.setdefault(name, []).append(extra)
    return out


def missing_distributions(required: dict[str, list[str]]) -> list[dict]:
    missing = []
    for name, extras in sorted(required.items()):
        try:
            distribution(name)
        except PackageNotFoundError:
            missing.append({"distribution": name, "required_by_extras": extras})
    return missing


def collection_skips() -> tuple[list[str], int]:
    """Modules skipped at collection, and the total node IDs collected."""
    env = {
        **{k: v for k, v in __import__("os").environ.items()},
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONPATH": str(ROOT),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-rs",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, env=env, check=False,
    )
    out = proc.stdout + proc.stderr
    skips = [ln.strip() for ln in out.splitlines() if ln.startswith("SKIPPED")]
    collected = 0
    m = re.search(r"(\d+)\s+tests? collected", out)
    if m:
        collected = int(m.group(1))
    return skips, collected


def build_report() -> dict:
    extras = workflow_extras()
    required = required_distributions(extras)
    missing = missing_distributions(required)
    skips, collected = collection_skips()
    return {
        "schema": "jaxfne.environment_parity.v1",
        "release_workflow_extras": sorted(extras),
        "required_distribution_count": len(required),
        "missing_distributions": missing,
        "collection_skips": skips,
        "tests_collected": collected,
        "pass": not missing and not skips,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit the full report as JSON")
    args = parser.parse_args(argv)

    report = build_report()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"release workflow extras : {', '.join(report['release_workflow_extras'])}")
        print(f"required distributions  : {report['required_distribution_count']}")
        print(f"tests collected         : {report['tests_collected']}")
        if report["missing_distributions"]:
            print("\nMISSING distributions that release CI installs:")
            for item in report["missing_distributions"]:
                print(f"  {item['distribution']} (extras: {', '.join(item['required_by_extras'])})")
        if report["collection_skips"]:
            print("\nModules SKIPPED at collection (their tests produce no node IDs):")
            for line in report["collection_skips"]:
                print(f"  {line}")
        print()
        print("environment parity:", "pass" if report["pass"] else "FAIL")

    if not report["pass"]:
        print(
            "\nThe gate environment cannot execute everything release CI executes, so a "
            "PASS here would not establish PRE_RELEASE_GATE >= RELEASE_CI_GATE.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
