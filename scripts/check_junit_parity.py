#!/usr/bin/env python3
"""Require RC-vs-CI per-node outcome parity as a blocking gate check.

``compare_pytest_junit.py`` can answer the question; until this wrapper existed
nothing *asked* it, so the comparison was advisory and a divergence could reach
a release unnoticed. This module is invoked from the RC gate, so a divergence
now fails the gate.

Failing conditions (any one of them):
  * an RC node absent from CI, or a CI node absent from RC
  * an outcome difference not classified INTENTIONAL_PLATFORM_DIFFERENCE
  * a difference classified ENVIRONMENT_DEFECT or UNKNOWN
  * any failure or error on either side

Justified platform differences (POSIX-only shell gates and executable-bit
checks, skipped on Windows and passing on Linux) are preserved, not suppressed:
they are reported by node ID and counted, and only that classification passes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_pytest_junit import compare, load, load_many  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ATTESTATIONS = ROOT / "artifacts" / "attestations"
RC_JUNIT = ("rc-pytest-broad.xml", "rc-pytest-slow.xml", "rc-pytest-notebook.xml")
CI_WORKFLOW = "release_ci.yml"


def _head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, check=True, text=True
    ).stdout.strip()


def _gh(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gh", *args], cwd=ROOT, capture_output=True, text=True)


def fetch_ci_junit(sha: str, dest: Path) -> list[Path]:
    """Download the release_ci JUnit artifacts for one exact commit.

    No fallback to another commit's evidence: comparing against a different
    SHA's results would compare the RC to something it is not.
    """
    listing = _gh(
        "run", "list", "--workflow", CI_WORKFLOW, "--commit", sha,
        "--status", "success", "--limit", "1", "--json", "databaseId",
        "--jq", ".[0].databaseId // empty",
    )
    if listing.returncode != 0:
        raise SystemExit(f"gh run list failed: {listing.stderr.strip()}")
    run_id = listing.stdout.strip()
    if not run_id:
        raise SystemExit(
            f"No successful {CI_WORKFLOW} run for {sha}. JUnit parity cannot be "
            "evaluated, and an unevaluated comparison is not a pass."
        )
    print(f"+ CI evidence from {CI_WORKFLOW} run {run_id} @ {sha}", flush=True)
    dest.mkdir(parents=True, exist_ok=True)
    download = _gh("run", "download", run_id, "--pattern", "pytest-results-*", "--dir", str(dest))
    if download.returncode != 0:
        raise SystemExit(f"gh run download failed: {download.stderr.strip()}")
    found = sorted(dest.rglob("pytest-results-*.xml"))
    if not found:
        raise SystemExit(f"No pytest-results-*.xml in run {run_id} artifacts")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci-junit", type=Path, nargs="*", default=None,
                        help="CI JUnit XML file(s); downloaded for HEAD when omitted")
    parser.add_argument("--sha", default=None, help="commit to fetch CI evidence for")
    parser.add_argument("--out", type=Path, default=ATTESTATIONS / "junit-parity.json")
    args = parser.parse_args(argv)

    rc_paths = [ATTESTATIONS / name for name in RC_JUNIT]
    missing = [p.name for p in rc_paths if not p.is_file()]
    if missing:
        raise SystemExit(
            f"Missing RC JUnit evidence: {missing}. Run the RC gate sweeps first."
        )
    rc = load_many(rc_paths)

    sha = args.sha or _head_sha()
    ci_paths = args.ci_junit or fetch_ci_junit(sha, ATTESTATIONS / "ci-junit")

    reports = {}
    ok = True
    for path in ci_paths:
        report = compare(rc, load(path))
        justified = [d for d in report["outcome_differences"]
                     if d["classification"] == "INTENTIONAL_PLATFORM_DIFFERENCE"]
        reports[path.name] = {
            "rc_count": report["rc_count"],
            "ci_count": report["ci_count"],
            "rc_only": report["rc_only"],
            "ci_only": report["ci_only"],
            "justified_platform_differences": [d["node_id"] for d in justified],
            "unjustified_differences": report["unjustified_differences"],
            "failures_or_errors": report["failures_or_errors"],
            "pass": report["pass"],
        }
        status = "PASS" if report["pass"] else "FAIL"
        print(
            f"{status}  {path.name}: rc={report['rc_count']} ci={report['ci_count']} "
            f"rc_only={len(report['rc_only'])} ci_only={len(report['ci_only'])} "
            f"justified={len(justified)} unjustified={len(report['unjustified_differences'])} "
            f"failures={len(report['failures_or_errors'])}",
            flush=True,
        )
        for diff in justified:
            print(f"    justified platform difference: {diff['node_id']}", flush=True)
        for diff in report["unjustified_differences"]:
            print(
                f"    UNJUSTIFIED [{diff['classification']}/{diff['kind']}] {diff['node_id']}",
                file=sys.stderr,
            )
        ok = ok and report["pass"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"sha": sha, "pass": ok, "comparisons": reports}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"+ wrote {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
