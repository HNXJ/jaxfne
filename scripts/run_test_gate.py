#!/usr/bin/env python3
"""Executable test gates for jaxfne.

Gate vocabulary (single source of truth for local runs and CI references):

  dev          Curated architectural gate (~1 minute)
  broad        Repository-wide ``-m "not slow"`` (~8 minutes)
  release      Broad + slow markers + release examples/docs checks
  publication  Frozen scientific experiments and artifact validation

Documentation must refer to these gate names and this script (or Makefile
targets), not duplicate file lists in prose.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PYTEST_ENV = {
    **os.environ,
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "PYTHONPATH": str(ROOT),
}

# Curated dev gate — owned here, not in docs/ci_policy.md prose.
DEV_PYTEST_TARGETS = [
    "tests/test_api_smoke.py",
    "tests/test_root_import_lightweight.py",
    "tests/test_signals_get_v0329.py",
    "tests/test_neuronal_tensor_connectivity.py",
    "tests/test_neuronal_tensor.py",
    "tests/test_connection_rule_compile_v0330.py",
    "tests/test_continuation_contract.py",
    "tests/test_mcc.py",
]

BROAD_PYTEST_IGNORE = [
    "tests/test_multi_area_source_projector.py",
    "tests/test_multi_area_spectrolaminar_objective.py",
]

RELEASE_EXAMPLES = [
    "examples/00_minimal_column.py",
    "examples/01_source_field_manifest.py",
    "examples/01_generalized_readout_smoke.py",
    "examples/02_generalized_vis_smoke.py",
    "examples/02_omission_scaffold.py",
    "examples/02_spectrolaminar_oddball_scaffold.py",
    "examples/03_objective_and_tune_smoke.py",
    "examples/03_jaxley_bridge_smoke.py",
    "examples/04_blackbox_tuning_loop.py",
    "examples/05_dataset_bridge_manifest.py",
    "examples/06_edge_list_recurrent_backend.py",
    "examples/07_jaxley_trace_bridge.py",
    "examples/08_neuronal_tensor_first.py",
]


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd or ROOT, check=True, env=PYTEST_ENV)


def gate_dev() -> None:
    _run([sys.executable, "-m", "compileall", "-q", "jaxfne", "tests", "scripts"])
    _run(
        [
            sys.executable,
            "-m",
            "pytest",
            *DEV_PYTEST_TARGETS,
            "-q",
            "-m",
            "not slow and not release",
            "--tb=short",
        ]
    )
    _run([sys.executable, "scripts/audit_public_docs_language.py", "--check"])


def gate_broad() -> None:
    _run([sys.executable, "-m", "compileall", "-q", "jaxfne", "tests", "scripts"])
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "-m",
        "not slow",
        "--tb=short",
    ]
    for path in BROAD_PYTEST_IGNORE:
        cmd.extend(["--ignore", path])
    _run(cmd)


def gate_release() -> None:
    gate_broad()
    slow_cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "-q",
        "-m",
        "slow",
        "--tb=short",
    ]
    for path in BROAD_PYTEST_IGNORE:
        slow_cmd.extend(["--ignore", path])
    _run(slow_cmd)
    _run([sys.executable, "-m", "mkdocs", "build", "--strict"])
    for example in RELEASE_EXAMPLES:
        _run([sys.executable, example])


def gate_publication() -> None:
    """Placeholder hook for frozen experiment manifests.

    Extend this gate when publication-facing receipts have a single entrypoint.
    """
    _run([sys.executable, "scripts/repo_state_snapshot.py"])
    print(
        "publication gate: repo_state_snapshot only; "
        "add frozen experiment runners here when contracted.",
        flush=True,
    )


GATES = {
    "dev": gate_dev,
    "broad": gate_broad,
    "release": gate_release,
    "publication": gate_publication,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gate",
        choices=sorted(GATES),
        help="which validation gate to run",
    )
    args = parser.parse_args(argv)
    GATES[args.gate]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
