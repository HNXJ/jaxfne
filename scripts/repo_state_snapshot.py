#!/usr/bin/env python3
"""Print volatile jaxfne checkout state as deterministic JSON.

This command is read-only. It is the preferred source for branch, SHA, version,
dependency, export-surface, and status observations; do not copy its values into
artifacts/AGENTS.md, artifacts/skills, or other persistent doctrine.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_DEPENDENCIES = ("jaxley", "jax_fem", "optax", "pynwb", "matplotlib", "plotly")


def _run_git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _remote_head(branch: str) -> str | None:
    output = _run_git("ls-remote", "origin", f"refs/heads/{branch}")
    if not output:
        return None
    return output.split()[0]


def _optional_dependency_state() -> dict[str, bool]:
    state: dict[str, bool] = {}
    for name in OPTIONAL_DEPENDENCIES:
        try:
            state[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            state[name] = False
    return state


def _package_state() -> tuple[dict[str, Any], list[str]]:
    sys.path.insert(0, str(ROOT))
    errors: list[str] = []
    try:
        import jax
        import jaxfne
    except Exception as exc:  # pragma: no cover - environment-dependent
        errors.append(f"package_import: {type(exc).__name__}: {exc}")
        return {
            "version": None,
            "jax_version": None,
            "root_export_count": None,
        }, errors

    return {
        "version": getattr(jaxfne, "__version__", None),
        "jax_version": getattr(jax, "__version__", None),
        "root_export_count": len(getattr(jaxfne, "__all__", ())),
    }, errors


def build_snapshot() -> dict[str, Any]:
    package, errors = _package_state()
    status_lines = _run_git("status", "--porcelain=v1") or ""
    snapshot: dict[str, Any] = {
        "schema_version": "jaxfne.repository_state.v1",
        "repository_root": str(ROOT),
        "git": {
            "branch": _run_git("branch", "--show-current"),
            "head_sha": _run_git("rev-parse", "HEAD"),
            "working_tree_clean": not bool(status_lines),
            "working_tree_status": status_lines.splitlines(),
            "remote_heads": {
                "main": _remote_head("main"),
                "dev": _remote_head("dev"),
            },
        },
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package": package,
            "optional_dependencies": _optional_dependency_state(),
        },
        "status_vocabulary": {
            "field_operator_type": [
                "linear_projection",
                "pde_solve",
                "not_computed",
            ],
            "field_solver_status": [
                "not_solved",
                "experimental_pde_solver",
                "validated_solver",
            ],
            "amplitude_status": ["relative", "calibrated"],
            "compatibility_metadata": {
                "field_solver_status": "linear_solver",
            },
        },
        "errors": errors,
    }
    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="return nonzero when package or Git observations are unavailable",
    )
    args = parser.parse_args(argv)
    snapshot = build_snapshot()
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    if args.check:
        return int(
            not snapshot["git"]["branch"]
            or not snapshot["git"]["head_sha"]
            or bool(snapshot["errors"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
