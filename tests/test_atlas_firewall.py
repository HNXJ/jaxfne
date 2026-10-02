"""Firewall gate (0.5.1 ATLAS item 8): AT scenarios use the public surface only.

Static check, no simulation:
1. Every ``artifacts/atlas/*.py`` scenario file imports only the top-level
   ``jaxfne`` package (and ``jaxfne.public_surface``) plus non-jaxfne
   modules. Any ``jaxfne.<submodule>`` import is refused.
2. No ``jaxfne/`` engine file contains an AT-specific branch or token.

Authority: 0.5.1 stack items 8/8b; public surface
``jaxfne/public_surface.py``.
"""

from __future__ import annotations

import ast
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
ATLAS_DIR = REPO / "artifacts" / "atlas"
ENGINE_DIR = REPO / "jaxfne"

# Only these jaxfne-rooted imports are the public surface.
_ALLOWED_JAXFNE_IMPORTS = {"jaxfne", "jaxfne.public_surface"}

# Tokens that mark an AT-specific branch inside the engine.
_AT_TOKEN = re.compile(r"AT-0\d\b|AT-10|at01_at10|atlas_toy|run_at0\d|run_at10")


def _scenario_files() -> list[pathlib.Path]:
    assert ATLAS_DIR.is_dir(), f"missing scenario dir {ATLAS_DIR}"
    files = sorted(ATLAS_DIR.glob("*.py"))
    assert files, "firewall gate must not pass vacuously: no scenario files found"
    return files


def _jaxfne_imports(tree: ast.AST) -> list[str]:
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "jaxfne" and alias.name not in _ALLOWED_JAXFNE_IMPORTS:
                    bad.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "jaxfne" or module == "jaxfne.public_surface":
                continue
            if module == "" or module.split(".")[0] == "jaxfne":
                bad.append(f"from {module} import ...")
    return bad


def test_at_scenarios_import_only_public_surface():
    violations: dict[str, list[str]] = {}
    for path in _scenario_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bad = _jaxfne_imports(tree)
        if bad:
            violations[path.name] = bad
    assert not violations, (
        "AT scenario files must import only the public surface "
        f"(jaxfne, jaxfne.public_surface); violations: {violations}"
    )


def test_no_at_specific_branch_in_engine():
    hits: list[str] = []
    for path in sorted(ENGINE_DIR.rglob("*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            if _AT_TOKEN.search(line):
                hits.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()[:120]}")
    assert not hits, f"AT-specific branches/tokens inside jaxfne/: {hits}"
