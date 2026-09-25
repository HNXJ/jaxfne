"""artifacts/memory.md stays true: every path and symbol it names resolves.

The brief is loaded into every delegated agent's context, so a stale path or
a renamed symbol misleads silently. Checked: backticked repo paths (globs need
one match; ``path:NAME`` also needs NAME in the file), backticked
``jaxfne.<...>`` dotted names, and each entry point listed in the package-map
table (section 3) against the module named in its row or the package root.
"""

from __future__ import annotations

import importlib
import pathlib
import re

import jaxfne

REPO = pathlib.Path(__file__).resolve().parent.parent
BRIEF = REPO / "artifacts" / "memory.md"
PATH_ROOTS = ("jaxfne/", "tests/", "scripts/", "artifacts/", "docs/", "examples/", ".github/")
ROOT_FILES = ("README.md", "MEMORY.md", "Makefile", "pyproject.toml", "mkdocs.yml", "CITATION.cff")
TICKS = re.compile(r"`([^`\n]+)`")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _text() -> str:
    return BRIEF.read_text(encoding="utf-8")


def _resolve_dotted(name: str) -> bool:
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:cut]))
        except ImportError:
            continue
        try:
            for attr in parts[cut:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return False
        return True
    return False


def test_brief_paths_exist():
    missing = []
    for token in TICKS.findall(_text()):
        path, _, name = token.partition(":")
        if not (path.startswith(PATH_ROOTS) or path in ROOT_FILES) or " " in path:
            continue
        hits = list(REPO.glob(path)) if "*" in path else [REPO / path]
        if not hits or not all(h.exists() for h in hits):
            missing.append(token)
        elif name and IDENT.fullmatch(name) and name not in hits[0].read_text(encoding="utf-8"):
            missing.append(token)
    assert not missing, f"memory.md names paths that do not exist: {missing}"


def test_brief_dotted_symbols_resolve():
    bad = [
        t
        for t in TICKS.findall(_text())
        if re.fullmatch(r"jaxfne(\.[A-Za-z_][A-Za-z0-9_]*)+", t) and not _resolve_dotted(t)
    ]
    assert not bad, f"memory.md names jaxfne symbols that do not resolve: {bad}"


def test_package_map_entry_points_resolve():
    section = _text().split("## 3. Package map", 1)[1].split("\n## ", 1)[0]
    bad = []
    rows = 0
    for line in section.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not cells[0].startswith("`jaxfne"):
            continue
        rows += 1
        modules = [
            importlib.import_module(m)
            for m in TICKS.findall(cells[0])
            if re.fullmatch(r"jaxfne(\.\w+)*", m)
        ]
        for token in TICKS.findall(cells[2]):
            if not IDENT.fullmatch(token):
                continue  # shorthand such as `cable_filter_sources/tau/report`
            if not any(hasattr(m, token) for m in [*modules, jaxfne]):
                bad.append(f"{cells[0]}: {token}")
    assert rows >= 10, "package-map table not found or reshaped; this test would pass vacuously"
    assert not bad, f"package-map entry points that do not resolve: {bad}"
