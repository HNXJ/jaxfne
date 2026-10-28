"""P-025: every script/artifact/example that (transitively) imports jaxfne
inserts the repo root into sys.path before that import can execute.

Rule:
- SCOPE: all ``*.py`` under ``scripts/``, ``artifacts/``, ``examples/``.
- COVERED: a file is covered if it imports jaxfne at module level (outside
  ``TYPE_CHECKING``, including try/except) or lazily in a function, or if it
  imports a local sibling module that is itself covered (transitive entry
  points such as evidence-figure scripts that pull jaxfne through helpers).
- GUARD: the file inserts ``Path(__file__).resolve().parents[N]`` (N reaches
  the repo root) into ``sys.path`` at a line before the first covering
  import. Lazy-only importers still need the module-level guard.
- ASSERT (scripts/ and artifacts/ only, never examples/): with a
  module-level jaxfne import outside try/except, the file asserts the import
  resolved to the repo tree, never site-packages (the silent-shadowing hole
  that voided bisect rounds in P-016/P-020).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = ["scripts", "artifacts", "examples"]
ASSERT_ROOTS = ["scripts", "artifacts"]


def _top_jaxfne(tree: ast.Module) -> tuple[bool, bool]:
    """(plain, in_try) module-level runtime jaxfne imports.

    Imports nested in a function/class body are lazy, not top-level, even
    when the def itself sits at module level.
    """
    plain = in_try = False
    parent = {c: m for m in ast.walk(tree) for c in ast.iter_child_nodes(m)}

    def _scope(nd: ast.AST) -> str:
        while nd in parent:
            nd = parent[nd]
            if isinstance(nd, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                return "lazy"
            if isinstance(nd, ast.Try):
                return "try"
        return "top"

    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            continue
        for sub in ast.walk(node):
            hit = isinstance(sub, ast.Import) and any(
                a.name.split(".")[0] == "jaxfne" for a in sub.names
            )
            hit = hit or (
                isinstance(sub, ast.ImportFrom) and (sub.module or "").split(".")[0] == "jaxfne"
            )
            if hit:
                scope = _scope(sub)
                if scope == "try":
                    in_try = True
                elif scope == "top":
                    plain = True
    return plain, in_try


def _lazy_jaxfne(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import) and any(
                    a.name.split(".")[0] == "jaxfne" for a in sub.names
                ):
                    return True
                if isinstance(sub, ast.ImportFrom) and (sub.module or "").split(".")[0] == "jaxfne":
                    return True
    return False


def _local_targets(path: Path, tree: ast.Module) -> set[Path]:
    """Top-level sibling imports resolvable to repo-local .py files."""
    out: set[Path] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                sib = path.parent / (a.name.split(".")[0] + ".py")
                if sib.is_file():
                    out.add(sib)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            sib = path.parent / (node.module.split(".")[0] + ".py")
            if sib.is_file():
                out.add(sib)
    return out


def _load(paths: list[Path]) -> dict[Path, tuple[ast.Module, str]]:
    info: dict[Path, tuple[ast.Module, str]] = {}
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8")
            info[p] = (ast.parse(text), text)
        except (OSError, SyntaxError):
            continue
    return info


def _covered(info: dict[Path, tuple[ast.Module, str]]) -> set[Path]:
    covered = {p for p, (t, _) in info.items() if any(_top_jaxfne(t)) or _lazy_jaxfne(t)}
    changed = True
    while changed:
        changed = False
        for p, (t, _) in info.items():
            if p in covered:
                continue
            if any(q in covered for q in _local_targets(p, t)):
                covered.add(p)
                changed = True
    return covered


def _root_names(tree: ast.Module, text: str, depth: int) -> set[str]:
    """Module-level names bound to the repo root via __file__."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            seg = ast.get_source_segment(text, node.value) or ""
            if "__file__" in seg and (
                f"parents[{depth + 1}]" in seg or (depth == 0 and "parent.parent" in seg)
            ):
                out.add(node.targets[0].id)
    return out


def _sys_names(tree: ast.Module) -> set[str]:
    """Top-level names bound to the sys module (plain or aliased)."""
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "sys":
                    out.add(a.asname or "sys")
    return out


def _guard_line(path: Path, tree: ast.Module, text: str) -> int | None:
    """Earliest line inserting the repo root into sys.path, else None."""
    parts = path.relative_to(REPO).parts
    depth = len(parts[1:-1])
    names = _root_names(tree, text, depth)
    sysnames = _sys_names(tree) or {"sys"}
    best = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr in ("insert", "append")
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id in sysnames
        ):
            continue
        args = node.args
        arg = args[1] if func.attr == "insert" and len(args) == 2 else args[-1]
        seg = (ast.get_source_segment(text, arg) or "").strip()
        ok = f"parents[{depth + 1}]" in seg and "__file__" in seg
        ok = ok or (depth == 0 and "parent.parent" in seg and "__file__" in seg)
        ok = ok or seg in {f"str({n})" for n in names} | names
        if ok and (best is None or node.lineno < best):
            best = node.lineno
    return best


def _first_cover_line(path: Path, tree: ast.Module, covered: set[Path]) -> int | None:
    """Earliest top-level import line that can pull jaxfne at runtime."""
    best = None
    for node in tree.body:
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.dump(node.test):
            continue
        hit = False
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                if any(a.name.split(".")[0] == "jaxfne" for a in sub.names):
                    hit = True
                for a in sub.names:
                    sib = path.parent / (a.name.split(".")[0] + ".py")
                    if sib in covered:
                        hit = True
            if isinstance(sub, ast.ImportFrom):
                if (sub.module or "").split(".")[0] == "jaxfne":
                    hit = True
                elif sub.level == 0 and sub.module:
                    sib = path.parent / (sub.module.split(".")[0] + ".py")
                    if sib in covered:
                        hit = True
        if hit and isinstance(node, (ast.Import, ast.ImportFrom, ast.Try, ast.If)):
            if best is None or node.lineno < best:
                best = node.lineno
    return best


def _all_files() -> list[Path]:
    # artifacts/_temp/ is the gitignored scratch area: local, never shipped, out of scope.
    temp = REPO / "artifacts" / "_temp"
    out: list[Path] = []
    for root in ROOTS:
        out.extend(p for p in sorted((REPO / root).rglob("*.py")) if temp not in p.parents)
    return out


def test_every_covered_file_guards_repo_root():
    info = _load(_all_files())
    covered = _covered(info)
    assert covered, "P-025 gate examined an empty file set"
    bad = []
    for p in sorted(covered):
        tree, text = info[p]
        gl = _guard_line(p, tree, text)
        fl = _first_cover_line(p, tree, covered)
        if gl is None or (fl is not None and not gl < fl):
            bad.append(str(p.relative_to(REPO)))
    assert not bad, f"P-025: {len(bad)} files pull jaxfne without a prior repo-root guard: {bad}"


def _has_p025_assert(tree: ast.Module) -> bool:
    """True when the file asserts ``"site-packages" not in <jaxfne module>.__file__``.

    The module is ``sys.modules["jaxfne"]`` or a name bound by ``import jaxfne [as X]``.
    Matched on the AST, so line breaks and quote style do not matter, while a comment or
    string that merely contains "site-packages" does not match.
    """
    sysnames = _sys_names(tree) or {"sys"}
    aliases = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
               for a in n.names if a.name == "jaxfne"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare) and len(test.ops) == 1
                and isinstance(test.ops[0], ast.NotIn)):
            continue
        left, right = test.left, test.comparators[0]
        if not (isinstance(left, ast.Constant) and left.value == "site-packages"):
            continue
        if not (isinstance(right, ast.Attribute) and right.attr == "__file__"):
            continue
        mod = right.value
        if isinstance(mod, ast.Name) and mod.id in aliases:
            return True
        if (isinstance(mod, ast.Subscript) and isinstance(mod.slice, ast.Constant)
                and mod.slice.value == "jaxfne" and isinstance(mod.value, ast.Attribute)
                and mod.value.attr == "modules" and isinstance(mod.value.value, ast.Name)
                and mod.value.value.id in sysnames):
            return True
    return False


def test_scripts_and_artifacts_assert_repo_package():
    info = _load(_all_files())
    covered = _covered(info)
    checked = 0
    bad = []
    for p in sorted(covered):
        if p.relative_to(REPO).parts[0] not in ASSERT_ROOTS:
            continue
        tree, _ = info[p]
        plain, _ = _top_jaxfne(tree)
        if plain:
            checked += 1
            if not _has_p025_assert(tree):
                bad.append(str(p.relative_to(REPO)))
    assert checked, "P-025 assert gate examined an empty file set"
    assert not bad, f"P-025: {len(bad)} files lack the site-packages assert: {bad}"


def test_p025_assert_matcher_rejects_text_without_the_assert():
    guard = 'assert "site-packages" not in {}.__file__, ("P-025")\n'
    assert not _has_p025_assert(ast.parse("import jaxfne\n# site-packages\n"))
    assert not _has_p025_assert(ast.parse('import jaxfne\nx = "site-packages"\n'))
    assert not _has_p025_assert(ast.parse("import numpy as np\n" + guard.format("np")))
    assert _has_p025_assert(ast.parse("import jaxfne as J\n" + guard.format("J")))
    assert _has_p025_assert(ast.parse("import sys\nimport jaxfne\n" + guard.format('sys.modules["jaxfne"]')))
