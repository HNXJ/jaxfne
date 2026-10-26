#!/usr/bin/env python3
"""Replication-checklist gate rows: R1, R4, R6, S1, S2, E1, E3, P4 (manifest).

Paper registry: ``dashboard/sources.json`` papers[] (``id``, ``stage``,
``stage_artifacts``, ``preregs``). Paper folders: ``artifacts/etudes/papers/*/``.

Stage-artifact paths are pinned to each paper's ``lane.sha``, where they
resolve; ``scripts/build_dashboard_data.py --check`` verifies them at that
SHA, so this gate does not re-check their working-tree existence.

- R1: a paper at stage S holds the records of every stage up to S in its folder
  (P1 ``SOURCE_P1.md``, P2 ``PORT_SPEC.md``, P3 ``PREREG_P3*.md`` or ``prereg/``
  holding ``P3_*.md``, P5 ``LEDGER.md`` and ``RESULT_P3*.md``, P6 ``RESULT_P6*.md``).
  Entries of class ``core method`` are analytic checks, not staged papers, and are skipped.
  The folder is ``artifacts/etudes/papers/<name>/``, where ``<name>`` is the path
  segment after ``artifacts/etudes/`` (or ``artifacts/etudes/papers/``) in the
  paper's own ``stage_artifacts`` paths; a paper with no ``stage_artifacts``
  maps to ``artifacts/etudes/papers/<id>/`` (then ``<id>`` before the first
  ``"-"``) when that folder exists. Disagreeing segments, or no folder at all,
  is a finding.
- R4: a paper counts as replicated with jaxfne only at P6: a paper whose
  recorded verdict is exactly ``reproduced`` holds its P6 stage result file
  (the P6 entry of its ``stage_artifacts``, resolved on disk the way R1
  resolves record files). A paper below P6, or with verdict ``reproduced
  (conditional)``, ``mixed``, ``in progress``, ``not started`` or ``failed``,
  is not a finding.
- R6: at most 7 files and 7 folders per paper folder (direct children, no ``__pycache__``).
- S1: no ``.py`` under a paper folder runs outside code: ``subprocess.*``,
  ``os.system``/``popen``/``startfile``/``fork``/``forkpty``/``exec*``/``spawn*``/
  ``posix_spawn*``, ``pty.spawn``, ``importlib.import_module``,
  ``importlib.util.spec_from_file_location`` loading outside the repository (or
  an unresolvable path), ``__import__``, ``getattr`` on an ``os``/``subprocess``
  module alias, ``runpy.*``, ``exec(``, ``eval``, or a ``sys.path`` addition
  resolving outside the repository. S1 is a lint for code written in good
  faith (a reviewer aid), not a sandbox: dynamic dispatch beyond these forms
  (computed attribute names, plugin loaders, ``os.exec*`` via ``getattr`` on
  anything but an ``os``/``subprocess`` alias) is out of scope.
  Imports are resolved per module, so aliases
  (``import subprocess as sp``, ``from subprocess import run``,
  ``from os import system as s``, ``import os as o``, ``import sys as s``,
  ``from sys import path``, ``from builtins import __import__``) do not bypass
  the rule. ``sys.path`` mutation in any form (``insert``/``append``/``extend``,
  ``+=``, subscript assignment) goes through the same resolver.
  A ``subprocess`` call whose argv is a list literal of string constants
  starting with ``"git"`` is exempt, unless an element is ``-c``/
  ``--exec-path*`` or a ``shell=``, ``executable=``, ``env=`` or
  ``preexec_fn=`` keyword is present; a non-constant argv element stays a
  finding, except the directory right after ``-C``.
- S2: a transcription of the authors' model (an oracle) is imported only by
  tests or gate scripts, never by the scored path: ``import X``,
  ``from X import``, ``from .X import``, ``importlib.import_module("X")``
  with a constant, and ``spec_from_file_location`` with a constant path that
  resolves to the oracle (module scope and function scope both count).
  Oracle modules are the paper's ``stage_artifacts`` entries labelled
  oracle/transcription/reference arm (mapped onto
  ``artifacts/etudes/papers/`` the way R1 maps them, so a stale path still
  maps; a label naming a ``.py`` file that does not exist is not an S2
  finding) and ``*_ref.py``, ``*_ref_*.py``, ``*reference*.py`` files under
  the paper folders. Allowed importers are ``test_*``/``gate_*`` files,
  files under a ``tests/`` directory, and sibling oracles of the same paper.
- E1: the scored run builds and simulates through jaxfne
  (``Configuration``/``CircuitSpec`` -> ``construct`` -> ``Model`` ->
  ``simulate``): the paper's P4 ``stage_artifacts`` entries (mapped onto
  ``artifacts/etudes/papers/`` the way oracles map; labels naming an oracle
  or a test, ``test_*`` basenames and ``*_ref*``/``*reference*`` names
  excluded; existing files only) hold at least one ``construct`` call (bare
  or attribute) and one attribute ``.simulate`` call, in any scope, across
  the modules. Papers below P4 and ``core method`` entries are skipped.
- E3: no top-level ``def NAME(`` under a paper folder shadows a public
  ``jaxfne.analysis`` name (collected recursively under ``jaxfne/analysis*``,
  including ``__all__`` re-exports; a method named like a core function is not
  a copy).
- P4 (manifest): every run writes a manifest with code SHA, backend, dtype,
  seeds and the activity-regime block: the paper's P4 ``stage_artifacts``
  entries (the same scored modules E1 checks; labels naming an oracle or a
  test, ``test_*`` basenames and ``*_ref*``/``*reference*`` names excluded)
  hold one manifest writer covering one key from each class. A writer is a
  dict display or ``dict(...)`` call assigned to a name containing
  ``manifest``, or passed to ``json.dump``/``json.dumps``/``write_text``
  (attribute calls; a name resolves to its first bound literal); the dicts
  passed to one call of a function named like ``*manifest*`` count together
  as one writer. Required
  classes, in finding order: code SHA {git_sha, code_sha, sha, commit, baseline_sha},
  backend {backend}, dtype {dtype}, seeds {seed, seeds}, regime {regime,
  activity_regime} (exact match, case-sensitive; every member is a key a real
  paper writer uses). Papers below P4 and ``core method`` entries are
  skipped; a paper with no scored module is E1's finding, not P4 (manifest)'s.

Known violations live in ``scripts/replication_known_violations.json`` as
``{"rule", "path", "message", "reason", "todo"}`` and match findings on
(rule, path, message). Under ``--check`` a missing allowlist file fails closed.


Usage:
    python scripts/check_replications.py             # report, always exit 0
    python scripts/check_replications.py --check    # exit 1 on unlisted or stale findings
    python scripts/check_replications.py --root DIR # check the tree under DIR (tests)

The repo root is the parent of this script's directory (never cwd, never an install).
Only the standard library is used; jaxfne is parsed with ``ast``, never imported.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath

ROOT = Path(__file__).resolve().parents[1]
KNOWN_PATH = Path(__file__).resolve().parent / "replication_known_violations.json"
SOURCES_REL = Path("dashboard") / "sources.json"
PAPERS_REL = Path("artifacts") / "etudes" / "papers"

STAGE_ORDER = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")

SYS_PATH_METHODS = ("insert", "append", "extend")


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_registry(root: Path) -> list[dict]:
    """Papers list from dashboard/sources.json (fail-closed on unreadable)."""
    try:
        raw = json.loads((root / SOURCES_REL).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"error: cannot read {SOURCES_REL.as_posix()}: {exc}")
    papers = raw.get("papers")
    if not isinstance(papers, list):
        raise SystemExit(f"error: {SOURCES_REL.as_posix()} holds no papers list")
    return [p for p in papers if isinstance(p, dict)]


def _artifact_paths(paper: dict) -> list[str]:
    paths = []
    stages = paper.get("stage_artifacts")
    if isinstance(stages, dict):
        for items in stages.values():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        paths.append(item["path"])
    return paths


def _paper_name(path: str) -> str | None:
    """The ``<name>`` segment of an etudes path, or None when unmappable.

    ``artifacts/etudes/papers/<name>/...`` and ``artifacts/etudes/<name>/...``
    both vote for ``<name>``. The working tree is never consulted: the paths
    are pinned to the lane SHA, only their folder segment is read here.
    """
    parts = Path(path).parts
    if len(parts) < 3 or parts[0] != "artifacts" or parts[1] != "etudes":
        return None
    rest = parts[2:]
    if rest[0] == "papers":
        rest = rest[1:]
    if len(rest) < 2 or not rest[0]:
        return None
    return rest[0]


def paper_names(paper: dict) -> set[str]:
    """Distinct ``<name>`` segments over the paper's stage_artifacts paths."""
    return {_paper_name(p) for p in _artifact_paths(paper)} - {None}


def paper_folder(paper: dict, root: Path) -> str | None:
    """The paper's folder, or None when unmappable.

    With ``stage_artifacts`` paths the segments must agree on one folder.
    Without them the paper id maps to ``artifacts/etudes/papers/<id>/`` when
    that folder exists, else ``<id>`` before the first ``"-"`` (so ``pcl-fig2``
    finds ``papers/pcl``). The working tree decides only the id fallback: the
    pinned paths are never existence-checked here.
    """
    names = paper_names(paper)
    if len(names) > 1:
        return None
    if len(names) == 1:
        return "artifacts/etudes/papers/" + next(iter(names))
    pid = paper.get("id", "")
    candidates = [pid, pid.split("-")[0]] if isinstance(pid, str) else []
    for cand in dict.fromkeys(candidates):
        if cand and (root / PAPERS_REL / cand).is_dir():
            return (PAPERS_REL / cand).as_posix()
    return None


def _r1_missing(folder: Path, stage: str) -> list[str]:
    """Record names missing for one stage in an existing folder."""
    missing = []
    if stage == "P1" and not (folder / "SOURCE_P1.md").is_file():
        missing.append("SOURCE_P1.md")
    if stage == "P2" and not (folder / "PORT_SPEC.md").is_file():
        missing.append("PORT_SPEC.md")
    if stage == "P3" and not list(folder.glob("PREREG_P3*.md")) and not list(
        (folder / "prereg").glob("P3_*.md")
    ):
        missing.append("PREREG_P3*.md (or prereg/P3_*.md)")
    if stage == "P5":
        if not (folder / "LEDGER.md").is_file():
            missing.append("LEDGER.md")
        if not list(folder.glob("RESULT_P3*.md")):
            missing.append("RESULT_P3*.md")
    if stage == "P6" and not list(folder.glob("RESULT_P6*.md")):
        missing.append("RESULT_P6*.md")
    return missing


def check_r1(papers: list[dict], root: Path) -> list[tuple[str, str, str]]:
    """R1 findings: unresolvable folders and missing stage records."""
    findings = []
    for paper in papers:
        if paper.get("class") == "core method":
            continue  # analytic checks of a core method, not a staged paper replication
        pid = paper.get("id", "?")
        stage = paper.get("stage")
        if stage not in STAGE_ORDER:
            findings.append(
                ("R1", SOURCES_REL.as_posix(), f"paper {pid!r} has unknown stage {stage!r}")
            )
            continue
        folder_rel = paper_folder(paper, root)
        if folder_rel is None:
            if not paper_names(paper):
                detail = "names no folder under artifacts/etudes/papers/"
            else:
                detail = (
                    "stage_artifacts paths disagree: "
                    f"{sorted(paper_names(paper))}"
                )
            findings.append(
                (
                    "R1",
                    SOURCES_REL.as_posix(),
                    f"paper {pid!r} (stage {stage}) {detail}",
                )
            )
            continue
        if not (root / folder_rel).is_dir():
            findings.append(
                (
                    "R1",
                    folder_rel,
                    f"paper {pid!r} (stage {stage}) folder is missing "
                    "in the working tree",
                )
            )
            continue
        folder = root / folder_rel
        for req in STAGE_ORDER[: STAGE_ORDER.index(stage) + 1]:
            for name in _r1_missing(folder, req):
                findings.append(
                    (
                        "R1",
                        folder_rel,
                        f"stage {stage} paper {pid!r} lacks the {req} record {name}",
                    )
                )
    return findings


def paper_folders(root: Path) -> list[Path]:
    """Every paper folder, sorted; absent papers root means nothing to check."""
    base = root / PAPERS_REL
    if not base.is_dir():
        return []
    return sorted(
        (p for p in base.iterdir() if p.is_dir() and p.name != "__pycache__"),
        key=lambda p: p.name,
    )


def check_r6(root: Path) -> list[tuple[str, str, str]]:
    """R6 findings: folders holding more than 7 files or 7 folders."""
    findings = []
    for folder in paper_folders(root):
        rel = _rel(folder, root)
        children = [c for c in folder.iterdir() if c.name != "__pycache__"]
        n_files = sum(1 for c in children if c.is_file())
        n_dirs = sum(1 for c in children if c.is_dir())
        if n_files > 7:
            findings.append(( "R6", rel, f"holds {n_files} files (at most 7)"))
        if n_dirs > 7:
            findings.append(( "R6", rel, f"holds {n_dirs} folders (at most 7)"))
    return findings


def _paper_py_files(root: Path) -> list[Path]:
    files = []
    for folder in paper_folders(root):
        files.extend(
            p
            for p in sorted(folder.rglob("*.py"))
            if "__pycache__" not in p.parts and p.is_file()
        )
    return files


def _collect_bindings(tree: ast.Module) -> dict[str, ast.AST]:
    """Name bindings (first binding wins), module level and function level.

    Function-local paths such as ``dl_path = _RUN_DIR / "data_labels.py"``
    resolve through module-level names, so sibling-module loads stay clean.
    Parameters are not bindings: a bare parameter never resolves.
    """
    bindings: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            bindings.setdefault(node.targets[0].id, node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            bindings.setdefault(node.target.id, node.value)
    return bindings


def _resolve_expr(node: ast.AST, bindings: dict[str, ast.AST], here: Path, depth: int = 0):
    """Resolve a path expression to an absolute Path, or None when unresolvable."""
    if depth > 25:
        return None
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            # A drive path ("C:/...") is absolute on every OS; on POSIX, Path() would read it as
            # relative and the finding would differ between Windows and Linux gates.
            if PureWindowsPath(node.value).drive:
                return PureWindowsPath(node.value)
            # Likewise a POSIX root path ("/tmp/x") is absolute on Windows too (UNC "//" excepted,
            # which both OSes already read the same way).
            if node.value.startswith("/") and not node.value.startswith("//"):
                return PurePosixPath(node.value)
            cand = Path(node.value)
            return cand if cand.is_absolute() else None
        return None
    if isinstance(node, ast.Name):
        if node.id == "__file__":
            return here
        if node.id in bindings:
            return _resolve_expr(bindings[node.id], bindings, here, depth + 1)
        return None
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _resolve_expr(node.value, bindings, here, depth + 1)
        return base.parent if base is not None else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = _resolve_expr(node.left, bindings, here, depth + 1)
        right = node.right
        if left is None or not (isinstance(right, ast.Constant) and isinstance(right.value, str)):
            return None
        return left / right.value
    if isinstance(node, ast.Subscript):
        inner = node.value
        if (
            isinstance(inner, ast.Attribute)
            and inner.attr == "parents"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            base = _resolve_expr(inner.value, bindings, here, depth + 1)
            if base is None:
                return None
            try:
                return base.parents[node.slice.value]
            except IndexError:
                return None
        return None
    if isinstance(node, ast.Call):
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id in ("str", "Path")
            and len(node.args) == 1
            and not node.keywords
        ):
            return _resolve_expr(node.args[0], bindings, here, depth + 1)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "resolve"
            and not node.args
            and not node.keywords
        ):
            return _resolve_expr(node.func.value, bindings, here, depth + 1)
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "with_name"
            and len(node.args) == 1
            and not node.keywords
        ):
            base = _resolve_expr(node.func.value, bindings, here, depth + 1)
            arg = node.args[0]
            if base is None or not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                return None
            return base.parent / arg.value
        if (
            isinstance(func, ast.Attribute)
            and func.attr in ("join", "dirname", "abspath")
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "path"
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "os"
            and not node.keywords
        ):
            parts = []
            for arg in node.args:
                resolved = _resolve_expr(arg, bindings, here, depth + 1)
                if resolved is None:
                    return None
                parts.append(str(resolved))
            if func.attr == "join":
                return Path(os.path.join(*parts)) if parts else None
            if len(parts) != 1:
                return None
            if func.attr == "dirname":
                return Path(os.path.dirname(parts[0]))
            return Path(os.path.abspath(parts[0]))
        return None
    return None


#: Read-only git subcommands a replication may call to stamp its provenance.
GIT_READ_SUBCOMMANDS = ("rev-parse", "status", "log", "show", "describe", "ls-files")
#: Element prefixes that let git run a command or load config (attached ``-cX=Y`` included).
GIT_BANNED_PREFIXES = ("-c", "--config", "--exec-path", "--upload-pack", "--receive-pack",
                       "--ext-diff", "--textconv", "-u")
GIT_BANNED_KEYWORDS = ("shell", "executable", "env", "preexec_fn")


def _is_git_literal_call(node: ast.Call) -> bool:
    """A ``subprocess`` call that cannot run authors' code.

    The argv is a list literal starting with ``"git"``, optionally one
    ``-C <dir>`` (the only element allowed to be computed), then a read-only
    subcommand from ``GIT_READ_SUBCOMMANDS`` and string constants, none
    starting with a ``GIT_BANNED_PREFIXES`` entry; no ``shell=``,
    ``executable=``, ``env=``, ``preexec_fn=`` or ``**`` keyword. Anything else
    (a string, a variable, another program, a writing or fetching subcommand,
    a second computed element) stays a finding.
    """
    if not node.args:
        return False
    argv = node.args[0]
    if not isinstance(argv, (ast.List, ast.Tuple)) or not argv.elts:
        return False
    elts = list(argv.elts)
    if not (isinstance(elts[0], ast.Constant) and elts[0].value == "git"):
        return False
    rest = elts[1:]
    if rest and isinstance(rest[0], ast.Constant) and rest[0].value == "-C":
        # ``git -C <dir>`` with a computed directory (the repo root) only picks where git runs.
        if len(rest) < 2 or isinstance(rest[1], ast.Starred):
            return False
        rest = rest[2:]
    if not rest:
        return False
    for elt in rest:
        if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
            return False
        if elt.value.startswith(GIT_BANNED_PREFIXES):
            return False
    if rest[0].value not in GIT_READ_SUBCOMMANDS:
        return False
    for kw in node.keywords:
        if kw.arg is None or kw.arg in GIT_BANNED_KEYWORDS:
            return False
    return True


def _collect_imports(tree: ast.Module) -> tuple[dict[str, str], dict[str, str]]:
    """(module aliases, name aliases) for one module.

    ``mods`` maps the bound name to the full module (``import os as o`` gives
    ``o -> os``); ``names`` maps it to the dotted source (``from os import
    system as s`` gives ``s -> os.system``). Function-level imports count:
    a bypass inside a function is still a bypass.
    """
    mods: dict[str, str] = {}
    names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods[a.asname or a.name.split(".")[0]] = a.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                names[a.asname or a.name] = node.module + "." + a.name
    return mods, names


def _call_target(
    func: ast.AST, mods: dict[str, str], names: dict[str, str]
) -> tuple[str | None, str | None]:
    """(top module, attribute) of a call's function, resolving aliases.

    ``import subprocess as sp; sp.run`` gives ``("subprocess", "run")``;
    ``from os import system as s; s(...)`` gives ``("os", "system")``; bare
    ``__import__``/``exec``/``eval`` give ``("builtin", name)``. Dotted chains
    such as ``importlib.util.spec_from_file_location`` resolve on the root
    (``("importlib", "spec_from_file_location")``), so local file loading is
    not mistaken for ``import_module``.
    """
    if isinstance(func, ast.Attribute):
        node = func.value
        while isinstance(node, ast.Attribute):
            node = node.value
        if not isinstance(node, ast.Name):
            return (None, None)
        full = mods.get(node.id, node.id)
        top = full.split(".")[0]
        if top == "builtins":
            top = "builtin"
        return (top, func.attr)
    if isinstance(func, ast.Name):
        if func.id in ("__import__", "exec", "eval"):
            return ("builtin", func.id)
        if func.id in names:
            src = names[func.id]
            mod, _, attr = src.rpartition(".")
            top = mod.split(".")[0] if mod else None
            if top == "builtins":
                top = "builtin"
            return (top, attr)
        return (None, None)
    return (None, None)


def _os_import_dangerous(name: str) -> bool:
    """``os`` names that execute outside code when imported or called."""
    return name in ("system", "popen", "startfile", "fork", "forkpty") or name.startswith(
        ("exec", "spawn", "posix_spawn")
    )


def _os_dangerous(attr: str) -> bool:
    """``os`` attributes that execute outside code."""
    return _os_import_dangerous(attr)


def _subprocess_use(
    tree: ast.Module, mods: dict[str, str], names: dict[str, str]
) -> tuple[bool, bool]:
    """(any ``subprocess`` call, any non-exempt ``subprocess`` call), alias-aware."""
    exempt = []
    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        mod, _ = _call_target(node.func, mods, names)
        if mod == "subprocess":
            found = True
            exempt.append(_is_git_literal_call(node))
    return (found, any(not e for e in exempt))


def _within_root(resolved: Path, root: Path) -> bool:
    if not resolved.is_absolute():
        return False
    norm = Path(os.path.normcase(os.path.normpath(str(resolved))))
    base = Path(os.path.normcase(os.path.normpath(str(root))))
    return norm == base or base in norm.parents


def _sys_path_target(target: ast.AST, mods: dict[str, str], names: dict[str, str]) -> bool:
    """Whether an assignment target (or call receiver) is ``sys.path``.

    ``import sys as s`` resolves through ``mods`` (``s.path`` counts);
    ``from sys import path`` (optionally aliased) resolves through ``names``.
    """
    if (
        isinstance(target, ast.Attribute)
        and target.attr == "path"
        and isinstance(target.value, ast.Name)
    ):
        return mods.get(target.value.id, target.value.id).split(".")[0] == "sys"
    if isinstance(target, ast.Name):
        return names.get(target.id) == "sys.path"
    return False


def _pty_spawn_called(tree: ast.Module, mods: dict[str, str], names: dict[str, str]) -> bool:
    """Whether the module calls ``pty.spawn`` (alias-aware)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        mod, attr = _call_target(node.func, mods, names)
        if mod == "pty" and attr == "spawn":
            return True
    return False


def _sys_path_args(node: ast.Call) -> list[ast.AST]:
    """Path expressions added by one sys.path.insert/append/extend call."""
    if node.func.attr == "insert":
        return list(node.args[1:2])
    if node.func.attr == "append":
        return list(node.args[0:1])
    arg = node.args[0:1]
    if arg and isinstance(arg[0], (ast.List, ast.Tuple)):
        return list(arg[0].elts)
    return arg


def _report_added(
    label: str,
    lineno: int,
    exprs: list[ast.AST],
    bindings: dict[str, ast.AST],
    here: Path,
    root: Path,
    issues: list[str],
) -> None:
    """Flag ``sys.path`` additions that are unresolvable or leave the repo."""
    if not exprs:
        issues.append(f"line {lineno}: {label} with no path argument")
        return
    for arg in exprs:
        resolved = _resolve_expr(arg, bindings, here)
        if resolved is None:
            issues.append(f"line {lineno}: {label} adds a path that cannot be resolved")
        elif not _within_root(resolved, root):
            issues.append(
                f"line {lineno}: {label} adds a path outside the repository "
                f"({resolved.as_posix()})"
            )


def check_s1_file(tree: ast.Module, rel: str, here: Path, root: Path) -> list[str]:
    """S1 issue descriptions for one parsed file (empty when clean)."""
    issues: list[str] = []
    bindings = _collect_bindings(tree)
    mods, names = _collect_imports(tree)
    any_sp, live_sp = _subprocess_use(tree, mods, names)
    pty_live = _pty_spawn_called(tree, mods, names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            tops = {a.name.split(".")[0] for a in node.names}
            if "subprocess" in tops and (live_sp or not any_sp):
                issues.append(f"line {node.lineno}: imports subprocess")
            if "runpy" in tops:
                issues.append(f"line {node.lineno}: imports runpy")
            if "pty" in tops and pty_live:
                issues.append(f"line {node.lineno}: imports pty")
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            top = node.module.split(".")[0]
            imported = {a.name for a in node.names}
            if top == "subprocess" and (live_sp or not any_sp):
                issues.append(f"line {node.lineno}: imports subprocess")
            elif top == "runpy":
                issues.append(f"line {node.lineno}: imports runpy")
            elif top == "os" and any(_os_import_dangerous(n) for n in imported):
                issues.append(f"line {node.lineno}: imports os process execution")
            elif top == "pty" and "spawn" in imported:
                issues.append(f"line {node.lineno}: imports pty process execution")
            elif top == "importlib" and "import_module" in imported:
                issues.append(f"line {node.lineno}: imports importlib.import_module")
        elif isinstance(node, ast.Call):
            mod, attr = _call_target(node.func, mods, names)
            if mod == "subprocess":
                if not _is_git_literal_call(node):
                    issues.append(f"line {node.lineno}: calls subprocess.{attr}")
            elif mod == "os" and attr is not None and _os_dangerous(attr):
                issues.append(f"line {node.lineno}: calls os.{attr}")
            elif mod == "pty" and attr == "spawn":
                issues.append(f"line {node.lineno}: calls pty.spawn")
            elif mod == "runpy":
                issues.append(f"line {node.lineno}: calls runpy.{attr}")
            elif mod == "importlib" and attr == "import_module":
                issues.append(f"line {node.lineno}: calls importlib.import_module")
            elif mod == "importlib" and attr == "spec_from_file_location":
                loc = None
                if len(node.args) >= 2:
                    loc = node.args[1]
                else:
                    for kw in node.keywords:
                        if kw.arg == "location":
                            loc = kw.value
                            break
                if loc is None:
                    issues.append(
                        f"line {node.lineno}: spec_from_file_location without a path argument"
                    )
                else:
                    resolved = _resolve_expr(loc, bindings, here)
                    if resolved is None:
                        issues.append(
                            f"line {node.lineno}: spec_from_file_location loads a path "
                            "that cannot be resolved"
                        )
                    elif not _within_root(resolved, root):
                        issues.append(
                            f"line {node.lineno}: spec_from_file_location loads a path "
                            f"outside the repository ({resolved.as_posix()})"
                        )
            elif mod == "builtin" and attr == "__import__":
                issues.append(f"line {node.lineno}: calls __import__")
            elif mod == "builtin" and attr == "exec":
                issues.append(f"line {node.lineno}: calls exec(")
            elif mod == "builtin" and attr == "eval":
                issues.append(f"line {node.lineno}: calls eval(")
            elif (
                mod is None
                and isinstance(node.func, ast.Name)
                and (node.func.id == "getattr" or names.get(node.func.id) == "builtins.getattr")
                and node.args
                and isinstance(node.args[0], ast.Name)
            ):
                target = mods.get(node.args[0].id, node.args[0].id).split(".")[0]
                if target in ("os", "subprocess"):
                    issues.append(f"line {node.lineno}: calls getattr({target}, ...)")
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in SYS_PATH_METHODS
            and _sys_path_target(node.func.value, mods, names)
        ):
            _report_added(
                f"sys.path.{node.func.attr}",
                node.lineno,
                _sys_path_args(node),
                bindings,
                here,
                root,
                issues,
            )
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.op, ast.Add)
            and _sys_path_target(node.target, mods, names)
        ):
            value = node.value
            exprs = list(value.elts) if isinstance(value, (ast.List, ast.Tuple)) else [value]
            _report_added("sys.path +=", node.lineno, exprs, bindings, here, root, issues)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and _sys_path_target(
                    target.value, mods, names
                ):
                    value = node.value
                    exprs = (
                        list(value.elts)
                        if isinstance(value, (ast.List, ast.Tuple))
                        else [value]
                    )
                    _report_added(
                        "sys.path[...] =",
                        node.lineno,
                        exprs,
                        bindings,
                        here,
                        root,
                        issues,
                    )
    return issues


#: Stage-artifact label fragments that mark a transcription of the authors' model.
ORACLE_LABEL_KEYWORDS = ("oracle", "transcription", "reference arm")


def _is_oracle_label(label: str) -> bool:
    """Whether a stage_artifacts label names an oracle (case-insensitive)."""
    low = label.lower()
    return any(key in low for key in ORACLE_LABEL_KEYWORDS)


def _label_oracle_rel(path: str) -> str | None:
    """Repo-relative oracle path for a stage_artifacts entry, or None.

    The ``<name>`` segment maps onto ``artifacts/etudes/papers/<name>/`` the
    way R1 maps it (the folders moved there), so a stale pinned path still
    maps. Only ``.py`` entries count.
    """
    if Path(path).suffix != ".py":
        return None
    parts = Path(path).parts
    if len(parts) < 3 or parts[0] != "artifacts" or parts[1] != "etudes":
        return None
    rest = parts[2:]
    if rest[0] == "papers":
        rest = rest[1:]
    if len(rest) < 2 or not rest[0]:
        return None
    return PAPERS_REL.joinpath(*rest).as_posix()


def _is_oracle_name(basename: str) -> bool:
    """Whether a file basename marks an oracle transcription."""
    return (
        fnmatch.fnmatchcase(basename, "*_ref.py")
        or fnmatch.fnmatchcase(basename, "*_ref_*.py")
        or fnmatch.fnmatchcase(basename, "*reference*.py")
    )


def oracle_files(papers: list[dict], root: Path) -> dict[str, Path]:
    """(repo-relative posix, absolute path) oracle modules, per paper folder.

    Labelled oracles resolve through the R1 folder mapping; a label naming a
    ``.py`` file that does not exist is not an oracle (R1 and the dashboard
    build own existence). Name oracles match ``*_ref.py``, ``*_ref_*.py`` or
    ``*reference*.py`` under the paper folders.
    """
    oracles: dict[str, Path] = {}
    for paper in papers:
        stages = paper.get("stage_artifacts")
        if isinstance(stages, dict):
            for items in stages.values():
                if isinstance(items, list):
                    for item in items:
                        if (
                            isinstance(item, dict)
                            and isinstance(item.get("label"), str)
                            and isinstance(item.get("path"), str)
                            and _is_oracle_label(item["label"])
                        ):
                            rel = _label_oracle_rel(item["path"])
                            if rel is not None and (root / rel).is_file():
                                oracles.setdefault(rel, root / rel)
    for path in _paper_py_files(root):
        if _is_oracle_name(path.name):
            oracles.setdefault(_rel(path, root), path)
    return oracles


def _norm(path: Path | PurePosixPath | PureWindowsPath) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def _s2_paper(rel: str) -> str | None:
    """The paper folder name for a repo-relative path under papers/, or None."""
    parts = PurePosixPath(rel).parts
    if (
        len(parts) > 3
        and parts[0] == "artifacts"
        and parts[1] == "etudes"
        and parts[2] == "papers"
    ):
        return parts[3]
    return None


def _s2_module_candidates(name: str, base: Path) -> list[Path]:
    """Candidate files for a dotted module name anchored at one directory."""
    stem = Path(*name.split("."))
    return [base / (str(stem) + ".py"), base / stem / "__init__.py"]


def _resolve_s2_module(
    name: str, here: Path, paper_root: Path, level: int
) -> Path | None:
    """The file a module name resolves to, or None when it resolves nowhere.

    Absolute names resolve beside the importer, then at the paper root (the
    paper folders run with their own directory on ``sys.path``); relative
    names resolve beside the importer. Only existing files count.
    """
    if level:
        base = here.parent
        for _ in range(level - 1):
            base = base.parent
        candidates = _s2_module_candidates(name, base)
    else:
        candidates = _s2_module_candidates(name, here.parent) + _s2_module_candidates(
            name, paper_root
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _s2_module_refs(tree: ast.Module) -> list[tuple[str, int]]:
    """(module name, relative level) for every import statement, any scope."""
    refs: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                refs.append((a.name, 0))
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                refs.append((node.module, node.level))
            else:
                for a in node.names:
                    if a.name != "*":
                        refs.append((a.name, node.level))
    return refs


def check_s2(
    parsed: list[tuple[str, Path, ast.Module | None]], papers: list[dict], root: Path
) -> list[tuple[str, str, str]]:
    """S2 findings: a non-test, non-gate file importing an oracle module."""
    oracles = oracle_files(papers, root)
    by_abs = {_norm(path): rel for rel, path in oracles.items()}
    by_paper: dict[str | None, set[str]] = {}
    for rel, path in oracles.items():
        by_paper.setdefault(_s2_paper(rel), set()).add(_norm(path))
    findings: set[tuple[str, str, str]] = set()
    for rel, here, tree in parsed:
        if tree is None:
            continue
        paper = _s2_paper(rel)
        if paper is None:
            continue
        if here.name.startswith(("test_", "gate_")):
            continue
        if "tests" in PurePosixPath(rel).parts:
            continue
        if _norm(here) in by_paper.get(paper, set()):
            continue  # a sibling oracle of the same paper
        paper_root = root / PAPERS_REL / paper
        bindings = _collect_bindings(tree)
        mods, names = _collect_imports(tree)
        hits: set[str] = set()
        for name, level in _s2_module_refs(tree):
            target = _resolve_s2_module(name, here, paper_root, level)
            if target is not None and _norm(target) in by_abs:
                hits.add(by_abs[_norm(target)])
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            mod, attr = _call_target(node.func, mods, names)
            if mod == "importlib" and attr == "import_module":
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    target = _resolve_s2_module(node.args[0].value, here, paper_root, 0)
                    if target is not None and _norm(target) in by_abs:
                        hits.add(by_abs[_norm(target)])
            elif mod == "importlib" and attr == "spec_from_file_location":
                loc = None
                if len(node.args) >= 2:
                    loc = node.args[1]
                else:
                    for kw in node.keywords:
                        if kw.arg == "location":
                            loc = kw.value
                            break
                if loc is not None:
                    resolved = _resolve_expr(loc, bindings, here)
                    if resolved is not None and _norm(resolved) in by_abs:
                        hits.add(by_abs[_norm(resolved)])
        for oracle_rel in hits:
            findings.add(("S2", rel, "imports oracle " + oracle_rel))
    return sorted(findings)


def _is_test_label(label: str) -> bool:
    """Whether a stage_artifacts label names a test module."""
    return "test" in label.lower()


def e1_scored_modules(paper: dict, root: Path) -> list[Path]:
    """The paper's scored P4 modules: existing .py files, oracles and tests excluded.

    P4 ``stage_artifacts`` entries map onto ``artifacts/etudes/papers/`` the
    way S2 oracles map. Entries whose label marks an oracle or a test,
    ``test_*`` basenames and oracle names do not count.
    """
    modules = []
    stages = paper.get("stage_artifacts")
    if not isinstance(stages, dict):
        return []
    items = stages.get("P4")
    if not isinstance(items, list):
        return []
    for item in items:
        if not (isinstance(item, dict) and isinstance(item.get("path"), str)):
            continue
        label = item.get("label", "")
        if isinstance(label, str) and (_is_oracle_label(label) or _is_test_label(label)):
            continue
        rel = _label_oracle_rel(item["path"])
        if rel is None:
            continue
        if Path(rel).name.startswith("test_") or _is_oracle_name(Path(rel).name):
            continue
        candidate = root / rel
        if candidate.is_file():
            modules.append(candidate)
    return modules


def _e1_calls(tree: ast.Module) -> tuple[bool, bool]:
    """(construct call, .simulate call) anywhere in one module, any scope."""
    saw_construct = saw_simulate = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "construct":
            saw_construct = True
        elif isinstance(func, ast.Attribute) and func.attr == "construct":
            saw_construct = True
        elif isinstance(func, ast.Attribute) and func.attr == "simulate":
            saw_simulate = True
    return saw_construct, saw_simulate


def check_e1(papers: list[dict], root: Path) -> list[tuple[str, str, str]]:
    """E1 findings: the scored P4 modules must call construct( and .simulate(."""
    findings = []
    for paper in papers:
        if paper.get("class") == "core method":
            continue
        stage = paper.get("stage")
        if stage not in STAGE_ORDER or STAGE_ORDER.index(stage) < STAGE_ORDER.index("P4"):
            continue
        folder_rel = paper_folder(paper, root)
        if folder_rel is None:
            continue  # R1 owns folders no stage_artifacts path names
        modules = e1_scored_modules(paper, root)
        if not modules:
            findings.append(("E1", folder_rel, "no scored P4 module listed"))
            continue
        saw_construct = saw_simulate = False
        for module in modules:
            try:
                tree = ast.parse(module.read_bytes().decode("utf-8"))
            except (OSError, ValueError, SyntaxError):
                continue
            has_construct, has_simulate = _e1_calls(tree)
            saw_construct = saw_construct or has_construct
            saw_simulate = saw_simulate or has_simulate
        if saw_construct and saw_simulate:
            continue
        if saw_construct:
            findings.append(
                ("E1", folder_rel, "scored P4 modules call construct( but not .simulate(")
            )
        elif saw_simulate:
            findings.append(
                ("E1", folder_rel, "scored P4 modules call .simulate( but not construct(")
            )
        else:
            findings.append(
                ("E1", folder_rel, "no scored P4 module calls construct( or .simulate(")
            )
    return findings


#: P4 (manifest) required key classes, in finding order. Each class is
#: satisfied by one exact (case-sensitive) key from its set.
P4_KEY_CLASSES = (
    ("code SHA", ("git_sha", "code_sha", "sha", "commit", "baseline_sha")),
    ("backend", ("backend",)),
    ("dtype", ("dtype",)),
    ("seeds", ("seed", "seeds")),
    ("regime", ("regime", "activity_regime")),
)

#: Attribute calls a manifest writer may be passed to (``json.dump``,
#: ``json.dumps``, ``Path.write_text`` and the like).
P4_SINK_CALLS = ("dump", "dumps", "write_text")


def _p4_dict_keys(node: ast.AST) -> set[str] | None:
    """String keys of a manifest-writer-shaped node, else None.

    A dict display contributes its constant string keys; a ``dict(...)``
    call contributes its keyword names. Anything else (a load, a merge, a
    name) is not a statically readable writer.
    """
    if isinstance(node, ast.Dict):
        return {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
    ):
        return {kw.arg for kw in node.keywords if kw.arg is not None}
    return None


def _p4_literal_bindings(tree: ast.Module) -> dict[str, ast.AST]:
    """Name to its first bound dict-display / ``dict(...)`` value."""
    bindings: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            targets = node.targets
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        ):
            targets = [node.target]
            value = node.value
        else:
            continue
        if (
            len(targets) == 1
            and isinstance(targets[0], ast.Name)
            and _p4_dict_keys(value) is not None
        ):
            bindings.setdefault(targets[0].id, value)
    return bindings


def _p4_writers(tree: ast.Module) -> list[tuple[int, int, set[str]]]:
    """(lineno, column, keys) manifest writers of one module, in file order.

    A dict display or ``dict(...)`` call counts when it is assigned to a
    name containing ``manifest``, or passed to ``json.dump`` /
    ``json.dumps`` / ``write_text`` (a passed name resolves to its first
    bound literal; subscript stores and loads never count).
    """
    found: dict[tuple[int, int], set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Name) and "manifest" in target.id
                for target in targets
            ):
                keys = _p4_dict_keys(node.value)
                if keys is not None:
                    found.setdefault(
                        (node.lineno, node.col_offset), keys
                    )
    bindings = _p4_literal_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        args: list[ast.AST] = list(node.args) + [kw.value for kw in node.keywords]
        name = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else "")
        if "manifest" in name.lower():
            # A manifest helper (write_manifest(out, record, static)) merges the dicts it is
            # given into one manifest: their keys count together as one writer.
            merged: set[str] = set()
            for arg in args:
                if isinstance(arg, ast.Name) and arg.id in bindings:
                    arg = bindings[arg.id]
                keys = _p4_dict_keys(arg)
                if keys is not None:
                    merged |= keys
            if merged:
                found.setdefault((node.lineno, node.col_offset), merged)
            continue
        if not (isinstance(func, ast.Attribute) and func.attr in P4_SINK_CALLS):
            continue
        for arg in args:
            if isinstance(arg, ast.Starred):
                arg = arg.value
            if isinstance(arg, ast.Name) and arg.id in bindings:
                arg = bindings[arg.id]
            keys = _p4_dict_keys(arg)
            if keys is not None:
                found.setdefault((arg.lineno, arg.col_offset), keys)
    return sorted((line, col, keys) for (line, col), keys in found.items())


def check_p4(papers: list[dict], root: Path) -> list[tuple[str, str, str]]:
    """P4 (manifest) findings: one scored-modules manifest covering every key class."""
    findings = []
    for paper in papers:
        if paper.get("class") == "core method":
            continue
        stage = paper.get("stage")
        if stage not in STAGE_ORDER or STAGE_ORDER.index(stage) < STAGE_ORDER.index("P4"):
            continue
        folder_rel = paper_folder(paper, root)
        if folder_rel is None:
            continue  # R1 owns folders no stage_artifacts path names
        modules = e1_scored_modules(paper, root)
        if not modules:
            continue  # no scored module is E1's finding, not P4 (manifest)'s
        best: tuple[set[str], int, int] | None = None
        for module in modules:
            try:
                tree = ast.parse(module.read_bytes().decode("utf-8"))
            except (OSError, ValueError, SyntaxError):
                continue
            for lineno, col, keys in _p4_writers(tree):
                covered = {
                    cls for cls, members in P4_KEY_CLASSES if keys & set(members)
                }
                if best is None or len(covered) > len(best[0]):
                    best = (covered, lineno, col)
        if best is None:
            findings.append(
                ("P4", folder_rel, "no manifest writer in the scored P4 modules")
            )
            continue
        covered, _, _ = best
        if len(covered) == len(P4_KEY_CLASSES):
            continue
        missing = [cls for cls, _ in P4_KEY_CLASSES if cls not in covered]
        findings.append(
            ("P4", folder_rel, "manifest writer lacks " + ", ".join(missing))
        )
    return findings


def check_r4(papers: list[dict], root: Path) -> list[tuple[str, str, str]]:
    """R4 findings: a paper claiming replication must hold its P6 result file."""
    findings = []
    for paper in papers:
        if paper.get("verdict") != "reproduced":
            continue  # only a full-replication claim counts, not a stage-level one
        pid = paper.get("id", "?")
        stage = paper.get("stage")
        if stage in STAGE_ORDER and STAGE_ORDER.index(stage) < STAGE_ORDER.index("P6"):
            continue  # a paper below P6 is not a finding
        stages = paper.get("stage_artifacts")
        items = stages.get("P6") if isinstance(stages, dict) else None
        paths = (
            [
                item["path"]
                for item in items
                if isinstance(item, dict) and isinstance(item.get("path"), str)
            ]
            if isinstance(items, list)
            else []
        )
        if any((root / path).is_file() for path in paths):
            continue
        folder_rel = paper_folder(paper, root)
        findings.append(
            (
                "R4",
                folder_rel if folder_rel is not None else SOURCES_REL.as_posix(),
                f"paper {pid!r} claims reproduced without a P6 result file",
            )
        )
    return findings


def _all_names(tree: ast.Module) -> set[str]:
    """Non-private names of one ``__all__`` list/tuple assignment."""
    names: set[str] = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "__all__"
            and isinstance(node.value, (ast.List, ast.Tuple))
        ):
            for elt in node.value.elts:
                if (
                    isinstance(elt, ast.Constant)
                    and isinstance(elt.value, str)
                    and not elt.value.startswith("_")
                ):
                    names.add(elt.value)
    return names


def analysis_names(root: Path) -> set[str]:
    """Public names of jaxfne.analysis, parsed with ast (never imported).

    Every ``*.py`` under each ``jaxfne/analysis*`` package is read
    recursively, so nested modules count; top-level functions plus every
    ``__all__`` entry count (re-exports included).
    """
    names: set[str] = set()
    for candidate in sorted((root / "jaxfne").glob("analysis*")):
        if candidate.is_file() and candidate.suffix == ".py":
            modules = [candidate]
        elif candidate.is_dir():
            modules = sorted(
                p
                for p in candidate.rglob("*.py")
                if "__pycache__" not in p.parts
            )
        else:
            continue
        for module in modules:
            try:
                tree = ast.parse(module.read_bytes().decode("utf-8"))
            except (OSError, ValueError, SyntaxError):
                continue
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        names.add(node.name)
            names |= _all_names(tree)
    return names


def check_trees(
    parsed: list[tuple[str, Path, ast.Module | None]], root: Path
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """S1 and E3 findings over parsed paper files (None tree means unparseable)."""
    s1, e3 = [], []
    wanted = analysis_names(root)
    for rel, here, tree in parsed:
        if tree is None:
            s1.append(( "S1", rel, "does not parse; banned calls cannot be excluded"))
            e3.append(( "E3", rel, "does not parse; shadowed definitions cannot be excluded"))
            continue
        issues = check_s1_file(tree, rel, here, root)
        if issues:
            s1.append(("S1", rel, "; ".join(sorted(set(issues)))))
        shadowed = sorted(
            {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in wanted
            }
        )
        if shadowed:
            e3.append(
                (
                    "E3",
                    rel,
                    "defines "
                    + ", ".join(f"{name}()" for name in shadowed)
                    + " shadowing jaxfne.analysis",
                )
            )
    return s1, e3


def parse_paper_files(root: Path) -> list[tuple[str, Path, ast.Module | None]]:
    """(rel, file path, parsed tree or None) for every .py under paper folders."""
    parsed = []
    for path in _paper_py_files(root):
        try:
            tree = ast.parse(path.read_bytes().decode("utf-8"))
        except (OSError, ValueError, SyntaxError):
            tree = None
        parsed.append((_rel(path, root), path, tree))
    return parsed


def collect_findings(root: Path) -> list[tuple[str, str, str]]:
    """Every (rule, path, message) finding for one tree, sorted and deduplicated."""
    papers = load_registry(root)
    parsed = parse_paper_files(root)
    s1, e3 = check_trees(parsed, root)
    findings = (
        check_r1(papers, root)
        + check_r6(root)
        + s1
        + check_s2(parsed, papers, root)
        + check_e1(papers, root)
        + check_p4(papers, root)
        + check_r4(papers, root)
        + e3
    )
    return sorted(set(findings))


def load_known(root: Path, known_path: Path | None = None) -> list[dict]:
    """Known-violation entries (absent file means none)."""
    path = known_path if known_path is not None else (
        root / "scripts" / "replication_known_violations.json"
    )
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        entries = json.loads(raw)
    except ValueError as exc:
        raise SystemExit(f"error: cannot parse {path}: {exc}")
    if not isinstance(entries, list):
        raise SystemExit(f"error: {path} holds no list")
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("rule"), str)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("message"), str)
        ):
            raise SystemExit(f"error: {path} holds an entry without rule/path/message strings")
    return entries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root if args.root is not None else ROOT

    findings = collect_findings(root)
    known_path = root / "scripts" / "replication_known_violations.json"
    if args.check and not known_path.is_file():
        print(f"error: allowlist is missing: {known_path.as_posix()}")
        print("replication-audit: findings=%d known=0 unlisted=%d stale=0" % (len(findings), len(findings)))
        return 1
    known = load_known(root)
    # One entry covers one exact finding, so a new finding on a listed path is unlisted.
    known_keys = {(e["rule"], e["path"], e["message"]) for e in known}
    finding_keys = set(findings)

    for rule, path, message in findings:
        print(f"{rule} {path}: {message}")
    for rule, path, message in sorted(known_keys - finding_keys):
        print(f"{rule} {path}: known violation is stale (listed but not occurring): {message}")

    unlisted = len(finding_keys - known_keys)
    stale = len(known_keys - finding_keys)
    print(
        f"replication-audit: findings={len(findings)} known={len(known)} "
        f"unlisted={unlisted} stale={stale}"
    )
    if args.check and (unlisted or stale):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
