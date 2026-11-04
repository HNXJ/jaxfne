#!/usr/bin/env python3
"""Fidelity-fix gate: rule F3 of ``artifacts/etudes/meta/RESIDUAL_PROTOCOL.md``.

A fidelity fix is a commit whose message has one or more trailer lines
``Fidelity-fix: <pytest node id>``; several ids may share a line, comma-separated.
Node ids are relative to the repository root (``tests/test_x.py::test_y``).
Untagged commits are ignored.

For each tagged commit C, oldest first:

- parent: the tree of C^ (``git archive``, extracted with ``tarfile``, no
  worktree), with the ``tests/`` files that C added or changed copied in from C,
  must make the named nodes fail (pytest exit 1 or 2);
- fix: the tree of C must make the same nodes pass (pytest exit 0).

Usage:
    python scripts/check_fidelity_fix.py --range A..B
    python scripts/check_fidelity_fix.py --range A..B --repo PATH --python PATH

Exit codes:
    0  ACCEPT (no findings; zero tagged commits is ACCEPT)
    1  REVISE (one or more findings)
    2  usage error (bad range, not a git repository) or a git failure

The repo root defaults to the nearest ancestor of this script holding
``pyproject.toml``. Only the standard library is used; jaxfne is never imported.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

TRAILER = re.compile(r"^Fidelity-fix:(.*)$")
PARENT_FAIL_EXITS = (1, 2)
NO_TESTS_EXITS = (4, 5)


class UsageError(Exception):
    """Bad arguments or repository state: exit 2."""


class GitFailure(Exception):
    """A git command failed: exit 2."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True)


def _git_out(repo: Path, *args: str) -> bytes:
    proc = _git(repo, *args)
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()
        raise GitFailure(f"git {args[0]} failed: {detail}")
    return proc.stdout


def _default_repo() -> Path | None:
    here = Path(__file__).resolve().parent
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return None


def _clear_readonly(func, path, _exc) -> None:
    os.chmod(path, stat.S_IWRITE)
    func(path)


def _rmtree(path: Path) -> None:
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=_clear_readonly)
    else:
        shutil.rmtree(path, onerror=_clear_readonly)


def _extract_archive(repo: Path, rev: str, dest: Path) -> None:
    data = _git_out(repo, "archive", "--format=tar", rev)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        if hasattr(tarfile, "data_filter"):
            tar.extractall(dest, filter="data")
        else:
            tar.extractall(dest)


def _run_tree(
    repo: Path,
    python: str,
    rev: str,
    nodes: list[str],
    overlay_rev: str | None = None,
    overlay_paths: tuple[str, ...] = (),
) -> int:
    """Extract ``rev`` into a fresh temp dir, copy ``overlay_paths`` from
    ``overlay_rev`` over it, run pytest there and return its exit code."""
    tmp = Path(tempfile.mkdtemp(prefix="fidelity_fix_"))
    try:
        _extract_archive(repo, rev, tmp)
        for path in overlay_paths:
            target = tmp.joinpath(*path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_git_out(repo, "show", f"{overlay_rev}:{path}"))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [python, "-m", "pytest", "-q", "-p", "no:cacheprovider", *nodes],
            cwd=tmp,
            capture_output=True,
            env=env,
        )
        return proc.returncode
    finally:
        _rmtree(tmp)


def _fix_values(repo: Path, sha: str) -> list[str]:
    """Value of each ``Fidelity-fix:`` trailer line in the commit message."""
    message = _git_out(repo, "log", "-1", "--format=%B", sha).decode("utf-8", "replace")
    values = []
    for line in message.splitlines():
        match = TRAILER.match(line.strip())
        if match:
            values.append(match.group(1).strip())
    return values


def _parent(repo: Path, sha: str) -> str | None:
    proc = _git(repo, "rev-parse", "--verify", "-q", f"{sha}^")
    return proc.stdout.decode("utf-8").strip() if proc.returncode == 0 else None


def _changed_tests(repo: Path, parent: str, sha: str) -> tuple[str, ...]:
    out = _git_out(
        repo, "diff", "--name-only", "-z", "--no-renames", "--diff-filter=AM",
        parent, sha, "--", "tests",
    )
    return tuple(p.decode("utf-8") for p in out.split(b"\0") if p)


def _finding(short: str, problem: str) -> str:
    return f"- [major] {short} — {problem}."


def _examine(repo: Path, commits: list[str], python: str) -> tuple[list[str], int]:
    findings: list[str] = []
    tagged = 0
    for sha in commits:
        values = _fix_values(repo, sha)
        if not values:
            continue
        tagged += 1
        short = sha[:8]
        nodes: list[str] = []
        for value in values:
            items = [item.strip() for item in value.split(",") if item.strip()]
            if not items:
                findings.append(_finding(short, "tag names no node id"))
            nodes.extend(items)
        if not nodes:
            continue

        absent = []
        for node in nodes:
            path = node.split("::", 1)[0]
            if not path or _git(repo, "cat-file", "-e", f"{sha}:{path}").returncode != 0:
                absent.append(node)
        for node in absent:
            findings.append(_finding(short, f"node file absent in C: {node}"))
        if absent:
            continue

        parent = _parent(repo, sha)
        if parent is None:
            findings.append(_finding(short, "no parent commit: parent check cannot run"))
            continue

        label = ", ".join(nodes)
        rc_parent = _run_tree(
            repo, python, parent, nodes,
            overlay_rev=sha, overlay_paths=_changed_tests(repo, parent, sha),
        )
        if rc_parent == 0:
            findings.append(_finding(short, f"passes on the parent: not a failing test ({label})"))
        elif rc_parent in NO_TESTS_EXITS:
            findings.append(_finding(short, f"node not found on the parent, pytest exit {rc_parent} ({label})"))
        elif rc_parent not in PARENT_FAIL_EXITS:
            findings.append(_finding(short, f"parent run exit {rc_parent}: expected 1 or 2 ({label})"))

        rc_fix = _run_tree(repo, python, sha, nodes)
        if rc_fix != 0:
            findings.append(_finding(short, f"C run exit {rc_fix}: expected 0 ({label})"))
    return findings, tagged


def _check(args: argparse.Namespace) -> int:
    if args.rng.startswith("-"):
        raise UsageError(f"bad range {args.rng!r}")
    repo = args.repo if args.repo is not None else _default_repo()
    if repo is None:
        raise UsageError("no pyproject.toml above this script; pass --repo")
    repo = repo.resolve()
    if not repo.is_dir():
        raise UsageError(f"not a directory: {repo}")
    if _git(repo, "rev-parse", "--git-dir").returncode != 0:
        raise UsageError(f"not a git repository: {repo}")
    try:
        commits = _git_out(repo, "rev-list", "--reverse", args.rng, "--").decode("utf-8").split()
    except GitFailure as exc:
        raise UsageError(f"bad range {args.rng!r}: {exc}") from exc

    findings, tagged = _examine(repo, commits, args.python)
    print(f"VERDICT: {'REVISE' if findings else 'ACCEPT'}")
    for line in findings:
        print(line)
    print(f"# examined: {tagged} tagged commits of {len(commits)}")
    return 1 if findings else 0


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--range", dest="rng", required=True, help="git rev range, e.g. A..B")
    parser.add_argument(
        "--repo", type=Path, default=None,
        help="repository root (default: nearest pyproject.toml above this script)",
    )
    parser.add_argument(
        "--python", default=sys.executable,
        help="interpreter that runs pytest (default: this one)",
    )
    args = parser.parse_args(argv)
    _force_utf8()
    try:
        return _check(args)
    except (UsageError, GitFailure) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
