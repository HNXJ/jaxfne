"""Tree-size gate tests: temp git repo + checker function."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_tree_size import check_tree

CAP = 1048576


def _make_repo(root: Path, payloads: dict[str, bytes]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(root), check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "t@t"], cwd=str(root), check=True
    )
    subprocess.run(["git", "config", "user.name", "t"], cwd=str(root), check=True)
    for name, data in payloads.items():
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"], cwd=str(root), check=True
    )
    return root


def _allow(**kw) -> dict:
    base: dict = {"budget_bytes": 10**12, "cap_bytes": CAP, "files": {}}
    base.update(kw)
    return base


def test_pass_case(tmp_path):
    repo = _make_repo(tmp_path / "pass", {"a.txt": b"x" * 100, "b.txt": b"y" * 200})
    violations, info = check_tree(repo, _allow())
    assert info["files"] == 2, "fixture must track two files"
    assert info["total"] == 300, "fixture must total 300 bytes"
    assert violations == []


def test_unlisted_big_file_fails(tmp_path):
    big = b"z" * 1_200_000
    repo = _make_repo(tmp_path / "big", {"a.txt": b"x", "big.bin": big})
    from scripts.check_tree_size import tracked_sizes

    sizes = tracked_sizes(repo)
    assert sizes["big.bin"] > CAP, "fixture must exceed the cap"
    violations, _ = check_tree(repo, _allow())
    assert any("over-cap-unlisted" in v and "big.bin" in v for v in violations)


def test_grown_listed_file_fails(tmp_path):
    data = b"w" * 2_000_000
    repo = _make_repo(tmp_path / "grown", {"big.bin": data})
    recorded = int(len(data) / 1.2)
    assert len(data) > recorded * 1.10, "fixture must exceed recorded + 10%"
    violations, _ = check_tree(repo, _allow(files={"big.bin": recorded}))
    assert any("over-cap-grown" in v for v in violations)


def test_total_over_budget_fails(tmp_path):
    repo = _make_repo(tmp_path / "budget", {"a.txt": b"x" * 100})
    from scripts.check_tree_size import tracked_sizes

    total = sum(tracked_sizes(repo).values())
    assert total == 100, "fixture must total 100 bytes"
    violations, _ = check_tree(repo, _allow(budget_bytes=99))
    assert any("over-budget" in v for v in violations)


def test_file_count_over_budget_fails(tmp_path):
    repo = _make_repo(tmp_path / "count", {"a.txt": b"x", "b.txt": b"y", "c.txt": b"z"})
    violations, info = check_tree(repo, _allow(budget_files=2))
    assert info["files"] == 3, "fixture must track three files"
    assert any("over-file-budget: files=3 budget_files=2" in v for v in violations)
    violations, _ = check_tree(repo, _allow(budget_files=3))
    assert violations == []


def test_stale_entry_fails(tmp_path):
    repo = _make_repo(tmp_path / "stale", {"a.txt": b"x" * 10})
    from scripts.check_tree_size import tracked_sizes

    assert "gone.txt" not in tracked_sizes(repo), "fixture must not track gone.txt"
    violations, _ = check_tree(repo, _allow(files={"gone.txt": 123}))
    assert any("stale-allowlist-entry" in v for v in violations)


def test_non_ascii_path_matches_allowlist(tmp_path):
    big = b"u" * 1_200_000
    repo = _make_repo(tmp_path / "utf8", {"déjà.bin": big})
    from scripts.check_tree_size import tracked_sizes

    assert "déjà.bin" in tracked_sizes(repo), "fixture must track the raw UTF-8 path"
    violations, _ = check_tree(repo, _allow(files={"déjà.bin": len(big)}))
    assert violations == []


def _rewrite_batch(monkeypatch, rewrite) -> None:
    """Run real git; replace only the cat-file --batch-check stdout with rewrite(stdout)."""
    real_run = subprocess.run

    def fake_run(args, **kw):
        out = real_run(args, **kw)
        if list(args[:2]) == ["git", "cat-file"]:
            return subprocess.CompletedProcess(args, out.returncode, stdout=rewrite(out.stdout))
        return out

    monkeypatch.setattr(subprocess, "run", fake_run)


def _three_files(root: Path) -> Path:
    return _make_repo(root, {"a.txt": b"x" * 10, "b.txt": b"y" * 20, "c.txt": b"z" * 30})


def test_short_batch_response_fails(tmp_path, monkeypatch):
    from scripts.check_tree_size import tracked_sizes

    repo = _three_files(tmp_path / "short")
    assert len(tracked_sizes(repo)) == 3, "fixture must track three files"
    _rewrite_batch(monkeypatch, lambda out: "".join(out.splitlines(keepends=True)[:2]))
    with pytest.raises(RuntimeError, match="2 response lines for 3 requested blobs"):
        tracked_sizes(repo)


def test_missing_blob_response_fails(tmp_path, monkeypatch):
    from scripts.check_tree_size import tracked_sizes

    repo = _three_files(tmp_path / "missing")
    assert len(tracked_sizes(repo)) == 3, "fixture must track three files"

    def rewrite(out: str) -> str:
        lines = out.splitlines()
        oid = lines[0].split()[0]
        lines[0] = f"{oid} missing"
        return "\n".join(lines) + "\n"

    _rewrite_batch(monkeypatch, rewrite)
    with pytest.raises(RuntimeError, match="missing"):
        tracked_sizes(repo)


def test_malformed_batch_line_fails(tmp_path, monkeypatch):
    from scripts.check_tree_size import tracked_sizes

    repo = _three_files(tmp_path / "malformed")
    assert len(tracked_sizes(repo)) == 3, "fixture must track three files"

    def rewrite(out: str) -> str:
        lines = out.splitlines()
        oid, kind = lines[0].split()[:2]
        lines[0] = f"{oid} {kind}"  # size field cut off
        return "\n".join(lines) + "\n"

    _rewrite_batch(monkeypatch, rewrite)
    with pytest.raises(RuntimeError, match="malformed"):
        tracked_sizes(repo)


def test_reordered_batch_response_fails(tmp_path, monkeypatch):
    from scripts.check_tree_size import tracked_sizes

    repo = _three_files(tmp_path / "reorder")
    assert len(tracked_sizes(repo)) == 3, "fixture must track three files"

    def rewrite(out: str) -> str:
        lines = out.splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        return "\n".join(lines) + "\n"

    _rewrite_batch(monkeypatch, rewrite)
    with pytest.raises(RuntimeError, match="does not match"):
        tracked_sizes(repo)


def test_duplicate_path_fails(tmp_path, monkeypatch):
    from scripts.check_tree_size import tracked_sizes

    repo = _three_files(tmp_path / "dup")
    assert len(tracked_sizes(repo)) == 3, "fixture must track three files"
    real_run = subprocess.run

    def fake_run(args, **kw):
        out = real_run(args, **kw)
        if list(args[:2]) == ["git", "ls-files"]:
            first = out.stdout.split("\0", 1)[0]
            return subprocess.CompletedProcess(args, out.returncode, stdout=out.stdout + first + "\0")
        return out

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="duplicate path"):
        tracked_sizes(repo)

