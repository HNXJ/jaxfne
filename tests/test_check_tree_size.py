"""Tree-size gate tests: temp git repo + checker function."""

from __future__ import annotations

import subprocess
from pathlib import Path

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
