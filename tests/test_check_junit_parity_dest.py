"""Destination handling for the CI JUnit download in scripts/check_junit_parity.py.

`gh run download` refuses to overwrite, so downloading every SHA into one
shared directory makes a rerun on a new SHA fail with "file exists". The
destination must be keyed by SHA (two SHAs never mix) and cleared for a
same-SHA rerun. No `gh` or network here: `_gh` is stubbed with a fake that
mimics the real refusal.
"""

from __future__ import annotations

from pathlib import Path

import scripts.check_junit_parity as parity

OLD_SHA = "old" * 13 + "00"  # 40 hex chars, like a real SHA
NEW_SHA = "new" * 13 + "11"
FRESH_PAYLOAD = "fresh evidence for this SHA"
STALE_PAYLOAD = "stale evidence from an earlier run"


class _Result:
    """Minimal stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _fake_gh(payload):
    """Mimic `gh`: list answers a run id; download refuses into a non-empty dir."""

    def _gh(*args):
        if args[:2] == ("run", "list"):
            return _Result(0, stdout="12345\n", stderr="")
        assert args[:2] == ("run", "download"), args[:2]
        dest = Path(args[args.index("--dir") + 1])
        existing = [p for p in dest.rglob("*") if p.is_file()] if dest.exists() else []
        if existing:
            return _Result(1, stdout="", stderr=f"file exists: {existing[0].name}")
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "pytest-results-3.11.xml").write_text(payload, encoding="utf-8")
        return _Result(0, stdout="", stderr="")

    return _gh


def test_rerun_on_new_sha_ignores_old_sha_files(tmp_path, monkeypatch):
    """A stale old-SHA file must not fail, nor leak into, a new-SHA run."""
    dest = tmp_path / "ci-junit"
    stale = dest / OLD_SHA / "pytest-results-3.11.xml"
    stale.parent.mkdir(parents=True)
    stale.write_text(STALE_PAYLOAD, encoding="utf-8")

    monkeypatch.setattr(parity, "_gh", _fake_gh(FRESH_PAYLOAD))
    found = parity.fetch_ci_junit(NEW_SHA, dest)

    assert [p.name for p in found] == ["pytest-results-3.11.xml"]
    assert all(OLD_SHA not in str(p) for p in found)
    assert (dest / NEW_SHA / "pytest-results-3.11.xml").read_text(encoding="utf-8") == (
        FRESH_PAYLOAD
    )


def test_rerun_on_same_sha_replaces_stale_files(tmp_path, monkeypatch):
    """A same-SHA rerun clears this SHA's directory before downloading."""
    dest = tmp_path / "ci-junit"
    stale = dest / NEW_SHA / "pytest-results-3.11.xml"
    stale.parent.mkdir(parents=True)
    stale.write_text(STALE_PAYLOAD, encoding="utf-8")

    monkeypatch.setattr(parity, "_gh", _fake_gh(FRESH_PAYLOAD))
    found = parity.fetch_ci_junit(NEW_SHA, dest)

    assert [p.name for p in found] == ["pytest-results-3.11.xml"]
    assert all(NEW_SHA in str(p) for p in found)
    assert found[0].read_text(encoding="utf-8") == FRESH_PAYLOAD
