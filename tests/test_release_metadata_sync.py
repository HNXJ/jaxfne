"""CLI contract for the release freeze and metadata scripts (T-33 tooling ruling).

Both scripts are check-only: neither has a write path, so neither takes a
``--check`` flag; invoking a script performs its check.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCRIPT = REPO_ROOT / "scripts" / "release" / "assert_release_freeze.py"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_release_metadata.py"


def run_script(script: Path, args: list, cwd: Path) -> tuple:
    """Run a script and return (exit_code, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(cwd),
    )
    return result.returncode, result.stdout, result.stderr


@pytest.mark.parametrize("script", [FREEZE_SCRIPT, SYNC_SCRIPT], ids=["freeze", "metadata"])
def test_check_flag_is_rejected(script, tmp_path):
    """There is no write path, so there is no --check flag to accept."""
    code, _, err = run_script(script, ["--check"], tmp_path)
    assert code != 0, "expected argparse to reject the removed --check flag"
    assert "unrecognized arguments: --check" in err


@pytest.mark.parametrize("script", [FREEZE_SCRIPT, SYNC_SCRIPT], ids=["freeze", "metadata"])
def test_help_has_no_check_flag(script):
    """--help must not advertise a flag the parser rejects."""
    code, out, _ = run_script(script, ["--help"], REPO_ROOT)
    assert code == 0
    assert "--check" not in out


def test_metadata_check_passes_on_this_tree():
    """The bare call checks the real version sources and passes when they agree."""
    code, out, _ = run_script(SYNC_SCRIPT, [], REPO_ROOT)
    assert code == 0, out
    assert "All versions consistent" in out
