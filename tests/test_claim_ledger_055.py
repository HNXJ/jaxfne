"""Tests for scripts/check_claim_ledger_055.py (manuscript item 11).

One test runs the checker on the repo as committed; one test mutates a
temporary copy of the ledger (never the repo file) and asserts the checker
fails on the real mismatch. No network, no jaxfne import.
"""

from __future__ import annotations

import pathlib
import shutil
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import check_claim_ledger_055  # noqa: E402


def test_checker_passes_on_repo():
    assert check_claim_ledger_055.main([]) == 0


def test_checker_fails_on_altered_state_in_temp_copy(tmp_path, capsys):
    ledger = tmp_path / "claim_ledger_055.md"
    coverage = tmp_path / "atlas_coverage.json"
    shutil.copyfile(REPO_ROOT / "artifacts" / "programme" / "claim_ledger_055.md", ledger)
    shutil.copyfile(REPO_ROOT / "artifacts" / "programme" / "atlas_coverage.json", coverage)
    text = ledger.read_text(encoding="utf-8")
    target = (
        "| AT-10-R6 | Reduced fast model with very long T | RESULT | SUPPORTED | "
        "`artifacts/atlas/results/at10_r6_long_055.json` | yes | supported only |"
    )
    altered = target.replace("| RESULT | SUPPORTED |", "| RESULT | VALIDATED |")
    assert target in text and altered != target
    ledger.write_text(text.replace(target, altered), encoding="utf-8", newline="")

    rc = check_claim_ledger_055.main(["--ledger", str(ledger), "--coverage", str(coverage)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "AT-10-R6" in out and "VALIDATED" in out