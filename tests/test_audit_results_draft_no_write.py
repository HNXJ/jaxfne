"""The results audit reports on every run but only rewrites its write-once
receipt with --write (the receipt is a committed record).

Runs the audit as a subprocess from the repo root without --write and asserts
the receipt's bytes are unchanged.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_results_draft.py"
RECEIPT = REPO_ROOT / "artifacts" / "publication" / "results_reconstruction" / "results_audit_receipt.json"


def test_default_run_does_not_rewrite_receipt():
    assert RECEIPT.exists(), "precondition: the committed receipt must exist"
    before = RECEIPT.read_bytes()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    assert proc.returncode == 0, proc.stdout[-500:] + proc.stderr[-500:]
    assert "0 failures" in proc.stdout, proc.stdout[-500:]
    assert RECEIPT.read_bytes() == before, "the audit rewrote the write-once receipt"