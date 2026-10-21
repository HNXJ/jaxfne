"""The docs-page audit reads pages as UTF-8 whatever the locale codec (O-51).

Under a cp1252 locale (Windows without UTF-8 mode) the audit crashed on the first page holding
a byte cp1252 cannot decode.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_audit_docs_pages_reads_utf8_under_legacy_locale():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["PYTHONUTF8"] = "0"
    run = subprocess.run(
        [sys.executable, "scripts/harness/audit_81_docs_pages.py"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    assert "UnicodeDecodeError" not in run.stderr, run.stderr[-500:]
    assert "AUDIT RESULT: PASS" in run.stdout, run.stdout[-500:]
