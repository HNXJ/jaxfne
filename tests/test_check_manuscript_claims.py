"""Tests for scripts/check_manuscript_claims.py.

Each failing case builds a small ledger and manuscript in tmp_path and asserts the
checker reports the defect. The passing case asserts what it examined. One test runs
the checker on the repo as committed and asserts only the shape of its counts. The
checker runs as a subprocess, so no repo import is needed.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_manuscript_claims.py"
COUNT_KEYS = {"cited_not_in_ledger", "uncited_ledger_ids", "status_outside_tier", "orphan_candidates"}

CLEAN_ABSTRACT = "# Abstract\n\nA registered validation found no traveling waves.\n"
CLEAN_RESULTS = "# Results\n\nThe estimator passed its synthetic controls [CL-01].\n"


def row(cid: str, status: str = "SUPPORTED") -> dict:
    return {"claim_id": cid, "status": status, "manuscript_destination": f"Results ({cid})"}


def make_case(tmp_path: pathlib.Path, rows: list[dict], files: dict[str, str]):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"ledger": rows}), encoding="utf-8")
    mdir = tmp_path / "manuscript"
    mdir.mkdir()
    texts = {"abstract.md": CLEAN_ABSTRACT, "results.md": CLEAN_RESULTS}
    texts.update(files)
    for name, text in texts.items():
        (mdir / name).write_text(text, encoding="utf-8")
    return ledger, mdir


def run(ledger: pathlib.Path, mdir: pathlib.Path, *flags: str):
    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--ledger", str(ledger), "--manuscript-dir", str(mdir), "--json", *flags,
        ],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    return proc.returncode, json.loads(proc.stdout)


def test_passing_case_examines_its_inputs(tmp_path):
    rows = [row("CL-01"), row("CL-02", "NEGATIVE")]
    files = {
        "results.md": (
            "# Results\n\n"
            "Fig. 2 shows the canonical source Q with 9 cells [CL-02].\n\n"
            "The toy readout [CL-01] uses 3 seeds [CL-01].\n"
        ),
        "discussion.md": "Both CL-01 and CL-02 are discussed here.\n",
    }
    ledger, mdir = make_case(tmp_path, rows, files)
    rc, rep = run(ledger, mdir, "--strict")
    assert rc == 0 and rep["failed"] == []
    assert rep["counts"] == {key: 0 for key in COUNT_KEYS}
    assert rep["examined"]["ledger_rows"] == 2
    assert rep["examined"]["cited_ids_distinct"] == 2
    # The figure and number paragraphs must have been examined and found cited.
    assert rep["examined"]["orphan_paragraphs"] >= 2


def test_cited_id_missing_from_ledger(tmp_path):
    rows = [row("CL-01")]
    files = {"discussion.md": "The claim CL-01 holds, and CL-09 is cited here.\n"}
    ledger, mdir = make_case(tmp_path, rows, files)
    rc, rep = run(ledger, mdir, "--check")
    assert rc == 1
    assert "cited_not_in_ledger" in rep["failed"]
    assert rep["counts"]["cited_not_in_ledger"] == 1
    hit = rep["cited_not_in_ledger"][0]
    assert hit["id"] == "CL-09" and hit["file"] == "discussion.md"


def test_uncited_ledger_id(tmp_path):
    rows = [row("CL-01"), row("CL-02")]
    files = {"discussion.md": "Only CL-01 is discussed here.\n"}
    ledger, mdir = make_case(tmp_path, rows, files)
    rc_check, rep = run(ledger, mdir, "--check")
    assert rc_check == 0, "uncited ids are not a --check failure"
    assert rep["counts"]["uncited_ledger_ids"] == 1
    assert rep["uncited_ledger_ids"][0] == {
        "id": "CL-02",
        "manuscript_destination": "Results (CL-02)",
    }
    rc_strict, strict_rep = run(ledger, mdir, "--strict")
    assert rc_strict == 1
    assert "uncited_ledger_ids" in strict_rep["failed"]


def test_status_outside_tier_set(tmp_path):
    rows = [row("CL-01", "MAYBE")]
    ledger, mdir = make_case(tmp_path, rows, {})
    rc, rep = run(ledger, mdir, "--check")
    assert rc == 1
    assert "status_outside_tier" in rep["failed"]
    assert rep["counts"]["status_outside_tier"] == 1
    assert rep["status_outside_tier"][0] == {"id": "CL-01", "status": "MAYBE"}


def test_orphan_paragraph(tmp_path):
    rows = [row("CL-01")]
    files = {
        "results.md": (
            "# Results\n\n"
            "Fig. 3 reports 0.0521 mm/s in the H4 assay.\n\n"
            "The estimator passed its controls [CL-01].\n"
        ),
    }
    ledger, mdir = make_case(tmp_path, rows, files)
    rc_check, rep = run(ledger, mdir, "--check")
    assert rc_check == 0, "orphan candidates are a --strict failure only"
    assert rep["counts"]["orphan_candidates"] == 1
    orphan = rep["orphan_candidates"][0]
    assert orphan["file"] == "results.md" and orphan["line"] == 3
    assert set(orphan["reasons"]) == {"figure", "number_with_unit"}
    rc_strict, strict_rep = run(ledger, mdir, "--strict")
    assert rc_strict == 1
    assert "orphan_candidates" in strict_rep["failed"]


def test_missing_ledger_is_unknown_not_clean(tmp_path):
    missing = tmp_path / "no_such_ledger.json"
    mdir = tmp_path / "manuscript"
    mdir.mkdir()
    (mdir / "abstract.md").write_text(CLEAN_ABSTRACT, encoding="utf-8")
    (mdir / "results.md").write_text(CLEAN_RESULTS, encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--ledger", str(missing), "--manuscript-dir", str(mdir)],
        capture_output=True, text=True, encoding="utf-8", check=False,
    )
    assert proc.returncode == 2
    assert proc.stdout == ""
    assert "cannot read ledger" in proc.stderr


def test_live_repo_reports_four_counts():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--json"],
        capture_output=True, text=True, encoding="utf-8", check=False, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    rep = json.loads(proc.stdout)
    assert set(rep["counts"]) == COUNT_KEYS
    assert all(isinstance(v, int) and v >= 0 for v in rep["counts"].values())
    assert rep["examined"]["ledger_rows"] > 0
