"""Tests for scripts/compare_pytest_junit.py — per-node JUnit comparison."""

from __future__ import annotations

from pathlib import Path

from scripts.compare_pytest_junit import compare, load, load_many

from scripts.run_test_gate import BROAD_PYTEST_IGNORE


def _junit(path: Path, cases: list[tuple[str, str, str]]) -> Path:
    """Write a minimal pytest-junit file. Each case: (node, outcome, reason)."""
    lines = ['<?xml version="1.0" ?>', '<testsuites><testsuite>']
    for node, outcome, reason in cases:
        cls, _, name = node.partition("::")
        if outcome == "skipped":
            lines.append(
                f'<testcase classname="{cls}" name="{name}">'
                f'<skipped message="{reason}"/></testcase>'
            )
        elif outcome in ("failure", "error"):
            lines.append(
                f'<testcase classname="{cls}" name="{name}">'
                f'<{outcome} message="{reason}"/></testcase>'
            )
        else:
            lines.append(f'<testcase classname="{cls}" name="{name}"/>')
    lines.append("</testsuite></testsuites>")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_identical_sets_pass(tmp_path):
    rc = _junit(tmp_path / "rc.xml", [("a::t1", "passed", ""), ("a::t2", "skipped", "x")])
    ci = _junit(tmp_path / "ci.xml", [("a::t1", "passed", ""), ("a::t2", "skipped", "x")])
    rep = compare(load(rc), load(ci))
    assert rep["pass"] is True
    assert rep["outcome_differences"] == []


def test_platform_skip_difference_is_justified(tmp_path):
    rc = _junit(tmp_path / "rc.xml", [("a::t1", "skipped", "POSIX shell-script gates run on Linux/macOS CI only")])
    ci = _junit(tmp_path / "ci.xml", [("a::t1", "passed", "")])
    rep = compare(load(rc), load(ci))
    assert len(rep["outcome_differences"]) == 1
    assert rep["outcome_differences"][0]["classification"] == "INTENTIONAL_PLATFORM_DIFFERENCE"
    assert rep["pass"] is True


def test_reportlab_difference_is_environment_defect(tmp_path):
    rc = _junit(tmp_path / "rc.xml", [("a::t1", "passed", "")])
    ci = _junit(tmp_path / "ci.xml", [("a::t1", "skipped", "reportlab not installed (viz extra not in dev CI)")])
    rep = compare(load(rc), load(ci))
    assert rep["outcome_differences"][0]["classification"] == "ENVIRONMENT_DEFECT"
    assert rep["pass"] is False


def test_node_set_mismatch_fails(tmp_path):
    rc = _junit(tmp_path / "rc.xml", [("a::t1", "passed", "")])
    ci = _junit(tmp_path / "ci.xml", [("a::t1", "passed", ""), ("a::t2", "passed", "")])
    rep = compare(load(rc), load(ci))
    assert rep["ci_only"] == ["a::t2"]
    assert rep["pass"] is False


def test_rc_sweeps_emit_mergable_junit_evidence():
    """Every RC pytest sweep must write gitignored JUnit with skip reasons.

    Without this, RC outcomes cannot be compared node-by-node against the
    blocking CI JUnit artifacts (the defect that hid the reportlab/POSIX
    swap behind equal totals).
    """
    from pathlib import Path as _Path

    text = (_Path(__file__).resolve().parents[1] / "scripts" / "run_test_gate.py").read_text(
        encoding="utf-8"
    )
    for sweep in ("rc-pytest-broad.xml", "rc-pytest-slow.xml", "rc-pytest-notebook.xml"):
        assert sweep in text, f"RC gate no longer emits {sweep}"
    assert '"-rs"' in text or "'-rs'" in text or "-rs" in text
    assert "artifacts" in text and "attestations" in text
    assert BROAD_PYTEST_IGNORE == []


def test_load_many_merges_disjoint_sweeps(tmp_path):
    r1 = _junit(tmp_path / "b.xml", [("a::t1", "passed", "")])
    r2 = _junit(tmp_path / "s.xml", [("a::t2", "skipped", "POSIX x")])
    merged = load_many([r1, r2])
    assert sorted(merged) == ["a::t1", "a::t2"]
