"""Tests for scripts/compare_pytest_junit.py — per-node JUnit comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


def test_python_311_jnwb_skip_is_a_version_difference(tmp_path):
    """jnwb is installed only for Python >= 3.12 (pyproject extra marker)."""
    cases = [
        ("tests.test_jnwb_view::test_jnwb_consumes_the_view",
         "could not import 'jnwb': No module named 'jnwb'"),
        ("tests.test_jnwb_view::test_welch_psd_through_jnwb_is_bit_identical",
         "installed jnwb has no compute_psd(axis=, nperseg=)"),
        ("tests.test_pynwb_compat::test_jnwb_reads_the_written_spikes",
         "could not import 'jnwb': No module named 'jnwb'"),
    ]
    rc = _junit(tmp_path / "rc.xml", [(n, "passed", "") for n, _ in cases])
    ci = _junit(tmp_path / "ci.xml", [(n, "skipped", reason) for n, reason in cases])
    excused = compare(load(rc), load(ci), ci_python="3.11")
    assert excused["pass"] is True
    assert {d["classification"] for d in excused["outcome_differences"]} == {
        "INTENTIONAL_VERSION_DIFFERENCE"
    }
    not_excused = compare(load(rc), load(ci), ci_python="3.14")
    assert not_excused["pass"] is False
    assert not_excused["outcome_differences"][0]["classification"] == "UNKNOWN"


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


# --- marker-classified differences (tests/conftest.py writes jaxfne_marker) ---

DATA_NODE = "tests.test_pcl_stage_c::test_stage_c_tensor_route_20clip_receipt"
DATA_REASON = ("Stage C needs the warehouse test7 inputs, S1 oracle outputs "
               "and the frozen cSTDP3 network")
ORACLE_NODE = "tests.test_head_forward::test_eeg_matches_mne_single_shell"
ORACLE_REASON = "could not import 'mne': No module named 'mne'"
DIAG_NODE = "tests.test_adex_route::test_adex_seed_batch_matches_serial_per_member"
DIAG_REASON = "bitwise V_m is platform/run-dependent; diagnostic evidence only"

# (node, marker, expected class, skip type on the skipped leg, skip reason)
KINDS = [
    (DATA_NODE, "data_gated", "INTENTIONAL_DATA_GATE", "pytest.skip", DATA_REASON),
    (ORACLE_NODE, "optional_oracle", "INTENTIONAL_OPTIONAL_ORACLE", "pytest.skip", ORACLE_REASON),
    (DIAG_NODE, "bitwise_diagnostic", "INTENTIONAL_DIAGNOSTIC", "pytest.xfail", DIAG_REASON),
]


def _case(node, outcome, reason="", markers=(), skip_type="pytest.skip") -> str:
    """One <testcase> shaped like pytest's xunit2 output for these outcomes."""
    cls, _, name = node.partition("::")
    props = "".join(f'<property name="jaxfne_marker" value="{m}"/>' for m in markers)
    body = f"<properties>{props}</properties>" if props else ""
    if outcome == "skipped":
        body += f'<skipped type="{skip_type}" message="{reason}"/>'
    elif outcome in ("failure", "error"):
        body += f'<{outcome} message="{reason}"/>'
    return f'<testcase classname="{cls}" name="{name}">{body}</testcase>'


def _xml(path: Path, *cases: str) -> Path:
    path.write_text(
        '<?xml version="1.0" ?><testsuites><testsuite>'
        + "".join(cases) + "</testsuite></testsuites>",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("node, marker, expected, skip_type, reason", KINDS)
def test_marker_excuses_a_passed_versus_skipped_pair(tmp_path, node, marker, expected, skip_type, reason):
    rc = _xml(tmp_path / "rc.xml", _case(node, "passed", markers=[marker]))
    ci = _xml(tmp_path / "ci.xml",
              _case(node, "skipped", reason, markers=[marker], skip_type=skip_type))
    for a, b in ((rc, ci), (ci, rc)):
        rep = compare(load(a), load(b))
        assert [d["classification"] for d in rep["outcome_differences"]] == [expected]
        assert rep["pass"] is True


@pytest.mark.parametrize("marked", [True, False])
def test_parity_gate_passes_marked_pairs_and_fails_unmarked(tmp_path, monkeypatch, marked):
    import scripts.check_junit_parity as parity

    att = tmp_path / "attestations"
    att.mkdir()
    monkeypatch.setattr(parity, "ATTESTATIONS", att)
    _xml(att / parity.RC_JUNIT[0],
         *[_case(node, "passed", markers=[marker]) for node, marker, *_ in KINDS])
    for name in parity.RC_JUNIT[1:]:
        _xml(att / name)
    ci = _xml(tmp_path / "ci.xml", *[
        _case(node, "skipped", reason, markers=[marker] if marked else (), skip_type=skip_type)
        for node, marker, _, skip_type, reason in KINDS
    ])
    out = tmp_path / "out.json"
    rc = parity.main(["--ci-junit", str(ci), "--sha", "c" * 40, "--out", str(out)])
    assert rc == (0 if marked else 1)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["pass"] is marked
    by_class = written["comparisons"]["ci.xml"]["justified_by_class"]
    for node, _, expected, *_ in KINDS:
        assert (node in by_class[expected]) is marked


@pytest.mark.parametrize("node, marker, expected, skip_type, reason", KINDS)
def test_same_skip_without_marker_property_stays_unknown(tmp_path, node, marker, expected, skip_type, reason):
    rc = _xml(tmp_path / "rc.xml", _case(node, "passed"))
    ci = _xml(tmp_path / "ci.xml", _case(node, "skipped", reason, skip_type=skip_type))
    rep = compare(load(rc), load(ci))
    assert [d["classification"] for d in rep["outcome_differences"]] == ["UNKNOWN"]
    assert rep["pass"] is False


def test_marker_on_one_leg_only_stays_unknown(tmp_path):
    node, marker, _, _, reason = KINDS[0]
    rc = _xml(tmp_path / "rc.xml", _case(node, "passed", markers=[marker]))
    ci = _xml(tmp_path / "ci.xml", _case(node, "skipped", reason))
    rep = compare(load(rc), load(ci))
    assert [d["classification"] for d in rep["outcome_differences"]] == ["UNKNOWN"]
    assert rep["pass"] is False


@pytest.mark.parametrize("bad", ["failure", "error"])
@pytest.mark.parametrize("node, marker, expected, skip_type, reason", KINDS)
def test_failure_with_marker_is_never_excused(tmp_path, bad, node, marker, expected, skip_type, reason):
    rc = _xml(tmp_path / "rc.xml", _case(node, "passed", markers=[marker]))
    ci = _xml(tmp_path / "ci.xml", _case(node, bad, "boom", markers=[marker]))
    rep = compare(load(rc), load(ci))
    assert [d["classification"] for d in rep["outcome_differences"]] == ["UNKNOWN"]
    assert rep["pass"] is False
    assert node in rep["failures_or_errors"]


def test_failure_on_rc_leg_with_marker_is_never_excused(tmp_path):
    node, marker, _, _, reason = KINDS[0]
    rc = _xml(tmp_path / "rc.xml", _case(node, "error", "boom", markers=[marker]))
    ci = _xml(tmp_path / "ci.xml", _case(node, "skipped", reason, markers=[marker]))
    rep = compare(load(rc), load(ci))
    assert rep["outcome_differences"][0]["classification"] == "UNKNOWN"
    assert rep["pass"] is False


def test_diagnostic_needs_the_xfail_skip_not_a_plain_skip(tmp_path):
    node, marker, _, _, reason = KINDS[2]
    rc = _xml(tmp_path / "rc.xml", _case(node, "passed", markers=[marker]))
    ci = _xml(tmp_path / "ci.xml",
              _case(node, "skipped", reason, markers=[marker], skip_type="pytest.skip"))
    rep = compare(load(rc), load(ci))
    assert rep["outcome_differences"][0]["classification"] == "UNKNOWN"
    assert rep["pass"] is False


def test_environment_reason_beats_a_marker(tmp_path):
    node, marker, _, _, _ = KINDS[0]
    rc = _xml(tmp_path / "rc.xml", _case(node, "passed", markers=[marker]))
    ci = _xml(tmp_path / "ci.xml", _case(
        node, "skipped", "reportlab not installed (viz extra not in dev CI)", markers=[marker]))
    rep = compare(load(rc), load(ci))
    assert rep["outcome_differences"][0]["classification"] == "ENVIRONMENT_DEFECT"
    assert rep["pass"] is False


def test_load_reads_marker_property_and_skip_type(tmp_path):
    node, marker, _, skip_type, reason = KINDS[2]
    path = _xml(tmp_path / "ci.xml",
                _case(node, "skipped", reason, markers=[marker], skip_type=skip_type))
    rec = load(path)[node]
    assert rec["markers"] == ["bitwise_diagnostic"]
    assert rec["skip_type"] == "pytest.xfail"
    assert load(_xml(tmp_path / "plain.xml", _case("a::t1", "passed")))["a::t1"]["markers"] == []
