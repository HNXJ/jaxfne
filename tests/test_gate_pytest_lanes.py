"""The paper-lane pytest family (``pytest_lanes``): discovery, known-failing handling, wiring.

Lane tests under ``artifacts/etudes/papers/`` run one pytest process per lane. The
lane runner is stubbed here so these tests do not execute the lanes themselves.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import scripts.run_test_gate as rtg

ROOT = Path(__file__).resolve().parents[1]


def _tracked_lanes_from_git() -> set[str]:
    """Lane folders named by git: a tracked ``test_*.py`` at any depth under the papers root."""
    listing = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", f"{rtg.PAPER_LANES_DIR}/*test_*.py"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    prefix = rtg.PAPER_LANES_DIR + "/"
    lanes: set[str] = set()
    for path in listing:
        rel = path[len(prefix) :].split("/")
        if len(rel) >= 2 and rel[-1].startswith("test_") and rel[-1].endswith(".py"):
            lanes.add(rel[0])
    return lanes


def test_discovery_matches_tracked_lane_test_files():
    tracked = _tracked_lanes_from_git()
    assert tracked, "git lists no tracked lane test files; the comparison examined nothing"
    discovered = rtg.discover_pytest_lanes()
    assert discovered == sorted(tracked), (
        f"discovered {discovered}, git lists lanes {sorted(tracked)}"
    )


def test_discovery_ignores_non_test_files_and_top_level_tests(monkeypatch):
    listing = "\n".join(
        [
            "artifacts/etudes/papers/alpha/test_a.py",
            "artifacts/etudes/papers/alpha/data.json",
            "artifacts/etudes/papers/beta/deep/er/test_b.py",
            "artifacts/etudes/papers/gamma/notes.md",
            "artifacts/etudes/papers/loose_test_c.py",
            "artifacts/etudes/papers/delta/helper_test.py",
            "artifacts/etudes/other/test_z.py",
        ]
    )
    monkeypatch.setattr(rtg, "_git_out", lambda *args: listing)
    assert rtg.discover_pytest_lanes() == ["alpha", "beta"]


def test_lane_command_is_one_process_per_lane_dir():
    argv = rtg._lane_argv("pcl")
    assert argv[1:4] == ["-m", "pytest", "-q"]
    assert argv[-1] == f"{rtg.PAPER_LANES_DIR}/pcl"
    assert "no:cacheprovider" in argv


def test_slow_lanes_run_under_xdist_and_short_ones_serial():
    # Profiled 2026-10-27: pcl 542 s serial, 121 s with xdist; caminos 247 s, 65 s.
    for lane in ("pcl", "caminos"):
        argv = rtg._lane_argv(lane)
        assert argv[-1] == f"{rtg.PAPER_LANES_DIR}/{lane}"
        assert all(a in argv for a in rtg.PYTEST_XDIST_ARGS), lane
    assert "-n" not in rtg._lane_argv("vogels2011")


def _control_lanes(monkeypatch, lanes, returncodes, known):
    monkeypatch.setattr(rtg, "discover_pytest_lanes", lambda: list(lanes))
    monkeypatch.setattr(rtg, "LANE_PYTEST_KNOWN_FAILING", dict(known))
    monkeypatch.setattr(rtg, "_run_lane", lambda lane: returncodes[lane])
    rtg.reset_observations()


def test_stale_known_failing_entry_fails_the_family(monkeypatch, capsys):
    _control_lanes(monkeypatch, ["brunel2000"], {"brunel2000": 0}, {"brunel2000": "old reason"})
    assert rtg.LANE_PYTEST_KNOWN_FAILING["brunel2000"]  # fixture: the entry is listed
    try:
        with pytest.raises(RuntimeError, match="stale"):
            rtg.gate_pytest_lanes()
        assert rtg._OBSERVATIONS["pytest_lanes"]["status"] == "FAIL"
    finally:
        rtg.reset_observations()


def test_known_failing_entry_for_a_lane_with_no_tests_is_stale(monkeypatch):
    _control_lanes(monkeypatch, ["vogels2011"], {"vogels2011": 5}, {"vogels2011": "old reason"})
    try:
        with pytest.raises(RuntimeError, match="stale"):
            rtg.gate_pytest_lanes()
    finally:
        rtg.reset_observations()


def test_known_failing_lane_that_still_fails_is_skipped_with_reason(monkeypatch, capsys):
    _control_lanes(monkeypatch, ["pcl"], {"pcl": 1}, {"pcl": "guard aborts collection"})
    try:
        rtg.gate_pytest_lanes()
        out = capsys.readouterr().out
        assert "SKIP known-failing lane pcl (rc=1): guard aborts collection" in out
        assert rtg.observed_pass_families() == {"pytest_lanes"}
    finally:
        rtg.reset_observations()


def test_unlisted_failing_lane_fails_the_family(monkeypatch):
    _control_lanes(monkeypatch, ["caminos", "qlearning"], {"caminos": 1, "qlearning": 0}, {})
    try:
        with pytest.raises(RuntimeError, match="caminos: rc=1"):
            rtg.gate_pytest_lanes()
        assert rtg._OBSERVATIONS["pytest_lanes"]["status"] == "FAIL"
    finally:
        rtg.reset_observations()


def test_no_tests_collected_is_accepted(monkeypatch):
    _control_lanes(monkeypatch, ["vogels2011"], {"vogels2011": 5}, {})
    try:
        rtg.gate_pytest_lanes()
        assert rtg.observed_pass_families() == {"pytest_lanes"}
    finally:
        rtg.reset_observations()


def test_all_lanes_passing_records_one_passing_family(monkeypatch):
    _control_lanes(monkeypatch, ["a", "b"], {"a": 0, "b": 0}, {})
    try:
        rtg.gate_pytest_lanes()
        rec = rtg._OBSERVATIONS["pytest_lanes"]
        assert rec["status"] == "PASS"
        assert len(rec["commands"]) == 2
    finally:
        rtg.reset_observations()


def test_family_is_in_broad_and_rc_but_not_report_or_docs():
    assert "pytest_lanes" in rtg.CHECK_FAMILIES
    for gate in ("broad", "rc"):
        assert "pytest_lanes" in rtg.GATE_CHECK_FAMILIES[gate], gate
    for gate in ("report", "docs"):
        assert "pytest_lanes" not in rtg.GATE_CHECK_FAMILIES[gate], gate


def test_broad_gate_runs_the_lane_family_after_the_broad_sweep(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(rtg, "_run", lambda cmd, family=None, **kw: calls.append(family))
    monkeypatch.setattr(rtg, "gate_report", lambda: calls.append("report"))
    monkeypatch.setattr(rtg, "gate_pytest_lanes", lambda: calls.append("pytest_lanes"))
    rtg.gate_broad()
    assert calls.count("pytest_lanes") == 1
    assert calls[-2:] == ["pytest_broad", "pytest_lanes"]
