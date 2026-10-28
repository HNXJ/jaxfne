"""The paper-lanes gate entry in run_test_gate.py selects the broad gate's lanes.

The CI job ``paper-lanes`` calls ``run_test_gate.py paper-lanes``; these tests pin that
the CLI runs the same lane list the broad gate runs, refuses an unknown lane without
running anything, and fails the run with the lane's name when a lane fails.
"""

from __future__ import annotations

import re

import pytest

import scripts.run_test_gate as rtg


@pytest.fixture
def ran(monkeypatch):
    """Replace the per-lane pytest process with a recorder that returns rc 0."""
    calls: list[str] = []
    monkeypatch.setattr(rtg, "_run_lane", lambda lane: calls.append(lane) or 0)
    rtg.reset_observations()
    yield calls
    rtg.reset_observations()


def test_cli_selects_the_same_lanes_as_the_broad_gate(ran):
    rtg.gate_pytest_lanes()  # the call gate_broad makes
    broad = list(ran)
    ran.clear()
    assert rtg.main(["paper-lanes"]) == 0
    assert broad == rtg.discover_pytest_lanes(), "broad gate does not run every discovered lane"
    assert ran == broad, "paper-lanes CLI selects different lanes from the broad gate"
    assert ran, "examined set is empty: no paper lanes ran"


def test_cli_lane_flag_runs_only_the_named_lane(ran):
    lane = rtg.discover_pytest_lanes()[0]
    assert rtg.main(["paper-lanes", "--lane", lane]) == 0
    assert ran == [lane]


def test_cli_refuses_an_unknown_lane_before_running_any(ran):
    with pytest.raises(ValueError, match="unknown paper lane"):
        rtg.main(["paper-lanes", "--lane", "no_such_lane"])
    assert ran == [], "an unknown lane must refuse the run, not start the other lanes"


def test_lane_flag_is_refused_on_other_gates(ran, capsys):
    with pytest.raises(SystemExit):
        rtg.main(["dev", "--lane", rtg.discover_pytest_lanes()[0]])
    assert "applies to the paper-lanes gate only" in capsys.readouterr().err
    assert ran == []


def test_failing_lane_fails_the_run_and_names_the_lane(monkeypatch, ran):
    target = rtg.discover_pytest_lanes()[0]
    monkeypatch.setattr(rtg, "_run_lane", lambda lane: ran.append(lane) or (1 if lane == target else 0))
    with pytest.raises(RuntimeError, match=re.escape(f"{target}: rc=1")):
        rtg.main(["paper-lanes"])
    assert set(ran) == set(rtg.discover_pytest_lanes()), "a failing lane must not stop the other lanes"
