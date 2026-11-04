"""The docs gate and the dev gate run the prose hygiene tests.

2026-10-27: a docs edit passed the docs gate and CI Fast with three "X, not Y" hits that only
the broad gate's ``tests/test_public_docs_hygiene.py`` caught.
"""

from __future__ import annotations

import scripts.run_test_gate as rtg

HYGIENE = "tests/test_public_docs_hygiene.py"


def test_dev_gate_collects_the_docs_hygiene_tests():
    assert HYGIENE in rtg.DEV_PYTEST_TARGETS


def test_docs_gate_runs_the_docs_hygiene_tests(monkeypatch):
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(rtg, "_run", lambda cmd, family=None, **kw: calls.append((family, cmd)))
    rtg.gate_docs()
    assert calls, "the docs gate ran nothing"
    pytest_calls = [cmd for family, cmd in calls if family == "pytest_docs"]
    assert len(pytest_calls) == 1
    assert HYGIENE in pytest_calls[0]
    assert "pytest_docs" in rtg.CHECK_FAMILIES
    assert "pytest_docs" in rtg.GATE_CHECK_FAMILIES["docs"]
