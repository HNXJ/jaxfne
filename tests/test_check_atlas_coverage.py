"""Coverage gate: rows owned by a sealed release may not stay PLANNED."""

from __future__ import annotations

import importlib.util
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_atlas_coverage", REPO / "scripts" / "check_atlas_coverage.py"
)
cac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cac)


def _row(rid: str, release: str, candidate: bool) -> dict:
    return {
        "id": rid,
        "sim": f"S{int(rid.split('-')[1])}",
        "requirement": "r",
        "capability": "c",
        "release": release,
        "candidate": candidate,
        "evidence": None,
        "state": "PLANNED",
    }


def _doc(rows: list[dict]) -> dict:
    return {
        "schema": cac.SCHEMA_TAG,
        "states": {s: s for s in cac.KNOWN_STATES},
        "seal_rule": "x",
        "sealed_releases": ["0.5.3"],
        "requirements": rows,
    }


def test_planned_in_sealed_release_fails_candidate_warns(tmp_path):
    doc = _doc(
        [
            _row("AT-07-R1", "0.5.3", candidate=False),  # sealed, not candidate
            _row("AT-07-R4", "0.5.3", candidate=True),  # sealed, candidate
            _row("AT-10-R1", "0.5.5", candidate=False),  # open release
        ]
    )
    path = tmp_path / "cov.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    warnings: list[str] = []
    errors = cac.check(path, tmp_path, warnings)
    assert errors == ["AT-07-R1: PLANNED in sealed release 0.5.3"]
    assert len(warnings) == 1 and warnings[0].startswith("AT-07-R4:")


def test_repository_coverage_valid():
    errors = cac.check(REPO / "artifacts" / "programme" / "atlas_coverage.json", REPO)
    assert errors == []
