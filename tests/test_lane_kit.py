"""Lane kit and lane scaffold tests (R1).

The kit holds the one copy of the guards every paper lane re-implements;
``scripts/new_lane.py`` scaffolds a new lane from it. Fixtures live under
``tmp_path``: no test writes into the real papers folder.
"""

from __future__ import annotations

import json
import py_compile
import subprocess
from pathlib import Path

import pytest

from artifacts.etudes.meta import lane_kit
from scripts import new_lane as nl

REPO = Path(__file__).resolve().parent.parent
SHA = "testsha"


def test_refuse_scored_dir_refuses_unsigned():
    with pytest.raises(SystemExit, match="scored"):
        lane_kit.refuse_scored_dir("some/scored/runs")
    lane_kit.refuse_scored_dir("some/scored/runs", signed=True)
    lane_kit.refuse_scored_dir("some/runs")


def test_refuse_scored_dir_matches_scored_parts_only():
    for bad in ("scored", "scored/v1", "Scored", "SCORED",
                "out/scored_v1", "out/scored-v1", "out/Scored_x"):
        with pytest.raises(SystemExit, match="scored"):
            lane_kit.refuse_scored_dir("some/%s/runs" % bad)
    for ok in ("unscored", "out/unscored/pilots", "out/rescored",
               "out/my_scored_mix"):
        lane_kit.refuse_scored_dir("some/%s/runs" % ok)


def test_refuse_existing_signed_needs_provenance(tmp_path):
    with pytest.raises(SystemExit, match="provenance"):
        lane_kit.refuse_existing(tmp_path, "f.npz", "unknown", signed=True)
    assert lane_kit.refuse_existing(tmp_path, "f.npz", "unknown") is None
    assert lane_kit.refuse_existing(tmp_path, "g.npz", SHA, signed=True) is None


def test_refuse_existing_file_refuses(tmp_path):
    (tmp_path / "f.npz").write_text("x")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        lane_kit.refuse_existing(tmp_path, "f.npz", SHA)


def test_refuse_existing_other_sha_refuses(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"git_sha": "other", "runs": []}))
    with pytest.raises(SystemExit, match="one code SHA"):
        lane_kit.refuse_existing(tmp_path, "f.npz", SHA)


def test_refuse_existing_same_sha_returns_prior(tmp_path):
    prior = {"git_sha": SHA, "runs": [{"file": "old.npz"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(prior))
    assert lane_kit.refuse_existing(tmp_path, "new.npz", SHA) == prior
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        lane_kit.refuse_existing(tmp_path, "old.npz", SHA)


def test_git_sha_matches_repo(tmp_path):
    out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    assert lane_kit.git_sha(REPO) == out.stdout.strip()
    assert lane_kit.git_sha(tmp_path) == "unknown"


def test_write_manifest_round_trip(tmp_path):
    path = lane_kit.write_manifest(tmp_path, "manifest.json",
                                   {"runs": [{"file": "f.npz"}]}, sha=SHA)
    text = path.read_text()
    doc = json.loads(text)
    assert doc["git_sha"] == SHA
    assert doc["runs"] == [{"file": "f.npz"}]
    assert doc["start_time"] and doc["jaxfne_version"] and doc["platform"]
    assert list(doc) == sorted(doc), "manifest keys are sorted"


def _lane_args(tmp_path, *extra):
    return ["demo_lane", "--paper", "Author (2026), Title, Journal",
            "--root", str(tmp_path), *extra]


def test_new_lane_dry_run_creates_nothing(tmp_path, capsys):
    assert nl.main(_lane_args(tmp_path)) == 0
    assert not (tmp_path / "artifacts" / "etudes" / "papers" / "demo_lane").exists()
    out = capsys.readouterr().out
    assert "README.md" in out and "PREREG_P3.md" in out


def test_new_lane_write_then_refuse(tmp_path, capsys):
    assert nl.main(_lane_args(tmp_path, "--write")) == 0
    folder = tmp_path / "artifacts" / "etudes" / "papers" / "demo_lane"
    names = sorted(p.name for p in folder.iterdir() if p.is_file())
    assert len(names) <= 7, names
    for want in ("README.md", "SOURCE_P1.md", "PREREG_P3.md", "CALIBRATION.md",
                 "demo_lane_api.py", "test_demo_lane_api.py"):
        assert want in names, names
    api = (folder / "demo_lane_api.py").read_text()
    assert "from artifacts.etudes.meta import lane_kit" in api
    assert "import jaxfne as J" in api
    assert "PORT_SPEC" not in api
    prereg = (folder / "PREREG_P3.md").read_text()
    assert "UNSIGNED" in prereg and "no scored run may use it" in prereg
    assert "## Pilot (B8)" in prereg
    for name in names:
        if name.endswith(".py"):
            py_compile.compile(str(folder / name), doraise=True)
    assert nl.main(_lane_args(tmp_path, "--write")) == 1
    assert "refusing existing lane folder" in capsys.readouterr().out


def test_scaffold_passes_checker_r1_r6(tmp_path):
    """The scaffolded lane holds its P1 record and passes the R1/R6 gates.

    ``scripts/check_replications.py`` exposes per-root gates (``--root``),
    so the scaffold is checked in a tmp tree, never in the real papers
    folder: ``check_r1`` with a one-paper registry and ``check_r6``.
    """
    from scripts import check_replications as cr

    assert nl.main(_lane_args(tmp_path, "--write")) == 0
    papers = [{"id": "demo_lane", "stage": "P1"}]
    assert cr.check_r1(papers, tmp_path) == []
    assert cr.check_r6(tmp_path) == []
