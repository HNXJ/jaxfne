"""evidence_checklist_path points at the checklist's current home, artifacts/programme/paper/publication/."""

import json
from pathlib import Path

import jaxfne.vis.evidence_manifest as em

CURRENT = ("artifacts", "programme", "paper", "publication", "evidence_checklist.json")


def test_missing_checklist_resolves_to_current_home(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "repo_root", lambda: tmp_path)
    assert not (tmp_path / "artifacts").exists(), "fixture must have no checklist"
    assert em.evidence_checklist_path() == tmp_path.joinpath(*CURRENT)


def test_present_checklist_resolves_to_current_home(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "repo_root", lambda: tmp_path)
    target = tmp_path.joinpath(*CURRENT)
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert em.evidence_checklist_path() == target


def test_evidence_dirs_default_under_artifacts_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "repo_root", lambda: tmp_path)
    dirs = em.ensure_evidence_dirs()
    temp = tmp_path / "artifacts" / "_temp"
    assert dirs["outputs"] == temp / "outputs" / "evidence"
    assert dirs["figures"] == temp / "figures" / "evidence"
    assert dirs["outputs"].is_dir() and dirs["figures"].is_dir()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["artifacts"], "nothing may be created at the root"


def _strings(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)
    elif isinstance(node, str):
        yield node


def test_checklist_manifest_and_inventory_paths_live_in_artifacts_temp():
    checklist_path = Path(__file__).resolve().parents[1].joinpath(*CURRENT)
    checklist = json.loads(checklist_path.read_text(encoding="utf-8"))
    named = [s for s in _strings(checklist) if s.endswith(("_manifest.json", "inventory.json"))]
    assert any(s.endswith("_manifest.json") for s in named), "fixture must name manifests"
    assert any(s.endswith("inventory.json") for s in named), "fixture must name the inventory"
    for path in named:
        assert path.startswith("artifacts/_temp/"), path
