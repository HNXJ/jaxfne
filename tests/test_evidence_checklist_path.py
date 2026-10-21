"""evidence_checklist_path points at the checklist's current home, docs/theory/publication/."""

import jaxfne.vis.evidence_manifest as em

CURRENT = ("docs", "theory", "publication", "evidence_checklist.json")


def test_missing_checklist_resolves_to_current_home(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "repo_root", lambda: tmp_path)
    assert not (tmp_path / "docs").exists(), "fixture must have no checklist"
    assert em.evidence_checklist_path() == tmp_path.joinpath(*CURRENT)


def test_present_checklist_resolves_to_current_home(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "repo_root", lambda: tmp_path)
    target = tmp_path.joinpath(*CURRENT)
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    assert em.evidence_checklist_path() == target
