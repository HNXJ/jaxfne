"""save_figure_manifest records only source files that exist on disk."""

from __future__ import annotations

import pytest

import jaxfne.vis.evidence_manifest as em


def _output(tmp_path, monkeypatch):
    monkeypatch.setattr(em, "repo_root", lambda: tmp_path)
    out = tmp_path / "fig.png"
    out.write_bytes(b"fake-png")
    return out


def test_missing_source_raises_naming_it(tmp_path, monkeypatch):
    out = _output(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="no-such-source"):
        em.save_figure_manifest(
            figure_id="t",
            title="t",
            output_path=out,
            source_files=["docs/no-such-source.json"],
        )


def test_existing_source_passes(tmp_path, monkeypatch):
    out = _output(tmp_path, monkeypatch)
    (tmp_path / "checklist.json").write_text("{}", encoding="utf-8")
    manifest = em.save_figure_manifest(
        figure_id="t",
        title="t",
        output_path=out,
        source_files=["checklist.json"],
    )
    assert manifest["source_files"] == ["checklist.json"]
