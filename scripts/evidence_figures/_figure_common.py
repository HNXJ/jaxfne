"""Shared helpers for jaxfne evidence figure generators.

Dependency-light manifest helpers delegate to ``jaxfne.vis.evidence_manifest``.
Matplotlib save/close delegates to ``jaxfne.vis.evidence_export``.
"""

from __future__ import annotations

from jaxfne.vis.evidence_export import save_matplotlib_evidence_figure
from jaxfne.vis.evidence_manifest import (
    ensure_evidence_dirs,
    evidence_checklist_path,
    jaxfne_version,
    manifest_path_for,
    repo_root,
    repo_sha,
    save_figure_manifest,
    sha256_file,
    truth_gates,
    utc_now_iso,
    write_json_strict,
)

__all__ = [
    "ensure_evidence_dirs",
    "evidence_checklist_path",
    "jaxfne_version",
    "manifest_path_for",
    "repo_root",
    "repo_sha",
    "save_figure_manifest",
    "save_matplotlib_figure",
    "sha256_file",
    "truth_gates",
    "utc_now_iso",
    "write_json_strict",
]


def save_matplotlib_figure(fig, path, *, dpi: int = 150) -> str:
    """Save and close a matplotlib figure via ``jaxfne.vis`` (grammar rule 2 bridge)."""
    return save_matplotlib_evidence_figure(
        fig, path, dpi=dpi, bbox_inches="tight", facecolor="white", close=True,
    )
