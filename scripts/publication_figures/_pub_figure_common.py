"""Shared helpers for jaxfne publication figure generators (0.4.17)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from jaxfne.vis.evidence_export import save_matplotlib_evidence_figure

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUBLICATION_ARTIFACTS = _REPO_ROOT / "artifacts" / "publication"
_PUBLICATION_FIGURES = _REPO_ROOT / "figures" / "publication"


def repo_root() -> Path:
    return _REPO_ROOT


def repo_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_publication_dirs() -> dict[str, Path]:
    _PUBLICATION_FIGURES.mkdir(parents=True, exist_ok=True)
    _PUBLICATION_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": _REPO_ROOT,
        "figures": _PUBLICATION_FIGURES,
        "artifacts": _PUBLICATION_ARTIFACTS,
    }


def save_matplotlib_figure(fig, path: Path, *, dpi: int = 200) -> str:
    return save_matplotlib_evidence_figure(
        fig, path, dpi=dpi, bbox_inches="tight", facecolor="white", close=True
    )


def write_json_strict(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
