"""Shared helpers for jaxfne publication figure generators (0.4.17)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


_FROZEN_MANIFEST: dict | None = None
_METADATA_IGNORE_KEYS = ("repo_head", "audited_at_utc")


def _frozen_path_set() -> set[str]:
    global _FROZEN_MANIFEST
    if _FROZEN_MANIFEST is None:
        manifest_path = _REPO_ROOT / ".opencode" / "frozen_paths.json"
        _FROZEN_MANIFEST = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = _FROZEN_MANIFEST.get("files") or _FROZEN_MANIFEST.get("paths") or []
    return {str(p) for p in files}


def _describe_diff(old: Any, new: Any, at: str) -> str:
    if isinstance(old, dict) and isinstance(new, dict):
        for key in sorted(set(old) | set(new), key=str):
            if at == "" and key in _METADATA_IGNORE_KEYS:
                continue
            if key not in old:
                return f"{at}.{key}: new-only key"
            if key not in new:
                return f"{at}.{key}: missing in regeneration"
            found = _describe_diff(old[key], new[key], f"{at}.{key}")
            if found:
                return found
        return ""
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            return f"{at}: list length {len(old)} != {len(new)}"
        for i, (a, b) in enumerate(zip(old, new)):
            found = _describe_diff(a, b, f"{at}[{i}]")
            if found:
                return found
        return ""
    if old != new:
        return f"{at}: {str(old)[:80]!r} != {str(new)[:80]!r}"
    return ""


def write_json_strict(path: Path, obj: dict) -> None:
    root = _REPO_ROOT
    rel = path.resolve().relative_to(root).as_posix()
    if rel in _frozen_path_set() and path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        regenerated = json.loads(json.dumps(obj, indent=2, allow_nan=False))
        drift = _describe_diff(existing, regenerated, "")
        if not drift:
            return
        raise RuntimeError(
            f"frozen artifact drift at {rel}: {drift}. "
            "Write-once: frozen files must not be regenerated in place. "
            "Restore with `git checkout -- <path>` before re-running."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
