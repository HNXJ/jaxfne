"""Shared helpers for jaxfne evidence figure generators.

Dependency-light: no matplotlib, no optional packages, no import-time side effects.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


_MANIFEST_SCHEMA = "jaxfne.evidence_figure_manifest.v0.1.0"


def repo_root() -> Path:
    """Repository root (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent.parent


def repo_sha() -> str:
    """Current git HEAD SHA, or ``unknown`` if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def jaxfne_version() -> str:
    """Installed jaxfne version, or pyproject.toml fallback."""
    try:
        import jaxfne as jtfne

        return str(getattr(jtfne, "__version__", "unknown"))
    except Exception:
        content = (repo_root() / "pyproject.toml").read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.strip().startswith("version"):
                return line.split("=", 1)[1].strip().strip('"')
        return "unknown"


def sha256_file(path: Path) -> str:
    """SHA256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    """UTC timestamp in ISO-8601 compact form."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")



def evidence_checklist_path() -> Path:
    """Return the active evidence checklist path with compatibility fallback.

    v0.3.42 public generators refer to ``docs/evidence/evidence_checklist.json``.
    Some repo bundles place the same checklist under ``docs/evidence_artifacts``.
    Prefer the canonical path when present, but allow the compatibility path so
    local audit and figure scripts do not false fail on zipped source bundles.
    """
    root = repo_root()
    candidates = [
        root / "docs" / "evidence" / "evidence_checklist.json",
        root / "docs" / "evidence_artifacts" / "evidence_checklist.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]

def truth_gates() -> dict:
    """Default evidence truth gates (copy safe for manifests)."""
    return {
        "truth_mode": "truth_safe_unverified",
        "claim_level": "computational_scaffold",
        "field_solver_status": "laminar_proxy_no_pde",
        "field_claim_level": "proxy_readout_only",
        "physical_amplitude_claim_allowed": False,
    }


def write_json_strict(path: Path, obj: dict) -> None:
    """Write JSON with NaN/Inf rejected."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ensure_evidence_dirs() -> dict[str, Path]:
    """Create and return standard evidence output directories."""
    root = repo_root()
    figures = root / "figures" / "evidence"
    outputs = root / "outputs" / "evidence"
    figures.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return {
        "repo_root": root,
        "figures": figures,
        "outputs": outputs,
    }


def manifest_path_for(output_path: Path) -> Path:
    """Standard manifest path: outputs/evidence/<stem>_manifest.json."""
    dirs = ensure_evidence_dirs()
    return dirs["outputs"] / f"{output_path.stem}_manifest.json"


def save_figure_manifest(
    *,
    figure_id: str,
    title: str,
    output_path: Path,
    source_files: list[str],
    extra: dict | None = None,
) -> dict:
    """Build and write a strict JSON manifest for a generated figure PNG."""
    root = repo_root()
    output_path = Path(output_path)
    if not output_path.is_file():
        raise FileNotFoundError(f"Figure output not found: {output_path}")

    sha = sha256_file(output_path)
    manifest: dict = {
        "schema_version": _MANIFEST_SCHEMA,
        "figure_id": figure_id,
        "title": title,
        "output_path": str(output_path.relative_to(root)),
        "sha256": sha,
        "generated_at_utc": utc_now_iso(),
        "jaxfne_version": jaxfne_version(),
        "repo_sha": repo_sha(),
        "truth_gates": truth_gates(),
        "source_files": list(source_files),
        "physical_amplitude_claim_allowed": False,
    }
    if extra:
        manifest.update(extra)

    manifest_path = manifest_path_for(output_path)
    write_json_strict(manifest_path, manifest)
    return manifest
