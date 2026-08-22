"""SHA256 content-identity helpers for jaxfne artifacts, configs, notebooks, and release files.

Extracted from skills/jaxfne-sha256-artifact-integrity/SKILL.md (2026-07-14) so the skill can
reference a real contract instead of re-embedding ~150 lines of function bodies in prose. See
that skill for when/why to use each helper, the debug decision table, and the status-gate
wording rules (SHA256 proves content identity, never scientific/biological correctness).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def sha256_file(path: PathLike, *, block_size: int = 1024 * 1024) -> str:
    """Return SHA256 hex digest for a file's exact bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def stable_json_bytes(obj: Any) -> bytes:
    """Return stable strict-JSON bytes for deterministic hashing."""
    return json.dumps(
        obj,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(obj: Any) -> str:
    """Return SHA256 for stable strict JSON representation."""
    return hashlib.sha256(stable_json_bytes(obj)).hexdigest()


def notebook_source_sha256(path: PathLike) -> str:
    """Hash notebook cell type + source only, ignoring outputs/execution counts."""
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    source_only = [
        {
            "cell_type": cell.get("cell_type"),
            "source": cell.get("source", []),
        }
        for cell in nb.get("cells", [])
    ]
    return sha256_json(source_only)


def make_asset_hashes(
    root: PathLike, patterns: tuple[str, ...] = ("*.json", "*.png", "*.html", "*.npz")
) -> dict[str, str]:
    """Hash generated artifacts under a directory."""
    root = Path(root)
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    out: dict[str, str] = {}
    for path in sorted(set(paths)):
        if path.is_file():
            out[str(path.relative_to(root))] = sha256_file(path)
    return out


def write_asset_hashes(root: PathLike, output_name: str = "asset_hashes.json") -> dict[str, str]:
    """Write asset_hashes.json with strict sorted JSON."""
    root = Path(root)
    hashes = make_asset_hashes(root)
    (root / output_name).write_text(json.dumps(hashes, indent=2, sort_keys=True, allow_nan=False))
    return hashes


def diff_hashes(old: dict[str, str], new: dict[str, str]) -> dict[str, dict[str, str | None]]:
    """Return changed/missing/new file hashes."""
    keys = sorted(set(old) | set(new))
    return {
        k: {"old": old.get(k), "new": new.get(k)}
        for k in keys
        if old.get(k) != new.get(k)
    }


def candidate_sha256(params: dict) -> str:
    """Stable hash for an AGSDR/training candidate's parameter dict, for dedup/caching."""
    return sha256_json(params)


def load_weight_artifact(ref: dict, root: PathLike = "."):
    """Load a weight array from an artifact_ref dict, verifying its recorded SHA256 first."""
    import numpy as np

    path = Path(root) / ref["path"]
    expected = str(ref["sha256"])
    if expected.startswith("sha256:"):
        expected = expected[len("sha256:"):]
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"Weight artifact hash mismatch: {path}\n"
            f"expected={expected}\n"
            f"actual={actual}"
        )
    data = np.load(path)
    return data[ref["array_name"]]
