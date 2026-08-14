"""Frozen Experiment A protocol loader (B0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_ID = "experiment_a_v0417_b"
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SPEC_PATH = _REPO_ROOT / "artifacts" / "etudes" / "experiment_a" / "b0_protocol_spec.json"


def load_protocol_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen B0 protocol specification."""
    spec_path = path or PROTOCOL_SPEC_PATH
    return json.loads(spec_path.read_text())
