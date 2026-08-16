"""Frozen Protocol C wave specification loader (C0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_ID = "protocol_c_wave_v0417"
_REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_c" / "c0_wave_protocol_spec.json"


def load_protocol_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen C0 wave protocol specification."""
    spec_path = path or PROTOCOL_SPEC_PATH
    return json.loads(spec_path.read_text())
