"""Frozen Protocol E0 integrated composition specification (spec only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_ID = "protocol_e_integration_v0417"
_REPO_ROOT = Path(__file__).resolve().parents[2]
E0_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e0_composition_spec.json"


def load_e0_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or E0_SPEC_PATH
    return json.loads(spec_path.read_text())


def validate_e0_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_e0_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("E0 spec must have status FROZEN")
    if spec.get("checkpoint") != "E0":
        raise ValueError("E0 spec must have checkpoint E0")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")
    if spec.get("experiment_type") != "composition_not_phenotype_manufacturing":
        raise ValueError("E0 must be composition_not_phenotype_manufacturing")

    blocks = spec["explicit_blocks"]
    if blocks["W3_closed_loop_HDP"]["status"] != "unresolved":
        raise ValueError("W3 must remain unresolved")
    if blocks["D4_second_RBS_class"]["status"] != "not_authorized":
        raise ValueError("D4 must not be authorized in E0")
    if blocks["D3_adaptation_requirement"]["status"] != "NO_ADAPTATION_frozen":
        raise ValueError("D3 NO_ADAPTATION must be frozen")

    if not spec["execution_authorization"].get("specification_only"):
        raise ValueError("E0 must be specification_only at freeze")
    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("E0 must not authorize implementation at specification")

    banned = set(spec["explicit_prohibitions"])
    if "no_closed_loop_HDP" not in banned:
        raise ValueError("must prohibit closed-loop HDP")
    if "no_D4_authorization" not in banned:
        raise ValueError("must prohibit D4 authorization")
