"""Frozen Protocol E1 hierarchy/runtime specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_e_integration.e0_protocol import PROTOCOL_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
E1_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e1_hierarchy_runtime_spec.json"


def load_e1_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or E1_SPEC_PATH
    return json.loads(spec_path.read_text())


def validate_e1_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_e1_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("E1 spec must have status FROZEN")
    if spec.get("checkpoint") != "E1":
        raise ValueError("E1 spec must have checkpoint E1")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    if spec["rbs"].get("enabled") is not False:
        raise ValueError("E1 must not enable RBS")
    if int(spec["connectivity"]["delay_steps"]) != 0:
        raise ValueError("E1 requires zero delay")

    areas = spec["hierarchy"]["areas"]
    if len(areas) != 2:
        raise ValueError("E1 requires exactly two areas")

    if spec["hierarchy"]["heterogeneity_semantics"] != "population_parameter_heterogeneity_not_different_emitter_equations":
        raise ValueError("E1 heterogeneity semantics must be frozen")

    if not spec["execution_authorization"].get("implementation_authorized"):
        raise ValueError("E1 must authorize implementation")

    banned = set(spec["explicit_prohibitions"])
    if "no_RBS_at_E1" not in banned:
        raise ValueError("must prohibit RBS at E1")
    if "no_nonzero_edge_delays" not in banned:
        raise ValueError("must prohibit nonzero delays at E1")
