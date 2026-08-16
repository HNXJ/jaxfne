"""Frozen Protocol E0.1 implementation ladder and reduction contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_e_integration.e0_protocol import PROTOCOL_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
E0_1_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e0_1_implementation_ladder_spec.json"


def load_e0_1_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or E0_1_SPEC_PATH
    return json.loads(spec_path.read_text())


def e0_1_ladder_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e0_1_spec()
    return tuple(step["id"] for step in spec["implementation_ladder"]["ordered_checkpoints"])


def validate_e0_1_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_e0_1_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("E0.1 spec must have status FROZEN")
    if spec.get("checkpoint") != "E0.1":
        raise ValueError("E0.1 spec must have checkpoint E0.1")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    ladder = e0_1_ladder_ids(spec)
    if ladder != ("E1", "E2", "E3", "E4", "E5"):
        raise ValueError("E0.1 ladder must be E1..E5")

    reductions = {r["id"] for r in spec["reduction_contracts"]}
    if reductions != {"R_E2_to_E1", "R_E3_to_E2", "R_E4_to_E3"}:
        raise ValueError("E0.1 reduction contracts incomplete")

    if not spec["integration_monotonicity_principle"]["explicit_reduction_tests_required"]:
        raise ValueError("reduction tests must be required")

    if spec["authorization_policy"]["E1"] != "implementation_authorized_after_E0_1_freeze":
        raise ValueError("E1 authorization policy mismatch")

    if not spec["execution_authorization"].get("specification_only"):
        raise ValueError("E0.1 must be specification_only at freeze")
    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("E0.1 must not authorize its own implementation")

    banned = set(spec["explicit_prohibitions"])
    if "no_monolithic_E_implementation" not in banned:
        raise ValueError("must prohibit monolithic E implementation")
