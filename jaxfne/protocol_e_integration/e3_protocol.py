"""Frozen Protocol E3 RBS composition specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_e_integration.e0_protocol import PROTOCOL_ID
from jaxfne.protocol_e_integration.e1_protocol import E1_EXECUTION_RECEIPT_PATH

_REPO_ROOT = Path(__file__).resolve().parents[2]
E3_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e3_rbs_composition_spec.json"


def load_e3_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or E3_SPEC_PATH
    return json.loads(spec_path.read_text())


def e3_gate_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e3_spec()
    return tuple(spec["gates"])


def e3_owner_flat_indices(spec: dict[str, Any] | None = None) -> tuple[int, ...]:
    spec = spec or load_e3_spec()
    return tuple(int(i) for i in spec["ownership"]["flat_indices"])


def resolve_owner_indices_from_e1_identity(
    spec: dict[str, Any] | None = None,
    *,
    e1_receipt: dict[str, Any] | None = None,
) -> list[int]:
    """Resolve owner flat indices from the frozen E1 identity table."""
    spec = spec or load_e3_spec()
    selector = spec["ownership"]["selector"]
    receipt = e1_receipt or json.loads(E1_EXECUTION_RECEIPT_PATH.read_text())
    rows = receipt["identity_map"]
    matched = [
        int(row["flat_index"])
        for row in rows
        if row["area"] == selector["area"]
        and row["layer"] == selector["layer"]
        and row["cell_type"] == selector["cell_type"]
    ]
    return sorted(matched)


def validate_e3_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_e3_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("E3 spec must have status FROZEN")
    if spec.get("checkpoint") != "E3":
        raise ValueError("E3 spec must have checkpoint E3")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    if spec["reduction_contract"]["id"] != "R_E3_to_E2":
        raise ValueError("E3 must declare R_E3_to_E2 reduction contract")

    primitive = spec["rbs_primitive"]
    if "D2b" not in primitive["deferred_not_used"]:
        raise ValueError("E3 must defer D2b explicitly")
    if primitive["coupling_map"]["display"] != "b_eff = H_K * b":
        raise ValueError("E3 must use D1 coupling map b_eff = H_K * b")
    if int(primitive["coupling_map"]["dot_W"]) != 0:
        raise ValueError("E3 requires dot_W = 0")

    owner = spec["ownership"]
    if owner["n_nodes"] < 1:
        raise ValueError("E3 owner population must have at least one node")
    frozen_indices = e3_owner_flat_indices(spec)
    if len(frozen_indices) != int(owner["n_nodes"]):
        raise ValueError("ownership n_nodes must match flat_indices length")

    resolved = resolve_owner_indices_from_e1_identity(spec)
    if resolved != list(frozen_indices):
        raise ValueError(
            "E3 owner flat_indices must match E1 identity_map lookup "
            f"(expected {resolved}, got {list(frozen_indices)})"
        )

    modes = {m["id"] for m in spec["execution_modes"]}
    if modes != {"E3-null", "E3-dynamic"}:
        raise ValueError("E3 requires E3-null and E3-dynamic modes")

    tau_k = float(spec["rbs_primitive"]["dynamics_F_H"]["dynamic_mode"]["tau_K_ms"])
    if tau_k <= 0.0:
        raise ValueError("tau_K_ms must be positive")
    if float(tau_k) != 100.0:
        raise ValueError("E3 must inherit D2a tau_K_ms=100")

    gates = e3_gate_ids(spec)
    if len(gates) != 9:
        raise ValueError("E3 requires exactly nine gates G1–G9")
    if gates[0] != "G1_e2_reduction":
        raise ValueError("G1 must be E2 reduction gate")
    if gates[-1] != "G9_no_phenotype_claim":
        raise ValueError("G9 must prohibit phenotype claims")

    if not spec["execution_authorization"].get("specification_only"):
        raise ValueError("E3 must be specification_only at freeze")
    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("E3 must not authorize implementation at specification freeze")

    banned = set(spec["explicit_prohibitions"])
    if "no_D2b_H_A_activity_writing" not in banned:
        raise ValueError("must prohibit D2b activity writing at E3")
    if "no_hierarchy_modification_for_RBS_ownership" not in banned:
        raise ValueError("must prohibit hierarchy modification for RBS ownership")
