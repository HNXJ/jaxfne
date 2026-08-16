"""Frozen Protocol E5 causal perturbation specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_e_integration.e0_protocol import PROTOCOL_ID
from jaxfne.protocol_e_integration.e3_protocol import e3_owner_flat_indices, load_e3_spec

_REPO_ROOT = Path(__file__).resolve().parents[2]
E5_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e5_causal_perturbation_spec.json"
E5_EXECUTION_RECEIPT_PATH = (
    _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e5_execution_receipt.json"
)
E5_INTERPRETATION_RECEIPT_PATH = (
    _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e5_interpretation_receipt.json"
)


def load_e5_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or E5_SPEC_PATH
    return json.loads(spec_path.read_text())


def e5_gate_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e5_spec()
    return tuple(spec["gates"])


def e5_arm_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e5_spec()
    return tuple(str(arm["id"]) for arm in spec["experimental_arms"])


def e5_result_classes(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e5_spec()
    return tuple(spec["result_classification"]["enum"])


def validate_e5_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_e5_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("E5 spec must have status FROZEN")
    if spec.get("checkpoint") != "E5":
        raise ValueError("E5 spec must have checkpoint E5")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    arms = e5_arm_ids(spec)
    if arms != ("N0", "N1", "D"):
        raise ValueError("E5 must freeze arms N0, N1, D")

    arm_by_id = {arm["id"]: arm for arm in spec["experimental_arms"]}
    if arm_by_id["N0"]["H_K_initial_on_O_H"] != 1.0:
        raise ValueError("N0 must use H_K=1 reference")
    if arm_by_id["N1"]["Gamma_H"] != "identity_disabled":
        raise ValueError("N1 must disable Gamma_H")
    if arm_by_id["D"]["Gamma_H"] != "enabled":
        raise ValueError("D must enable Gamma_H")
    if float(arm_by_id["N1"]["H_K_initial_on_O_H"]) != 1.2:
        raise ValueError("N1 must use H_K(0)=1.2 on owners")
    if float(arm_by_id["D"]["H_K_initial_on_O_H"]) != float(arm_by_id["N1"]["H_K_initial_on_O_H"]):
        raise ValueError("N1 and D must share H_K initial condition")

    contrast = spec["causal_contrast"]
    if contrast["primary"] != "D - N1":
        raise ValueError("E5 primary contrast must be D - N1")

    owner = spec["perturbation_target"]
    e3_owners = e3_owner_flat_indices(load_e3_spec())
    frozen = tuple(int(i) for i in owner["flat_indices"])
    if frozen != e3_owners:
        raise ValueError("E5 owner flat_indices must match frozen E3 ownership")

    classes = e5_result_classes(spec)
    if classes != ("NO_EFFECT", "LOCAL_EXPRESSION", "HIERARCHICAL_PROPAGATION", "UNRESOLVED"):
        raise ValueError("E5 result classification enum mismatch")

    if spec["success_criteria"].get("not_required") is None:
        raise ValueError("E5 must declare outcomes not required for close")
    if "HIERARCHICAL_PROPAGATION" not in spec["success_criteria"]["not_required"]:
        raise ValueError("HIERARCHICAL_PROPAGATION must not be required for E5 close")

    sim = spec["simulation_policy"]
    if [int(s) for s in sim["seeds"]] != [11, 12, 13]:
        raise ValueError("E5 must inherit seeds [11, 12, 13]")
    if "do not add seeds" not in sim.get("sample_size_policy", "").lower():
        raise ValueError("E5 must freeze sample size policy")

    gates = e5_gate_ids(spec)
    if len(gates) != 10:
        raise ValueError("E5 requires exactly ten gates G1–G10")

    arch = spec["scientific_scope"].get("architecture_rule", "")
    if "zero" not in arch.lower() and "only" not in arch.lower():
        raise ValueError("E5 must declare zero-architecture evidence rule")

    if not spec["execution_authorization"].get("specification_only"):
        raise ValueError("E5 must be specification_only at freeze")
    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("E5 must not authorize implementation at specification freeze")

    banned = set(spec["explicit_prohibitions"])
    if "no_new_TFNE_architecture" not in banned:
        raise ValueError("must prohibit new TFNE architecture at E5")
    if "no_spectral_analysis_pipeline" not in banned:
        raise ValueError("must prohibit spectral pipeline at E5")
    if "no_E6_checkpoint" not in banned:
        raise ValueError("must prohibit E6 checkpoint")
