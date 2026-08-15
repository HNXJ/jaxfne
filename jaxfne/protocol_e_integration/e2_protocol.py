"""Frozen Protocol E2 typed delayed-coupling specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_e_integration.e0_protocol import PROTOCOL_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
E2_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_e_integration" / "e2_delayed_coupling_spec.json"


def load_e2_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or E2_SPEC_PATH
    return json.loads(spec_path.read_text())


def e2_delay_class_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e2_spec()
    return tuple(row["edge_class"] for row in spec["delay_values"]["classes"])


def e2_gate_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_e2_spec()
    return tuple(spec["gates"])


def validate_e2_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_e2_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("E2 spec must have status FROZEN")
    if spec.get("checkpoint") != "E2":
        raise ValueError("E2 spec must have checkpoint E2")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    if spec["reduction_contract"]["id"] != "R_E2_to_E1":
        raise ValueError("E2 must declare R_E2_to_E1 reduction contract")

    p_local = spec["e1_receipt_derived_constants"]["p_local"]
    if float(p_local["value"]) != 0.2:
        raise ValueError("E2 must freeze p_local=0.2 from E1 receipt")
    if "E1-receipt-derived" not in p_local.get("provenance", ""):
        raise ValueError("p_local must document E1-receipt provenance")

    classes = e2_delay_class_ids(spec)
    if classes != ("local_A1", "local_A2", "FF_A1_to_A2", "FB_A2_to_A1"):
        raise ValueError("E2 delay classes must match E1 provenance classes")

    delay_rows = spec["delay_values"]["classes"]
    tau_by_class = {row["edge_class"]: float(row["tau_ms"]) for row in delay_rows}
    tau_local = max(tau_by_class["local_A1"], tau_by_class["local_A2"])
    tau_ff = tau_by_class["FF_A1_to_A2"]
    tau_fb = tau_by_class["FB_A2_to_A1"]
    if not (tau_local <= tau_ff < tau_fb):
        raise ValueError("E2 requires tau_local <= tau_FF < tau_FB")
    if tau_ff <= 0.0 or tau_fb <= 0.0:
        raise ValueError("E2 nonzero-delay realization requires tau_FF, tau_FB > 0")

    dt_ms = float(spec["delay_values"]["dt_ms"])
    for row in delay_rows:
        expected_steps = int(round(float(row["tau_ms"]) / dt_ms))
        if int(row["delay_steps"]) != expected_steps:
            raise ValueError(f"delay_steps mismatch for {row['edge_class']}")

    gates = e2_gate_ids(spec)
    if len(gates) != 8:
        raise ValueError("E2 requires exactly eight gates G1–G8")
    if gates[0] != "G1_e1_reduction":
        raise ValueError("G1 must be E1 reduction gate")
    if gates[-1] != "G8_no_scientific_overinterpretation":
        raise ValueError("G8 must prohibit scientific overinterpretation")

    if not spec["execution_authorization"].get("specification_only"):
        raise ValueError("E2 must be specification_only at freeze")
    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("E2 must not authorize implementation at specification freeze")

    banned = set(spec["explicit_prohibitions"])
    if "no_RBS_at_E2" not in banned:
        raise ValueError("must prohibit RBS at E2")
    if "no_retroactive_p_local_as_E1_preregistration" not in banned:
        raise ValueError("must prohibit retroactive p_local as E1 preregistration")
