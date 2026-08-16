"""Frozen Protocol D1 static H_K expression specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_d_biological_rbs.d0_protocol import PROTOCOL_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
D1_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d1_static_h_k_expression_spec.json"
D1_EXECUTION_RECEIPT_PATH = (
    _REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d1_static_expression_receipt.json"
)


def load_d1_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen D1 static expression specification."""
    spec_path = path or D1_SPEC_PATH
    return json.loads(spec_path.read_text())


def d1_h_k_sweep_values(spec: dict[str, Any] | None = None) -> tuple[float, ...]:
    """Return preregistered static H_K sweep levels."""
    spec = spec or load_d1_spec()
    return tuple(float(v) for v in spec["static_sweep"]["values"])


def validate_d1_spec(spec: dict[str, Any] | None = None) -> None:
    """Raise ValueError when required D1 contract fields are missing or inconsistent."""
    spec = spec or load_d1_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("D1 spec must have status FROZEN")
    if spec.get("checkpoint") != "D1":
        raise ValueError("D1 spec must have checkpoint D1")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    impl = spec["implementation"]
    if impl.get("dH_K_dt") != 0:
        raise ValueError("D1 requires dH_K/dt=0")
    if impl.get("dot_W") != 0:
        raise ValueError("D1 requires dot_W=0")
    if impl.get("coupling") != "b_eff = H_K * b":
        raise ValueError("D1 coupling must be b_eff = H_K * b")

    sweep = spec["static_sweep"]
    delta = float(sweep["delta"])
    values = [float(v) for v in sweep["values"]]
    expected = [1.0 - delta, 1.0, 1.0 + delta]
    if values != expected:
        raise ValueError(f"static sweep values must be {expected}, got {values}")

    sim = spec["simulation_policy"]
    n_steps = int(round(sim["duration_ms"] / sim["dt_ms"]))
    if n_steps != int(sim["n_steps"]):
        raise ValueError("duration_ms/dt_ms inconsistent with n_steps")
    if float(sim["noise_scale"]) != 0.0:
        raise ValueError("D1 requires noise_scale=0 for containment gate")

    if not spec["execution_authorization"].get("implementation_authorized"):
        raise ValueError("D1 spec must authorize implementation")

    for gate in ("G1_containment", "G2_static_state_integrity", "G3_parameter_locality", "G4_bidirectional_sensitivity"):
        if gate not in spec["gates"]:
            raise ValueError(f"missing gate {gate}")
