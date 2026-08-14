"""Frozen Protocol D2a autonomous H_K relaxation specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_d_biological_rbs.d0_protocol import PROTOCOL_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
D2A_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d2a_autonomous_h_k_relaxation_spec.json"
D2A_EXECUTION_RECEIPT_PATH = (
    _REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d2a_autonomous_relaxation_receipt.json"
)


def load_d2a_spec(path: Path | None = None) -> dict[str, Any]:
    spec_path = path or D2A_SPEC_PATH
    return json.loads(spec_path.read_text())


def d2a_h_k0_values(spec: dict[str, Any] | None = None) -> tuple[float, ...]:
    spec = spec or load_d2a_spec()
    return tuple(float(v) for v in spec["initial_conditions"]["H_K0_values"])


def validate_d2a_spec(spec: dict[str, Any] | None = None) -> None:
    spec = spec or load_d2a_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("D2a spec must have status FROZEN")
    if spec.get("checkpoint") != "D2a":
        raise ValueError("D2a spec must have checkpoint D2a")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    dyn = spec["dynamics_F_H"]
    if dyn.get("kappa_K") != 0:
        raise ValueError("D2a requires kappa_K=0")
    if dyn.get("family") != "F1_linear":
        raise ValueError("D2a requires F1_linear dynamics")

    sim = spec["simulation_policy"]
    n_steps = int(round(sim["duration_ms"] / sim["dt_ms"]))
    if n_steps != int(sim["n_steps"]):
        raise ValueError("duration_ms/dt_ms inconsistent with n_steps")
    if float(sim["tau_k_ms"]) <= 0:
        raise ValueError("tau_k_ms must be positive")

    if not spec["execution_authorization"].get("implementation_authorized"):
        raise ValueError("D2a spec must authorize implementation")

    if spec["d2b_deferred"].get("status") != "specified_not_authorized":
        raise ValueError("D2b must remain specified_not_authorized at D2a")
