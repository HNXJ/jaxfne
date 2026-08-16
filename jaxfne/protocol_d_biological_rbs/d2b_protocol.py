"""Frozen Protocol D2b activity-to-H_K coupling specification (spec only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_d_biological_rbs.d0_protocol import PROTOCOL_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
D2B_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d2b_activity_h_k_coupling_spec.json"
D2B_EXECUTION_RECEIPT_PATH = (
    _REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d2b_implementation_receipt.json"
)


def load_d2b_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen D2b two-coordinate RBS specification."""
    spec_path = path or D2B_SPEC_PATH
    return json.loads(spec_path.read_text())


def validate_d2b_spec(spec: dict[str, Any] | None = None) -> None:
    """Raise ValueError when required D2b contract fields are missing or inconsistent."""
    spec = spec or load_d2b_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("D2b spec must have status FROZEN")
    if spec.get("checkpoint") != "D2b":
        raise ValueError("D2b spec must have checkpoint D2b")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    coords = {c["id"]: c for c in spec["rbs_vector"]["coordinates"]}
    if coords["H_A"]["reference"] != 0:
        raise ValueError("H_A reference must be 0")
    if coords["H_K"]["reference"] != 1:
        raise ValueError("H_K reference must be 1")
    if coords["H_A"].get("is_ion") is not False:
        raise ValueError("H_A must not be classified as an ion")

    ts = spec["dynamics"]["timescales"]
    if float(ts["tau_A_ms"]) >= float(ts["tau_K_ms"]):
        raise ValueError("D2b requires tau_A < tau_K")
    if float(spec["dynamics"]["coupling_constant"]["value"]) <= 0:
        raise ValueError("D2b primary candidate requires kappa_AK > 0")

    ordering = spec["dynamics"]["discrete_update_ordering"]["steps"]
    if "H_A^n" not in ordering[1] or "H_K^n" not in ordering[1]:
        raise ValueError("H_K update must use H_A^n causal ordering")

    if not spec["execution_authorization"].get("specification_only"):
        raise ValueError("D2b must be specification_only at freeze")
    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("D2b must not authorize implementation at specification")

    banned = set(spec["explicit_prohibitions"])
    if "no_preregister_H_K_up_implies_firing_down" not in banned:
        raise ValueError("must prohibit preregistered firing-down phenotype")
