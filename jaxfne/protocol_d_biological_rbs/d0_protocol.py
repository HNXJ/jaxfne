"""Frozen Protocol D0 biological RBS specification (spec only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_ID = "protocol_d_biological_rbs_v0417"
_REPO_ROOT = Path(__file__).resolve().parents[2]
D0_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d0_intrinsic_ionic_rbs_spec.json"


def load_d0_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen D0 biological RBS specification."""
    spec_path = path or D0_SPEC_PATH
    return json.loads(spec_path.read_text())


def d0_first_coordinate_id(spec: dict[str, Any] | None = None) -> str:
    """Return the coordinate authorized for D1 implementation."""
    spec = spec or load_d0_spec()
    for coord in spec["rbs_vector_grammar"]["typed_coordinates"]:
        if coord.get("d1_implementation"):
            return str(coord["id"])
    raise ValueError("no D1 coordinate declared in rbs_vector_grammar")


def d0_static_sweep_values(spec: dict[str, Any] | None = None) -> tuple[float, ...]:
    """Return preregistered static H_K sweep levels."""
    spec = spec or load_d0_spec()
    values = spec["static_sweep_before_dynamics"]["frozen_levels"]["values"]
    return tuple(float(v) for v in values)


def validate_d0_spec(spec: dict[str, Any] | None = None) -> None:
    """Raise ValueError when required D0 contract fields are missing or inconsistent."""
    spec = spec or load_d0_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("D0 spec must have status FROZEN")
    if spec.get("checkpoint") != "D0":
        raise ValueError("D0 spec must have checkpoint D0")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    runtime = spec["runtime_contract"]
    if runtime.get("enable_hdp") is not False:
        raise ValueError("D0 requires enable_hdp=false")
    if runtime.get("dot_W") != 0:
        raise ValueError("D0 requires dot_W=0")

    coords = spec["rbs_vector_grammar"]["typed_coordinates"]
    d1_coords = [c for c in coords if c.get("d1_implementation")]
    if len(d1_coords) != 1:
        raise ValueError("D0 must declare exactly one D1 coordinate")
    if d1_coords[0]["id"] != "H_K":
        raise ValueError("D1 coordinate must be H_K per frozen specification")

    sweep = spec["static_sweep_before_dynamics"]["frozen_levels"]
    delta = float(sweep["delta"])
    values = [float(v) for v in sweep["values"]]
    expected = [1.0 - delta, 1.0, 1.0 + delta]
    if values != expected:
        raise ValueError(f"static sweep values must be {expected}, got {values}")

    ladder = spec["checkpoint_ladder"]
    if ladder["D4"].get("mandatory_for_0417") is not False:
        raise ValueError("D4 must not be mandatory for 0.4.17")
    for ck in ("D1", "D2", "D3", "D4"):
        if ladder[ck].get("implementation_authorized") is not False:
            raise ValueError(f"{ck} must freeze implementation_authorized=false at D0")

    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("D0 must freeze implementation_authorized=false at specification")

    h_k = spec["first_coordinate_H_K"]
    is_not = set(h_k["physical_interpretation"]["is_not"])
    for forbidden in (
        "potassium concentration",
        "STDP or synaptic plasticity coordinate",
        "neurotransmitter availability",
    ):
        if forbidden not in is_not:
            raise ValueError(f"H_K is_not must include {forbidden!r}")
