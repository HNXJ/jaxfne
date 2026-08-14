"""Frozen Protocol D3 adaptation/recovery phenotype specification (spec only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jaxfne.protocol_d_biological_rbs.d0_protocol import PROTOCOL_ID

_REPO_ROOT = Path(__file__).resolve().parents[2]
D3_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_d_biological_rbs" / "d3_adaptation_recovery_phenotype_spec.json"


def load_d3_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen D3 adaptation/recovery phenotype specification."""
    spec_path = path or D3_SPEC_PATH
    return json.loads(spec_path.read_text())


def d3_null_arm_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    spec = spec or load_d3_spec()
    return tuple(arm["id"] for arm in spec["null_hierarchy"]["arms"])


def d3_recovery_interval_ms(spec: dict[str, Any] | None = None) -> tuple[float, ...]:
    spec = spec or load_d3_spec()
    return tuple(float(level["T_recovery_ms"]) for level in spec["recovery_intervals"]["levels"])


def validate_d3_spec(spec: dict[str, Any] | None = None) -> None:
    """Raise ValueError when required D3 contract fields are missing or inconsistent."""
    spec = spec or load_d3_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("D3 spec must have status FROZEN")
    if spec.get("checkpoint") != "D3":
        raise ValueError("D3 spec must have checkpoint D3")
    if spec.get("protocol_id") != PROTOCOL_ID:
        raise ValueError(f"protocol_id must be {PROTOCOL_ID!r}")

    if "fatigue" in spec.get("phenomenon_label", "").lower():
        raise ValueError("D3 phenomenon_label must not use fatigue")
    avoid = spec.get("terminology", {}).get("avoid_as_formal_mechanism", [])
    if "fatigue" not in avoid:
        raise ValueError("D3 must list fatigue in avoid_as_formal_mechanism")

    labels = set(spec["classification"]["labels"])
    if labels != {"ADAPTATION", "NO_ADAPTATION", "UNRESOLVED"}:
        raise ValueError("D3 classification labels must be ADAPTATION, NO_ADAPTATION, UNRESOLVED")

    arms = {a["id"] for a in spec["null_hierarchy"]["arms"]}
    if arms != {"N0", "N1", "N2", "D"}:
        raise ValueError("D3 null hierarchy must be N0, N1, N2, D")

    if spec["null_hierarchy"]["primary_contrast"]["id"] != "D_minus_N2":
        raise ValueError("D3 primary contrast must be D_minus_N2")

    levels = spec["recovery_intervals"]["levels"]
    if len(levels) != 3:
        raise ValueError("D3 requires three prospective recovery intervals")
    tau_a = float(spec["recovery_intervals"]["tau_A_ms"])
    tau_k = float(spec["recovery_intervals"]["tau_K_ms"])
    t_short = float(levels[0]["T_recovery_ms"])
    t_med = float(levels[1]["T_recovery_ms"])
    t_long = float(levels[2]["T_recovery_ms"])
    if not (t_short < t_med < t_long):
        raise ValueError("recovery intervals must be strictly increasing")
    if abs(t_short - 2.0 * tau_a) > 1e-6:
        raise ValueError("short recovery interval must equal 2*tau_A")
    if abs(t_med - tau_k) > 1e-6:
        raise ValueError("medium recovery interval must equal tau_K")

    sim = spec["simulation_policy"]
    n_steps = int(round(sim["duration_ms"] / sim["dt_ms"]))
    if n_steps != int(sim["n_steps"]):
        raise ValueError("duration_ms/dt_ms inconsistent with n_steps")

    n_cells = int(sim["cell_grid"]["n_cells"])
    n_expected = len(sim["seeds"]) * len(levels) * len(arms)
    if n_cells != n_expected:
        raise ValueError(f"cell_grid n_cells must be {n_expected}")

    if not spec["execution_authorization"].get("specification_only"):
        raise ValueError("D3 must be specification_only at freeze")
    if spec["execution_authorization"].get("implementation_authorized") is not False:
        raise ValueError("D3 must not authorize implementation at specification")

    banned = set(spec["explicit_prohibitions"])
    if "no_D3_implementation_in_this_checkpoint" not in banned:
        raise ValueError("must prohibit D3 implementation in specification checkpoint")
    if "no_fatigue_as_formal_mechanism_label" not in banned:
        raise ValueError("must prohibit fatigue as formal mechanism label")
