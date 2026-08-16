"""Frozen Protocol C3 neural geometry/delay experiment specification (spec only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROTOCOL_ID = "protocol_c_wave_v0417"
_REPO_ROOT = Path(__file__).resolve().parents[2]
C3_SPEC_PATH = _REPO_ROOT / "artifacts" / "protocol_c" / "c3_neural_experiment_spec.json"


def load_c3_spec(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen C3 neural experiment specification."""
    spec_path = path or C3_SPEC_PATH
    return json.loads(spec_path.read_text())


def c3_condition_ids(spec: dict[str, Any] | None = None) -> tuple[str, ...]:
    """Return preregistered condition identifiers in frozen order."""
    spec = spec or load_c3_spec()
    return tuple(c["id"] for c in spec["design_matrix"]["conditions"])


def c3_total_cells(spec: dict[str, Any] | None = None) -> int:
    """Number of prospective seed × condition cells."""
    spec = spec or load_c3_spec()
    n_cond = int(spec["design_matrix"]["n_conditions"])
    n_seeds = int(spec["simulation_policy"]["n_seeds"])
    return n_cond * n_seeds


def validate_c3_spec(spec: dict[str, Any] | None = None) -> None:
    """Raise ValueError when required C3 contract fields are missing or inconsistent."""
    spec = spec or load_c3_spec()
    if spec.get("status") != "FROZEN":
        raise ValueError("C3 spec must have status FROZEN")
    runtime = spec["runtime_contract"]
    if runtime.get("enable_hdp") is not False:
        raise ValueError("C3 requires enable_hdp=false")
    if runtime.get("enable_rbd") is not False:
        raise ValueError("C3 requires enable_rbd=false")
    conditions = spec["design_matrix"]["conditions"]
    if len(conditions) != int(spec["design_matrix"]["n_conditions"]):
        raise ValueError("design_matrix.n_conditions mismatch")
    ids = {c["id"] for c in conditions}
    if len(ids) != len(conditions):
        raise ValueError("duplicate condition ids")
    for policy in ("delay_shuffled",):
        for c in conditions:
            if c.get("delay_policy") == policy and "delay_shuffle_reference" not in c:
                raise ValueError(f"{c['id']}: delay_shuffled requires delay_shuffle_reference")
    n_steps = int(round(spec["simulation_policy"]["duration_ms"] / spec["simulation_policy"]["dt_ms"]))
    if n_steps != int(spec["simulation_policy"]["n_steps"]):
        raise ValueError("duration_ms/dt_ms inconsistent with n_steps")
    if not spec["execution_authorization"].get("prospective_run_authorized") is False:
        raise ValueError("C3 spec must freeze prospective_run_authorized=false at specification")
