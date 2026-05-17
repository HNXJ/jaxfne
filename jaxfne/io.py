"""I/O, hashing, strict JSON, and manifest helpers."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


def json_safe(obj: Any) -> Any:
    """Convert common scientific Python/JAX objects into strict JSON values."""

    if dataclasses.is_dataclass(obj):
        return json_safe(dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if hasattr(obj, "shape") and hasattr(obj, "tolist"):
        return json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def config_hash(cfg: Any) -> str:
    """Return a compact SHA256 hash for a configuration-like object."""

    payload = json.dumps(json_safe(cfg), sort_keys=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def sha256_text(text: str) -> str:
    """Return SHA256 for a text payload."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return SHA256 for a file."""

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(
    cfg: Any,
    signals: Optional[Any] = None,
    readout: Optional[dict[str, Any]] = None,
    runtime_config: Optional[Any] = None,
    paradigm: Optional[dict[str, Any]] = None,
    objective: Optional[dict[str, Any]] = None,
    evaluation: Optional[dict[str, Any]] = None,
    tuning: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a strict JSON-safe run manifest.

    Optional v0.0.5 arguments (paradigm, objective, evaluation, tuning) extend the
    manifest without changing the base schema version.  All v0.0.4 truth gates are
    always emitted.  Objective/evaluation/tuning sections carry explicit claim labels
    so downstream readers cannot mistake them for empirical validation results.
    """
    from .fields import validate_source_field_status

    cfg_metadata = dict(getattr(cfg, "metadata", {}) or {})
    data: dict[str, Any] = {
        "package": "jaxfne",
        "manifest_schema_version": cfg_metadata.get("manifest_schema_version", "0.0.4"),
        "truth_mode": cfg_metadata.get("truth_mode", "truth_safe_unverified"),
        "claim_level": cfg_metadata.get("claim_level", "computational_scaffold"),
        "source_calibration_status": cfg_metadata.get(
            "source_calibration_status", "uncalibrated_izhikevich_native_current"
        ),
        "source_projection_mode": cfg_metadata.get("source_projection_mode", "proxy_no_field_solve"),
        "source_decomposition": cfg_metadata.get("source_decomposition", "proxy_reduced_emitter"),
        "boundary_condition": cfg_metadata.get("boundary_condition", "mean_zero_neumann"),
        "gauge": cfg_metadata.get("gauge", "mean_zero"),
        "csd_sign_convention": cfg_metadata.get(
            "csd_sign_convention", "proxy_positive_equals_extracellular_source_like"
        ),
        "field_solver_status": cfg_metadata.get("field_solver_status", "laminar_proxy_no_pde"),
        "operator_status": cfg_metadata.get("operator_status", {}),
        "config_hash": config_hash(cfg),
    }
    if runtime_config is not None:
        data["runtime"] = runtime_config.runtime_report()
        data["runtime_report"] = data["runtime"]  # v0.0.3 compatibility
    if signals is not None:
        data["signals"] = {
            "n_time": int(signals.time_ms.shape[0]),
            "n_neurons": int(signals.V_m.shape[1]),
            "has_field": signals.field is not None,
            "metadata": signals.metadata,
            "dtype": str(signals.V_m.dtype),
        }
        if signals.field is not None:
            requested_modes = []
            if readout is not None:
                requested_modes = list(readout.get("requested_modes", []))
            data["field_diagnostics"] = signals.field.diagnostics
            data["source_field_status"] = validate_source_field_status(
                signals.field, cfg_metadata, requested_modes=requested_modes
            )
    # v0.0.5 optional extension blocks — always include claim labels to prevent
    # downstream misreading of computational reports as empirical validation.
    _v005_present = any(x is not None for x in (paradigm, objective, evaluation, tuning))
    if _v005_present:
        data["v005_claim_labels"] = {
            "objective_status": "computational_diagnostic",
            "tuning_status": "metadata_only_v0.0.5",
            "optimizer_claim_level": "metadata_only",
            "empirical_validation_status": "not_empirically_validated",
            "mechanism_claim_status": "not_claimed",
            "field_claim_level": "proxy_readout_only",
            "physical_amplitude_claim_allowed": False,
        }
    if paradigm is not None:
        data["paradigm"] = paradigm
    if objective is not None:
        data["objective"] = objective
    if evaluation is not None:
        data["evaluation"] = evaluation
    if tuning is not None:
        data["tuning"] = tuning
    return json_safe(data)


def save_json(obj: Any, path: str | Path) -> None:
    """Save strict JSON with ``allow_nan=False``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def load_json(path: str | Path) -> Any:
    """Load JSON from disk."""

    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

# Backwards-compatible name from v0.0.1.
hash_file = sha256_file
