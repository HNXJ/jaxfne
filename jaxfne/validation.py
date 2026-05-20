"""Field admissibility validation for v0.2.0 theoretical gates.

This module provides validation functions for conductivity tensors, field finiteness,
source conservation, gauge/boundary metadata, and JSON-safe field admissibility reports.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np


def _to_numpy(arr: Any) -> np.ndarray:
    """Convert JAX array or similar to numpy for numerical operations."""
    if hasattr(arr, "toarray"):  # scipy sparse
        return arr.toarray()
    if hasattr(arr, "tolist"):
        return np.asarray(arr.tolist())
    return np.asarray(arr)


def _is_finite_value(val: Any) -> bool:
    """Check if a value is finite (not NaN, not Inf)."""
    if val is None:
        return True  # None is acceptable in optional fields
    if isinstance(val, bool):
        return True
    if isinstance(val, (int, str)):
        return True
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return True


def validate_scalar_conductivity(
    sigma: Any, *, tolerance: float = 1e-10
) -> dict[str, Any]:
    """Validate scalar conductivity σ.

    Returns:
        dict with keys: is_valid, is_positive, is_finite, value, status, evidence
    """
    try:
        sigma_float = float(sigma)
    except (TypeError, ValueError):
        return {
            "is_valid": False,
            "is_positive": False,
            "is_finite": False,
            "value": None,
            "status": "not_numeric",
            "evidence": f"Cannot convert to float: {type(sigma)}",
        }

    is_finite = math.isfinite(sigma_float)
    is_positive = sigma_float > tolerance if is_finite else False
    is_valid = is_finite and is_positive

    return {
        "is_valid": is_valid,
        "is_positive": is_positive,
        "is_finite": is_finite,
        "value": sigma_float if is_finite else None,
        "status": "passed" if is_valid else ("non_positive" if is_finite else "non_finite"),
        "evidence": None if is_valid else f"sigma={sigma_float}",
    }


def validate_diagonal_conductivity(
    sigma: Any, *, tolerance: float = 1e-10
) -> dict[str, Any]:
    """Validate diagonal conductivity tensor σ = diag([σ_x, σ_y, σ_z]).

    Returns:
        dict with keys: is_valid, is_spd, is_finite, diag_values, min_eigenvalue, status, evidence
    """
    try:
        arr = _to_numpy(sigma)
    except (TypeError, ValueError):
        return {
            "is_valid": False,
            "is_spd": False,
            "is_finite": False,
            "diag_values": None,
            "min_eigenvalue": None,
            "status": "not_convertible_to_array",
            "evidence": f"Cannot convert to array: {type(sigma)}",
        }

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return {
            "is_valid": False,
            "is_spd": False,
            "is_finite": False,
            "diag_values": None,
            "min_eigenvalue": None,
            "status": "not_square_matrix",
            "evidence": f"Shape {arr.shape}; expected (N, N)",
        }

    # Check if diagonal by verifying off-diagonal elements are near zero
    off_diag = arr - np.diag(np.diag(arr))
    max_off_diag = float(np.max(np.abs(off_diag)))
    is_diagonal = bool(max_off_diag < tolerance)

    if not is_diagonal:
        return {
            "is_valid": False,
            "is_spd": False,
            "is_finite": False,
            "diag_values": None,
            "min_eigenvalue": None,
            "status": "not_diagonal",
            "evidence": f"Max off-diagonal element: {max_off_diag}",
        }

    diag_vals = np.diag(arr)
    is_finite = bool(np.all(np.isfinite(diag_vals)))
    is_positive = bool(np.all(diag_vals > tolerance)) if is_finite else False
    min_eig = float(np.min(diag_vals)) if is_finite else None
    is_spd = is_finite and is_positive
    is_valid = is_spd

    return {
        "is_valid": is_valid,
        "is_spd": is_spd,
        "is_finite": is_finite,
        "diag_values": [float(v) if np.isfinite(v) else None for v in diag_vals],
        "min_eigenvalue": min_eig,
        "status": "passed" if is_valid else ("non_positive" if is_finite else "non_finite"),
        "evidence": None if is_valid else f"min_eigenvalue={min_eig}",
    }


def validate_full_spd_conductivity(
    sigma: Any, *, tolerance: float = 1e-10
) -> dict[str, Any]:
    """Validate full symmetric positive-definite (SPD) conductivity tensor σ.

    Returns:
        dict with keys: is_valid, is_spd, is_symmetric, is_finite, min_eigenvalue,
                       condition_number, status, evidence
    """
    try:
        arr = _to_numpy(sigma)
    except (TypeError, ValueError):
        return {
            "is_valid": False,
            "is_spd": False,
            "is_symmetric": False,
            "is_finite": False,
            "min_eigenvalue": None,
            "condition_number": None,
            "status": "not_convertible_to_array",
            "evidence": f"Cannot convert to array: {type(sigma)}",
        }

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return {
            "is_valid": False,
            "is_spd": False,
            "is_symmetric": False,
            "is_finite": False,
            "min_eigenvalue": None,
            "condition_number": None,
            "status": "not_square_matrix",
            "evidence": f"Shape {arr.shape}; expected (N, N)",
        }

    is_finite = bool(np.all(np.isfinite(arr)))
    if not is_finite:
        return {
            "is_valid": False,
            "is_spd": False,
            "is_symmetric": False,
            "is_finite": False,
            "min_eigenvalue": None,
            "condition_number": None,
            "status": "non_finite_elements",
            "evidence": f"Found NaN or Inf in tensor",
        }

    # Check symmetry
    diff = float(np.max(np.abs(arr - arr.T)))
    is_symmetric = bool(diff < tolerance)

    if not is_symmetric:
        return {
            "is_valid": False,
            "is_spd": False,
            "is_symmetric": False,
            "is_finite": True,
            "min_eigenvalue": None,
            "condition_number": None,
            "status": "not_symmetric",
            "evidence": f"Max asymmetry: {diff}",
        }

    # Check positive-definiteness via eigenvalues
    try:
        evals = np.linalg.eigvalsh(arr)
        min_eig = float(np.min(evals))
        max_eig = float(np.max(evals))
        cond = max_eig / (min_eig + 1e-16) if min_eig > 0 else np.inf
        is_spd = bool(min_eig > tolerance)
        is_valid = is_spd
    except Exception as e:
        return {
            "is_valid": False,
            "is_spd": False,
            "is_symmetric": True,
            "is_finite": True,
            "min_eigenvalue": None,
            "condition_number": None,
            "status": "eigenvalue_computation_failed",
            "evidence": str(e),
        }

    return {
        "is_valid": is_valid,
        "is_spd": is_spd,
        "is_symmetric": is_symmetric,
        "is_finite": is_finite,
        "min_eigenvalue": min_eig if np.isfinite(min_eig) else None,
        "condition_number": float(cond) if np.isfinite(cond) else None,
        "status": "passed" if is_valid else "not_positive_definite",
        "evidence": None if is_valid else f"min_eigenvalue={min_eig}",
    }


def validate_field_arrays_finite(
    phi_e: Optional[Any] = None,
    J_e: Optional[Any] = None,
    CSD: Optional[Any] = None,
) -> dict[str, Any]:
    """Check that field arrays are finite (no NaN, no Inf).

    Returns:
        dict with keys: all_finite, phi_e_finite, J_e_finite, CSD_finite, evidence
    """
    results = {
        "phi_e_finite": None,
        "J_e_finite": None,
        "CSD_finite": None,
    }
    evidence = []

    if phi_e is not None:
        try:
            arr = _to_numpy(phi_e)
            finite = bool(np.all(np.isfinite(arr)))
            results["phi_e_finite"] = finite
            if not finite:
                evidence.append("phi_e contains NaN or Inf")
        except Exception as e:
            evidence.append(f"phi_e validation error: {e}")

    if J_e is not None:
        try:
            arr = _to_numpy(J_e)
            finite = bool(np.all(np.isfinite(arr)))
            results["J_e_finite"] = finite
            if not finite:
                evidence.append("J_e contains NaN or Inf")
        except Exception as e:
            evidence.append(f"J_e validation error: {e}")

    if CSD is not None:
        try:
            arr = _to_numpy(CSD)
            finite = bool(np.all(np.isfinite(arr)))
            results["CSD_finite"] = finite
            if not finite:
                evidence.append("CSD contains NaN or Inf")
        except Exception as e:
            evidence.append(f"CSD validation error: {e}")

    all_provided = all(v is not None for v in results.values())
    all_finite = all(v is True for v in results.values() if v is not None)

    return {
        "all_finite": all_finite,
        "phi_e_finite": results["phi_e_finite"],
        "J_e_finite": results["J_e_finite"],
        "CSD_finite": results["CSD_finite"],
        "evidence": evidence if evidence else None,
    }


def build_field_admissibility_report(
    field_output: Optional[Any] = None,
    cfg_metadata: Optional[dict[str, Any]] = None,
    signals_field: Optional[Any] = None,
) -> dict[str, Any]:
    """Build a comprehensive field admissibility report for v0.2.0 compliance.

    Returns a JSON-safe dict with all required v0.2.0 field admissibility fields.
    """
    metadata = dict(cfg_metadata or {})
    field_diag = {}

    if field_output is not None and hasattr(field_output, "diagnostics"):
        field_diag = dict(field_output.diagnostics)
    elif signals_field is not None and hasattr(signals_field, "diagnostics"):
        field_diag = dict(signals_field.diagnostics)

    report: dict[str, Any] = {
        "field_solver_status": metadata.get(
            "field_solver_status", field_diag.get("field_solver_status", "laminar_proxy_no_pde")
        ),
        "field_claim_level": metadata.get(
            "field_claim_level", field_diag.get("field_claim_level", "proxy_readout_only")
        ),
        "boundary_condition": metadata.get("boundary_condition", "mean_zero_neumann"),
        "gauge": metadata.get("gauge", "mean_zero"),
        "CSD_sign_convention": metadata.get(
            "CSD_sign_convention",
            field_diag.get("CSD_sign_convention", "proxy_positive_equals_extracellular_source_like"),
        ),
        "conductivity_status": "proxy_not_solved",  # For laminar proxy paths
        "conductivity_is_spd": None,  # Not applicable for proxy mode
        "conductivity_min_eigenvalue": None,  # Not applicable for proxy mode
        "passivity_check_passed": None,  # Not enforced for proxy mode
        "source_conservation_status": "not_applicable_proxy_mode",
        "integrated_source": None,
        "integrated_boundary_flux": None,
        "source_flux_residual": None,
        "source_flux_residual_tolerance": None,
        "mean_phi_e": None,
        "gauge_residual": None,
        "finite_phi_e": field_diag.get("finite_phi_e_proxy", field_diag.get("finite_phi_e", True)),
        "finite_J_e": None,  # Not computed in proxy mode
        "finite_CSD": field_diag.get("finite_csd_proxy", field_diag.get("finite_CSD", True)),
        "solver_residual_l2_relative": None,  # Not applicable for proxy mode
        "physical_amplitude_claim_allowed": False,  # Always False for uncalibrated proxy
    }

    return report


# ─── v0.2.5 Calibration Specification and Reporting ──────────────────────────────


class CalibrationSpec:
    """Calibration specification contract for v0.2.5.

    Allows users to declare calibration state without upgrading physical amplitude claims.
    All proxy readouts remain computational proxies unless a workflow supplies full
    calibration, geometry, and validation evidence.
    """

    ALLOWED_MODES = {
        "uncalibrated_native",
        "toy_scale",
        "relative_normalized",
        "empirical_gain_candidate",
        "physical_units_candidate",
        "calibrated_empirical",
    }

    def __init__(
        self,
        *,
        name: str,
        target: str,
        mode: str = "uncalibrated_native",
        scale: float | None = None,
        units: str | None = None,
        reference: str | None = None,
        description: str | None = None,
    ):
        """Initialize a calibration specification.

        Parameters
        ----------
        name : str
            Name of the calibration spec.
        target : str
            Target: 'source', 'field', 'probe', 'readout', or 'objective'.
        mode : str, optional
            Calibration mode. Default 'uncalibrated_native'.
        scale : float, optional
            Scale factor if applicable.
        units : str, optional
            Physical or proxy units.
        reference : str, optional
            Reference dataset, method, or publication.
        description : str, optional
            Human-readable description.

        Raises
        ------
        ValueError
            If mode is not in ALLOWED_MODES or target is empty.
        """
        if mode not in self.ALLOWED_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Allowed: {self.ALLOWED_MODES}"
            )
        if not target:
            raise ValueError("target must be a non-empty string")

        self.name = str(name)
        self.target = str(target)
        self.mode = str(mode)
        self.scale = float(scale) if scale is not None else None
        self.units = str(units) if units is not None else None
        self.reference = str(reference) if reference is not None else None
        self.description = str(description) if description is not None else None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe representation of the calibration spec."""
        return {
            "name": self.name,
            "target": self.target,
            "mode": self.mode,
            "scale": self.scale,
            "units": self.units,
            "reference": self.reference,
            "description": self.description,
        }


def make_calibration_report(
    spec: CalibrationSpec | dict,
    *,
    readout_kind: str | None = None,
) -> dict[str, Any]:
    """Create a calibration status report from a CalibrationSpec.

    Parameters
    ----------
    spec : CalibrationSpec or dict
        Calibration specification.
    readout_kind : str, optional
        Type of readout (e.g., 'lfp_proxy', 'spk', 'vm').

    Returns
    -------
    dict
        JSON-safe calibration report with status, claim levels, and warnings.
    """
    # Convert dict to CalibrationSpec if needed
    if isinstance(spec, dict):
        spec = CalibrationSpec(**spec)

    mode = spec.mode
    target = spec.target
    has_scale = spec.scale is not None
    has_units = spec.units is not None
    has_reference = spec.reference is not None

    # v0.2.5: Always keep physical_amplitude_claim_allowed false
    # Empirical calibration is declared metadata only, not validated.
    physical_amplitude_claim_allowed = False
    calibration_claim_level = "computational_proxy_with_declared_metadata"

    # Generate warnings for incomplete high-claim modes
    warnings: list[str] = []
    if mode == "calibrated_empirical":
        if not (has_reference and has_units and has_scale and target):
            warnings.append(
                "calibrated_empirical mode requires reference, units, scale, and target; treating as metadata_declared_not_validated"
            )
        if not has_units:
            warnings.append("calibrated_empirical missing units")
        if not has_reference:
            warnings.append("calibrated_empirical missing reference")
        if not has_scale:
            warnings.append("calibrated_empirical missing scale")

    if mode in {"physical_units_candidate", "empirical_gain_candidate"}:
        warnings.append(f"mode {mode} is candidate status, not validated")

    return {
        "calibration_name": spec.name,
        "target": target,
        "mode": mode,
        "status": "metadata_declared",
        "units": spec.units,
        "scale": spec.scale,
        "reference": spec.reference,
        "description": spec.description,
        "readout_kind": readout_kind,
        "physical_amplitude_claim_allowed": physical_amplitude_claim_allowed,
        "calibration_claim_level": calibration_claim_level,
        "empirical_reference_declared": has_reference,
        "empirical_units_declared": has_units,
        "empirical_scale_declared": has_scale,
        "assumptions": [
            "proxy readouts remain computational proxies in v0.2.5",
            "empirical calibration metadata is declared but not validated",
            "physical amplitude claims require separate calibration, geometry, and validation evidence",
        ],
        "warnings": warnings,
    }
