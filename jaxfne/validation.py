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
