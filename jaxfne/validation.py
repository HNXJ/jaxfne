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
            "proxy readouts are simulated computational mathematical proxies",
            "empirical calibration parameters are metadata declarations",
            "physical amplitude claims are excluded under proxy mode",
        ],
        "warnings": warnings,
    }


# ─── v0.2.6 Field/Proxy Diagnostic Scaffolding ────────────────────────────────


def make_source_balance_diagnostic(
    sources: Any | None = None,
    *,
    operator_path: str = "proxy",
    residual: float | None = None,
) -> dict[str, Any]:
    """Field/proxy source-balance diagnostic report.

    In proxy mode, source balance is not applicable (no PDE solve).
    In physical_candidate mode, users can supply an observed residual.

    Parameters
    ----------
    sources : array-like, optional
        Source array for future physical path integration.
    operator_path : str
        Path designation: 'proxy' (default) or 'physical_candidate'.
    residual : float, optional
        Observed source balance residual (for candidate mode).

    Returns
    -------
    dict
        JSON-safe source-balance diagnostic report.
    """
    if operator_path == "proxy":
        status = "not_applicable_proxy_mode"
        residual_value = None
    elif operator_path == "physical_candidate":
        status = "candidate_only" if residual is None else "checked"
        residual_value = float(residual) if residual is not None else None
    else:
        raise ValueError(f"Invalid operator_path: {operator_path}")

    return {
        "diagnostic_kind": "source_balance",
        "operator_path": operator_path,
        "status": status,
        "residual": residual_value,
        "physical_amplitude_claim_allowed": False,
        "assumptions": [
            "proxy mode: source balance is not applicable",
            "physical_candidate mode: residual is declared metadata only, not validated",
        ],
    }


def make_gauge_diagnostic(
    field_array: Any | None = None,
    *,
    operator_path: str = "proxy",
    gauge_mode: str = "mean_zero",
) -> dict[str, Any]:
    """Field/proxy gauge diagnostic report.

    In proxy mode, gauge is declared metadata only.
    In physical_candidate mode, users can supply a field array to check.

    Parameters
    ----------
    field_array : array-like, optional
        Field array (phi_e or similar) for gauge checking.
    operator_path : str
        Path designation: 'proxy' (default) or 'physical_candidate'.
    gauge_mode : str
        Declared gauge mode: 'mean_zero' (default) or other.

    Returns
    -------
    dict
        JSON-safe gauge diagnostic report.
    """
    if operator_path == "proxy":
        status = "declared_metadata_only"
        residual_value = None
    elif operator_path == "physical_candidate":
        status = "not_tested"
        if field_array is not None:
            arr = _to_numpy(field_array)
            residual_value = float(jnp.mean(arr))
            status = "checked"
        else:
            residual_value = None
    else:
        raise ValueError(f"Invalid operator_path: {operator_path}")

    return {
        "diagnostic_kind": "gauge",
        "operator_path": operator_path,
        "gauge_mode": gauge_mode,
        "status": status,
        "residual": residual_value,
        "physical_amplitude_claim_allowed": False,
        "assumptions": [
            "proxy mode: gauge is declared metadata only",
            "physical_candidate mode: residual is diagnostic scaffold, not validation",
        ],
    }


def make_boundary_diagnostic(
    operator_path: str = "proxy",
) -> dict[str, Any]:
    """Field/proxy boundary-condition diagnostic report.

    Parameters
    ----------
    operator_path : str
        Path designation: 'proxy' (default) or 'physical_candidate'.

    Returns
    -------
    dict
        JSON-safe boundary diagnostic report.
    """
    if operator_path not in {"proxy", "physical_candidate"}:
        raise ValueError(f"Invalid operator_path: {operator_path}")

    status = "declared_metadata_only" if operator_path == "proxy" else "candidate_only"

    return {
        "diagnostic_kind": "boundary",
        "operator_path": operator_path,
        "status": status,
        "boundary_condition_status": status,
        "physical_amplitude_claim_allowed": False,
        "assumptions": [
            "proxy mode: boundary condition is declared metadata only",
            "physical_candidate mode: boundary condition is candidate, not validated",
        ],
    }


def make_manufactured_residual_diagnostic(
    *,
    operator_path: str = "proxy",
    residual_l2_relative: float | None = None,
) -> dict[str, Any]:
    """Field/proxy manufactured-residual diagnostic report.

    Manufactured residual is a diagnostic scaffold for future PDE solver validation.
    In v0.2.6, it remains scaffolding only.

    Parameters
    ----------
    operator_path : str
        Path designation: 'proxy' (default) or 'physical_candidate'.
    residual_l2_relative : float, optional
        L2 relative residual (for physical_candidate mode).

    Returns
    -------
    dict
        JSON-safe manufactured-residual diagnostic report.
    """
    if operator_path == "proxy":
        status = "not_applicable_proxy_mode"
        residual_value = None
    elif operator_path == "physical_candidate":
        status = "candidate_only" if residual_l2_relative is None else "checked"
        residual_value = float(residual_l2_relative) if residual_l2_relative is not None else None
    else:
        raise ValueError(f"Invalid operator_path: {operator_path}")

    return {
        "diagnostic_kind": "manufactured_residual",
        "operator_path": operator_path,
        "status": status,
        "residual_l2_relative": residual_value,
        "physical_amplitude_claim_allowed": False,
        "assumptions": [
            "proxy mode: manufactured residual is not applicable",
            "physical_candidate mode: residual is diagnostic scaffold, not validation",
            "v0.2.6: no full PDE solver exists yet",
        ],
    }


def make_field_operator_status(
    *,
    operator_path: str,
) -> dict[str, Any]:
    """Field operator path and status crosswalk for proxy vs physical.

    Parameters
    ----------
    operator_path : str
        Path designation: 'proxy' or 'physical_candidate'.

    Returns
    -------
    dict
        JSON-safe operator status report.

    Raises
    ------
    ValueError
        If operator_path is invalid.
    """
    if operator_path not in {"proxy", "physical_candidate"}:
        raise ValueError(
            f"Invalid operator_path: {operator_path}. "
            "Allowed: 'proxy', 'physical_candidate'"
        )

    if operator_path == "proxy":
        field_solver_status = "laminar_proxy_no_pde"
        field_solver_selected = False
    else:  # physical_candidate
        field_solver_status = "physical_field_solver_candidate"
        field_solver_selected = False  # Still candidate in v0.2.6

    return {
        "diagnostic_kind": "field_operator_status",
        "operator_path": operator_path,
        "field_solver_selected": field_solver_selected,
        "field_solver_status": field_solver_status,
        "physical_field_solver_status": "not_selected",
        "physical_amplitude_claim_allowed": False,
        "assumptions": [
            "proxy path: laminar proxy projection, no PDE solve",
            "physical_candidate path: designated for future PDE integration, not yet implemented",
            "v0.2.6: no full PDE solver exists; all physical claims remain false",
        ],
    }


# ─── v0.2.15 Poisson Admissibility Specification ────────────────────────────────


def validate_poisson_spd_conductivity(
    conductivity: Any,
    tolerance: float = 1e-8,
) -> tuple[bool, str]:
    """Validate that conductivity tensor is Symmetric Positive Definite (SPD).

    For future Poisson solver: conductivity must be SPD to ensure well-posedness
    and unique solution existence.

    Parameters
    ----------
    conductivity : Any
        Conductivity tensor or matrix (any shape).
    tolerance : float, optional
        Tolerance for negative eigenvalues (default 1e-8).

    Returns
    -------
    tuple[bool, str]
        (is_spd, diagnostic_message)
    """
    try:
        cond_np = _to_numpy(conductivity)
        if cond_np.ndim == 0:
            return False, "conductivity is scalar; requires matrix/tensor"

        # Check symmetry (for square matrices)
        if cond_np.ndim >= 2 and cond_np.shape[-2] == cond_np.shape[-1]:
            sym_err = np.linalg.norm(cond_np - cond_np.T) / (
                np.linalg.norm(cond_np) + 1e-16
            )
            if sym_err > tolerance:
                return False, f"conductivity not symmetric; relative error {sym_err:.2e}"

        # Check positive definiteness (compute eigenvalues)
        evals = np.linalg.eigvalsh(cond_np)
        min_eval = float(np.min(evals))
        if min_eval < -tolerance:
            return False, f"conductivity not positive definite; min eigenvalue {min_eval:.2e}"

        return True, f"SPD verified; min eigenvalue {min_eval:.2e}"
    except Exception as e:
        return False, f"SPD check failed: {str(e)}"


def validate_poisson_source_conservation(
    integrated_source: Optional[float],
    integrated_boundary_flux: Optional[float],
    tolerance: float = 1e-6,
) -> tuple[bool, str, Optional[float]]:
    """Validate source conservation: source integral + boundary flux integral ≈ 0.

    For future Poisson solver: integrated source must equal negative integrated
    boundary flux (within tolerance) for charge/current conservation.

    Mathematical form: ∫ I_src = -∫ σ ∇φ_e · n̂ (by divergence theorem)
    Or equivalently: ∫ I_src + ∫ σ ∇φ_e · n̂ = 0

    Parameters
    ----------
    integrated_source : float or None
        Integral of source/sink over domain.
    integrated_boundary_flux : float or None
        Integral of normal flux over boundary (≈ ∫ σ ∇φ_e · n̂).
    tolerance : float, optional
        Absolute tolerance for residual (default 1e-6).

    Returns
    -------
    tuple[bool, str, float or None]
        (is_conserved, diagnostic_message, residual)
    """
    if integrated_source is None or integrated_boundary_flux is None:
        return True, "source conservation: not tested (data not available)", None

    try:
        src_val = float(integrated_source)
        flux_val = float(integrated_boundary_flux)
        # Conservation: source + flux = 0 (flux already includes the negative)
        residual = abs(src_val + flux_val)

        if residual > tolerance:
            return (
                False,
                f"source not conserved; residual {residual:.2e} > tol {tolerance:.2e}",
                residual,
            )

        return True, f"source conserved; residual {residual:.2e}", residual
    except Exception as e:
        return False, f"conservation check failed: {str(e)}", None


def validate_poisson_gauge_condition(
    mean_potential: Optional[float],
    gauge: str = "mean_zero",
    tolerance: float = 1e-6,
) -> tuple[bool, str]:
    """Validate gauge condition (e.g., mean-zero potential).

    For future Poisson solver with mean-zero gauge: mean(phi_e) should be ≈ 0.

    Parameters
    ----------
    mean_potential : float or None
        Mean of potential field over domain.
    gauge : str, optional
        Gauge type ('mean_zero', 'other'). Default 'mean_zero'.
    tolerance : float, optional
        Tolerance for mean potential (default 1e-6).

    Returns
    -------
    tuple[bool, str]
        (gauge_satisfied, diagnostic_message)
    """
    if mean_potential is None:
        return True, f"gauge ({gauge}): not tested (data not available)"

    try:
        mean_val = float(mean_potential)

        if gauge == "mean_zero":
            if abs(mean_val) > tolerance:
                return False, f"gauge {gauge} violated; |mean(phi_e)| {abs(mean_val):.2e} > tol"
            return True, f"gauge {gauge} satisfied; |mean(phi_e)| {abs(mean_val):.2e}"
        else:
            raise NotImplementedError(
                f"Unsupported Poisson gauge validation: {gauge!r}. "
                "Only gauge='mean_zero' is implemented."
            )
    except NotImplementedError:
        raise
    except Exception as e:
        return False, f"gauge check failed: {str(e)}"


def validate_poisson_field_arrays(
    phi_e: Optional[Any] = None,
    J_e: Optional[Any] = None,
    CSD: Optional[Any] = None,
) -> dict[str, bool]:
    """Validate finiteness of Poisson field solution arrays.

    All outputs must be finite (no NaN, no Inf) for valid solver output.

    Parameters
    ----------
    phi_e : Any, optional
        Extracellular potential field.
    J_e : Any, optional
        Extracellular current density.
    CSD : Any, optional
        Current source density.

    Returns
    -------
    dict[str, bool]
        Finiteness status for each array.
    """
    results = {}

    if phi_e is not None:
        phi_np = _to_numpy(phi_e)
        results["finite_phi_e"] = bool(np.all(np.isfinite(phi_np)))
    else:
        results["finite_phi_e"] = True  # Not provided

    if J_e is not None:
        J_np = _to_numpy(J_e)
        results["finite_J_e"] = bool(np.all(np.isfinite(J_np)))
    else:
        results["finite_J_e"] = True  # Not provided

    if CSD is not None:
        csd_np = _to_numpy(CSD)
        results["finite_CSD"] = bool(np.all(np.isfinite(csd_np)))
    else:
        results["finite_CSD"] = True  # Not provided

    return results


def build_poisson_admissibility_report(
    *,
    conductivity: Optional[Any] = None,
    integrated_source: Optional[float] = None,
    integrated_boundary_flux: Optional[float] = None,
    mean_potential: Optional[float] = None,
    phi_e: Optional[Any] = None,
    J_e: Optional[Any] = None,
    CSD: Optional[Any] = None,
    solver_residual_l2: Optional[float] = None,
    n_iterations: Optional[int] = None,
    converged: bool = False,
    gauge: str = "mean_zero",
    boundary_condition: str = "dirichlet",
    csd_sign_convention: str = "positive_equals_extracellular_source",
) -> dict[str, Any]:
    """Build comprehensive Poisson solver admissibility report for v0.2.15+.

    This report contract specifies what a Poisson solver must output to be
    admissible. Used for validating solver implementations before integration.

    **v0.2.15 invariant:** physical_amplitude_claim_allowed is ALWAYS false
    because v0.2.15 is specification-only (no solver implemented yet).
    All gates may pass on synthetic data, but no physical amplitude claims
    are allowed until a solver exists and calibration/units are validated.

    Parameters
    ----------
    conductivity : Any, optional
        Conductivity tensor/matrix.
    integrated_source : float, optional
        Integral of source over domain.
    integrated_boundary_flux : float, optional
        Integral of boundary flux.
    mean_potential : float, optional
        Mean of potential field.
    phi_e : Any, optional
        Extracellular potential field.
    J_e : Any, optional
        Extracellular current density.
    CSD : Any, optional
        Current source density.
    solver_residual_l2 : float, optional
        Relative L2 residual of solution.
    n_iterations : int, optional
        Number of solver iterations.
    converged : bool, optional
        Whether solver converged (default False).
    gauge : str, optional
        Gauge condition applied (default 'mean_zero').
    boundary_condition : str, optional
        Boundary condition type (default 'dirichlet').
    csd_sign_convention : str, optional
        CSD sign convention (default 'positive_equals_extracellular_source').

    Returns
    -------
    dict
        JSON-safe Poisson admissibility report with validation results.
    """
    # Check SPD conductivity
    cond_spd, cond_msg = validate_poisson_spd_conductivity(conductivity)

    # Check source conservation
    cons_ok, cons_msg, cons_residual = validate_poisson_source_conservation(
        integrated_source, integrated_boundary_flux
    )

    # Check gauge
    gauge_ok, gauge_msg = validate_poisson_gauge_condition(
        mean_potential, gauge=gauge
    )

    # Check field arrays finiteness
    field_finite = validate_poisson_field_arrays(phi_e=phi_e, J_e=J_e, CSD=CSD)

    # All gates passed?
    all_gates_pass = cond_spd and cons_ok and gauge_ok and all(field_finite.values())

    return {
        "diagnostic_kind": "poisson_admissibility",
        "admissibility_status": "admissible" if all_gates_pass else "not_admissible",
        "gates": {
            "conductivity_spd": {
                "passed": cond_spd,
                "message": cond_msg,
            },
            "source_conservation": {
                "passed": cons_ok,
                "message": cons_msg,
                "residual": cons_residual,
            },
            "gauge_condition": {
                "passed": gauge_ok,
                "message": gauge_msg,
                "gauge": gauge,
            },
            "field_finiteness": {
                "passed": all(field_finite.values()),
                "phi_e_finite": field_finite.get("finite_phi_e", True),
                "J_e_finite": field_finite.get("finite_J_e", True),
                "CSD_finite": field_finite.get("finite_CSD", True),
            },
        },
        "solver_metadata": {
            "solver_residual_l2_relative": solver_residual_l2,
            "n_iterations": n_iterations,
            "converged": converged,
            "boundary_condition": boundary_condition,
            "gauge": gauge,
            "csd_sign_convention": csd_sign_convention,
        },
        "physical_amplitude_claim_allowed": False,  # v0.2.15: specification-only, no solver yet
        "v0215_note": (
            "Admissibility report for Poisson solver specification (v0.2.15). "
            "v0.2.15 is specification-only: no solver implemented, physical amplitude claims remain false. "
            "v0.2.16+: Poisson solver implementation will validate all gates and optionally allow "
            "physical amplitude claims only after: (1) solver exists, (2) all gates pass, (3) calibration/units validated."
        ),
    }


# ──────────────────────────────────────────────────────────────
# v0.2.26 computation-basis validation
# ──────────────────────────────────────────────────────────────


def validate_basis_spec(spec: Any) -> dict[str, Any]:
    """Validate a BasisSpec or basis dict against computation-basis contracts.

    Parameters
    ----------
    spec : BasisSpec or dict
        A BasisSpec dataclass (preferred) or dict with equivalent fields.

    Returns
    -------
    dict
        JSON-safe report with keys: valid, issues, warnings, field_by_field.
        Never raises; invalid inputs produce ``valid: False``.
    """
    # Import here to avoid circular import; core imports validation, not vice versa
    try:
        from .core import (
            BasisSpec,
            _AXIS_STATUS_VALUES,
            _SPACE_BASIS_VALUES,
            _TIME_BASIS_VALUES,
            _FIELD_REGIME_VALUES,
            _SOURCE_MODE_BASIS_VALUES,
            _PROBE_BASIS_VALUES,
            _FUTURE_FIELD_REGIMES,
        )
    except ImportError:
        return {"valid": False, "issues": ["import_error_core_unavailable"], "warnings": []}

    # Normalize dict to BasisSpec for uniform validation
    if isinstance(spec, dict):
        try:
            spec = BasisSpec(
                space_basis=spec.get("space_basis", "laminar_depth"),
                time_basis=spec.get("time_basis", "continuous_ms"),
                field_regime=spec.get("field_regime", "laminar_proxy"),
                source_mode=spec.get("source_mode", "proxy_no_field_solve"),
                probe_basis=spec.get("probe_basis", "multimodal_proxy"),
            )
        except Exception as e:
            return {"valid": False, "issues": [f"dict_normalization_error:{e}"], "warnings": []}

    if not isinstance(spec, BasisSpec):
        return {"valid": False, "issues": ["not_a_BasisSpec_instance"], "warnings": []}

    validation = spec.validate()
    warnings: list[str] = []

    # Extra warnings (non-fatal)
    if spec.field_regime in _FUTURE_FIELD_REGIMES:
        warnings.append(
            f"future_regime_{spec.field_regime}:implemented=False,claim_allowed=False"
        )
    if spec.field_regime == "solved_poisson":
        warnings.append(
            "solved_poisson:not_implemented_in_v0.2.x:treated_as_future_specification"
        )
    if spec.source_mode != "proxy_no_field_solve":
        warnings.append(
            f"source_mode_{spec.source_mode}:not_active_in_v0.2.x:proxy_no_field_solve_only"
        )

    validation["warnings"] = warnings
    # Guarantee physical_amplitude_claim_allowed is never escalated
    validation["physical_amplitude_claim_allowed"] = False
    return validation


def basis_claim_gate(
    spec: Any,
    *,
    source_calibration_status: str,
    field_solver_status: str,
) -> dict[str, Any]:
    """Evaluate claim eligibility given a BasisSpec and current runtime status.

    Parameters
    ----------
    spec : BasisSpec or dict
        Computation basis specification.
    source_calibration_status : str
        From manifest (e.g. ``"uncalibrated_izhikevich_native_current"``).
    field_solver_status : str
        From manifest (e.g. ``"laminar_proxy_no_pde"``).

    Returns
    -------
    dict
        JSON-safe gate result. ``physical_amplitude_claim_allowed`` is always
        ``False`` in v0.2.x.
    """
    validation = validate_basis_spec(spec)
    issues = list(validation.get("issues", []))
    warnings = list(validation.get("warnings", []))

    # Physical claim gate: requires solved field AND calibrated source
    proxy_solver = field_solver_status in (
        "laminar_proxy_no_pde",
        "not_computed",
        None,
        "",
    )
    uncalibrated_source = source_calibration_status.startswith("uncalibrated") if isinstance(
        source_calibration_status, str
    ) else True

    if proxy_solver:
        warnings.append("claim_gate:field_not_solved:laminar_proxy")
    if uncalibrated_source:
        warnings.append(f"claim_gate:source_uncalibrated:{source_calibration_status}")

    return {
        "valid": validation.get("valid", False),
        "issues": issues,
        "warnings": warnings,
        "physical_amplitude_claim_allowed": False,
        "claim_rationale": (
            "proxy_scaffold_no_physical_claim_allowed"
            if (proxy_solver or uncalibrated_source)
            else "calibration_required_for_future_claim"
        ),
        "source_calibration_status": source_calibration_status,
        "field_solver_status": field_solver_status,
        "basis_implemented": validation.get("implemented", False),
        "basis_future_regime": validation.get("future_regime", False),
    }
