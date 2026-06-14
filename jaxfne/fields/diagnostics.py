"""Field projection diagnostics and validation helpers.

Separated from proxy.py to keep mathematical projection code clean.
"""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import jax
import jax.numpy as jnp


def _dtype_name(array: jax.Array) -> str:
    return str(array.dtype)


def _finite_bool(array: jax.Array) -> bool:
    return bool(jnp.all(jnp.isfinite(array)))


def _test_kernel_row_normalization(kernel: jax.Array, tol: float = 1e-6) -> dict[str, Any]:
    """Verify that projection kernel is row-stochastic (rows sum to 1.0)."""
    row_sums = jnp.sum(kernel, axis=1)
    max_abs_error_arr = jnp.max(jnp.abs(row_sums - 1.0))

    if isinstance(max_abs_error_arr, jax.core.Tracer):
        is_valid = max_abs_error_arr < tol
        return {
            "kernel_row_normalization_valid": is_valid,
            "kernel_row_sum_max_abs_error": max_abs_error_arr,
            "kernel_row_sum_tolerance": tol,
            "kernel_row_stochastic_valid": is_valid,
        }
    else:
        max_abs_error = float(max_abs_error_arr)
        is_valid = max_abs_error < tol
        return {
            "kernel_row_normalization_valid": is_valid,
            "kernel_row_sum_max_abs_error": max_abs_error,
            "kernel_row_sum_tolerance": tol,
            "kernel_row_stochastic_valid": is_valid,
        }


def validate_projection_invariants(
    *,
    sources: jax.Array,
    positions: jax.Array,
    kernel: jax.Array,
    source_proxy: jax.Array,
    phi_e_proxy: jax.Array,
    csd_proxy: jax.Array,
    lfp_proxy: jax.Array,
) -> dict[str, Any]:
    """Check structural invariants of the laminar proxy projection.

    Verifies kernel row-normalization (row-stochastic to ``tol=1e-6``) and the
    finiteness/consistency of the ``source_proxy``/``phi_e_proxy``/``csd_proxy``/
    ``lfp_proxy`` arrays for the given ``sources``/``positions``/``kernel``.
    Returns a JSON-safe dict of per-invariant pass/fail diagnostics. This checks
    the proxy operator's internal consistency only — it makes no claim of
    physical correctness.
    """
    kernel_norm_tests = _test_kernel_row_normalization(kernel, tol=1e-6)
    kernel_row_sum_max_abs_error = kernel_norm_tests["kernel_row_sum_max_abs_error"]

    if isinstance(kernel_row_sum_max_abs_error, jax.core.Tracer):
        kernel_row_sum_max_abs_error_val = kernel_row_sum_max_abs_error
        normalization_valid = kernel_row_sum_max_abs_error < 1e-6
    else:
        kernel_row_sum_max_abs_error_val = float(kernel_row_sum_max_abs_error)
        normalization_valid = kernel_row_sum_max_abs_error_val < 1e-6

    t_steps = int(sources.shape[0])
    n_emitters = int(sources.shape[1])
    n_contacts = int(kernel.shape[0])
    warnings: list[str] = []
    if positions.shape != (n_emitters, 3):
        warnings.append("positions_shape_not_N_by_3")
    if source_proxy.shape != (t_steps, n_contacts):
        warnings.append("source_proxy_shape_mismatch")
    if not _finite_bool(source_proxy):
        warnings.append("non_finite_source_proxy")
    if not _finite_bool(csd_proxy):
        warnings.append("non_finite_csd_proxy")

    return {
        "source_shape": tuple(int(x) for x in sources.shape),
        "positions_shape": tuple(int(x) for x in positions.shape),
        "kernel_shape": tuple(int(x) for x in kernel.shape),
        "source_proxy_shape": tuple(int(x) for x in source_proxy.shape),
        "phi_e_proxy_shape": tuple(int(x) for x in phi_e_proxy.shape),
        "csd_proxy_shape": tuple(int(x) for x in csd_proxy.shape),
        "lfp_proxy_shape": tuple(int(x) for x in lfp_proxy.shape),
        "dtype": _dtype_name(sources),
        "kernel_row_sum_max_abs_error": kernel_row_sum_max_abs_error_val,
        "finite_sources": _finite_bool(sources),
        "finite_positions": _finite_bool(positions),
        "finite_kernel": _finite_bool(kernel),
        "finite_source_proxy": _finite_bool(source_proxy),
        "finite_phi_e_proxy": _finite_bool(phi_e_proxy),
        "finite_csd_proxy": _finite_bool(csd_proxy),
        "finite_lfp_proxy": _finite_bool(lfp_proxy),
        "finite_phi_e": _finite_bool(phi_e_proxy),
        "finite_CSD": _finite_bool(csd_proxy),
        "field_admissibility": {
            "field_arrays_finite": {
                "phi_e_finite": _finite_bool(phi_e_proxy),
                "csd_finite": _finite_bool(csd_proxy),
                "lfp_finite": _finite_bool(lfp_proxy),
            },
            "kernel_normalization_valid": normalization_valid,
            "source_conservation_status": "proxy_not_solved",
            "kernel_row_stochastic_valid": normalization_valid,
            "kernel_normalization_definition": "contact_rows_sum_to_one_proxy",
            "source_current_conservation_status": "not_applicable_proxy_mode",
            "source_current_conservation_test": "not_applicable_proxy_mode",
            "boundary_condition_status": "declared_metadata_only",
            "gauge_status": "declared_metadata_only",
            "kernel_row_normalization_valid": kernel_norm_tests["kernel_row_normalization_valid"],
            "kernel_row_sum_max_abs_error_v024": kernel_norm_tests["kernel_row_sum_max_abs_error"],
            "kernel_row_sum_tolerance_v024": kernel_norm_tests["kernel_row_sum_tolerance"],
        },
        "warnings": warnings,
    }


def _make_field_solution_report(
    field_solver_status: str = "laminar_proxy_no_pde",
    solver_name: str = "laminar_proxy",
    boundary_condition: str = "declared_metadata_only",
    gauge: str = "declared_metadata_only",
    csd_sign_convention: str = "positive_equals_extracellular_source",
    current_density_layout: str = "not_applicable",
    solver_residual_l2_relative: float | None = None,
    n_iterations: int | None = None,
    converged: bool | None = None,
    finite_phi_e: bool = True,
    finite_J_e: bool = True,
    finite_CSD: bool = True,
    field_claim_level: str = "proxy_readout_only",
    physical_amplitude_claim_allowed: bool = False,
    source_projection_mode: str = "proxy_no_field_solve",
    source_current_conservation_status: str = "not_applicable_proxy_mode",
    source_conservation_tested: bool = False,
    source_conservation_claim_allowed: bool = False,
) -> dict:
    return {
        "field_solver_status": field_solver_status,
        "solver_name": solver_name,
        "boundary_condition": boundary_condition,
        "gauge": gauge,
        "csd_sign_convention": csd_sign_convention,
        "current_density_layout": current_density_layout,
        "solver_residual_l2_relative": solver_residual_l2_relative,
        "n_iterations": n_iterations,
        "converged": converged,
        "finite_phi_e": finite_phi_e,
        "finite_J_e": finite_J_e,
        "finite_CSD": finite_CSD,
        "field_claim_level": field_claim_level,
        "physical_amplitude_claim_allowed": physical_amplitude_claim_allowed,
        "source_projection_mode": source_projection_mode,
        "source_current_conservation_status": source_current_conservation_status,
        "source_conservation_tested": source_conservation_tested,
        "source_conservation_claim_allowed": source_conservation_claim_allowed,
    }
