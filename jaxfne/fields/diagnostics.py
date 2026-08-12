"""Field projection diagnostics and validation helpers.

Separated from proxy.py to keep mathematical projection code clean.
"""
from __future__ import annotations
from typing import Any
import jax
import jax.numpy as jnp


def _dtype_name(array: jax.Array) -> str:
    return str(array.dtype)


def _finite_bool(array: jax.Array) -> bool:
    return bool(jnp.all(jnp.isfinite(array)))


def _finite_flag(array: jax.Array):
    """Finiteness flag usable inside a traced region.

    Returns a Python ``bool`` in eager mode (reporting stays unchanged) and a
    traced boolean under ``jax.jit``/``vmap``, where forcing a Python bool
    would raise ``TracerBoolConversionError``. The output dtype is identical
    in both modes; only the value's host vs. trace status differs.
    """
    finite = jnp.all(jnp.isfinite(array))
    if isinstance(finite, jax.core.Tracer):
        return finite
    return bool(finite)


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
    mode: str = "row_normalize",
) -> dict[str, Any]:
    """Check structural invariants of the laminar proxy projection.

    Verifies the selected kernel normalization contract and the
    finiteness/consistency of the ``source_proxy``/``phi_e_proxy``/
    ``csd_proxy``/``lfp_proxy`` arrays for the given
    ``sources``/``positions``/``kernel``. Row sums are checked to
    ``tol=1e-6`` for ``row_normalize``; density-preserving mode reports the
    row-normalization check as not applicable.
    Returns a JSON-safe dict of per-invariant pass/fail diagnostics. This checks
    the proxy operator's internal consistency only — it makes no claim of
    physical correctness.
    """
    kernel_norm_tests = _test_kernel_row_normalization(kernel, tol=1e-6)
    kernel_row_sum_max_abs_error = kernel_norm_tests["kernel_row_sum_max_abs_error"]

    if isinstance(kernel_row_sum_max_abs_error, jax.core.Tracer):
        kernel_row_sum_max_abs_error_val = kernel_row_sum_max_abs_error
        row_normalization_valid = (
            kernel_row_sum_max_abs_error < 1e-6
            if mode != "density_preserving"
            else None
        )
    else:
        kernel_row_sum_max_abs_error_val = float(kernel_row_sum_max_abs_error)
        row_normalization_valid = (
            kernel_row_sum_max_abs_error_val < 1e-6
            if mode != "density_preserving"
            else None
        )
    normalization_valid = (
        True if mode == "density_preserving" else row_normalization_valid
    )

    t_steps = int(sources.shape[0])
    n_emitters = int(sources.shape[1])
    n_contacts = int(kernel.shape[0])
    finite_sources = _finite_flag(sources)
    finite_positions = _finite_flag(positions)
    finite_kernel = _finite_flag(kernel)
    finite_source_proxy = _finite_flag(source_proxy)
    finite_phi_e_proxy = _finite_flag(phi_e_proxy)
    finite_csd_proxy = _finite_flag(csd_proxy)
    finite_lfp_proxy = _finite_flag(lfp_proxy)
    warnings: list[str] = []
    if positions.shape != (n_emitters, 3):
        warnings.append("positions_shape_not_N_by_3")
    if source_proxy.shape != (t_steps, n_contacts):
        warnings.append("source_proxy_shape_mismatch")
    if not isinstance(finite_source_proxy, jax.core.Tracer) and not finite_source_proxy:
        warnings.append("non_finite_source_proxy")
    if not isinstance(finite_csd_proxy, jax.core.Tracer) and not finite_csd_proxy:
        warnings.append("non_finite_csd_proxy")

    return {
        "operator_type": "linear_projection",
        "representation": "relative",
        "normalization_mode": mode,
        "validation_status": "computational",
        "calibration_transform": "explicit_boundary_transform",
        "source_shape": tuple(int(x) for x in sources.shape),
        "positions_shape": tuple(int(x) for x in positions.shape),
        "kernel_shape": tuple(int(x) for x in kernel.shape),
        "source_proxy_shape": tuple(int(x) for x in source_proxy.shape),
        "phi_e_proxy_shape": tuple(int(x) for x in phi_e_proxy.shape),
        "csd_proxy_shape": tuple(int(x) for x in csd_proxy.shape),
        "lfp_proxy_shape": tuple(int(x) for x in lfp_proxy.shape),
        "dtype": _dtype_name(sources),
        "kernel_row_sum_max_abs_error": kernel_row_sum_max_abs_error_val,
        "finite_sources": finite_sources,
        "finite_positions": finite_positions,
        "finite_kernel": finite_kernel,
        "finite_source_proxy": finite_source_proxy,
        "finite_phi_e_proxy": finite_phi_e_proxy,
        "finite_csd_proxy": finite_csd_proxy,
        "finite_lfp_proxy": finite_lfp_proxy,
        "finite_phi_e": finite_phi_e_proxy,
        "finite_CSD": finite_csd_proxy,
        "field_admissibility": {
            "field_arrays_finite": {
                "phi_e_finite": finite_phi_e_proxy,
                "csd_finite": finite_csd_proxy,
                "lfp_finite": finite_lfp_proxy,
            },
            "source_conservation_status": "proxy_not_solved",
            "kernel_normalization_definition": "contact_rows_density_preserving" if mode == "density_preserving" else "contact_rows_sum_to_one_proxy",
            "source_current_conservation_status": "not_applicable_proxy_mode",
            "source_current_conservation_test": "not_applicable_proxy_mode",
            "boundary_condition_status": "declared_metadata_only",
            "gauge_status": "declared_metadata_only",
            "kernel_row_normalization_applied": mode == "row_normalize",
            "kernel_row_normalization_valid": row_normalization_valid,
            "kernel_normalization_valid": normalization_valid,
            # Compatibility key: this is validity of the selected mode, not a
            # claim that density-preserving rows sum to one.
            "kernel_row_stochastic_valid": normalization_valid,
            "kernel_row_stochastic_status": (
                "mode_valid_density_preserving"
                if mode == "density_preserving"
                else "row_sum_valid"
            ),
            "kernel_row_sum_max_abs_error_v024": kernel_norm_tests["kernel_row_sum_max_abs_error"],
            "kernel_row_sum_tolerance_v024": kernel_norm_tests["kernel_row_sum_tolerance"],
        },
        "warnings": warnings,
    }


def _make_field_solution_report(
    field_solver_status: str = "linear_solver",
    operator_type: str = "linear_projection",
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
    field_claim_level: str = "proxy_readout",
    physical_amplitude_calibrated: bool = False,
    representation: str = "relative",
    validation_status: str = "computational",
    calibration_transform: str = "explicit_boundary_transform",
    normalization_mode: str | None = None,
    source_projection_mode: str = "proxy_no_field_solve",
    source_current_conservation_status: str = "not_applicable_proxy_mode",
    source_conservation_tested: bool = False,
    source_conservation_claim_allowed: bool = False,
) -> dict:
    return {
        "operator_type": operator_type,
        "representation": representation,
        "validation_status": validation_status,
        "calibration_transform": calibration_transform,
        "normalization_mode": normalization_mode,
        "csd_operator": {
            "type": "negative_second_difference",
            "spacing": "relative_contact_depth_dz",
            "boundary": "edge_padded",
            "input": "phi_e_proxy",
            "output": "csd_proxy",
            "representation": representation,
        },
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
        "physical_amplitude_calibrated": physical_amplitude_calibrated,
        "source_projection_mode": source_projection_mode,
        "source_current_conservation_status": source_current_conservation_status,
        "source_conservation_tested": source_conservation_tested,
        "source_conservation_claim_allowed": source_conservation_claim_allowed,
    }


def observation_operator_chain(
    *,
    execution_form: str,
    source: dict[str, Any],
    field: dict[str, Any],
    probe: dict[str, Any],
    amplitude_semantics: str = "relative",
    validation_status: str = "computational",
    physical_claim: str = "proxy_readout",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-safe observation receipt: conceptual S→F→P even when execution is fused.

    Orthogonal axes are stored separately and must not be collapsed:
    operator identity (chain), numerical validation, amplitude semantics,
    physical claim. ``amplitude_semantics`` uses the NeuronalTensor vocabulary
    ``relative | calibrated_proxy | calibrated``.
    """
    if execution_form not in {"fused", "materialized"}:
        raise ValueError(
            f"execution_form must be 'fused' or 'materialized', got {execution_form!r}"
        )
    receipt = {
        "execution_form": execution_form,
        "operator_chain": {
            "source": dict(source),
            "field": dict(field),
            "probe": dict(probe),
        },
        "amplitude_semantics": amplitude_semantics,
        "validation_status": validation_status,
        "physical_claim": physical_claim,
    }
    if extra:
        receipt.update(extra)
    return receipt


def laminar_projection_observation_receipt(
    *,
    n_sources: int,
    n_contacts: int,
    width: float,
    normalization_mode: str,
    contact_z_min: float,
    contact_z_max: float,
) -> dict[str, Any]:
    """Receipt for ``project_laminar_sources``: fused KQ plus fused CSD D_zz(KQ)."""
    source = {
        "identity": "canonical_relative_source",
        "representation": "relative",
        "n_sources": int(n_sources),
        "geometry": "model_or_caller_positions_z_relative",
    }
    field = {
        "identity": "gaussian_laminar_projection",
        "geometry": "source_positions_z_relative",
        "normalization": str(normalization_mode),
        "kernel_width_relative": float(width),
        "operator_type": "linear_projection",
    }
    lfp_probe = {
        "identity": "contact_sampling_compiled_into_kernel",
        "geometry": {
            "kind": "relative_laminar_contacts",
            "n_contacts": int(n_contacts),
            "z_min": float(contact_z_min),
            "z_max": float(contact_z_max),
            "construction": "linspace_relative_depth",
        },
    }
    csd_probe = {
        "identity": "laminar_second_derivative",
        "input": "phi_e_proxy",
        "stencil": "negative_second_difference_edge_padded",
        "operator_type": "spatial_derivative",
    }
    shared = dict(
        amplitude_semantics="relative",
        validation_status="computational",
        physical_claim="proxy_readout",
    )
    return {
        "execution_form": "fused",
        "operator_chain": {
            "source": source,
            "field": field,
            "probe": lfp_probe,
        },
        "csd": observation_operator_chain(
            execution_form="fused",
            source=source,
            field=field,
            probe=csd_probe,
            **shared,
        ),
        "output_identities": {
            "source_proxy": "KQ_relative_field_and_fused_lfp",
            "phi_e_proxy": "alias_of_source_proxy",
            "lfp_proxy": "alias_of_source_proxy",
            "csd_proxy": "Dzz_of_phi_e_proxy",
        },
        **shared,
    }


def linear_readout_observation_receipt(
    *,
    name: str,
    leadfield_status: str,
    matrix_shape: tuple[int, int],
) -> dict[str, Any]:
    """Receipt for ``LinearReadout`` / EEG-MEG-like ``Y = Q Wᵀ`` fused maps."""
    lowered = str(name).lower()
    is_meg = "meg" in lowered
    is_eeg = "eeg" in lowered
    source = {
        "identity": "canonical_relative_source",
        "representation": "relative",
        "n_sources": int(matrix_shape[1]),
    }
    if is_meg:
        field = {
            "identity": "not_separately_materialized",
            "note": "relative_linear_map_on_scalar_Q",
        }
        probe = {
            "identity": "relative_linear_map",
            "orientation_claim": "none",
            "leadfield_status": leadfield_status,
            "matrix_shape": [int(matrix_shape[0]), int(matrix_shape[1])],
        }
        extra = {"output_identity": "meg_relative_proxy"}
    elif is_eeg:
        field = {
            "identity": "compiled_into_leadfield",
            "note": "P_circ_F_not_separately_materialized",
        }
        probe = {
            "identity": "linear_leadfield",
            "leadfield_status": leadfield_status,
            "matrix_shape": [int(matrix_shape[0]), int(matrix_shape[1])],
        }
        extra = {"output_identity": "eeg_proxy"}
    else:
        field = {
            "identity": "compiled_into_linear_map",
            "note": "P_circ_F_not_separately_materialized",
        }
        probe = {
            "identity": "linear_readout_matrix",
            "leadfield_status": leadfield_status,
            "matrix_shape": [int(matrix_shape[0]), int(matrix_shape[1])],
        }
        extra = {"output_identity": str(name)}
    return observation_operator_chain(
        execution_form="fused",
        source=source,
        field=field,
        probe=probe,
        extra=extra,
    )
