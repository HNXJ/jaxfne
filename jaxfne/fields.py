"""Source-to-field and probe utilities.

This file intentionally implements a tiny laminar placeholder, not a full PDE solver.
It is designed to be replaced/hardened while preserving the public object API.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class FieldOutput:
    phi_e: jnp.ndarray
    J_e: jnp.ndarray
    csd: jnp.ndarray
    lfp: jnp.ndarray
    diagnostics: dict


def validate_source_field_status(field_output: FieldOutput, cfg_metadata: dict | None = None) -> dict:
    """Validate and report source-field projection status.

    Returns dict with field_claim_level, physical_amplitude_claim_allowed, warnings.
    Used by manifest in v0.0.3 to document the confidence level of field outputs.
    """
    cfg_metadata = cfg_metadata or {}
    warnings = []

    # Check finiteness
    if not field_output.diagnostics.get("finite_phi_e", False):
        warnings.append("non_finite_phi_e")
    if not field_output.diagnostics.get("finite_J_e", False):
        warnings.append("non_finite_J_e")
    if not field_output.diagnostics.get("finite_CSD", False):
        warnings.append("non_finite_CSD")

    # Field solver status from diagnostics
    field_solver = field_output.diagnostics.get("field_solver", "unknown")
    is_proxy = "proxy" in field_solver or "no_pde" in field_solver

    # Claim level depends on solver type and source calibration
    source_calib = cfg_metadata.get("source_calibration_status", "uncalibrated_izhikevich_native_current")
    is_calibrated = "calibrated" in source_calib.lower() and not "uncalibrated" in source_calib.lower()

    # All proxy and PDE paths use proxy_readout_only claim level (not calibrated for amplitude yet)
    field_claim_level = "proxy_readout_only"
    physical_amplitude_claim_allowed = False

    return {
        "field_claim_level": field_claim_level,
        "physical_amplitude_claim_allowed": physical_amplitude_claim_allowed,
        "is_proxy": is_proxy,
        "is_calibrated": is_calibrated,
        "warnings": warnings,
    }


def project_sources_to_laminar_field(sources: jnp.ndarray, positions: jnp.ndarray, n_contacts: int = 16) -> FieldOutput:
    """Project source traces to simple laminar contacts.

    Args:
        sources: [T, N] native/proxy source traces.
        positions: [N, 3], with z/depth in column 2.
        n_contacts: number of laminar probe contacts.

    Returns:
        FieldOutput with placeholder phi_e, J_e, CSD, LFP arrays.
    """
    depth = positions[:, 2]
    contacts = jnp.linspace(0.0, 1.0, n_contacts)
    width = 0.10
    kernel = jnp.exp(-0.5 * ((contacts[:, None] - depth[None, :]) / width) ** 2)
    kernel = kernel / (jnp.sum(kernel, axis=1, keepdims=True) + 1e-8)
    lfp = sources @ kernel.T
    phi_e = lfp
    dz = contacts[1] - contacts[0] if n_contacts > 1 else 1.0
    grad = jnp.gradient(phi_e, dz, axis=1)
    J_e = -grad
    csd = -jnp.gradient(J_e, dz, axis=1)
    diagnostics = {
        "field_solver": "laminar_proxy_no_pde",
        "source_projection_mode": "proxy_no_field_solve",
        "source_calibration_status": "uncalibrated_izhikevich_native_current",
        "CSD_sign_convention": "proxy_positive_equals_extracellular_source_like",
        "finite_phi_e": bool(jnp.all(jnp.isfinite(phi_e))),
        "finite_J_e": bool(jnp.all(jnp.isfinite(J_e))),
        "finite_CSD": bool(jnp.all(jnp.isfinite(csd))),
    }
    return FieldOutput(phi_e=phi_e, J_e=J_e, csd=csd, lfp=lfp, diagnostics=diagnostics)
