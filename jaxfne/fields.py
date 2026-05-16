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
