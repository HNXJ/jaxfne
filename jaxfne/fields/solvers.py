"""Experimental PDE field solvers for jaxfne."""

import jax
import jax.numpy as jnp
from typing import Any, Tuple, Dict

def experimental_poisson_1d(
    sources: jax.Array,
    conductivity: float,
    dx: float,
    boundary: str = "mean_zero_neumann",
    gauge: str = "mean_zero",
) -> Tuple[jax.Array, jax.Array, Dict[str, Any]]:
    """Experimental 1D Poisson solver with Neumann boundaries and mean-zero gauge.

    Solves: d/dx (conductivity * d/dx phi) = -sources

    Parameters
    ----------
    sources : jax.Array
        1D array of current/charge sources.
    conductivity : float
        Conductivity scalar.
    dx : float
        Grid spacing.
    boundary : str, default "mean_zero_neumann"
        Boundary condition choice.
    gauge : str, default "mean_zero"
        Gauge choice.

    Returns
    -------
    phi : jax.Array
        Solved potential array.
    residual : jax.Array
        Residual array of system (A @ phi - b).
    manifest : dict
        Solver metadata manifest.
    """
    sources_jnp = jnp.asarray(sources, dtype=jnp.float32)
    N = sources_jnp.shape[0]

    main_diag = -2.0 * jnp.ones(N, dtype=jnp.float32)
    if N > 1:
        main_diag = main_diag.at[0].set(-1.0)
        main_diag = main_diag.at[N - 1].set(-1.0)
    off_diag = jnp.ones(N - 1, dtype=jnp.float32)

    A = jnp.diag(main_diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)

    A = (A / (dx ** 2)) * conductivity
    b = -sources_jnp

    # Solve minimal norm (which satisfies mean-zero gauge)
    phi, _, _, _ = jnp.linalg.lstsq(A, b)

    residual = A @ phi - b
    residual_norm = float(jnp.linalg.norm(residual))

    manifest = {
        "claim_level": "computational_scaffold",
        "field_solver_status": "experimental_pde_solver",
        "boundary_condition": boundary,
        "gauge_choice": gauge,
        "residual_norm": residual_norm,
        "convergence_status": "converged" if residual_norm < 1e-3 else "failed",
        "physical_amplitude_calibrated": False,
    }

    return phi, residual, manifest
