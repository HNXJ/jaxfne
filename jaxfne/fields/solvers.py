"""Experimental PDE field solvers for jaxfne."""

import jax
import jax.numpy as jnp
from typing import Any, Tuple, Dict

def experimental_poisson_1d(
    sources: jax.Array,
    conductivity: "float | jax.Array",
    dx: float,
    boundary: str = "mean_zero_neumann",
    gauge: str = "mean_zero",
) -> Tuple[jax.Array, jax.Array, Dict[str, Any]]:
    """Experimental 1D Poisson solver with Neumann boundaries and mean-zero gauge.

    Solves: d/dx (conductivity * d/dx phi) = -sources

    ``conductivity`` accepts either a scalar (uniform medium, the original
    behavior -- unchanged) or a per-face array of shape ``(N-1,)`` giving the
    conductivity between each pair of adjacent grid nodes (a piecewise-constant
    "layered" medium, e.g. distinct cortical-layer conductivities). The
    layered case discretizes the variable-coefficient flux divergence at cell
    faces (the standard finite-difference treatment for
    ``d/dx(sigma(x) dphi/dx)``, not a per-node conductivity, which would be
    ambiguous at a node sitting exactly on a layer boundary):
    ``[sigma_{i+1/2}(phi_{i+1}-phi_i) - sigma_{i-1/2}(phi_i-phi_{i-1})] / dx**2``.
    Passing a scalar reduces to exactly the original uniform-conductivity
    matrix (verified in tests/test_experimental_poisson_1d_layered.py) --
    zero behavior change for every existing caller.

    Parameters
    ----------
    sources : jax.Array
        1D array of current/charge sources.
    conductivity : float or jax.Array
        Conductivity scalar (uniform medium) or per-face array of shape
        (N-1,) (layered medium).
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

    cond_arr = jnp.asarray(conductivity, dtype=jnp.float32)
    layered = cond_arr.ndim > 0
    if layered:
        if cond_arr.shape[0] != N - 1:
            raise ValueError(
                f"conductivity array must have shape (N-1,)={N - 1}, got {cond_arr.shape}"
            )
        sigma_face = cond_arr  # sigma_{i+1/2} for i in [0, N-2]
        main_diag = jnp.zeros(N, dtype=jnp.float32)
        main_diag = main_diag.at[1:-1].add(-(sigma_face[:-1] + sigma_face[1:])) if N > 2 else main_diag
        main_diag = main_diag.at[0].set(-sigma_face[0])
        main_diag = main_diag.at[N - 1].set(-sigma_face[-1])
        A = jnp.diag(main_diag) + jnp.diag(sigma_face, 1) + jnp.diag(sigma_face, -1)
        A = A / (dx ** 2)
    else:
        main_diag = -2.0 * jnp.ones(N, dtype=jnp.float32)
        if N > 1:
            main_diag = main_diag.at[0].set(-1.0)
            main_diag = main_diag.at[N - 1].set(-1.0)
        off_diag = jnp.ones(N - 1, dtype=jnp.float32)
        A = jnp.diag(main_diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        A = (A / (dx ** 2)) * cond_arr

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
        "layered_conductivity": bool(layered),
    }

    return phi, residual, manifest
