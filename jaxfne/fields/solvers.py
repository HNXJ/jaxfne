"""Experimental PDE field solvers for jaxfne."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Mapping, Sequence, Tuple, Dict

def experimental_poisson_1d(
    sources: jax.Array,
    conductivity: "float | jax.Array",
    dx: float,
    boundary: str = "mean_zero_neumann",
    gauge: str = "mean_zero",
    precision: str = "float32",
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

    KNOWN LIMITATION (confirmed 2026-07-18, both conductivity paths): the
    dense ``jnp.linalg.lstsq`` solve's residual grows sharply above roughly
    N~150-200 grid points in float32 -- ``convergence_status`` correctly
    self-reports ``"failed"`` in that regime rather than returning a silently
    wrong answer (see tests/test_experimental_poisson_1d_convergence.py,
    which reproduces this on both the scalar and layered paths and confirms
    it is NOT fixed by removing the 1/dx**2 matrix-scale factor before
    solving -- that hypothesis was tested and rejected). Root cause is most
    likely the dense discrete-Laplacian solve's inherent conditioning at
    this size, not yet addressed; treat this function as validated only up
    to roughly N~150 until a better-conditioned or sparse solve replaces it.

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
    precision : str, default "float32"
        Numerical dtype for the assembly + solve. The default ``"float32"``
        is the original, unchanged behavior (bit-identical to every prior
        caller -- see tests/test_experimental_poisson_1d_layered.py's
        ``test_original_smoke_case_unchanged``). Pass ``"float64"`` to opt
        into the real double-precision solve path that resolves the
        float32 convergence ceiling documented above (float64 converges
        cleanly at every N up to 500 with the identical dense-Laplacian +
        ``jnp.linalg.lstsq`` method -- confirmed empirically, not merely
        expected). Requesting ``"float64"`` REQUIRES the caller to have
        already run ``jax.config.update("jax_enable_x64", True)`` before
        any array in this process was created (this repo's standing
        x64-before-arrays convention, see the
        ``neuro-biophysics-units-sanity`` skill) -- if x64 is not enabled,
        this raises ``ValueError`` rather than silently downgrading to
        float32 or silently returning truncated double-precision-looking
        output that is actually still float32 under the hood. This
        function never mutates the global JAX config itself -- enabling
        x64 is the caller's responsibility, not this function's.

    Returns
    -------
    phi : jax.Array
        Solved potential array.
    residual : jax.Array
        Residual array of system (A @ phi - b).
    manifest : dict
        Solver metadata manifest.
    """
    if precision not in ("float32", "float64"):
        raise ValueError(
            f"precision must be 'float32' or 'float64', got {precision!r}"
        )
    if precision == "float64" and not jax.config.jax_enable_x64:
        raise ValueError(
            "experimental_poisson_1d(precision='float64') requires x64 to already be "
            "enabled -- call jax.config.update('jax_enable_x64', True) before creating "
            "any array in this process (this repo's standing x64-before-arrays "
            "convention). Refusing to silently fall back to float32 or silently return "
            "float64-labeled-but-actually-truncated output."
        )
    solve_dtype = jnp.float64 if precision == "float64" else jnp.float32

    sources_jnp = jnp.asarray(sources, dtype=solve_dtype)
    N = sources_jnp.shape[0]

    cond_arr = jnp.asarray(conductivity, dtype=solve_dtype)
    layered = cond_arr.ndim > 0
    if layered:
        if cond_arr.shape[0] != N - 1:
            raise ValueError(
                f"conductivity array must have shape (N-1,)={N - 1}, got {cond_arr.shape}"
            )
        sigma_face = cond_arr  # sigma_{i+1/2} for i in [0, N-2]
        main_diag = jnp.zeros(N, dtype=solve_dtype)
        main_diag = main_diag.at[1:-1].add(-(sigma_face[:-1] + sigma_face[1:])) if N > 2 else main_diag
        main_diag = main_diag.at[0].set(-sigma_face[0])
        main_diag = main_diag.at[N - 1].set(-sigma_face[-1])
        A = jnp.diag(main_diag) + jnp.diag(sigma_face, 1) + jnp.diag(sigma_face, -1)
        A = A / (dx ** 2)
    else:
        main_diag = -2.0 * jnp.ones(N, dtype=solve_dtype)
        if N > 1:
            main_diag = main_diag.at[0].set(-1.0)
            main_diag = main_diag.at[N - 1].set(-1.0)
        off_diag = jnp.ones(N - 1, dtype=solve_dtype)
        A = jnp.diag(main_diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
        A = (A / (dx ** 2)) * cond_arr

    b = -sources_jnp

    # Solve minimal norm (which satisfies mean-zero gauge)
    phi, _, _, _ = jnp.linalg.lstsq(A, b)

    residual = A @ phi - b
    residual_norm = float(jnp.linalg.norm(residual))

    manifest = {
        "claim_level": "computational_scaffold",
        "operator_type": "pde_solve",
        "representation": "relative",
        "validation_status": "numerical",
        "calibration_transform": "explicit_boundary_transform",
        "field_solver_status": "experimental_pde_solver",
        "field_claim_level": "proxy_readout",
        "boundary_condition": boundary,
        "gauge_choice": gauge,
        "residual_norm": residual_norm,
        "convergence_status": "converged" if residual_norm < 1e-3 else "failed",
        "physical_amplitude_calibrated": False,
        "layered_conductivity": bool(layered),
        "precision": precision,
    }

    return phi, residual, manifest


def experimental_poisson_1d_from_neuron_table(
    neuron_table: Sequence[Mapping[str, Any]],
    sources: "jax.Array | np.ndarray",
    conductivity: "float | jax.Array",
    n_bins: int,
    z_min: "float | None" = None,
    z_max: "float | None" = None,
    boundary: str = "mean_zero_neumann",
    gauge: str = "mean_zero",
) -> Tuple[jax.Array, jax.Array, Dict[str, Any]]:
    """Bridge real per-neuron depth positions and source values into
    :func:`experimental_poisson_1d`'s 1D depth grid -- the first real
    integration of the (previously standalone) experimental solver with the
    Model/Signals object model, toward
    ``plans.json:novelty::tfne-differentiable-field-solver``. Does NOT
    touch the existing ``project_laminar_sources``/``simulate()`` dispatch
    (that machinery is unchanged) -- this is a separate, explicitly opt-in
    accessor a caller invokes after ``construct()``/``simulate()``, not a new
    default source-projection mode.

    Parameters
    ----------
    neuron_table : sequence of dict
        From ``Model.neuron_table()``. Each row must have a numeric ``"z"``
        (depth) field -- raises ``ValueError`` if any row's ``z`` is
        ``None`` (e.g. a model built without laminar position metadata).
    sources : array, shape (n_neurons,)
        Per-neuron source values -- e.g. a single time-slice of
        ``Signals.sources``, or a caller-computed time-average. Neurons
        falling in the same depth bin are SUMMED (not averaged): jaxfne's
        source values are current-like quantities, so summing multiple
        neurons' contributions at the same depth is the physically sensible
        aggregation. Per :func:`experimental_poisson_1d`'s own established
        convention (confirmed empirically in
        tests/test_experimental_poisson_1d_convergence.py: a node value
        ``Q`` produces a discrete flux jump of ``Q*dx``, not ``Q``), the
        summed value at each bin is used AS-IS as that bin's density-like
        input -- no additional density normalization is applied. This is an
        explicit modeling choice, not a hidden one: if a caller wants a
        specific total injected current at a depth, they should pre-scale
        their input accordingly.
    conductivity : float or jax.Array
        Conductivity scalar or per-face array of shape ``(n_bins-1,)`` --
        caller-supplied; jaxfne has no calibrated conductivity data, so this
        is never inferred from the network itself.
    n_bins : int
        Number of depth-grid nodes for the 1D solve. Per
        :func:`experimental_poisson_1d`'s documented limitation, keep this
        below roughly 150 for a reliably converged solve.
    z_min, z_max : float, optional
        Depth range for the grid; default to the min/max ``z`` actually
        present in ``neuron_table``.
    boundary, gauge : str
        Forwarded to :func:`experimental_poisson_1d`.

    Returns
    -------
    phi, residual : jax.Array
        As :func:`experimental_poisson_1d`.
    manifest : dict
        As :func:`experimental_poisson_1d`, plus ``"bin_edges"`` (the depth
        grid, shape ``(n_bins,)``) and ``"neurons_per_bin"`` (shape
        ``(n_bins,)``, for transparency about how many neurons contributed
        to each bin -- an empty bin gets source value 0, not an error).
    """
    z_values = np.array([row.get("z") for row in neuron_table], dtype=object)
    if any(z is None for z in z_values):
        raise ValueError(
            "experimental_poisson_1d_from_neuron_table requires every neuron_table row "
            "to have a numeric 'z' -- got at least one None (model built without laminar "
            "position metadata?)."
        )
    z_values = z_values.astype(np.float64)
    sources_np = np.asarray(sources, dtype=np.float64)
    if sources_np.shape[0] != z_values.shape[0]:
        raise ValueError(
            f"sources length {sources_np.shape[0]} must match neuron_table length "
            f"{z_values.shape[0]}"
        )

    z_lo = float(z_values.min()) if z_min is None else float(z_min)
    z_hi = float(z_values.max()) if z_max is None else float(z_max)
    if z_hi <= z_lo:
        raise ValueError(f"z_max ({z_hi}) must be greater than z_min ({z_lo})")

    bin_edges = np.linspace(z_lo, z_hi, n_bins)
    dx = float(bin_edges[1] - bin_edges[0])
    bin_index = np.clip(
        np.searchsorted(bin_edges, z_values, side="right") - 1, 0, n_bins - 1
    )

    binned_sources = np.zeros(n_bins, dtype=np.float64)
    neurons_per_bin = np.zeros(n_bins, dtype=np.int64)
    np.add.at(binned_sources, bin_index, sources_np)
    np.add.at(neurons_per_bin, bin_index, 1)

    phi, residual, manifest = experimental_poisson_1d(
        jnp.asarray(binned_sources, dtype=jnp.float32), conductivity, dx,
        boundary=boundary, gauge=gauge,
    )
    manifest = dict(manifest)
    manifest["bin_edges"] = bin_edges.tolist()
    manifest["neurons_per_bin"] = neurons_per_bin.tolist()
    return phi, residual, manifest
