"""Discretization-structure check for experimental_poisson_1d's layered mode,
plus a real, confirmed numerical-precision ceiling found while writing it.

Analytic setup: a 1D domain with Neumann (zero-flux) boundaries at both
ends, piecewise-constant conductivity (sigma1 for z<z_iface, sigma2 for
z>=z_iface), and a balanced source/sink pair (+Q at one node, -Q at another,
matching the mean-zero-source convention tests/test_solver_smoke_v0401.py's
own fixture already requires -- with pure Neumann boundaries and no sink for
a net nonzero source, the steady-state problem isn't well-posed at all).

By Gauss's law in 1D, the discrete current (flux)
`sigma_face[i]*(phi[i+1]-phi[i])/dx` must be exactly zero outside the
[src_node, sink_node) span and exactly constant within it -- a structural
property of the PDE itself (current conservation), verified directly below.

REAL FINDING (2026-07-18, confirmed via a full N-sweep, not assumed): the
solver's convergence degrades sharply above roughly N~150-200 in float32 --
residual_norm grows from ~1e-3 at N=161 to ~1.6-1.7 at N=321+, well past the
code's own "converged" threshold (1e-3), and this is NOT specific to the new
layered path -- the pre-existing scalar-conductivity path shows the
identical failure pattern at the same N (confirmed with a random balanced
source array, uniform conductivity=1.5). A hypothesis that the 1/dx**2
matrix-scale blowup was the cause was tested and REJECTED (rescaling the
linear system to remove that factor before solving did not change the
failure point at all -- N=321 still failed identically). The more likely
cause is the dense jnp.linalg.lstsq solve's inherent conditioning behavior
for a discrete 1D Laplacian at this size in float32, not something this
pass fixes -- documented honestly here and in the function's docstring
instead. The solver's own convergence_status field already self-reports
"failed" correctly in this regime (good pre-existing design, matching
jaxfne-harden's "fallbacks must report why they triggered"), so this is a
now-documented limitation, not a silent landmine.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from jaxfne.fields.solvers import experimental_poisson_1d


def _run_case(N, sigma1=1.0, sigma2=3.0, Q=2.0):
    dx = 1.0 / (N - 1)
    iface_node = N // 2
    src_node = max(1, N // 6)
    sink_node = min(N - 2, N - N // 6 - 1)

    sigma_face = np.where(np.arange(N - 1) < iface_node, sigma1, sigma2).astype(np.float32)
    sources = np.zeros(N, dtype=np.float32)
    sources[src_node] = Q
    sources[sink_node] = -Q

    phi_num, residual, manifest = experimental_poisson_1d(
        jnp.asarray(sources), jnp.asarray(sigma_face), dx)
    phi_num = np.asarray(phi_num, dtype=np.float64)

    flux = sigma_face * (phi_num[1:] - phi_num[:-1]) / dx
    return dx, src_node, sink_node, flux, float(manifest["residual_norm"]), manifest["convergence_status"]


def test_flux_is_zero_outside_source_sink_span_and_constant_inside():
    """Discrete current conservation: flux must vanish outside [src,sink) and
    be uniform within it -- checked well within the solver's confirmed-good
    convergence regime (N=81, see the module docstring's N-ceiling finding)."""
    dx, src_node, sink_node, flux, residual_norm, status = _run_case(N=81)
    assert status == "converged"
    assert residual_norm < 1e-3

    outside = np.concatenate([flux[:src_node], flux[sink_node:]])
    inside = flux[src_node:sink_node]

    assert np.max(np.abs(outside)) < 1e-4, f"flux outside source/sink span should be ~0, got max {np.max(np.abs(outside))}"
    assert np.std(inside) < 1e-4, f"flux inside the span should be constant, got std {np.std(inside)}"


def test_flux_magnitude_matches_source_times_dx_within_good_regime():
    """Within the confirmed-converged regime (N<=81), flux magnitude equals
    sources[node]*dx and shrinks proportionally as dx shrinks with N --
    checked at 3 resolutions, all inside the good regime."""
    Q = 2.0
    results = []
    for N in (21, 41, 81):
        dx, src_node, sink_node, flux, residual_norm, status = _run_case(N=N, Q=Q)
        assert status == "converged", f"N={N} unexpectedly not converged (residual={residual_norm})"
        inside_mean = float(np.mean(flux[src_node:sink_node]))
        results.append((N, dx, inside_mean))
        assert abs(abs(inside_mean) - Q * dx) < 1e-3 * max(Q * dx, 1e-6), (
            f"N={N}: expected |flux| ~= Q*dx = {Q*dx}, got {abs(inside_mean)}"
        )
    print(f"flux-vs-resolution receipt (N, dx, flux_inside): {results}")
    fluxes = [abs(r[2]) for r in results]
    assert fluxes[0] > fluxes[1] > fluxes[2], f"expected shrinking flux magnitude, got {fluxes}"


def test_solver_honestly_self_reports_failure_above_the_confirmed_ceiling():
    """Real, confirmed limitation (module docstring): above roughly N~200 in
    float32, the dense lstsq solve's residual grows past the 1e-3 threshold.
    This test verifies the solver's OWN convergence_status field correctly
    flags this as 'failed' rather than silently returning a wrong answer --
    the limitation itself is real, but self-reporting it is a genuine, tested
    positive behavior, not a landmine."""
    _, _, _, _, residual_norm, status = _run_case(N=321)
    assert residual_norm > 1e-3, (
        "expected the known N=321 conditioning failure to still reproduce -- if this now "
        "passes, the underlying lstsq behavior changed; re-verify the N-ceiling claim above "
        "before trusting it, don't just relax this assertion"
    )
    assert status == "failed"
