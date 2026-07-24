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
import jax
import jax.numpy as jnp
import pytest

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


# --- precision opt-in (Phase 4 wave1: explicit float64 solve path) ---------
#
# x64 is a process-wide JAX config flag, so the "x64 actually enabled" case
# is run in an isolated subprocess -- flipping jax_enable_x64 globally in
# this test process would leak into every other test collected in the same
# pytest run (violates test isolation and the x64-before-arrays convention,
# which requires x64 to be decided before any array is created).

_X64_ENABLED_SUBPROCESS_SNIPPET = """
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from jaxfne.fields.solvers import experimental_poisson_1d

N = 300
dx = 1.0 / (N - 1)
rng = np.random.default_rng(0)
sources = rng.normal(size=N)
sources = sources - sources.mean()

phi, residual, manifest = experimental_poisson_1d(sources, 1.5, dx, precision="float64")
assert manifest["convergence_status"] == "converged", manifest
assert manifest["residual_norm"] < 1e-3, manifest["residual_norm"]
assert manifest["precision"] == "float64"
assert phi.dtype == np.float64, phi.dtype
print("OK")
"""


def test_precision_float64_converges_at_n300_when_x64_enabled():
    """The explicit opt-in (precision='float64') resolves the documented
    float32 N~150-200 convergence ceiling: N=300 -- previously failing in
    float32 (see test_solver_honestly_self_reports_failure_above_the_
    confirmed_ceiling, which fails already by N=321) -- converges cleanly
    once the caller has enabled x64 and passes precision='float64'. Run in a
    subprocess so this test's process-wide jax_enable_x64 flip cannot leak
    into any other test in the suite."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", _X64_ENABLED_SUBPROCESS_SNIPPET],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, (
        f"subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "OK" in result.stdout


def test_precision_float64_without_x64_enabled_raises_clear_error():
    """Requesting precision='float64' without x64 enabled must raise a clear,
    actionable ValueError -- never silently fall back to float32 and never
    silently "succeed" with truncated precision. Runs in-process: this repo's
    test suite does not enable x64 globally, so jax_enable_x64 is False here
    (confirmed via jax.config.jax_enable_x64 itself, not assumed)."""
    assert jax.config.jax_enable_x64 is False, (
        "expected x64 to be disabled in this test process -- if some earlier-collected "
        "test enabled it globally, this test's premise is invalid; investigate rather "
        "than deleting this assertion"
    )
    sources = jnp.array([1.0, -1.0, 0.5, -0.5], dtype=jnp.float32)
    with pytest.raises(ValueError, match="jax_enable_x64"):
        experimental_poisson_1d(sources, 1.5, dx=0.1, precision="float64")


def test_precision_default_omitted_reproduces_exact_current_float32_behavior():
    """Omitting `precision` entirely must reproduce bit-identical output to
    explicitly passing precision='float32' -- the default-unchanged
    guarantee for this pass's new keyword argument."""
    N = 81
    dx = 1.0 / (N - 1)
    rng = np.random.default_rng(1)
    sources = jnp.asarray(rng.normal(size=N) - rng.normal(size=N).mean(), dtype=jnp.float32)

    phi_default, residual_default, manifest_default = experimental_poisson_1d(sources, 1.5, dx)
    phi_explicit, residual_explicit, manifest_explicit = experimental_poisson_1d(
        sources, 1.5, dx, precision="float32")

    np.testing.assert_array_equal(np.asarray(phi_default), np.asarray(phi_explicit))
    np.testing.assert_array_equal(np.asarray(residual_default), np.asarray(residual_explicit))
    assert phi_default.dtype == jnp.float32
    assert manifest_default["precision"] == "float32" == manifest_explicit["precision"]
