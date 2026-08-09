"""Solver acceptance tests for Phase F (roadmap F-03).

Scope: the existing linear proxy solver ``jaxfne.fields.project_laminar_sources``.
One behavioral test per acceptance-checklist item (see the checklist comment at
the top of ``jaxfne/fields/__init__.py``), plus a zero-source null control:

1. finite field outputs for finite inputs;
2. additive (linear) superposition;
3. jax.jit execution (behavioral, not source inspection);
4. field_claim_level metadata on FieldOutput.diagnostics;
5. amplitude truth gate (physical_amplitude_calibrated is False);
+ zero-source null control (exact zeros, shape, finiteness).

Tolerance note: the projection is float32. The kernel matmul residual is
~2e-7, but the CSD second-derivative stencil amplifies float32 rounding to
~3e-5 at the tested shapes; an ``atol/rtol`` of 1e-4 covers both with one
order of magnitude of headroom, so the superposition assertion uses from
the package's established ``jnp.allclose`` defaults with rtol + atol = 1e-4.

JIT note: ``jax.jit(project_laminar_sources)`` returns the proxy arrays; the
metadata report surface (str/int-carrying diagnostics) is eager-only by
design because JAX jit output validation permits array leaves only. The
metadata assertions therefore run on the eager call — the same surface the
rest of the field suite gates.

The experimental 1D Poisson entry points are intentionally not tested here;
their schema is gated by ``tests/test_phaseE_field_schema.py``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxfne.fields import project_laminar_sources

# float32 superposition tolerance (see module docstring for the derivation).
_RTOL = 1e-4
_ATOL = 1e-4

# Output components inheriting the trace/contact axes.
_COMPONENTS = ("source_proxy", "phi_e_proxy", "csd_proxy", "lfp_proxy")


def _positions(n: int = 4) -> jax.Array:
    return jnp.array(
        [[0.0, 0.0, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.75], [0.0, 0.0, 0.9]],
        dtype=jnp.float32,
    )[:n]


class TestLinearSolverAcceptance:
    """Acceptance aments for the standard linear proxy projection solver."""

    def test_01_finite_outputs_for_finite_inputs(self):
        t_steps, n = 10, 4
        sources = jnp.ones((t_steps, n), dtype=jnp.float32)
        positions = _positions()
        field = project_laminar_sources(
            sources, positions, n_contacts=8
        )
        assert field.source_proxy.shape == (t_steps, 8)
        assert field.phi_e_proxy.shape == (t_steps, 8)
        assert field.csd_proxy.shape == (t_steps, 8)
        assert field.lfp_proxy.shape == (t_steps, 8)
        assert field.kernel.shape == (8, n)
        assert field.contact_depths.shape == (8,)
        for name in _COMPONENTS:
            arr = getattr(field, name)
            assert jnp.all(jnp.isfinite(arr)), f"{name} not finite"

    def test_02_linear_superposition(self):
        n_steps, n = 10, 4
        positions = _positions()
        rng = np.random.default_rng(20260806)
        a = jnp.asarray(rng.random((n_steps, n), dtype=np.float32))
        b = jnp.asarray(rng.random((n_steps, n), dtype=np.float32))

        fa = project_laminar_sources(a, positions, n_contacts=8)
        fb = project_laminar_sources(b, positions, n_contacts=8)
        fab = project_laminar_sources(a + b, positions, n_contacts=8)

        for name in _COMPONENTS:
            lhs = getattr(fab, name)
            rhs = getattr(fa, name) + getattr(fb, name)
            assert jnp.allclose(lhs, rhs, rtol=_RTOL, atol=_ATOL), name

    def test_03_jit_executes_and_finite(self):
        n_steps, n = 10, 4
        sources = jnp.ones((n_steps, n), dtype=jnp.float32)
        positions = _positions()

        jit_solver = jax.jit(
            lambda s, p: project_laminar_sources(s, p, n_contacts=8)
        )
        field = jit_solver(sources, positions)
        for name in _COMPONENTS:
            arr = getattr(field, name)
            assert arr.shape == (n_steps, 8)
            assert jnp.all(jnp.isfinite(arr)), f"jitted {name} not finite"

    def test_04_field_claim_level_metadata(self):
        n_steps, n = 10, 4
        field = project_laminar_sources(
            jnp.ones((n_steps, n), dtype=jnp.float32), _positions(), n_contacts=8
        )
        assert field.diagnostics["field_claim_level"] == "proxy_readout"
        assert field.diagnostics["field_solver_status"] == "linear_solver"

    def test_05_amplitude_truth_gate(self):
        n_steps, n = 10, 4
        field = project_laminar_sources(
            jnp.ones((n_steps, n), dtype=jnp.float32), _positions(), n_contacts=8
        )
        assert field.diagnostics["physical_amplitude_calibrated"] is False

    def test_null_control_zero_sources(self):
        n_steps, n = 10, 4
        zeros = jnp.zeros((n_steps, n), dtype=jnp.float32)
        field = project_laminar_sources(zeros, _positions(), n_contacts=8)

        for name in _COMPONENTS:
            arr = getattr(field, name)
            assert arr.shape == (n_steps, 8)
            assert jnp.all(jnp.isfinite(arr)), f"{name} not finite"
            # zero input -> zero output exactly under the tested float32 dtype.
            assert jnp.all(arr == 0.0), f"{name} not exactly zero"
        assert jnp.all(jnp.isfinite(field.kernel))