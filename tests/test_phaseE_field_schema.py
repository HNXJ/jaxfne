"""Field schema stability tests for Phase E (roadmap E-04).

One test per public solver entry point, asserting the canonical metadata trio
on each solver's actual output surface:

1. ``field_claim_level`` exists and equals ``"proxy_readout"``
2. ``field_solver_status`` exists and equals the source-derived expected value
   (``"linear_solver"`` for the proxy projection, ``"experimental_pde_solver"``
   for both Poisson entry points)
3. ``physical_amplitude_calibrated`` is explicitly ``False``

Also asserts output arrays are finite on minimal deterministic inputs. This is
a metadata-contract suite, not a numerical-accuracy suite.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxfne.fields import (
    experimental_poisson_1d,
    experimental_poisson_1d_from_neuron_table,
    project_laminar_sources,
)


def _minimal_sources_positions():
    """Minimal deterministic 2D sources/positions for the proxy projection."""
    sources = jnp.ones((10, 4), dtype=jnp.float32)
    positions = jnp.array(
        [[0.0, 0.0, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.75], [0.0, 0.0, 0.9]],
        dtype=jnp.float32,
    )
    return sources, positions


def _minimal_neuron_table():
    """Minimal deterministic neuron-table rows with numeric z depth."""
    return [{"z": 0.25}, {"z": 0.5}, {"z": 0.75}, {"z": 0.9}]


class TestProjectLaminarSourcesSchema:
    """Canonical trio on the proxy projection's FieldOutput diagnostics."""

    def test_field_claim_level_proxy_readout(self):
        sources, positions = _minimal_sources_positions()
        field = project_laminar_sources(sources, positions, n_contacts=8)
        assert field.diagnostics["field_claim_level"] == "proxy_readout"

    def test_field_solver_status_linear_solver(self):
        sources, positions = _minimal_sources_positions()
        field = project_laminar_sources(sources, positions, n_contacts=8)
        assert field.diagnostics["field_solver_status"] == "linear_solver"

    def test_physical_amplitude_calibrated_false(self):
        sources, positions = _minimal_sources_positions()
        field = project_laminar_sources(sources, positions, n_contacts=8)
        assert field.diagnostics["physical_amplitude_calibrated"] is False

    def test_output_arrays_finite(self):
        sources, positions = _minimal_sources_positions()
        field = project_laminar_sources(sources, positions, n_contacts=8)
        assert bool(jnp.all(jnp.isfinite(field.source_proxy)))
        assert bool(jnp.all(jnp.isfinite(field.phi_e_proxy)))
        assert bool(jnp.all(jnp.isfinite(field.csd_proxy)))
        assert bool(jnp.all(jnp.isfinite(field.lfp_proxy)))


class TestExperimentalPoisson1dSchema:
    """Canonical trio on the Poisson solver's returned manifest."""

    def test_field_claim_level_proxy_readout(self):
        _, _, manifest = experimental_poisson_1d(
            jnp.ones(8, dtype=jnp.float32), 1.5, 0.1
        )
        assert manifest["field_claim_level"] == "proxy_readout"

    def test_field_solver_status_experimental_pde_solver(self):
        _, _, manifest = experimental_poisson_1d(
            jnp.ones(8, dtype=jnp.float32), 1.5, 0.1
        )
        assert manifest["field_solver_status"] == "experimental_pde_solver"

    def test_physical_amplitude_calibrated_false(self):
        _, _, manifest = experimental_poisson_1d(
            jnp.ones(8, dtype=jnp.float32), 1.5, 0.1
        )
        assert manifest["physical_amplitude_calibrated"] is False

    def test_output_arrays_finite(self):
        phi, residual, _ = experimental_poisson_1d(
            jnp.ones(8, dtype=jnp.float32), 1.5, 0.1
        )
        assert bool(jnp.all(jnp.isfinite(phi)))
        assert bool(jnp.all(jnp.isfinite(residual)))


class TestExperimentalPoisson1dFromNeuronTableSchema:
    """Canonical trio on the bridge's returned manifest (inherited from the
    base solver, plus bin bookkeeping)."""

    def test_field_claim_level_proxy_readout(self):
        _, _, manifest = experimental_poisson_1d_from_neuron_table(
            _minimal_neuron_table(), jnp.ones(4, dtype=jnp.float32), 1.0, 5
        )
        assert manifest["field_claim_level"] == "proxy_readout"

    def test_field_solver_status_experimental_pde_solver(self):
        _, _, manifest = experimental_poisson_1d_from_neuron_table(
            _minimal_neuron_table(), jnp.ones(4, dtype=jnp.float32), 1.0, 5
        )
        assert manifest["field_solver_status"] == "experimental_pde_solver"

    def test_physical_amplitude_calibrated_false(self):
        _, _, manifest = experimental_poisson_1d_from_neuron_table(
            _minimal_neuron_table(), jnp.ones(4, dtype=jnp.float32), 1.0, 5
        )
        assert manifest["physical_amplitude_calibrated"] is False

    def test_output_arrays_finite(self):
        phi, residual, manifest = experimental_poisson_1d_from_neuron_table(
            _minimal_neuron_table(), jnp.ones(4, dtype=jnp.float32), 1.0, 5
        )
        assert bool(jnp.all(jnp.isfinite(phi)))
        assert bool(jnp.all(jnp.isfinite(residual)))
        assert len(manifest["bin_edges"]) == 5
        assert sum(manifest["neurons_per_bin"]) == 4
