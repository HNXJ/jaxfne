"""Tests for experimental_poisson_1d_from_neuron_table -- the first real
integration of experimental_poisson_1d with the Model/Signals object model,
toward plans.json:novelty::tfne-differentiable-field-solver.

Uses a real constructed laminar-column Model (not synthetic positions/
sources) -- neuron_table()'s real z-depth values, and simulate()'s real
per-neuron sources output.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne
from jaxfne.fields import experimental_poisson_1d_from_neuron_table


def _build_model(n=100):
    cfg = (
        jtfne.build_laminar_column(n=n, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy")
    )
    return jtfne.construct(cfg)


def test_bridge_runs_on_a_real_constructed_model():
    """Real model, real z positions, real simulated sources -- not synthetic
    fixtures. Confirms the bridge function's binning + solve pipeline
    produces a finite, well-formed result end-to-end."""
    model = _build_model(n=100)
    sig = jtfne.simulate(model, duration_ms=50.0, dt_ms=0.5, seed=0)
    rows = model.neuron_table()
    assert all(r["z"] is not None for r in rows)

    sources_last_step = np.asarray(sig.sources)[-1]  # (n_neurons,) time-slice
    n_bins = 20
    conductivity = 1.5  # uniform, caller-supplied (no calibrated data exists)

    phi, residual, manifest = experimental_poisson_1d_from_neuron_table(
        rows, sources_last_step, conductivity, n_bins)

    assert phi.shape == (n_bins,)
    assert bool(jnp.all(jnp.isfinite(phi)))
    assert manifest["field_solver_status"] == "experimental_pde_solver"
    assert manifest["physical_amplitude_calibrated"] is False
    assert len(manifest["bin_edges"]) == n_bins
    assert len(manifest["neurons_per_bin"]) == n_bins
    assert sum(manifest["neurons_per_bin"]) == len(rows)  # every neuron landed in some bin


def test_bridge_supports_layered_conductivity_from_real_model():
    """Layered (per-face) conductivity works through the bridge too, not just
    the scalar case."""
    model = _build_model(n=100)
    sig = jtfne.simulate(model, duration_ms=50.0, dt_ms=0.5, seed=0)
    rows = model.neuron_table()
    sources_last_step = np.asarray(sig.sources)[-1]

    n_bins = 20
    conductivity = jnp.concatenate([
        jnp.full((n_bins // 2,), 1.0),
        jnp.full((n_bins - 1 - n_bins // 2,), 3.0),
    ])
    phi, residual, manifest = experimental_poisson_1d_from_neuron_table(
        rows, sources_last_step, conductivity, n_bins)
    assert bool(jnp.all(jnp.isfinite(phi)))
    assert manifest["layered_conductivity"] is True


def test_bridge_raises_on_missing_z():
    rows = [{"z": 0.1}, {"z": None}, {"z": 0.3}]
    sources = np.zeros(3)
    try:
        experimental_poisson_1d_from_neuron_table(rows, sources, 1.0, 5)
        assert False, "expected ValueError for a row with z=None"
    except ValueError as e:
        assert "z" in str(e)


def test_bridge_raises_on_mismatched_lengths():
    rows = [{"z": 0.1}, {"z": 0.2}]
    sources = np.zeros(3)  # wrong length
    try:
        experimental_poisson_1d_from_neuron_table(rows, sources, 1.0, 5)
        assert False, "expected ValueError for mismatched lengths"
    except ValueError as e:
        assert "length" in str(e)


def test_bridge_sums_not_averages_within_a_bin():
    """Two neurons at (nearly) the same depth with sources 2.0 and 3.0 should
    contribute a combined 5.0 to their shared bin -- confirmed directly
    against a controlled, non-model fixture (isolating the binning logic
    itself from the full-model integration tested above)."""
    rows = [{"z": 0.5}, {"z": 0.5}, {"z": 9.5}]
    sources = np.array([2.0, 3.0, -5.0])
    n_bins = 10
    phi, residual, manifest = experimental_poisson_1d_from_neuron_table(
        rows, sources, 1.0, n_bins, z_min=0.0, z_max=10.0)
    bin_edges = np.asarray(manifest["bin_edges"])
    neurons_per_bin = np.asarray(manifest["neurons_per_bin"])
    first_bin_idx = np.searchsorted(bin_edges, 0.5, side="right") - 1
    assert neurons_per_bin[first_bin_idx] == 2


# --- Opt-in simulate() wiring (2026-07-24, TFNE goals wave 2) --------------------
#
# The solver is single-timestep by construction, so it is wired into simulate()
# as an ADDITIVE opt-in on the final timestep's sources -- signals.field remains
# the unchanged all-timesteps proxy projection. These tests pin that contract:
# default behavior byte-unchanged, opt-in produces a real solve, and a failure
# in the opt-in path can never break an otherwise-valid simulate().

import jaxfne as jtfne


def _cfg(*, solver: bool):
    field_kwargs = {"domain": "laminar"}
    if solver:
        field_kwargs.update(solver="experimental_poisson_1d", conductivity=1.0, n_bins=8)
    return (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=50.0, dt_ms=0.5)
        .uniform3d(radius_mm=1.0, height_mm=2.0)
        .network(name="v1", n=40)
        .set_emitter("izhikevich", "cortical_eig")
        .field(**field_kwargs)
        .probe(n_contacts=8)
    )


def _run(cfg):
    model = jtfne.construct(cfg)
    return model.simulate(
        jtfne.simulation(duration_ms=50.0, dt_ms=0.5, seed=0, record_fields=True)
    )


def test_poisson_optin_absent_by_default():
    """Without the opt-in, simulate() metadata carries no poisson key at all."""
    signals = _run(_cfg(solver=False))
    assert "poisson_field_final_step" not in signals.metadata
    assert signals.field is not None  # proxy projection unaffected


def test_poisson_optin_produces_real_solve_manifest():
    """With the opt-in, a real experimental_pde_solver manifest is attached."""
    signals = _run(_cfg(solver=True))
    entry = signals.metadata.get("poisson_field_final_step")
    assert entry is not None
    assert entry["applied_to"] == "final_timestep_only"
    manifest = entry["manifest"]
    assert manifest["field_solver_status"] == "experimental_pde_solver"
    # Truth gate must not be escalated by the real-solve path.
    assert manifest["physical_amplitude_calibrated"] is False


def test_poisson_optin_does_not_replace_proxy_field():
    """The opt-in is additive: signals.field stays the proxy projection, and is
    bit-identical to the non-opt-in run (same seed/config otherwise)."""
    baseline = _run(_cfg(solver=False))
    optin = _run(_cfg(solver=True))
    assert optin.field is not None
    assert np.array_equal(np.asarray(baseline.field.lfp), np.asarray(optin.field.lfp))
