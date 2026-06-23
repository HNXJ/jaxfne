"""Cable-filter tensor: depth/cell-type-dependent passive-cable low-pass.

Validated standard pipeline stage: emitter -> source_scale gain -> source ->
cable_filter_sources -> readout (LFP/EEG/MEG). See jaxfne/fields/proxy.py.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne.fields import cable_filter_tau, cable_filter_sources, cable_filter_report


def test_cable_filter_tau_ordering_and_shape():
    """PV shortest tau, E deep longest, depth-graded E in between."""
    cell_type = np.array(["E", "E", "PV", "SST", "VIP"])
    depth_z = np.array([0.0, 1.0, 0.5, 0.5, 0.5])

    tau_s = cable_filter_tau(cell_type, depth_z)

    assert tau_s.shape == (5,)
    assert jnp.all(jnp.isfinite(tau_s))
    assert jnp.all(tau_s > 0.0)
    # PV is the shortest cable time constant (highest cutoff, passes gamma everywhere)
    assert float(tau_s[2]) < float(tau_s[3])
    assert float(tau_s[2]) < float(tau_s[0])
    # Superficial E (depth=0) has shorter tau than deep E (depth=1)
    assert float(tau_s[0]) < float(tau_s[1])


def test_cable_filter_tau_unrecognized_cell_type_falls_back():
    """Unrecognized cell types fall back to the SST/VIP tau, not zero/NaN."""
    cell_type = np.array(["E", "UNKNOWN"])
    depth_z = np.array([0.0, 0.5])
    tau_s = cable_filter_tau(cell_type, depth_z, tau_sst_ms=2.0)
    assert jnp.all(jnp.isfinite(tau_s))
    assert float(tau_s[1]) == pytest.approx(2.0 / 1000.0)


def test_cable_filter_tau_length_mismatch_raises():
    with pytest.raises(ValueError):
        cable_filter_tau(np.array(["E", "PV"]), np.array([0.0]))


def test_cable_filter_sources_shape_and_finite():
    rng = np.random.default_rng(0)
    T, N = 512, 6
    sources = rng.normal(size=(T, N)).astype(np.float32)
    cell_type = np.array(["E", "E", "PV", "SST", "VIP", "E"])
    depth_z = np.array([0.0, 1.0, 0.5, 0.3, 0.7, 0.4])
    tau_s = cable_filter_tau(cell_type, depth_z)

    filtered = cable_filter_sources(sources, tau_s, dt_ms=0.5, order=2)

    assert filtered.shape == sources.shape
    assert jnp.all(jnp.isfinite(filtered))


def test_cable_filter_sources_attenuates_high_frequency_more_than_low():
    """A long-tau neuron should attenuate a fast oscillation more than a slow one,
    relative to its own DC/low-frequency gain — the defining low-pass behavior."""
    T = 4000
    dt_ms = 0.5
    dt_s = dt_ms / 1000.0
    t = np.arange(T) * dt_s

    low_freq_hz = 5.0
    high_freq_hz = 100.0
    low = np.sin(2.0 * np.pi * low_freq_hz * t)
    high = np.sin(2.0 * np.pi * high_freq_hz * t)
    sources = np.stack([low, high], axis=1).astype(np.float32)

    # one neuron, long tau (low cutoff) applied to BOTH columns identically
    tau_s = jnp.asarray([0.02, 0.02], dtype=jnp.float32)
    filtered = np.asarray(cable_filter_sources(sources, tau_s, dt_ms, order=2))

    low_amp_ratio = np.std(filtered[:, 0]) / np.std(sources[:, 0])
    high_amp_ratio = np.std(filtered[:, 1]) / np.std(sources[:, 1])

    assert high_amp_ratio < low_amp_ratio


def test_cable_filter_sources_dimension_mismatch_raises():
    sources = jnp.ones((10, 4), dtype=jnp.float32)
    tau_s = jnp.ones((3,), dtype=jnp.float32)
    with pytest.raises(ValueError):
        cable_filter_sources(sources, tau_s, dt_ms=0.5)


def test_cable_filter_report_truth_gates():
    tau_s = jnp.asarray([0.001, 0.005, 0.0005], dtype=jnp.float32)
    report = cable_filter_report(tau_s, order=2)

    assert report["field_solver_status"] == "linear_solver"
    assert report["physical_amplitude_calibrated"] is False
    assert report["claim_level"] == "computational_scaffold"
    assert report["finite_tau"] is True
    assert report["order"] == 2


def test_cable_filter_composes_with_project_laminar_sources():
    """Standard pipeline shape: source -> cable_filter_sources -> project_laminar_sources."""
    rng = np.random.default_rng(1)
    T, N = 256, 10
    sources = rng.normal(size=(T, N)).astype(np.float32)
    cell_type = np.array(["E"] * 6 + ["PV", "SST", "VIP", "E"])
    depth_z = np.linspace(0.0, 1.0, N)
    positions = np.stack([np.zeros(N), np.zeros(N), depth_z], axis=1)

    tau_s = cable_filter_tau(cell_type, depth_z)
    filtered = cable_filter_sources(sources, tau_s, dt_ms=0.5, order=2)
    field_out = jtfne.project_laminar_sources(filtered, positions, n_contacts=8)

    assert jnp.all(jnp.isfinite(field_out.lfp_proxy))
    assert field_out.lfp_proxy.shape == (T, 8)
