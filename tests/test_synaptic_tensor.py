"""Synaptic Tensor: mechanism-correct per-edge tau + standalone synaptic filter.

Additive only: jaxfne.synaptic_tau_from_mechanism, jaxfne.synaptic_current_tensor,
jaxfne.synaptic_tensor_report. Does NOT change core._compile_connection_rules's
existing weight-sign-only tau inference -- that inertness fix is deferred.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne.emitters import (
    synaptic_tau_from_mechanism,
    synaptic_current_tensor,
    synaptic_tensor_report,
    standard_receptor_tau_table,
)


def test_synaptic_tau_from_mechanism_matches_standard_table():
    mechanisms = ["AMPA", "GABA_A", "NMDA", "GABA_B"]
    tau_ms = synaptic_tau_from_mechanism(mechanisms)
    table = standard_receptor_tau_table()

    assert tau_ms.shape == (4,)
    np.testing.assert_allclose(np.asarray(tau_ms), np.asarray(table))


def test_synaptic_tau_from_mechanism_unrecognized_raises():
    with pytest.raises(ValueError):
        synaptic_tau_from_mechanism(["AMPA", "BOGUS"])


def test_synaptic_current_tensor_shape_and_finite():
    rng = np.random.default_rng(0)
    T, E = 256, 4
    spikes = (rng.random((T, E)) < 0.05).astype(np.float32)
    tau_ms = jnp.asarray([2.0, 5.0, 100.0, 150.0], dtype=jnp.float32)

    trace = synaptic_current_tensor(jnp.asarray(spikes), tau_ms, dt_ms=0.5)

    assert trace.shape == (T, E)
    assert jnp.all(jnp.isfinite(trace))


def test_synaptic_current_tensor_dimension_mismatch_raises():
    spikes = jnp.ones((10, 3), dtype=jnp.float32)
    tau_ms = jnp.ones((4,), dtype=jnp.float32)
    with pytest.raises(ValueError):
        synaptic_current_tensor(spikes, tau_ms, dt_ms=0.5)


def test_synaptic_current_tensor_nmda_sustains_more_than_ampa():
    """Falsification: a long-tau (NMDA) channel must integrate/sustain a periodic
    spike train far more than a short-tau (AMPA) channel, given identical input."""
    dt_ms = 0.5
    n_steps = int(500.0 / dt_ms)
    spike_every = int(50.0 / dt_ms)
    spikes_1ch = np.zeros(n_steps, dtype=np.float32)
    spikes_1ch[::spike_every] = 1.0
    spikes = np.tile(spikes_1ch[:, None], (1, 4)).astype(np.float32)

    mechanisms = ["AMPA", "GABA_A", "NMDA", "GABA_B"]
    tau_ms = synaptic_tau_from_mechanism(mechanisms)
    trace = np.asarray(synaptic_current_tensor(jnp.asarray(spikes), tau_ms, dt_ms))

    ampa_mean = trace[:, 0].mean()
    nmda_mean = trace[:, 2].mean()
    assert nmda_mean / ampa_mean > 5.0


def test_synaptic_tensor_report_truth_gates():
    tau_ms = jnp.asarray([2.0, 100.0], dtype=jnp.float32)
    report = synaptic_tensor_report(tau_ms, mechanism=["AMPA", "NMDA"])

    assert report["finite_tau"] is True
    assert report["physical_amplitude_calibrated"] is False
    assert report["claim_level"] == "computational_scaffold"
    assert report["source_calibration_status"] == "metadata_only_uncalibrated"
    assert report["mechanism"] == ["AMPA", "NMDA"]


def test_synaptic_tensor_exported_at_top_level():
    assert jtfne.synaptic_tau_from_mechanism is synaptic_tau_from_mechanism
    assert jtfne.synaptic_current_tensor is synaptic_current_tensor
    assert jtfne.synaptic_tensor_report is synaptic_tensor_report
