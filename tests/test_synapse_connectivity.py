"""Tests for connectivity and synapse operators (Patch B)."""

import json

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.fields import (
    exponential_synaptic_trace,
    make_laminar_connectivity,
    synaptic_current,
)


class MockNeuronsDataFrame:
    """Mock neuron dataframe for testing."""

    def __init__(self, n=20, areas=None, layers=None, cell_types=None):
        self.n = n
        self.data = {
            "area": areas or (["V1"] * (n // 2) + ["PFC"] * (n - n // 2)),
            "layer": layers or (["L4"] * (n // 2) + ["L5"] * (n - n // 2)),
            "cell_type": cell_types or (["E"] * int(0.75 * n) + ["PV"] * (n - int(0.75 * n))),
        }

    def __len__(self):
        return self.n

    def get(self, key, default=None):
        return self.data.get(key, default)


def test_connectivity_matrix_shape():
    """Test that connectivity matrix has correct shape [N, N]."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)

    result = make_laminar_connectivity(neurons, positions, seed=42)

    assert result["W"].shape == (20, 20)
    assert result["E_mask"].shape == (20,)
    assert result["I_mask"].shape == (20,)


def test_connectivity_no_self_connections():
    """Test that diagonal (self-connections) is zero."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)

    result = make_laminar_connectivity(neurons, positions, seed=42)
    W = np.asarray(result["W"])

    # Diagonal should be zero (no self-connections)
    assert np.allclose(np.diag(W), 0.0)


def test_connectivity_e_presynaptic_nonnegative():
    """Test that E presynaptic columns are nonnegative."""
    neurons = MockNeuronsDataFrame(n=30)
    positions = np.random.randn(30, 3).astype(np.float32)

    result = make_laminar_connectivity(neurons, positions, seed=42)
    W = np.asarray(result["W"])
    E_mask = np.asarray(result["E_mask"])

    # Excitatory presynaptic columns should be >= 0
    for i in np.where(E_mask)[0]:
        assert np.all(W[:, i] >= -1e-6), f"E neuron {i} has negative outgoing weights"


def test_connectivity_i_presynaptic_nonpositive():
    """Test that I presynaptic columns are nonpositive."""
    neurons = MockNeuronsDataFrame(n=30)
    positions = np.random.randn(30, 3).astype(np.float32)

    result = make_laminar_connectivity(neurons, positions, seed=42)
    W = np.asarray(result["W"])
    I_mask = np.asarray(result["I_mask"])

    # Inhibitory presynaptic columns should be <= 0
    for i in np.where(I_mask)[0]:
        assert np.all(W[:, i] <= 1e-6), f"I neuron {i} has positive outgoing weights"


def test_connectivity_zero_weights_zero_input():
    """Test that zero weights produce zero synaptic input."""
    spikes = np.ones((100, 10), dtype=np.float32)  # Constant spikes
    W = np.zeros((10, 10), dtype=np.float32)  # Zero connectivity

    I_syn = synaptic_current(spikes, W, tau_ms=5.0, dt_ms=0.1)

    assert np.allclose(I_syn, 0.0), "Zero weights should produce zero input"


def test_connectivity_excitatory_spike_increases_input():
    """Test that excitatory spikes increase postsynaptic input."""
    # One E neuron driving one target
    spikes = np.zeros((100, 2), dtype=np.float32)
    spikes[10:20, 0] = 1.0  # E neuron spikes at t=10-20

    W = np.zeros((2, 2), dtype=np.float32)
    W[1, 0] = 0.5  # E neuron 0 → neuron 1 with weight +0.5

    I_syn = synaptic_current(spikes, W, tau_ms=5.0, dt_ms=0.1)

    # Neuron 1 should have positive input during and after spike
    assert np.any(I_syn[15:25, 1] > 0.1), "Excitatory input should be positive"


def test_connectivity_inhibitory_spike_decreases_input():
    """Test that inhibitory spikes decrease postsynaptic input."""
    # One I neuron driving one target
    spikes = np.zeros((100, 2), dtype=np.float32)
    spikes[10:20, 0] = 1.0  # I neuron spikes at t=10-20

    W = np.zeros((2, 2), dtype=np.float32)
    W[1, 0] = -0.5  # I neuron 0 → neuron 1 with weight -0.5

    I_syn = synaptic_current(spikes, W, tau_ms=5.0, dt_ms=0.1)

    # Neuron 1 should have negative input during and after spike
    assert np.any(I_syn[15:25, 1] < -0.1), "Inhibitory input should be negative"


def test_synaptic_trace_exponential_decay():
    """Test that synaptic trace decays exponentially without new spikes."""
    # Single spike at t=0
    spikes = np.zeros((100,), dtype=np.float32)
    spikes[0] = 1.0

    trace = exponential_synaptic_trace(spikes, tau_ms=10.0, dt_ms=0.1)

    # Trace should decay monotonically after the spike
    for i in range(1, len(trace) - 1):
        assert trace[i] >= trace[i + 1], f"Trace should decay at t={i}"


def test_synaptic_current_shape():
    """Test that synaptic current has correct shape [T, N]."""
    T, N = 100, 20
    spikes = np.random.rand(T, N).astype(np.float32) > 0.9
    W = np.random.randn(N, N).astype(np.float32)

    I_syn = synaptic_current(spikes, W, tau_ms=5.0, dt_ms=0.1)

    assert I_syn.shape == (T, N)


def test_connectivity_report_json_safe():
    """Test that connectivity audit report is JSON-safe."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)

    result = make_laminar_connectivity(neurons, positions, seed=42)
    audit = result["audit"]

    # Should be serializable to JSON
    try:
        json_str = json.dumps(audit)
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["total_neurons"] == 20
    except (TypeError, ValueError) as e:
        pytest.fail(f"Audit report not JSON-safe: {e}")


def test_connectivity_deterministic_from_seed():
    """Test that connectivity is deterministic given the same seed."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)

    result1 = make_laminar_connectivity(neurons, positions, seed=42)
    result2 = make_laminar_connectivity(neurons, positions, seed=42)

    assert np.allclose(result1["W"], result2["W"]), "Same seed should produce same connectivity"


def test_connectivity_weights_finite():
    """Test that all weights are finite (no NaN or Inf)."""
    neurons = MockNeuronsDataFrame(n=30)
    positions = np.random.randn(30, 3).astype(np.float32)

    result = make_laminar_connectivity(neurons, positions, seed=42)
    W = np.asarray(result["W"])

    assert np.all(np.isfinite(W)), "All weights must be finite"


def test_synaptic_trace_finite():
    """Test that synaptic trace is finite."""
    spikes = np.random.rand(100, 20).astype(np.float32) > 0.9
    trace = exponential_synaptic_trace(spikes, tau_ms=5.0, dt_ms=0.1)

    assert np.all(np.isfinite(trace)), "Synaptic trace must be finite"
    assert np.all(trace >= 0.0), "Synaptic trace must be non-negative"


def test_connectivity_control_gains_applied():
    """Test that control gains modify connectivity correctly."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)

    control_default = {"local_exc_gain": 1.0, "local_inh_gain": 1.0, "feedforward_gain": 1.0, "feedback_gain": 1.0}
    control_scaled = {"local_exc_gain": 2.0, "local_inh_gain": 0.5, "feedforward_gain": 1.5, "feedback_gain": 1.0}

    result_default = make_laminar_connectivity(neurons, positions, control_params=control_default, seed=42)
    result_scaled = make_laminar_connectivity(neurons, positions, control_params=control_scaled, seed=42)

    # Scaled weights should differ from default (assuming non-zero connectivity)
    W_default = np.asarray(result_default["W"])
    W_scaled = np.asarray(result_scaled["W"])

    if np.count_nonzero(W_default) > 0:
        assert not np.allclose(W_default, W_scaled), "Control gains should change weights"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
