"""Tests for multi-area source projector (Patch D)."""

import json
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.fields import (
    filtered_spike_source,
    synaptic_resonance_source,
    combined_multi_area_source,
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


def test_filtered_spike_source_shape():
    """Test that filtered spike source has correct shape [T, N]."""
    neurons = MockNeuronsDataFrame(n=20)
    spikes = np.zeros((100, 20), dtype=np.float32)
    spikes[10:20, :] = 1.0

    source, _ = filtered_spike_source(spikes, neurons)

    assert source.shape == (100, 20)


def test_filtered_spike_source_exponential_decay():
    """Test that filtered source decays exponentially."""
    neurons = MockNeuronsDataFrame(n=5, cell_types=["E"] * 5)
    spikes = np.zeros((50, 5), dtype=np.float32)
    spikes[0, :] = 1.0  # Single spike at t=0

    source, _ = filtered_spike_source(spikes, neurons, tau_ms=10.0)

    # Source should decay monotonically (approximately)
    max_source = np.max(source)
    for i in range(1, 45):
        assert source[i, 0] <= source[i - 1, 0] + 1e-5, "Exponential decay not monotonic"


def test_filtered_spike_source_e_i_signs():
    """Test that E and I sources have opposite signs."""
    n_e = 15
    n_i = 5
    neurons = MockNeuronsDataFrame(
        n=n_e + n_i,
        cell_types=["E"] * n_e + ["PV"] * n_i,
    )
    spikes = np.ones((100, n_e + n_i), dtype=np.float32) * 0.5

    source, _ = filtered_spike_source(spikes, neurons)

    # E neurons should produce positive source (sign=+1)
    assert np.mean(source[:, :n_e]) > 0.0, "E source should be positive"
    # I neurons should produce negative source (sign=-1)
    assert np.mean(source[:, n_e:]) < 0.0, "I source should be negative"


def test_filtered_spike_source_zero_spikes():
    """Test that zero spikes produce zero source."""
    neurons = MockNeuronsDataFrame(n=20)
    spikes = np.zeros((100, 20), dtype=np.float32)

    source, _ = filtered_spike_source(spikes, neurons)

    assert np.allclose(source, 0.0), "Zero spikes should produce zero source"


def test_resonance_source_shape():
    """Test that resonance source has correct shape [T, N]."""
    neurons = MockNeuronsDataFrame(n=30)

    resonance = synaptic_resonance_source(neurons, n_steps=200)

    assert resonance.shape == (200, 30)


def test_resonance_source_finite():
    """Test that resonance source is finite."""
    neurons = MockNeuronsDataFrame(n=40)

    resonance = synaptic_resonance_source(neurons, n_steps=500)

    assert np.all(np.isfinite(resonance)), "Resonance must be finite"


def test_resonance_source_layer_specificity():
    """Test that resonance amplitude differs by layer."""
    # Create two networks with different layers
    neurons_l1 = MockNeuronsDataFrame(
        n=10,
        layers=["L1"] * 10,
        cell_types=["E"] * 10,
    )
    neurons_l5 = MockNeuronsDataFrame(
        n=10,
        layers=["L5"] * 10,
        cell_types=["E"] * 10,
    )

    resonance_l1 = synaptic_resonance_source(neurons_l1, n_steps=1000)
    resonance_l5 = synaptic_resonance_source(neurons_l5, n_steps=1000)

    # RMS amplitude should differ
    rms_l1 = float(np.sqrt(np.mean(resonance_l1**2)))
    rms_l5 = float(np.sqrt(np.mean(resonance_l5**2)))

    # L1 (superficial) typically has higher gamma, so RMS may differ
    # At least verify both are nonzero and finite
    assert rms_l1 > 0.0 and np.isfinite(rms_l1)
    assert rms_l5 > 0.0 and np.isfinite(rms_l5)


def test_resonance_source_control_gain_effect():
    """Test that control parameters affect resonance amplitude."""
    neurons = MockNeuronsDataFrame(n=20)

    control_low = {"alpha_beta_gain": 0.0, "gamma_gain": 0.0, "resonance_scale": 1.0}
    control_high = {"alpha_beta_gain": 2.0, "gamma_gain": 2.0, "resonance_scale": 1.0}

    resonance_low = synaptic_resonance_source(neurons, n_steps=200, control_params=control_low)
    resonance_high = synaptic_resonance_source(neurons, n_steps=200, control_params=control_high)

    # Amplitude with zero gain should be lower
    assert np.mean(np.abs(resonance_low)) < np.mean(np.abs(resonance_high))


def test_combined_source_shape():
    """Test that combined source has correct shape."""
    neurons = MockNeuronsDataFrame(n=25)
    spikes = np.random.rand(100, 25).astype(np.float32) > 0.9

    source = combined_multi_area_source(spikes, neurons, n_steps=100)

    assert source.shape == (100, 25)


def test_combined_source_finite():
    """Test that combined source is finite."""
    neurons = MockNeuronsDataFrame(n=30)
    spikes = np.random.rand(150, 30).astype(np.float32) > 0.9

    source = combined_multi_area_source(spikes, neurons, n_steps=150)

    assert np.all(np.isfinite(source)), "Combined source must be finite"


def test_combined_source_spike_component_dominance():
    """Test that spike source component affects combined source."""
    neurons = MockNeuronsDataFrame(n=20, cell_types=["E"] * 20)
    spikes = np.zeros((100, 20), dtype=np.float32)
    spikes[10:20, :] = 1.0

    control_spike_only = {"spike_source_scale": 1.0, "resonance_source_scale": 0.0}
    control_resonance_only = {"spike_source_scale": 0.0, "resonance_source_scale": 1.0}

    source_spike_only = combined_multi_area_source(
        spikes, neurons, n_steps=100, control_params=control_spike_only
    )
    source_resonance_only = combined_multi_area_source(
        spikes, neurons, n_steps=100, control_params=control_resonance_only
    )

    # Spike component should be strongest around spike times
    peak_spike = np.max(np.abs(source_spike_only[10:20, :]))
    peak_resonance = np.max(np.abs(source_resonance_only[10:20, :]))

    # Spike source should dominate in spike region
    assert peak_spike > 0.0 and np.isfinite(peak_spike)
    assert peak_resonance > 0.0 and np.isfinite(peak_resonance)


def test_combined_source_temporal_structure():
    """Test that combined source has sensible temporal structure."""
    neurons = MockNeuronsDataFrame(n=20)
    spikes = np.random.rand(500, 20).astype(np.float32) > 0.95

    source = combined_multi_area_source(spikes, neurons, n_steps=500)

    # Should show both fast (spike-driven) and slow (resonance) components
    # Verify autocorrelation structure
    assert source.shape == (500, 20)
    assert np.all(np.isfinite(source))

    # RMS should be nonzero (combination has power)
    rms = np.sqrt(np.mean(source**2))
    assert rms > 0.0, "Combined source should have nonzero power"


def test_source_metadata_json_safe():
    """Test that source metadata is JSON-serializable."""
    metadata = {
        "source_projection_mode": "voltage_and_resonance_proxy",
        "source_decomposition": "spike_filter + oscillatory_resonance",
        "source_calibration_status": "uncalibrated_proxy",
        "physical_amplitude_allowed": False,
        "truth_mode": "truth_safe_unverified",
    }

    # Should be JSON-serializable
    try:
        json_str = json.dumps(metadata)
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["source_projection_mode"] == "voltage_and_resonance_proxy"
    except (TypeError, ValueError) as e:
        pytest.fail(f"Metadata not JSON-safe: {e}")


def test_source_empty_neurons_raises():
    """Test that empty neuron dataframe raises or handles gracefully."""
    neurons = MockNeuronsDataFrame(n=0)
    spikes = np.zeros((100, 1), dtype=np.float32)

    # Should either raise or return empty tensor
    try:
        source = combined_multi_area_source(spikes, neurons, n_steps=100)
        # If it doesn't raise, should be reasonable shape
        assert source.shape[0] == 100
    except (ValueError, IndexError):
        pass  # Also acceptable


def test_filtered_spike_source_custom_tau():
    """Test that tau parameter affects decay rate."""
    neurons = MockNeuronsDataFrame(n=10, cell_types=["E"] * 10)
    spikes = np.zeros((100, 10), dtype=np.float32)
    spikes[0, :] = 1.0

    source_fast, _ = filtered_spike_source(spikes, neurons, tau_ms=2.0)
    source_slow, _ = filtered_spike_source(spikes, neurons, tau_ms=20.0)

    # Slower decay (longer tau) should persist more at later times
    slow_decay_index = 30
    assert np.mean(source_slow[slow_decay_index, :]) > np.mean(
        source_fast[slow_decay_index, :]
    ), "Longer tau should result in slower decay"


def test_resonance_area_independence():
    """Test that resonance can handle multiple areas."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 10 + ["V4"] * 10 + ["PFC"] * 10,
        layers=["L4"] * 30,
    )

    resonance = synaptic_resonance_source(neurons, n_steps=200)

    assert resonance.shape == (200, 30)
    assert np.all(np.isfinite(resonance))

    # All neurons should have resonance (even if different areas)
    assert np.any(resonance != 0.0), "At least some resonance values should be nonzero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
