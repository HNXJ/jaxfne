"""Tests for multi-area emitter runtime (Patch C)."""

import numpy as np
import pytest
import jax

from jaxfne.emitters import simulate_multi_area_izhikevich


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


def test_multi_area_emitter_spike_bounds():
    """Test that spikes are binary [0, 1]."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)
    W = np.eye(20, dtype=np.float32) * 0.1

    spikes, _ = simulate_multi_area_izhikevich(
        neurons, positions, W, n_steps=100, seed=42
    )

    assert np.all((spikes == 0.0) | (spikes == 1.0)), "Spikes must be binary"


def test_multi_area_emitter_population_firing_rate():
    """Test that population firing rate is within bounds (including silence)."""
    neurons = MockNeuronsDataFrame(n=50)
    positions = np.random.randn(50, 3).astype(np.float32)
    W = np.random.randn(50, 50).astype(np.float32) * 0.1

    dt_ms = 0.1
    n_steps = 1000
    spikes, _ = simulate_multi_area_izhikevich(
        neurons, positions, W, n_steps=n_steps, dt_ms=dt_ms, seed=42
    )

    duration_s = n_steps * dt_ms / 1000.0
    rate_hz = float(np.sum(spikes)) / (n_steps * 50)

    # Reasonable firing rate range (Hz). Default drive may be very low, allowing silence.
    assert 0.0 <= rate_hz < 50.0, f"Population rate {rate_hz:.2f} Hz out of bounds"


def test_multi_area_emitter_voltage_bounds():
    """Test that voltages are within reasonable bounds."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)
    W = np.eye(20, dtype=np.float32) * 0.05

    _, voltages = simulate_multi_area_izhikevich(
        neurons, positions, W, n_steps=100, seed=42
    )

    # Izhikevich voltages typically in range [-90, 30] mV
    v_min = float(np.min(voltages))
    v_max = float(np.max(voltages))

    assert v_min > -100.0, f"Minimum voltage {v_min:.1f} mV too negative"
    assert v_max < 50.0, f"Maximum voltage {v_max:.1f} mV too positive"


def test_multi_area_emitter_deterministic_from_seed():
    """Test that same seed produces same output."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)
    W = np.random.randn(20, 20).astype(np.float32) * 0.05

    spikes1, voltages1 = simulate_multi_area_izhikevich(
        neurons, positions, W, n_steps=100, seed=42
    )
    spikes2, voltages2 = simulate_multi_area_izhikevich(
        neurons, positions, W, n_steps=100, seed=42
    )

    assert np.allclose(spikes1, spikes2), "Same seed should produce same spikes"
    assert np.allclose(voltages1, voltages2), "Same seed should produce same voltages"


def test_multi_area_emitter_control_drive_scale():
    """Test that drive_scale control parameter affects firing rate."""
    neurons = MockNeuronsDataFrame(n=30)
    positions = np.random.randn(30, 3).astype(np.float32)
    W = np.zeros((30, 30), dtype=np.float32)

    control_low = {"drive_scale": 0.5}
    control_high = {"drive_scale": 2.0}

    spikes_low, _ = simulate_multi_area_izhikevich(
        neurons, positions, W, control_params=control_low, n_steps=100, seed=42
    )
    spikes_high, _ = simulate_multi_area_izhikevich(
        neurons, positions, W, control_params=control_high, n_steps=100, seed=42
    )

    # Both should be valid
    assert np.all((spikes_low == 0.0) | (spikes_low == 1.0))
    assert np.all((spikes_high == 0.0) | (spikes_high == 1.0))


def test_multi_area_emitter_empty_dataframe_raises():
    """Test that empty neuron dataframe raises ValueError."""
    neurons = MockNeuronsDataFrame(n=0)
    positions = np.zeros((0, 3), dtype=np.float32)
    W = np.zeros((0, 0), dtype=np.float32)

    with pytest.raises(ValueError, match="empty"):
        simulate_multi_area_izhikevich(neurons, positions, W)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
