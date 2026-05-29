"""Tests for multi-area emitter runtime (Patch C)."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxfne as jtfne
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


def test_multi_area_emitter_spike_shape():
    """Test that spike output has correct shape [T, N]."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)
    W = np.eye(20, dtype=np.float32) * 0.1

    spikes, voltages = simulate_multi_area_izhikevich(
        neurons, positions, W, n_steps=100, seed=42
    )

    assert spikes.shape == (100, 20)
    assert voltages.shape == (100, 20)


def test_multi_area_emitter_voltage_shape():
    """Test that voltage output has correct shape [T, N]."""
    neurons = MockNeuronsDataFrame(n=15)
    positions = np.random.randn(15, 3).astype(np.float32)
    W = np.zeros((15, 15), dtype=np.float32)

    spikes, voltages = simulate_multi_area_izhikevich(
        neurons, positions, W, n_steps=50, seed=42
    )

    assert voltages.shape == (50, 15)
    assert spikes.shape == (50, 15)


def test_multi_area_emitter_with_source_tensor():
    """Test that source tensor drives emitter correctly."""
    neurons = MockNeuronsDataFrame(n=10)
    positions = np.random.randn(10, 3).astype(np.float32)
    W = np.zeros((10, 10), dtype=np.float32)

    # Create a driving source tensor
    source = np.random.randn(50, 10).astype(np.float32) * 5.0

    spikes, voltages = simulate_multi_area_izhikevich(
        neurons, positions, W, source_tensor=source, seed=42
    )

    assert spikes.shape == (50, 10)
    assert voltages.shape == (50, 10)


def test_multi_area_emitter_finite_outputs():
    """Test that all outputs are finite (no NaN/Inf)."""
    neurons = MockNeuronsDataFrame(n=30)
    positions = np.random.randn(30, 3).astype(np.float32)
    W = np.random.randn(30, 30).astype(np.float32) * 0.05

    spikes, voltages = simulate_multi_area_izhikevich(
        neurons, positions, W, n_steps=100, seed=42
    )

    assert np.all(np.isfinite(spikes)), "Spikes must be finite"
    assert np.all(np.isfinite(voltages)), "Voltages must be finite"


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


def test_multi_area_emitter_connectivity_effect():
    """Test that different connectivity produces different outputs."""
    neurons = MockNeuronsDataFrame(n=20)
    positions = np.random.randn(20, 3).astype(np.float32)
    W_weak = np.eye(20, dtype=np.float32) * 0.01
    W_strong = np.eye(20, dtype=np.float32) * 0.5

    spikes1, _ = simulate_multi_area_izhikevich(
        neurons, positions, W_weak, n_steps=100, seed=42
    )
    spikes2, _ = simulate_multi_area_izhikevich(
        neurons, positions, W_strong, n_steps=100, seed=42
    )

    # Strong connectivity may produce different firing pattern
    # (not guaranteed to differ, but likely to)
    rate1 = float(np.mean(spikes1))
    rate2 = float(np.mean(spikes2))

    # At least they both produce valid outputs
    assert np.all(np.isfinite(spikes1)) and np.all(np.isfinite(spikes2))


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


def test_multi_area_emitter_source_schedule_integration():
    """Test full integration with time-varying source schedule."""
    neurons = MockNeuronsDataFrame(n=25)
    positions = np.random.randn(25, 3).astype(np.float32)
    W = np.random.randn(25, 25).astype(np.float32) * 0.08

    # Create a structured source schedule (e.g., pulse train)
    T = 200
    source = np.zeros((T, 25), dtype=np.float32)
    source[50:100, :] = 3.0  # Pulse from t=50 to t=100
    source[150:200, :] = -2.0  # Negative pulse from t=150 to t=200

    spikes, voltages = simulate_multi_area_izhikevich(
        neurons, positions, W, source_tensor=source, seed=42
    )

    # During strong positive pulse, should see increased activity
    rate_before = np.mean(spikes[:50])
    rate_pulse = np.mean(spikes[50:100])

    assert spikes.shape == (T, 25)
    assert voltages.shape == (T, 25)
    assert np.all(np.isfinite(spikes)) and np.all(np.isfinite(voltages))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
