"""Tests for spectrolaminar probe and readout (Patch E)."""

import json
import numpy as np
import pytest
import jax
import jax.numpy as jnp

import jaxfne as jtfne
from jaxfne.fields import (
    spectrolaminar_psd,
    spectrolaminar_readout,
    multi_area_spectrolaminar_readout,
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


def test_spectrolaminar_psd_shape():
    """Test that PSD output has correct shape."""
    signal = np.random.randn(1000, 5).astype(np.float32)

    freqs, psd = spectrolaminar_psd(signal, n_freqs=128)

    assert freqs.shape == (128,)
    assert psd.shape == (128, 5)


def test_spectrolaminar_psd_1d_signal():
    """Test that PSD handles 1D signals."""
    signal = np.random.randn(1000).astype(np.float32)

    freqs, psd = spectrolaminar_psd(signal, n_freqs=64)

    assert freqs.shape == (64,)
    assert psd.shape == (64, 1)


def test_spectrolaminar_psd_finite():
    """Test that PSD output is finite."""
    signal = np.random.randn(500, 8).astype(np.float32)

    freqs, psd = spectrolaminar_psd(signal)

    assert np.all(np.isfinite(freqs)), "Frequencies must be finite"
    assert np.all(np.isfinite(psd)), "PSD must be finite"


def test_spectrolaminar_psd_freq_bounds():
    """Test that frequency axis respects bounds."""
    signal = np.random.randn(500, 4).astype(np.float32)
    freq_min, freq_max = 5.0, 100.0

    freqs, _ = spectrolaminar_psd(signal, freq_min=freq_min, freq_max=freq_max)

    assert float(freqs[0]) >= freq_min - 1e-5
    assert float(freqs[-1]) <= freq_max + 1e-5


def test_spectrolaminar_psd_sine_wave():
    """Test that PSD correctly identifies sine wave frequencies."""
    # Create sine wave at 20 Hz
    dt_ms = 0.1
    t = np.arange(1000) * dt_ms / 1000.0
    signal = np.sin(2.0 * np.pi * 20.0 * t).reshape(-1, 1).astype(np.float32)

    freqs, psd = spectrolaminar_psd(signal, dt_ms=dt_ms, n_freqs=128)

    # Find peak frequency
    peak_idx = np.argmax(psd[:, 0])
    peak_freq = float(freqs[peak_idx])

    # Should be close to 20 Hz
    assert 18.0 < peak_freq < 22.0, f"Peak frequency {peak_freq:.1f} Hz not near 20 Hz"


def test_spectrolaminar_readout_shape_and_keys():
    """Test that readout has correct structure."""
    neurons = MockNeuronsDataFrame(n=20)
    signal = np.random.randn(500, 20).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    required_keys = {
        "freq_hz",
        "pos_from_l4",
        "relative_power",
        "alpha_beta",
        "gamma",
        "contact_depths_m",
        "n_contacts",
        "n_neurons",
        "area",
    }
    assert set(readout.keys()) >= required_keys, f"Missing keys: {required_keys - set(readout.keys())}"


def test_spectrolaminar_readout_finite():
    """Test that readout arrays are finite."""
    neurons = MockNeuronsDataFrame(n=30, areas=["V1"] * 30)
    signal = np.random.randn(500, 30).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    assert np.all(np.isfinite(readout["freq_hz"]))
    assert np.all(np.isfinite(readout["relative_power"]))
    assert np.all(np.isfinite(readout["alpha_beta"]))
    assert np.all(np.isfinite(readout["gamma"]))


def test_spectrolaminar_readout_normalized_power():
    """Test that band powers sum to less than total power."""
    neurons = MockNeuronsDataFrame(n=25)
    signal = np.random.randn(500, 25).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # alpha_beta + gamma should be <= 1 (relative power)
    total_profile = readout["alpha_beta"] + readout["gamma"]
    assert np.all(total_profile <= 1.01), "Profile sum should be <= 1"


def test_spectrolaminar_readout_missing_area():
    """Test that readout handles missing area gracefully."""
    neurons = MockNeuronsDataFrame(n=20, areas=["V1"] * 20)
    signal = np.random.randn(500, 20).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="PFC")

    # Should return empty readout
    assert readout["n_neurons"] == 0
    assert readout["n_contacts"] == 0
    assert readout["area"] == "PFC"


def test_spectrolaminar_readout_layer_specificity():
    """Test that readout captures layer information."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 30,
        layers=["L1"] * 5 + ["L2"] * 5 + ["L3"] * 5 + ["L4"] * 5 + ["L5"] * 5 + ["L6"] * 5,
    )
    signal = np.random.randn(500, 30).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1", n_contacts=6)

    # Should have position information relative to L4
    assert "pos_from_l4" in readout
    assert len(readout["pos_from_l4"]) == 6


def test_spectrolaminar_readout_json_serializable():
    """Test that readout is JSON-serializable."""
    neurons = MockNeuronsDataFrame(n=20)
    signal = np.random.randn(300, 20).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # Convert arrays to lists for JSON
    readout_json_safe = {
        k: v.tolist() if hasattr(v, "tolist") else v for k, v in readout.items()
    }

    try:
        json_str = json.dumps(readout_json_safe)
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["area"] == "V1"
    except (TypeError, ValueError) as e:
        pytest.fail(f"Readout not JSON-serializable: {e}")


def test_multi_area_readout_shape():
    """Test that multi-area readout covers all areas."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 10 + ["V4"] * 10 + ["PFC"] * 10,
    )
    signal = np.random.randn(500, 30).astype(np.float32)

    readouts = multi_area_spectrolaminar_readout(signal, neurons)

    assert set(readouts.keys()) == {"V1", "V4", "PFC"}
    assert all(readouts[area]["n_contacts"] > 0 for area in ["V1", "V4", "PFC"])


def test_multi_area_readout_finite():
    """Test that multi-area readout has all finite values."""
    neurons = MockNeuronsDataFrame(
        n=40,
        areas=["V1"] * 20 + ["PFC"] * 20,
    )
    signal = np.random.randn(500, 40).astype(np.float32)

    readouts = multi_area_spectrolaminar_readout(signal, neurons)

    for area, readout in readouts.items():
        assert np.all(np.isfinite(readout["relative_power"]))
        assert np.all(np.isfinite(readout["alpha_beta"]))
        assert np.all(np.isfinite(readout["gamma"]))


def test_spectrolaminar_readout_n_contacts_limit():
    """Test that n_contacts parameter limits contact number."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 30,
        layers=["L1"] * 5 + ["L2"] * 5 + ["L3"] * 5 + ["L4"] * 5 + ["L5"] * 5 + ["L6"] * 5,
    )
    signal = np.random.randn(500, 30).astype(np.float32)

    readout_3 = spectrolaminar_readout(signal, neurons, area="V1", n_contacts=3)
    readout_6 = spectrolaminar_readout(signal, neurons, area="V1", n_contacts=6)

    assert readout_3["n_contacts"] == 3
    assert readout_6["n_contacts"] == 6


def test_spectrolaminar_readout_psd_shape_matches_contacts():
    """Test that PSD shape matches number of contacts."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 30,
        layers=["L1"] * 5 + ["L2"] * 5 + ["L3"] * 5 + ["L4"] * 5 + ["L5"] * 5 + ["L6"] * 5,
    )
    signal = np.random.randn(500, 30).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1", n_freqs=128, n_contacts=5)

    assert readout["relative_power"].shape == (128, 5)
    assert len(readout["alpha_beta"]) == 5
    assert len(readout["gamma"]) == 5


def test_spectrolaminar_readout_band_specific_content():
    """Test that bands are computed for all contacts."""
    # Create signal with both alpha and gamma components
    dt_ms = 0.1
    t = np.arange(1000) * dt_ms / 1000.0
    alpha_signal = np.sin(2.0 * np.pi * 15.0 * t)  # 15 Hz = alpha/beta
    gamma_signal = np.sin(2.0 * np.pi * 90.0 * t)  # 90 Hz = gamma
    mixed_signal = (alpha_signal + 0.5 * gamma_signal).reshape(-1, 1).astype(np.float32)

    neurons = MockNeuronsDataFrame(n=1, areas=["V1"])
    readout = spectrolaminar_readout(mixed_signal, neurons, area="V1")

    # Bands should sum to approximately 1 (normalized)
    total = readout["alpha_beta"][0] + readout["gamma"][0]
    assert 0.0 <= total <= 1.01, f"Band sum {total} out of normalized range"


def test_spectrolaminar_readout_zero_signal():
    """Test that zero signal produces zero power."""
    neurons = MockNeuronsDataFrame(n=10)
    signal = np.zeros((500, 10), dtype=np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # Zero signal should have near-zero power
    assert np.allclose(readout["relative_power"], 0.0, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
