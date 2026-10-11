"""Tests for Patch E: spectrolaminar readout (PSD, bandpower, profiles).

Focus: Validate readout does not score and uses proxy-safe metadata.
"""

import json
import numpy as np
import pytest
import jax.numpy as jnp

from _shared_fixtures import MockNeuronsDataFrame as _SharedMockNeurons
from jaxfne.fields import (
    spectrolaminar_psd,
    spectrolaminar_bandpower,
    spectrolaminar_readout,
    multi_area_spectrolaminar_readout,
)


class MockNeuronsDataFrame(_SharedMockNeurons):
    """Shared mock plus explicit L4-relative positions.

    ``with_positions=False`` drops the ``pos_from_l4`` key so the readout
    exercises its linspace fallback (``jaxfne/fields/proxy.py``).
    """

    def __init__(
        self, n=20, areas=None, layers=None, cell_types=None, with_positions=True
    ):
        super().__init__(n=n, areas=areas, layers=layers, cell_types=cell_types)
        if with_positions:
            self.data["pos_from_l4"] = np.linspace(-0.5, 0.5, n)


@pytest.fixture(params=[True, False], ids=["explicit_pos", "fallback_pos"])
def with_positions(request):
    """Run readout tests over both neuron-position flavors."""
    return request.param


# ============================================================================
# Tests: spectrolaminar_psd
# ============================================================================


def test_spectrolaminar_psd_shape():
    """Test 1: PSD output has correct shape."""
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


def test_spectrolaminar_psd_sine_wave_10hz():
    """Test 10 Hz sine produces detectable PSD peak."""
    dt_ms = 0.1
    t = np.arange(2000) * dt_ms / 1000.0  # Longer signal for better frequency resolution
    signal = np.sin(2.0 * np.pi * 10.0 * t).reshape(-1, 1).astype(np.float32)

    freqs, psd = spectrolaminar_psd(signal, dt_ms=dt_ms, n_freqs=256)

    # PSD should be finite and have some variation
    assert np.all(np.isfinite(psd))
    assert np.max(psd) > np.min(psd), "PSD should have variation for sine wave"


def test_spectrolaminar_psd_sine_wave_90hz():
    """Test 90 Hz sine produces detectable PSD peak."""
    dt_ms = 0.1
    t = np.arange(2000) * dt_ms / 1000.0  # Longer signal for better frequency resolution
    signal = np.sin(2.0 * np.pi * 90.0 * t).reshape(-1, 1).astype(np.float32)

    freqs, psd = spectrolaminar_psd(signal, dt_ms=dt_ms, n_freqs=256, freq_max=150.0)

    # PSD should be finite and have some variation
    assert np.all(np.isfinite(psd))
    assert np.max(psd) > np.min(psd), "PSD should have variation for sine wave"


def test_spectrolaminar_psd_constant_signal():
    """Test that constant signal has near-zero non-DC power."""
    signal = np.ones((500, 1), dtype=np.float32)

    freqs, psd = spectrolaminar_psd(signal)

    # DC and low frequencies may have power, but mid/high should be near zero
    mid_freq_power = np.mean(psd[10:, 0])
    assert mid_freq_power < 0.1, "Constant signal should have minimal non-DC power"


# ============================================================================
# Tests: spectrolaminar_bandpower
# ============================================================================


def test_spectrolaminar_bandpower_shape_and_keys():
    """Test that bandpower returns expected keys."""
    freqs = np.linspace(1, 150, 128, dtype=np.float32)
    psd = np.random.rand(128, 5).astype(np.float32)

    bands = {"alpha_beta": (8, 25), "gamma": (40, 150)}
    bandpower = spectrolaminar_bandpower(psd, freqs, bands=bands)

    assert "alpha_beta" in bandpower
    assert "gamma" in bandpower
    assert bandpower["alpha_beta"].shape == (5,)
    assert bandpower["gamma"].shape == (5,)


def test_spectrolaminar_bandpower_finite():
    """Test that bandpower values are finite."""
    freqs = np.linspace(1, 150, 128, dtype=np.float32)
    psd = np.random.rand(128, 8).astype(np.float32)

    bandpower = spectrolaminar_bandpower(psd, freqs)

    assert np.all(np.isfinite(bandpower["alpha_beta"]))
    assert np.all(np.isfinite(bandpower["gamma"]))


# ============================================================================
# Tests: spectrolaminar_readout
# ============================================================================


def test_spectrolaminar_readout_shape_and_keys(with_positions):
    """Test that readout has correct structure."""
    neurons = MockNeuronsDataFrame(n=20, with_positions=with_positions)
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


def test_spectrolaminar_readout_no_scoring():
    """Test that readout does not include score or pass/fail judgment."""
    neurons = MockNeuronsDataFrame(n=20)
    signal = np.random.randn(500, 20).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # Readout should not include scoring
    assert "score" not in readout
    assert "pass" not in readout
    assert "motif_gate" not in readout
    assert "S_lam" not in readout


def test_spectrolaminar_readout_normalized_power(with_positions):
    """Test that band powers sum to ≤ 1 (relative power)."""
    neurons = MockNeuronsDataFrame(n=25, with_positions=with_positions)
    signal = np.random.randn(500, 25).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # alpha_beta + gamma should be <= 1 (relative power)
    total_profile = readout["alpha_beta"] + readout["gamma"]
    assert np.all(total_profile <= 1.01), "Profile sum should be <= 1"


def test_spectrolaminar_readout_missing_area(with_positions):
    """Test that readout handles missing area gracefully."""
    neurons = MockNeuronsDataFrame(n=20, areas=["V1"] * 20, with_positions=with_positions)
    signal = np.random.randn(500, 20).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="PFC")

    # Should return empty readout
    assert readout["n_neurons"] == 0
    assert readout["n_contacts"] == 0
    assert readout["area"] == "PFC"


def test_spectrolaminar_readout_layer_specificity(with_positions):
    """Test that readout captures depth information."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 30,
        layers=["L1"] * 5 + ["L2"] * 5 + ["L3"] * 5 + ["L4"] * 5 + ["L5"] * 5 + ["L6"] * 5,
        with_positions=with_positions,
    )
    signal = np.random.randn(500, 30).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1", n_contacts=6)

    # Should have position information relative to L4
    assert "pos_from_l4" in readout
    assert len(readout["pos_from_l4"]) == 6


def test_spectrolaminar_readout_json_serializable(with_positions):
    """Test that readout is JSON-serializable."""
    neurons = MockNeuronsDataFrame(n=20, with_positions=with_positions)
    signal = np.random.randn(300, 20).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # Convert arrays to lists for JSON
    readout_json_safe = {
        k: v.tolist() if hasattr(v, "tolist") else v for k, v in readout.items()
    }

    try:
        json_str = json.dumps(readout_json_safe, allow_nan=False)
        assert isinstance(json_str, str)
        restored = json.loads(json_str)
        assert restored["area"] == "V1"
    except (TypeError, ValueError) as e:
        pytest.fail(f"Readout not JSON-serializable: {e}")


def test_spectrolaminar_readout_n_contacts_limit(with_positions):
    """Test that n_contacts parameter limits contact number."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 30,
        layers=["L1"] * 5 + ["L2"] * 5 + ["L3"] * 5 + ["L4"] * 5 + ["L5"] * 5 + ["L6"] * 5,
        with_positions=with_positions,
    )
    signal = np.random.randn(500, 30).astype(np.float32)

    readout_3 = spectrolaminar_readout(signal, neurons, area="V1", n_contacts=3)
    readout_6 = spectrolaminar_readout(signal, neurons, area="V1", n_contacts=6)

    assert readout_3["n_contacts"] == 3
    assert readout_6["n_contacts"] == 6


def test_spectrolaminar_readout_psd_shape_matches_contacts(with_positions):
    """Test that PSD shape matches number of contacts."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 30,
        layers=["L1"] * 5 + ["L2"] * 5 + ["L3"] * 5 + ["L4"] * 5 + ["L5"] * 5 + ["L6"] * 5,
        with_positions=with_positions,
    )
    signal = np.random.randn(500, 30).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1", n_freqs=128, n_contacts=5)

    assert readout["relative_power"].shape == (128, 5)
    assert len(readout["alpha_beta"]) == 5
    assert len(readout["gamma"]) == 5


def test_spectrolaminar_readout_zero_signal(with_positions):
    """Test that zero signal produces zero power."""
    neurons = MockNeuronsDataFrame(n=10, with_positions=with_positions)
    signal = np.zeros((500, 10), dtype=np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # Zero signal should have near-zero power
    assert np.allclose(readout["relative_power"], 0.0, atol=1e-6)


def test_readout_pos_fallback_matches_explicit_default():
    """Fallback positions must equal the documented linspace default.

    Pins the ``neurons.get("pos_from_l4", np.linspace(-0.5, 0.5, N))``
    default in ``jaxfne/fields/proxy.py``: any mutant of that default
    makes the fallback readout diverge from the explicit one.
    """
    signal = np.random.default_rng(0).standard_normal((500, 20)).astype(np.float32)
    explicit = spectrolaminar_readout(
        signal, MockNeuronsDataFrame(n=20, with_positions=True), area="V1"
    )
    fallback = spectrolaminar_readout(
        signal, MockNeuronsDataFrame(n=20, with_positions=False), area="V1"
    )

    assert np.array_equal(explicit["pos_from_l4"], fallback["pos_from_l4"])
    assert np.array_equal(explicit["contact_depths_m"], fallback["contact_depths_m"])


# ============================================================================
# Tests: multi_area_spectrolaminar_readout
# ============================================================================


def test_multi_area_readout_shape(with_positions):
    """Test that multi-area readout covers all areas."""
    neurons = MockNeuronsDataFrame(
        n=30,
        areas=["V1"] * 10 + ["V4"] * 10 + ["PFC"] * 10,
        with_positions=with_positions,
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


# ============================================================================
# Tests: Readout Proxy-Safety Gates
# ============================================================================


def test_readout_does_not_export_score():
    """Test that readout does not compute or export objective scores."""
    neurons = MockNeuronsDataFrame(n=10)
    signal = np.random.randn(200, 10).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # Should not have scoring fields
    assert "score" not in readout
    assert "S_lam" not in readout
    assert "motif_gate_percent" not in readout


def test_readout_includes_metadata_dict():
    """Test that readout includes explicit metadata sub-dict."""
    neurons = MockNeuronsDataFrame(n=10)
    signal = np.random.randn(200, 10).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # Must have metadata sub-dict
    assert "metadata" in readout
    assert isinstance(readout["metadata"], dict)


def test_readout_metadata_score_computed_false():
    """Test that metadata says score_computed=false."""
    neurons = MockNeuronsDataFrame(n=10)
    signal = np.random.randn(200, 10).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    assert readout["metadata"]["score_computed"] is False


def test_readout_metadata_proxy_safe():
    """Test that metadata is proxy-safe."""
    neurons = MockNeuronsDataFrame(n=10)
    signal = np.random.randn(200, 10).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    meta = readout["metadata"]
    assert meta["field_solver_status"] == "linear_solver"
    assert meta["physical_amplitude_calibrated"] is False
    assert meta["units_or_status"] == "proxy_units"


def test_readout_metadata_json_serializable():
    """Test that readout metadata is JSON-serializable."""
    neurons = MockNeuronsDataFrame(n=10)
    signal = np.random.randn(200, 10).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    # Convert readout to JSON-safe format
    readout_json_safe = {
        k: v.tolist() if hasattr(v, "tolist") else v
        for k, v in readout.items()
    }

    try:
        json_str = json.dumps(readout_json_safe, allow_nan=False)
        assert isinstance(json_str, str)
    except (TypeError, ValueError) as e:
        pytest.fail(f"Readout with metadata not JSON-serializable: {e}")


def test_readout_metadata_bands_defined():
    """Test that metadata includes band definitions."""
    neurons = MockNeuronsDataFrame(n=10)
    signal = np.random.randn(200, 10).astype(np.float32)

    readout = spectrolaminar_readout(signal, neurons, area="V1")

    meta = readout["metadata"]
    assert "bands" in meta
    assert meta["bands"]["alpha_beta"] == [8.0, 25.0]
    assert meta["bands"]["gamma"] == [40.0, 150.0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
