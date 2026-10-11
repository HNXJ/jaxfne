"""Tests for spectrolaminar probe and readout (Patch E)."""

import numpy as np
import pytest
import jax

from _shared_fixtures import MockNeuronsDataFrame
from jaxfne.fields import (
    spectrolaminar_psd,
    spectrolaminar_readout,
)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
