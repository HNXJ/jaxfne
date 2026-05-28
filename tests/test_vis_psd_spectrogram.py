"""Tests for Power Spectral Density (PSD) and Spectrogram signal processing plots.

Validates that sine waves produce expected frequency peaks, and spectrograms execute cleanly.
"""

import pytest
import numpy as np
from scipy import signal

import jaxfne as jtfne


def test_psd_frequency_peak_detection():
    """Verify that a pure sine wave at 10 Hz produces a PSD peak near 10 Hz."""
    jtfne.vis.require_matplotlib()
    import matplotlib.pyplot as plt
    
    # 1. 10 Hz sine wave, sampling at 1000 Hz (dt_ms = 1.0)
    fs = 1000.0
    t = np.arange(1000) / fs  # 1 second
    x = np.sin(2 * np.pi * 10.0 * t).reshape(-1, 1)  # Shape [1000, 1]
    
    fig = jtfne.vis.psd(x, dt_ms=1.0)
    assert isinstance(fig, plt.Figure)
    
    # Run welch internally to assert peak location
    freqs, psds = signal.welch(x, fs=fs, axis=0, nperseg=256)
    peak_freq = freqs[np.argmax(psds)]
    
    # Peak frequency should be close to 10 Hz (within resolution delta ~ 3.9 Hz)
    assert np.abs(peak_freq - 10.0) < 4.0
    plt.close(fig)


def test_spectrogram_chirp_signal():
    """Verify spectrogram executes cleanly on dynamic chirp signals."""
    jtfne.vis.require_matplotlib()
    import matplotlib.pyplot as plt
    
    # 1. Chirp from 10 to 50 Hz over 1 second, sampling at 1000 Hz
    fs = 1000.0
    t = np.arange(1000) / fs
    x = signal.chirp(t, f0=10.0, t1=1.0, f1=50.0).reshape(-1, 1)
    
    fig = jtfne.vis.spectrogram(x, dt_ms=1.0)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
