"""Differential-oracle tests: package tutorial-spectral helpers vs the exact
formulas previously inlined in tutorials (suite_no_3, etude_no_7).

The inline originals are reproduced verbatim here as oracles; the package
functions must match them bit-exactly so notebook delegation preserves every
reported number.
"""

import numpy as np
import pytest

from jaxfne.tutorial_utils import (
    absolute_band_power,
    hann_absolute_psd,
    log_log_power_law_exponent,
)

FIT_BAND = (1.0, 40.0)


def _oracle_absolute_power(y, dt_ms):
    y = np.asarray(y, dtype=np.float64)
    y = y - np.mean(y)
    fs = 1000.0 / dt_ms
    window = np.hanning(y.size)
    yw = y * window
    scale = np.sum(window**2) * fs
    spec = np.abs(np.fft.rfft(yw)) ** 2 / max(scale, 1e-15)
    freq = np.fft.rfftfreq(y.size, d=1.0 / fs)
    return freq, spec


def _oracle_fit(freq, psd):
    mask = (freq >= FIT_BAND[0]) & (freq <= FIT_BAND[1]) & (freq > 0)
    log_f = np.log10(freq[mask])
    log_p = np.log10(psd[mask])
    p = np.polyfit(log_f, log_p, 1)
    return float(-p[0])


def _oracle_bandpow(field, dt_ms, lo, hi):
    F = np.fft.rfftfreq(field.shape[0], d=dt_ms / 1000.0)
    P = np.abs(np.fft.rfft(np.asarray(field), axis=0)) ** 2
    m = (F >= lo) & (F < hi)
    return P[m].sum(axis=0)


@pytest.mark.parametrize("n", [128, 1000])
def test_hann_absolute_psd_matches_inline_oracle_bit_exact(n):
    rng = np.random.default_rng(7)
    y = rng.normal(size=n) + 0.3 * np.sin(np.linspace(0, 40 * np.pi, n))
    f1, p1 = hann_absolute_psd(y, dt_ms=0.5)
    f2, p2 = _oracle_absolute_power(y, 0.5)
    assert np.array_equal(f1, f2)
    assert np.array_equal(p1, p2)


def test_hann_absolute_psd_rectangular_and_validation():
    rng = np.random.default_rng(1)
    y = rng.normal(size=256)
    f_r, p_r = hann_absolute_psd(y, 1.0, window="rectangular")
    n = y.size
    fs = 1000.0 / 1.0
    want = np.abs(np.fft.rfft(y - y.mean())) ** 2 / (n * fs)
    assert np.allclose(p_r, want, rtol=0, atol=0)
    with pytest.raises(ValueError, match="unsupported window"):
        hann_absolute_psd(y, 1.0, window="blackman")
    with pytest.raises(ValueError, match="1-D"):
        hann_absolute_psd(np.zeros((4, 4)), 1.0)


def test_log_log_exponent_matches_inline_oracle():
    rng = np.random.default_rng(3)
    freq = np.linspace(0.5, 100.0, 512)
    psd = 10 ** (-2.0 * np.log10(freq)) * rng.uniform(0.9, 1.1, size=freq.size)
    a_pkg = log_log_power_law_exponent(freq, psd, FIT_BAND)
    a_oracle = _oracle_fit(freq, psd)
    assert a_pkg == a_oracle
    assert abs(a_pkg - 2.0) < 0.05
    assert log_log_power_law_exponent(freq, psd, (500.0, 600.0)) == 0.0


@pytest.mark.parametrize("n_ch", [None, 8])
def test_absolute_band_power_matches_inline_oracle(n_ch):
    rng = np.random.default_rng(11)
    t = np.hanning(400)
    field = rng.normal(size=400) * t if n_ch is None else (
        rng.normal(size=(400, n_ch)) * t[:, None]
    )
    for lo, hi in [(10.0, 25.0), (40.0, 150.0)]:
        got = absolute_band_power(field, 0.5, lo, hi)
        want = _oracle_bandpow(field, 0.5, lo, hi)
        assert np.array_equal(got, want)
