"""jnwb short-list item 2: plain Welch sites route through ``welch_psd`` bit-identically.

Each case recomputes the site's exact ``scipy.signal.welch`` call on fixed
seeded inputs and asserts ``np.array_equal`` against ``welch_psd`` (exact;
both make the same scipy call). Passes before the rewire (contract holds)
and after (outputs unchanged). Fixture asserts nonzero power so the
comparison is not vacuous.
"""

import numpy as np
from scipy import signal as _signal

from jaxfne.fields.proxy import cross_area_coherence, spectrolaminar_psd
from jaxfne.vis.core import welch_psd


def _fixture_2d(dtype, t=500, c=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((t, c)).astype(dtype)
    x = x + np.sin(np.arange(t, dtype=np.float64)[:, None] * 0.05)
    return x


def _check(site, x, fs, nperseg):
    seg = int(min(nperseg, x.shape[0]))
    ef, epxx = _signal.welch(np.asarray(x), fs=fs, axis=0, nperseg=seg)
    assert np.asarray(epxx).sum() > 0, f"{site}: fixture must carry nonzero power"
    gf, gpxx = welch_psd(x, fs, nperseg=int(nperseg))
    assert np.array_equal(np.asarray(gf), np.asarray(ef)), site
    assert np.array_equal(np.asarray(gpxx), np.asarray(epxx)), site


def test_spectra_psd_site():
    for dtype in (np.float64, np.float32):
        _check("spectra.py:75", _fixture_2d(dtype), 10000.0, 256)


def test_fields_spectrolaminar_site():
    for dtype in (np.float64, np.float32):
        _check("fields.py:162", _fixture_2d(dtype), 20000.0, 256)


def test_fields_suite_site():
    for dtype in (np.float64, np.float32):
        _check("fields.py:866", _fixture_2d(dtype, t=800), 10000.0, 512)
    _check("fields.py:866-custom", _fixture_2d(np.float64, t=800), 10000.0, 128)


def test_tutorial_panels_site():
    for dtype in (np.float64, np.float32):
        _check("tutorial_panels.py:305", _fixture_2d(dtype), 10000.0, 256)


def test_short_input_clamp_site():
    # T shorter than nperseg: welch_psd clamps exactly like each site's min().
    for dtype in (np.float64, np.float32):
        _check("short-T", _fixture_2d(dtype, t=64, c=3), 10000.0, 256)
        _check("short-T-1d", _fixture_2d(dtype, t=100, c=1).ravel(), 10000.0, 1024)


def test_proxy_per_channel_site():
    # proxy.py:913 per-channel 1-D call, float32 path (function downcasts).
    rng = np.random.default_rng(0)
    t, n_ch, fs = 1200, 3, 10000.0
    sig = (rng.standard_normal((t, n_ch)) + np.sin(np.arange(t)[:, None] * 0.05)).astype(
        np.float32
    )
    freqs, psd = spectrolaminar_psd(sig, dt_ms=0.1)
    assert np.asarray(psd).sum() > 0, "fixture must carry nonzero power"
    grid = np.linspace(1.0, 150.0, 128)
    for ch in range(n_ch):
        f, pxx = _signal.welch(sig[:, ch], fs=fs, nperseg=min(1024, t))
        np.testing.assert_array_equal(
            np.asarray(psd)[:, ch], np.interp(grid, f, pxx).astype(np.float32)
        )


def test_proxy_coherence_site():
    # proxy.py:1106 two welch calls; csd call stays scipy (checked separately).
    rng = np.random.default_rng(0)
    t = 300
    x = rng.standard_normal(t) + np.sin(np.arange(t) * 0.05)
    y = rng.standard_normal(t) + np.cos(np.arange(t) * 0.05)
    out = cross_area_coherence(x, y, dt_ms=0.5)
    assert out["method"] == "welch"
    assert bool(np.asarray(out["valid"]).any()), "fixture must carry nonzero power"
    fs, seg = 2000.0, min(1024, t)
    _f, pxx = _signal.welch(np.asarray(x, dtype=np.float64), fs=fs, nperseg=seg)
    _f, pyy = _signal.welch(np.asarray(y, dtype=np.float64), fs=fs, nperseg=seg)
    from scipy.signal import csd as _csd

    _f, pxy = _csd(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), fs=fs, nperseg=seg
    )
    grid = np.linspace(1.0, 150.0, 128)
    pxx_i = np.interp(grid, _f, pxx)
    pyy_i = np.interp(grid, _f, pyy)
    pxy_i = np.interp(grid, _f, pxy.real) + 1j * np.interp(grid, _f, pxy.imag)
    denom = pxx_i * pyy_i
    ceiling = float(np.max(denom))
    valid = denom > (ceiling * 1e-12)
    coh = np.zeros_like(denom)
    pha = np.zeros_like(denom)
    coh[valid] = (np.abs(pxy_i[valid]) ** 2) / denom[valid]
    pha[valid] = np.angle(pxy_i[valid])
    np.testing.assert_array_equal(
        np.asarray(out["coherence"]), np.clip(coh, 0.0, 1.0).astype(np.float32)
    )
    np.testing.assert_array_equal(np.asarray(out["cross_phase_rad"]), pha.astype(np.float32))
    np.testing.assert_array_equal(np.asarray(out["valid"]), valid)
