"""jnwb item 2, remaining spectral sites: each differs from ``jnwb.compute_psd`` by declaration.

The four plain-Welch vis sites (``vis/spectra.py`` ``psd``, ``vis/fields.py``
``spectrolaminar`` and ``spectrolaminar_suite``, ``vis/tutorial_panels.py``
``activity_trace_suite``) route float64 input through ``jnwb.compute_psd`` via
``jaxfne/vis/core.py`` ``welch_psd`` and are tested in
``tests/test_jnwb_item2_welch_sites.py``. Each site below pins its current output
against a scipy/numpy call reproducing it and states why it is not a
``jnwb.compute_psd`` spectrum (jnwb 0.2.10: ``(lfp_data, fs, axis=0, *,
nperseg=None)``, a ``scipy.signal.welch`` wrapper):

- fields/proxy.py ``spectrolaminar_psd`` Welch branch: float32, interpolated onto 128 bins
- fields/proxy.py ``spectrolaminar_psd`` fallback: unwindowed periodogram
- fields/proxy.py ``cross_area_coherence``: a coherence ratio
- tutorial_utils.py ``hann_absolute_psd``: one symmetric Hann taper over the whole trace
- protocol_c/estimator.py ``estimate_traveling_wave``: a scalar peak frequency
- analysis/spectral.py ``spectrolaminar_psd_jax``: DFT magnitude projection on 96 bins
- analysis/spectral.py ``bandpower_jax``: a max-normalized band profile
"""

import jnwb
import numpy as np
from scipy import signal as _signal

from jaxfne.vis import core as _core


def test_proxy_spectrolaminar_differs_from_jnwb_by_float32_interp_grid():
    # fields/proxy.py:911 downcasts to float32 (so welch_psd takes its scipy
    # path, never the jnwb float64 path) and :926-927 interpolates each
    # channel's Welch PSD onto a fixed 128-point linspace grid.
    from jaxfne.fields.proxy import spectrolaminar_psd

    rng = np.random.default_rng(0)
    t, n_ch, fs = 1200, 3, 10000.0
    sig = rng.standard_normal((t, n_ch)) + np.sin(np.arange(t)[:, None] * 0.05)
    assert float(np.abs(sig).sum()) > 0, "fixture must carry nonzero power"
    freqs, psd = spectrolaminar_psd(sig, dt_ms=0.1)
    grid = np.linspace(1.0, 150.0, 128)
    np.testing.assert_array_equal(np.asarray(freqs), grid.astype(np.float32))
    sig32 = sig.astype(np.float32)
    expect = []
    for ch in range(n_ch):
        f, pxx = _signal.welch(sig32[:, ch], fs=fs, nperseg=min(1024, t))
        expect.append(np.interp(grid, f, pxx))
    np.testing.assert_array_equal(np.asarray(psd), np.stack(expect, axis=1).astype(np.float32))
    assert np.asarray(psd).sum() > 0, "fixture must carry nonzero power"
    assert np.asarray(psd).shape[0] > 10, "compared arrays must have more than 10 bins"
    jf, jpxx = jnwb.compute_psd(sig[:, 0], fs, axis=0, nperseg=min(1024, t))
    assert len(np.asarray(jf)) > 10, "compared arrays must have more than 10 bins"
    assert np.asarray(psd).shape != np.asarray(jpxx).shape, (
        "site returns a 128-bin interp grid, jnwb the 513-bin Welch grid"
    )


def test_proxy_spectrolaminar_fallback_differs_from_jnwb_by_periodogram(monkeypatch):
    # fields/proxy.py:929-931 except-branch periodogram: |rfft(x - mean)|**2 / T
    # on the raw grid, interpolated onto the linspace grid. No window, no
    # segment averaging, no density scaling: not a Welch PSD.
    from jaxfne.fields import proxy as _proxy

    def _raise(x, fs_hz, *, nperseg=256, freq_max_hz=None):
        raise ValueError("forced periodogram fallback")

    monkeypatch.setattr(_core, "welch_psd", _raise)
    rng = np.random.default_rng(0)
    t, n_ch, dt_ms = 300, 2, 0.5
    fs = 1000.0 / dt_ms
    dt_s = dt_ms / 1000.0
    sig = rng.standard_normal((t, n_ch)) + np.sin(np.arange(t)[:, None] * 0.05)
    assert float(np.abs(sig).sum()) > 0, "fixture must carry nonzero power"
    freqs, psd = _proxy.spectrolaminar_psd(sig, dt_ms=dt_ms)
    grid = np.linspace(1.0, 150.0, 128)
    expect = []
    for ch in range(n_ch):
        col = sig[:, ch].astype(np.float32)
        fft_vals = np.abs(np.fft.rfft(col - col.mean())) ** 2
        fft_freqs = np.fft.rfftfreq(t, dt_s)
        expect.append(np.interp(grid, fft_freqs, fft_vals / t))
    np.testing.assert_array_equal(np.asarray(psd), np.stack(expect, axis=1).astype(np.float32))
    assert np.asarray(psd).sum() > 0, "fixture must carry nonzero power"
    assert np.asarray(psd).shape[0] > 10, "compared arrays must have more than 10 bins"
    jf, jpxx = jnwb.compute_psd(sig[:, 0], fs, axis=0, nperseg=t)
    assert len(np.asarray(jf)) > 10, "compared arrays must have more than 10 bins"
    jpxx_i = np.interp(grid, np.asarray(jf), np.asarray(jpxx))
    assert not np.allclose(np.asarray(psd)[:, 0].astype(np.float64), jpxx_i, rtol=1e-10, atol=0), (
        "periodogram fallback must differ from Welch beyond rtol=1e-10"
    )


def test_proxy_coherence_differs_from_jnwb_by_coherence_ratio():
    # fields/proxy.py:1121-1126 Welch autospectra plus scipy csd, interpolated
    # onto the linspace grid, then :1140-1146 form |Pxy|^2/(Pxx*Pyy) with
    # valid-gating. The output is a coherence ratio, not a PSD.
    from jaxfne.fields.proxy import cross_area_coherence

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
    assert len(grid) > 10, "compared arrays must have more than 10 bins"
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
    jf, jpxx = jnwb.compute_psd(np.asarray(x, dtype=np.float64), fs, axis=0, nperseg=seg)
    assert len(np.asarray(jf)) > 10, "compared arrays must have more than 10 bins"
    assert np.asarray(out["coherence"]).shape != np.asarray(jpxx).shape, (
        "site returns a 128-bin coherence ratio, jnwb a 151-bin PSD"
    )


def test_tutorial_hann_psd_differs_from_jnwb_by_single_symmetric_taper():
    # tutorial_utils.py:87-96: demean, one symmetric np.hanning taper over the
    # whole trace, |rfft|^2 / (sum(w^2) * fs). jnwb.compute_psd is segmented
    # Welch with scipy's periodic Hann window, so the two differ.
    from jaxfne.tutorial_utils import hann_absolute_psd

    rng = np.random.default_rng(0)
    t = 512
    y = rng.standard_normal(t) + np.sin(np.arange(t) * 0.05)
    assert float(np.abs(y).sum()) > 0, "fixture must carry nonzero power"
    dt_ms, fs = 0.1, 10000.0
    freq, spec = hann_absolute_psd(y, dt_ms, window="hann")
    w = np.hanning(t)
    yw = (y - np.mean(y)) * w
    expect = np.abs(np.fft.rfft(yw)) ** 2 / (np.sum(w**2) * fs)
    np.testing.assert_array_equal(np.asarray(freq), np.fft.rfftfreq(t, d=1.0 / fs))
    np.testing.assert_array_equal(np.asarray(spec), expect)
    assert np.asarray(spec).sum() > 0, "fixture must carry nonzero power"
    assert len(np.asarray(spec)) > 10, "compared arrays must have more than 10 bins"
    jf, jpxx = jnwb.compute_psd(y, fs, axis=0, nperseg=t)
    assert len(np.asarray(jf)) > 10, "compared arrays must have more than 10 bins"
    assert not np.allclose(np.asarray(spec), np.asarray(jpxx), rtol=1e-10, atol=0), (
        "symmetric single taper must differ from periodic-Hann Welch beyond rtol=1e-10"
    )


def test_estimator_differs_from_jnwb_by_peak_frequency():
    # protocol_c/estimator.py:73-79: summed |rfft|^2 site power, argmax in band.
    # The public estimate_traveling_wave returns a scalar peak frequency, not a
    # PSD, so there is no spectrum to equate with jnwb.compute_psd.
    from jaxfne.protocol_c.estimator import estimate_traveling_wave

    dt_ms, fs, t, n = 1.0, 1000.0, 1000, 8
    band = (8.0, 13.0)
    positions = np.arange(n, dtype=np.float64)[:, None]
    tt = np.arange(t, dtype=np.float64) / fs
    phi = np.sin(2.0 * np.pi * 10.0 * tt[:, None] + 0.5 * positions.T)
    assert float(np.var(phi)) > 0, "fixture must carry nonzero power"
    est = estimate_traveling_wave(phi, positions, dt_ms=dt_ms)
    nyq = 0.5 * fs
    b, a = _signal.butter(
        4, [max(band[0] / nyq, 1e-6), min(band[1] / nyq, 0.999)], btype="band"
    )
    phi_bp = np.stack([_signal.filtfilt(b, a, phi[:, i]) for i in range(n)], axis=1)
    freqs = np.fft.rfftfreq(t, d=1.0 / fs)
    assert len(freqs) > 10, "compared arrays must have more than 10 bins"
    spec = np.sum(np.abs(np.fft.rfft(phi_bp, axis=0)) ** 2, axis=1)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    f_hat = float(freqs[mask][np.argmax(spec[mask])])
    assert est.frequency_hz == f_hat == 10.0
    assert est.classification == "TRAVELING_WAVE"
    jf, jpxx = jnwb.compute_psd(phi[:, 0], fs, axis=0, nperseg=t)
    assert len(np.asarray(jf)) > 10, "compared arrays must have more than 10 bins"
    assert np.ndim(est.frequency_hz) == 0 and np.asarray(jpxx).ndim == 1, (
        "site returns a scalar peak frequency, jnwb a PSD spectrum"
    )


def _jax_dft_psd_numpy(signal_f32, fs=1000.0):
    """Numpy mirror of analysis/spectral.py:61-82 (rectangular window)."""
    freqs = np.linspace(1.0, 150.0, 96, dtype=np.float32)
    n_steps = signal_f32.shape[1]
    sample_idx = np.arange(n_steps, dtype=freqs.dtype)
    basis = np.exp(-1j * 2.0 * np.pi * (freqs[:, None] / fs) * sample_idx[None, :])
    spec = np.einsum("fs,tsc->tfc", basis, signal_f32)
    return freqs, (np.mean(np.abs(spec), axis=0) / float(n_steps)).astype(np.float32)


def test_jax_dft_psd_differs_from_jnwb_by_dft_projection():
    # analysis/spectral.py:23-82 spectrolaminar_psd_jax: unnormalized DFT
    # projection onto a 96-point grid, trial-averaged magnitude (not squared
    # power, no Welch segmentation). Output is (96, n_contacts) float32.
    from jaxfne.analysis.spectral import spectrolaminar_psd_jax

    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    n_trials, n_steps, n_ch, fs = 2, 400, 3, 1000.0
    sig = (
        rng.standard_normal((n_trials, n_steps, n_ch))
        + np.sin(np.arange(n_steps)[None, :, None] * 0.05)
    ).astype(np.float32)
    assert float(np.abs(sig).sum()) > 0, "fixture must carry nonzero power"
    psd = np.asarray(spectrolaminar_psd_jax(jnp.asarray(sig), fs))
    _, expect = _jax_dft_psd_numpy(sig, fs)
    np.testing.assert_allclose(psd, expect, rtol=1e-4, atol=1e-6)
    assert float(psd.sum()) > 0, "fixture must carry nonzero power"
    assert psd.shape[0] > 10, "compared arrays must have more than 10 bins"
    jf, jpxx = jnwb.compute_psd(sig[0, :, 0].astype(np.float64), fs, axis=0, nperseg=n_steps)
    assert len(np.asarray(jf)) > 10, "compared arrays must have more than 10 bins"
    assert psd.shape != np.asarray(jpxx).shape, (
        "site returns a (96, n_contacts) DFT grid, jnwb a 201-bin Welch spectrum"
    )


def test_bandpower_differs_from_jnwb_by_normalized_profile():
    # analysis/spectral.py:86-114 bandpower_jax: in-band mean normalized by the
    # peak channel value. Output is an (n_contacts,) profile in [0, 1] with
    # max 1, not a PSD.
    from jaxfne.analysis.spectral import bandpower_jax

    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    n_trials, n_steps, n_ch, fs = 2, 400, 3, 1000.0
    sig = (
        rng.standard_normal((n_trials, n_steps, n_ch))
        + np.sin(np.arange(n_steps)[None, :, None] * 0.05)
    ).astype(np.float32)
    assert float(np.abs(sig).sum()) > 0, "fixture must carry nonzero power"
    freqs, psd_np = _jax_dft_psd_numpy(sig, fs)
    assert psd_np.shape[0] > 10, "compared arrays must have more than 10 bins"
    band = jnp.array([10.0, 25.0])
    prof = np.asarray(bandpower_jax(jnp.asarray(psd_np), jnp.asarray(freqs), band))
    mask = (freqs >= 10.0) & (freqs <= 25.0)
    band_power = (psd_np * mask.astype(psd_np.dtype)[:, None]).sum(axis=0) / (
        mask.sum() + 1e-12
    )
    expect = (band_power / (band_power.max() + 1e-12)).astype(np.float32)
    np.testing.assert_allclose(prof, expect, rtol=1e-6, atol=1e-7)
    assert prof.shape == (n_ch,)
    assert float(prof.max()) == 1.0
    jf, jpxx = jnwb.compute_psd(sig[0, :, 0].astype(np.float64), fs, axis=0, nperseg=n_steps)
    assert len(np.asarray(jf)) > 10, "compared arrays must have more than 10 bins"
    assert prof.shape != np.asarray(jpxx).shape, (
        "site returns an (n_contacts,) normalized profile, jnwb a 201-bin PSD"
    )
