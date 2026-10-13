"""jnwb short-list item 2: plain Welch sites route through ``welch_psd`` bit-identically.

Each site test calls the public function that contains the site on a fixed
seeded input, records the ``welch_psd`` call the site makes (shape, fs,
nperseg, input and output), and checks it against ``scipy.signal.welch``
called the pre-rewire way (exact; same scipy call). A mutated nperseg at a
site fails that site's test. Fixtures assert nonzero power so the comparison
is not vacuous. The ``_check`` cases pin the ``welch_psd`` contract for the
clamp and proxy paths.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as _signal

from jaxfne.fields.proxy import cross_area_coherence, spectrolaminar_psd
from jaxfne.vis import core as _core
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


@pytest.fixture
def agg():
    """Matplotlib on the Agg backend; skips without matplotlib; closes figures after."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    yield
    _core.close_all()


def _spy_welch(monkeypatch, owner, calls):
    """Replace ``owner.welch_psd`` with a recorder that runs the real ``welch_psd``."""
    real = welch_psd

    def spy(x, fs_hz, *, nperseg=256, freq_max_hz=None):
        arr = np.array(x, copy=True)
        freqs, pxx = real(x, fs_hz, nperseg=nperseg, freq_max_hz=freq_max_hz)
        calls.append(
            {
                "shape": arr.shape,
                "fs": float(fs_hz),
                "nperseg": nperseg,
                "x": arr,
                "freqs": np.array(freqs, copy=True),
                "pxx": np.array(pxx, copy=True),
            }
        )
        return freqs, pxx

    monkeypatch.setattr(owner, "welch_psd", spy)


def _assert_site(calls, site, *, shape, fs, nperseg):
    """Exactly one welch_psd call at ``site`` with this shape, fs and nperseg, whose
    output equals scipy.signal.welch called the pre-rewire way (nperseg clamped to T)."""
    assert len(calls) == 1, f"{site}: expected one welch_psd call, got {len(calls)}"
    call = calls[0]
    assert call["shape"] == shape, f"{site}: input shape {call['shape']} != {shape}"
    assert call["fs"] == fs, f"{site}: fs {call['fs']} != {fs}"
    assert call["nperseg"] == nperseg, f"{site}: nperseg {call['nperseg']} != {nperseg}"
    ef, epxx = _signal.welch(call["x"], fs=fs, axis=0, nperseg=int(min(nperseg, shape[0])))
    assert np.asarray(epxx).sum() > 0, f"{site}: fixture must carry nonzero power"
    assert np.array_equal(call["freqs"], np.asarray(ef)), site
    assert np.array_equal(call["pxx"], np.asarray(epxx)), site


def test_spectra_psd_site(monkeypatch, agg):
    # spectra.py psd() holds the site (welch_psd nperseg=256).
    from jaxfne.vis import spectra

    calls = []
    _spy_welch(monkeypatch, spectra, calls)
    for dtype in (np.float64, np.float32):
        calls.clear()
        x = _fixture_2d(dtype)
        spectra.psd(x, dt_ms=0.1)
        _assert_site(calls, "spectra.py psd", shape=x.shape, fs=1000.0 / 0.1, nperseg=256)


def test_fields_spectrolaminar_site(monkeypatch, agg):
    # fields.py spectrolaminar() holds the site (welch_psd nperseg=256).
    from jaxfne.vis import fields

    calls = []
    _spy_welch(monkeypatch, fields, calls)
    for dtype in (np.float64, np.float32):
        calls.clear()
        lfp = _fixture_2d(dtype)
        signals = {
            "time_ms": np.arange(lfp.shape[0]) * 0.05,
            "field": {
                "lfp_proxy": lfp,
                "csd_proxy": lfp,
                "contact_depths": np.arange(lfp.shape[1]) * 0.1,
            },
            "metadata": {"dt_ms": 0.05},
        }
        fields.spectrolaminar(signals)
        _assert_site(
            calls, "fields.py spectrolaminar", shape=lfp.shape, fs=1000.0 / 0.05, nperseg=256
        )


def test_fields_suite_site(monkeypatch, agg):
    # fields.py spectrolaminar_suite() holds the site: nperseg = psd_nperseg, else 512.
    from jaxfne.vis import fields

    calls = []
    _spy_welch(monkeypatch, fields, calls)
    rng = np.random.default_rng(1)
    for dtype in (np.float64, np.float32):
        lfp = _fixture_2d(dtype, t=800)
        signals = SimpleNamespace(
            spikes=(rng.random(lfp.shape) < 0.05).astype(float),
            time_ms=np.arange(lfp.shape[0]) * 0.2,
            field=SimpleNamespace(lfp_proxy=lfp, csd_proxy=lfp),
            metadata={"dt_ms": 0.2},
        )
        for kwargs, nperseg in (({}, 512), ({"psd_nperseg": 128}, 128)):
            calls.clear()
            fields.spectrolaminar_suite(signals, **kwargs)
            _assert_site(
                calls,
                f"fields.py spectrolaminar_suite {kwargs}",
                shape=lfp.shape,
                fs=1000.0 / 0.2,
                nperseg=nperseg,
            )


def test_tutorial_panels_site(monkeypatch, agg):
    # tutorial_panels.py activity_trace_suite() holds the site (welch_psd nperseg=256).
    # It imports welch_psd from core inside the function, so the spy goes on core.
    from jaxfne.vis import tutorial_panels

    calls = []
    _spy_welch(monkeypatch, _core, calls)
    t, n_trials, n_ch = 800, 2, 4
    rng = np.random.default_rng(2)
    for dtype in (np.float64, np.float32):
        calls.clear()
        base = _fixture_2d(dtype, t=t, c=n_ch)
        lfp = np.stack([base, 0.5 * base]).astype(dtype)
        trials = {
            "spikes": (rng.random((n_trials, t, 3)) < 0.05).astype(float),
            "time_ms": np.arange(t) * 0.2,
            "lfp_contacts": lfp,
            "csd_contacts": lfp,
        }
        tutorial_panels.activity_trace_suite(trials, SimpleNamespace(dt_ms=0.2))
        _assert_site(
            calls,
            "tutorial_panels.py activity_trace_suite",
            shape=(t, n_ch),
            fs=1000.0 / 0.2,
            nperseg=256,
        )


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
