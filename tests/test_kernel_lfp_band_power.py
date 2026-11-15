"""P-056 measurement chain: kernel_lfp -> Welch -> 8-25 Hz band power, checked on a known source.

A deep-class spike-count rate modulated at 15 Hz with a known amplitude must produce, at the deep contact,
a band power equal to (A |H(15 Hz)|)^2 / 2, with H the kernel's frequency response. Zero modulation and a
time-shuffled drive are the known negatives; doubling the drive must quadruple the power.
"""

import jax.numpy as jnp
import numpy as np

from jaxfne.analysis.spectral import welch_spectrum
from jaxfne.fields.kernel_lfp import (
    cable_cell,
    kernel_lfp,
    synapse_kernel,
    synapse_weights_in_depth_range,
)

PASSIVE = dict(cm_uF_cm2=1.0, rm_ohm_cm2=20000.0, ra_ohm_cm=150.0)
GEOM = dict(radii_um=np.array([20.0, 60.0, 120.0]), sigma_S_m=0.3, min_distance_um=1.0)
DT_MS, K = 0.2, 500
FS = 1000.0 / DT_MS
BAND = (8.0, 25.0)
F0, T_STEPS = 15.0, 10000
CONTACTS = np.array([150.0, 950.0])  # superficial, deep


def _cell():
    return cable_cell([
        dict(z0=900.0, z1=880.0, diam=20.0, n=1, parent=-1),
        dict(z0=900.0, z1=1000.0, diam=4.0, n=4, parent=0),
        dict(z0=880.0, z1=300.0, diam=3.0, n=12, parent=0),
        dict(z0=300.0, z1=50.0, diam=1.5, n=5, parent=2),
    ])


def _deep_kernel():
    cell = _cell()
    w = synapse_weights_in_depth_range(cell, 900.0, 1000.0)
    k = np.asarray(synapse_kernel(cell, w, CONTACTS, tau_syn_ms=2.0, dt_ms=DT_MS, n_steps=K, **PASSIVE, **GEOM))
    assert np.abs(k).max(1).min() > 0, "fixture must reach both contacts"
    return k  # [contacts, K]


def _band_power(x):
    f, p = welch_spectrum(np.asarray(x)[2 * K:], FS, nperseg=int(FS), noverlap=int(FS) // 2)
    m = (f >= BAND[0]) & (f <= BAND[1])
    return float(p[m].sum() * (f[1] - f[0]))


def _lfp(counts, kern):
    return np.asarray(kernel_lfp(jnp.asarray(counts[:, None], jnp.float32), jnp.asarray(kern[None], jnp.float32)))


def _drive(amp, rng=None):
    t = np.arange(T_STEPS) / FS
    c = 1.0 + amp * np.sin(2 * np.pi * F0 * t)
    return c if rng is None else rng.permutation(c)


def test_tone_band_power_matches_analytic_frequency_response():
    kern = _deep_kernel()
    amp = 0.5
    h = (kern * np.exp(-2j * np.pi * F0 * np.arange(K) / FS)).sum(1)
    expected = (amp * np.abs(h)) ** 2 / 2
    got = np.array([_band_power(_lfp(_drive(amp), kern)[:, p]) for p in range(2)])
    assert expected.min() > 0, "fixture must have a nonzero response"
    np.testing.assert_allclose(got, expected, rtol=0.03)


def test_power_scales_as_the_square_of_the_drive():
    kern = _deep_kernel()
    p1 = _band_power(_lfp(_drive(0.5), kern)[:, 1])
    p2 = _band_power(_lfp(_drive(1.0), kern)[:, 1])
    assert p1 > 0
    np.testing.assert_allclose(p2 / p1, 4.0, rtol=1e-3)


def test_zero_modulation_and_shuffled_drive_are_negatives():
    kern = _deep_kernel()
    pos = _band_power(_lfp(_drive(0.5), kern)[:, 1])
    zero = _band_power(_lfp(_drive(0.0), kern)[:, 1])
    shuf = _band_power(_lfp(_drive(0.5, np.random.default_rng(0)), kern)[:, 1])
    assert pos > 0
    assert zero < 1e-6 * pos
    assert shuf < 0.1 * pos, "a time-shuffled drive must lose the 15 Hz peak"
