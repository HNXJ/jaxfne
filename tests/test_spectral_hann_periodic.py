"""The ``"hann"`` window of ``spectrolaminar_psd_jax`` is the periodic Hann (T-56A).

w[k] = 0.5 - 0.5 cos(2 pi k / N), as scipy's ``hann`` with ``fftbins=True``
(the convention used by ``welch_spectrum`` and ``welch_numpy``).
"""

import jax.numpy as jnp
import numpy as np
from scipy.signal import get_window

from jaxfne.analysis.spectral import spectrolaminar_psd_jax

FS = 1000.0


def test_hann_window_is_periodic():
    n_trials, n_steps, n_contacts = 2, 256, 3
    freqs = np.linspace(1.0, 150.0, 96, dtype=np.float32)
    rng = np.random.default_rng(7)
    signal = rng.standard_normal((n_trials, n_steps, n_contacts)).astype(np.float32)
    # precondition: the tapered signal carries nonzero power
    w32 = get_window("hann", n_steps, fftbins=True).astype(np.float32)
    tapered_probe = signal * w32[None, :, None]
    assert float((tapered_probe**2).sum()) > 0.0

    got = np.asarray(
        spectrolaminar_psd_jax(jnp.asarray(signal), FS, jnp.asarray(freqs), window="hann")
    )

    # same computation by hand with the periodic Hann taper (float64 reference)
    w64 = get_window("hann", n_steps, fftbins=True).astype(np.float64)
    tapered = signal.astype(np.float64) * w64[None, :, None]
    basis = np.exp(
        -1j * 2.0 * np.pi * (freqs.astype(np.float64)[:, None] / FS) * np.arange(n_steps)[None, :]
    )
    spec = np.einsum("fs,tsc->tfc", basis, tapered)
    ref = np.abs(spec).mean(axis=0) / n_steps

    assert got.shape == ref.shape == (freqs.size, n_contacts)
    np.testing.assert_allclose(got, ref, rtol=1e-5)