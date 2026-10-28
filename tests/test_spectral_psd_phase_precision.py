"""spectrolaminar_psd_jax keeps its DFT phase exact in float32 over long recordings (T-55 item 11).

The phase of sample n at frequency f is f * n / fs cycles. Formed as (f / fs) * n in float32 it
lost ~log2(n) bits, and the rounding of f / fs shifted every frequency by ~6e-8 relative: on
broadband noise the PSD missed a float64 DFT at the same float32 frequencies by 2.4e-4 (10 s) to
2e-2 (1000 s) of its peak. The exact remainder reduction stays near 1e-6.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.analysis.spectral import spectrolaminar_psd_jax

FS = 1000.0
FREQS = np.linspace(1.0, 150.0, 96, dtype=np.float32)  # the default band


def _float64_psd(x: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    f64 = freqs.astype(np.float64)
    basis = np.exp(-1j * 2.0 * np.pi * (f64[:, None] / FS) * np.arange(n)[None, :])
    return np.abs(basis @ x.astype(np.float64)) / n


@pytest.mark.parametrize("duration_s", [10.0, 100.0])
def test_psd_matches_float64_dft_on_broadband_noise(duration_s):
    n = int(duration_s * FS)
    rng = np.random.default_rng(0)
    brown = np.cumsum(rng.standard_normal(n))
    x = np.stack([rng.standard_normal(n), (brown - brown.mean()) / brown.std()], axis=1)
    x = x.astype(np.float32)
    got = np.asarray(spectrolaminar_psd_jax(jnp.asarray(x[None]), FS, jnp.asarray(FREQS)), dtype=np.float64)
    ref = _float64_psd(x, FREQS)
    assert got.shape == ref.shape == (FREQS.size, 2)
    rel = np.abs(got - ref).max(axis=0) / ref.max(axis=0)
    assert rel.max() < 2e-5, rel


def test_psd_phase_reduction_keeps_dtype_and_short_inputs():
    out = spectrolaminar_psd_jax(jnp.ones((2, 64, 3), dtype=jnp.float32), FS, jnp.asarray(FREQS))
    assert out.dtype == jnp.float32 and out.shape == (FREQS.size, 3)
    one = spectrolaminar_psd_jax(jnp.ones((1, 1, 2), dtype=jnp.float32))
    assert one.shape == (96, 2)
    np.testing.assert_allclose(np.asarray(one), 1.0, rtol=1e-6)
