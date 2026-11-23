"""P-068: the multiscale claims check reads mm positions through the column height.

`_verify_multiscale_claims` (scripts/rerun_etude_figures.py) compares r90 with the frozen value
within rel 1e-6 (Hamm 2026-11-23). A unit error (height omitted, i.e. 1.0 for mm positions) must
abort; a value inside the tolerance must pass.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from jaxfne.fields import project_laminar_sources
from scripts.rerun_etude_figures import _verify_multiscale_claims
from scripts.run_multiscale_observation_etude import FS, mean_r90, psd_of, spectral_centroid

H = 1.6


def _fixture():
    rng = np.random.default_rng(7)
    n, t = 40, 640
    frac = np.sort(rng.uniform(0.02, 0.98, n)).astype(np.float32)
    positions = np.zeros((n, 3), dtype=np.float32)
    positions[:, 0] = rng.uniform(-0.2, 0.2, n)
    positions[:, 1] = rng.uniform(-0.2, 0.2, n)
    positions[:, 2] = frac * np.float32(H)
    tt = np.arange(t, dtype=np.float32)[:, None]
    Q = (np.sin(2 * np.pi * 0.02 * tt + frac[None, :] * 3.0) + 0.1 * rng.standard_normal((t, n))).astype(np.float32)
    spikes = rng.random((t, n)) < 0.01
    assert spikes.sum() > 0, "fixture must spike"
    V = rng.normal(-60.0, 2.0, (t, n)).astype(np.float32)
    return spikes, V, Q, positions


def _metrics(spikes, V, Q, positions, height):
    fo = project_laminar_sources(jnp.asarray(Q), jnp.asarray(positions), n_contacts=16, width=0.10,
                                 column_height_mm=height)
    z = np.asarray(positions)[:, 2] / height
    freqs = np.linspace(1.0, 150.0, 96).astype(np.float32)
    burn = int(round(200.0 / 0.5))
    cent = {key: float(spectral_centroid(psd_of(np.asarray(a)[burn:], freqs), freqs))
            for key, a in (("Q", Q), ("LFP", np.asarray(fo.lfp_proxy)), ("CSD", np.asarray(fo.csd_proxy)))}
    return {
        "spike_count": int(np.sum(spikes)),
        "mean_rate_hz": float(np.mean(spikes) * FS),
        "mean_V_m": float(np.mean(V)),
        "r90": {"lfp_ref": mean_r90(np.asarray(fo.kernel), z, np.asarray(fo.contact_depths))},
        "spectral_centroid_hz": cent,
    }


def test_claims_pass_with_the_column_height():
    spikes, V, Q, pos = _fixture()
    m = _metrics(spikes, V, Q, pos, H)
    assert np.isfinite(m["r90"]["lfp_ref"]) and m["r90"]["lfp_ref"] > 0
    _verify_multiscale_claims(m, spikes, V, Q, pos, H)
    m["r90"]["lfp_ref"] *= 1.0 + 5e-7  # inside rel 1e-6
    _verify_multiscale_claims(m, spikes, V, Q, pos, H)


def test_claims_abort_on_a_unit_error_or_an_r90_outside_tolerance():
    spikes, V, Q, pos = _fixture()
    m = _metrics(spikes, V, Q, pos, H)
    with pytest.raises(SystemExit, match="r90"):
        _verify_multiscale_claims(m, spikes, V, Q, pos, 1.0)  # mm positions read as fractions
    m["r90"]["lfp_ref"] *= 1.0 + 2e-6
    with pytest.raises(SystemExit, match="r90"):
        _verify_multiscale_claims(m, spikes, V, Q, pos, H)
