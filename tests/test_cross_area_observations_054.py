"""0.5.4 item 3 — local vs inter-area observations as declared operators.

Per-area Q/Phi (`population_rate`, `multi_area_spectrolaminar_readout`)
plus cross-area coupling (`cross_area_coherence`: magnitude-squared
coherence with cross-spectrum phase) — all computed in the observation
layer (`jaxfne.fields`), never plotting-side. Adversarial silence
(zero/constant inputs) is a declared `valid=False`, not a silent NaN.
"""

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.fields import (
    cross_area_coherence,
    multi_area_spectrolaminar_readout,
    population_rate,
)

DT_MS = 0.5
FS = 1000.0 / DT_MS


def _sine(freq_hz, t, phase=0.0):
    return np.sin(2.0 * np.pi * freq_hz * t / FS + phase)


def test_population_rate_exact_and_refusals():
    spikes = np.ones((100, 8), dtype=np.float32)
    assert np.allclose(np.asarray(population_rate(spikes, DT_MS)), 1000.0 / DT_MS)
    assert np.allclose(np.asarray(population_rate(np.zeros((50, 3)), DT_MS)), 0.0)
    with pytest.raises(ValueError):
        population_rate(np.ones((10, 4)), 0.0)
    with pytest.raises(ValueError):
        population_rate(np.ones(10))
    bad = np.ones((10, 4))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        population_rate(bad, DT_MS)


def test_coherence_identical_sines_is_one():
    t = np.arange(4000)
    out = cross_area_coherence(_sine(10.0, t), _sine(10.0, t), DT_MS)
    # Identical signals cohere everywhere: check the 10 Hz bin, not argmax.
    peak = int(np.argmin(np.abs(out["freq_hz"] - 10.0)))
    assert float(out["freq_hz"][peak]) == pytest.approx(10.0, abs=2.0)
    assert float(out["coherence"][peak]) > 0.99
    assert bool(out["valid"][peak])


def test_cross_phase_tracks_delay():
    t = np.arange(4000)
    d_ms, f = 5.0, 10.0
    d_steps = int(round(d_ms / DT_MS))
    x = _sine(f, t)
    y = np.concatenate([np.zeros(d_steps), x[: len(t) - d_steps]])
    out = cross_area_coherence(x, y, DT_MS)
    peak = int(np.argmax(out["coherence"]))
    # csd(x, y) angle = phi_y - phi_x = -2*pi*f*d for a pure delay.
    assert float(out["cross_phase_rad"][peak]) == pytest.approx(
        -2.0 * np.pi * f * d_ms / 1000.0, abs=0.3
    )


def test_coherence_independent_noise_is_low():
    rng = np.random.default_rng(0)
    out = cross_area_coherence(rng.normal(size=4000), rng.normal(size=4000), DT_MS)
    assert float(np.max(out["coherence"])) < 0.7


def test_silent_inputs_declared_not_nan():
    out = cross_area_coherence(np.zeros(500), np.zeros(500), DT_MS)
    assert not bool(out["valid"].any())
    assert np.allclose(np.asarray(out["coherence"]), 0.0)
    assert np.allclose(np.asarray(out["cross_phase_rad"]), 0.0)
    const = cross_area_coherence(np.ones(500), np.ones(500), DT_MS)
    assert not bool(const["valid"].any())


def test_coherence_refusals():
    with pytest.raises(ValueError):
        cross_area_coherence(np.ones(100), np.ones(50), DT_MS)
    bad = np.ones(100)
    bad[3] = np.nan
    with pytest.raises(ValueError):
        cross_area_coherence(bad, np.ones(100), DT_MS)
    with pytest.raises(ValueError):
        cross_area_coherence(np.ones(7), np.ones(7), DT_MS)


def test_ensemble_per_area_and_cross_observations():
    def _col(name, n, seed):
        cfg = (
            jtfne.Configuration()
            .runtime(duration_ms=40, dt_ms=DT_MS, seed=seed, recurrent_backend="edge_list")
            .column(name, layers=["L2/3", "L4"], n=n)
            .cell_types({"E": 0.8, "PV": 0.2})
            .connectivity()
            .set_emitter(family="izhikevich")
            .probes(["spikes", "V_m"], n_contacts=8)
        )
        return jtfne.construct(cfg)

    m1, m2 = _col("V1", 12, 0), _col("V2", 8, 1)
    ens = jtfne.connect(m1, m2, namespace=("A", "B"))
    sig = jtfne.simulate(ens, duration_ms=40.0, dt_ms=DT_MS, seed=7)
    spikes = np.asarray(sig.V_m) * 0.0 + np.asarray(sig.spikes)
    r0 = np.asarray(population_rate(spikes[:, :12], DT_MS))
    r1 = np.asarray(population_rate(spikes[:, 12:], DT_MS))
    assert r0.shape == r1.shape == (80,)
    assert bool(np.isfinite(r0).all()) and bool(np.isfinite(r1).all())
    neurons = {"area": np.array(["A/V1"] * 12 + ["A/V2"] * 8)}
    ros = multi_area_spectrolaminar_readout(np.asarray(sig.V_m), neurons, dt_ms=DT_MS)
    assert set(ros) == {"A/V1", "A/V2"}
    assert all(bool(np.isfinite(np.asarray(r["relative_power"])).all()) for r in ros.values())
    out = cross_area_coherence(r0, r1, DT_MS)
    assert bool(np.isfinite(np.asarray(out["coherence"])).all())
    assert out["cross_phase_rad"].shape == out["freq_hz"].shape
