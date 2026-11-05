"""Acceptance tests for P-BALANCED preset and BackgroundNoise/baseline_drive schema.

Guarantees:
1. P-BALANCED #113 bounds at 1k (1000 ms, dt=0.5 ms, seed=0):
   - Mean E rate in [1.0, 10.0] Hz
   - Mean CV_ISI in [0.70, 1.30] (for active E cells with >= 5 spikes)
   - Pairwise spike correlation |r_sc| < 0.10 (20 ms bins)
   - Inhibition x2 responsiveness: E rate shifts >= 20%
   Retuned at the 0.5.10 port (Hamm 2026-11-05) under unchanged bounds: noise
   amplitude 60 -> 80, drive E/PV/SST 2.5/4/1 -> 2/8/6. Seeds 0-4 (noise seeds
   42-46): E rate 8.91-9.14 Hz, CV 0.742-0.755, |r_sc| <= 0.078, shift 0.236-0.257.
2. Stored-is-not-consumed: each field perturbed (0, None, boundary) changes output or is refused.
   - BackgroundNoise requires explicit seed.
3. Caller override: caller Simulation argument overrides preset noise in dynamics and manifest.
4. Canonical bit-identity:
   - canonical-v1-column-1000n.json SHA256 and config_hash pinned
   - Probe numbers (E rate 11.00 Hz, CV 0.0585) pinned; the CV moved from 0.036
     with the P-048 Izhikevich default u0 = b * v0 (b1d9dfe1, Hamm 2026-11-02)
     and later dev changes, re-pinned at the 0.5.10 port
   - Absent fields produce bit-identical output with no extraneous manifest fields
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jaxfne as jtfne
from jaxfne.neuronal_tensor import BackgroundNoise

CANONICAL_JSON_SHA = "1a33aca34f543d25efc17657946ebea427cbce99d2c88c6870b3488e2b03981b"
CANONICAL_CONFIG_HASH = "91f037cf383e0bdb"


def test_balanced_preset_acceptance_bounds_at_1k():
    """Verify that canonical-v1-column-1000n-balanced satisfies all 4 acceptance bounds at 1k."""
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n-balanced")
    assert tensor.name == "canonical-v1-column-1000n-balanced"
    assert tensor.baseline_drive == {"E": 2.0, "PV": 8.0, "SST": 6.0, "VIP": 2.5}
    assert tensor.background_noise == BackgroundNoise(rate_hz=10.0, amplitude=80.0, seed=42, target="all")

    model = jtfne.construct_neuronal_tensor(tensor, duration_ms=1000.0, dt_ms=0.5, seed=0)
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)

    # Verify realized values in metadata and manifest
    assert signals.metadata.get("realized_drive") == {"E": 2.0, "PV": 8.0, "SST": 6.0, "VIP": 2.5}
    assert signals.metadata.get("realized_noise") == {
        "rate_hz": 10.0,
        "amplitude": 80.0,
        "seed": 42,
        "target": "all",
    }
    manifest = model.manifest(signals=signals)
    assert manifest.get("realized_drive") == {"E": 2.0, "PV": 8.0, "SST": 6.0, "VIP": 2.5}
    assert manifest.get("realized_noise") == {
        "rate_hz": 10.0,
        "amplitude": 80.0,
        "seed": 42,
        "target": "all",
    }

    spikes = np.asarray(signals.spikes)
    if spikes.shape[0] != 2000 and spikes.shape[1] == 2000:
        spikes = spikes.T

    table = model.neuron_table()
    cell_types = np.array([r["cell_type"] for r in table])
    dt_s = 0.5 / 1000.0
    e_mask = (cell_types == "E")

    e_rates = spikes[:, e_mask].sum(axis=0) / 1.0
    mean_e_rate = float(np.mean(e_rates))
    assert 1.0 <= mean_e_rate <= 10.0, f"E rate {mean_e_rate:.2f} Hz not in [1.0, 10.0]"

    # CV ISI on active E cells with >= 5 spikes
    e_spikes = spikes[:, e_mask]
    cv_list = []
    for i in range(e_spikes.shape[1]):
        st = np.where(e_spikes[:, i] > 0)[0] * dt_s
        if len(st) >= 5:
            isis = np.diff(st)
            cv_list.append(float(np.std(isis) / np.mean(isis)))
    assert len(cv_list) >= 500, f"Too few active E cells: {len(cv_list)}"
    mean_cv = float(np.mean(cv_list))
    assert 0.70 <= mean_cv <= 1.30, f"CV_ISI {mean_cv:.3f} not in [0.70, 1.30]"

    # Pairwise spike count correlation (20 ms bins)
    steps_20ms = int(20.0 / 0.5)
    n_bins = spikes.shape[0] // steps_20ms
    b20 = spikes[: n_bins * steps_20ms].reshape(n_bins, steps_20ms, -1).sum(axis=1)
    e_b20 = b20[:, e_mask]
    act_e = np.where(e_b20.std(axis=0) > 0)[0]
    corr = np.corrcoef(e_b20[:, act_e].T)
    triu = np.triu_indices(len(act_e), k=1)
    r_sc = float(np.nanmean(corr[triu]))
    assert abs(r_sc) < 0.10, f"|r_sc| {abs(r_sc):.4f} >= 0.10"

    # Inhibition x2 responsiveness
    new_areas_x2 = []
    for area in tensor.areas:
        new_conns = []
        for ic in area.inter_connections:
            if ic.source_neuron_type != "E":
                new_p = replace(ic.plastic, w_mech=float(ic.plastic.w_mech) * 2.0)
                new_conns.append(replace(ic, plastic=new_p))
            else:
                new_conns.append(ic)
        new_areas_x2.append(replace(area, inter_connections=new_conns))
    tensor_x2 = replace(tensor, areas=new_areas_x2)
    model_x2 = jtfne.construct_neuronal_tensor(tensor_x2, duration_ms=1000.0, dt_ms=0.5, seed=0)
    signals_x2 = jtfne.simulate(model_x2, duration_ms=1000.0, dt_ms=0.5, seed=0)
    spikes_x2 = np.asarray(signals_x2.spikes)
    if spikes_x2.shape[0] != 2000 and spikes_x2.shape[1] == 2000:
        spikes_x2 = spikes_x2.T
    mean_e_rate_x2 = float(np.mean(spikes_x2[:, e_mask].sum(axis=0) / 1.0))
    shift = abs(mean_e_rate - mean_e_rate_x2) / mean_e_rate
    assert shift >= 0.20, f"Inhibition x2 shift {shift*100:.1f}% < 20%"


def test_background_noise_validation_and_refusals():
    """Verify stored-is-not-consumed refusal boundary checks for BackgroundNoise."""
    # rate_hz: > 0, finite real
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=0.0, amplitude=10.0, seed=42)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=-1.0, amplitude=10.0, seed=42)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=None, amplitude=10.0, seed=42)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=float("inf"), amplitude=10.0, seed=42)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=True, amplitude=10.0, seed=42)

    # amplitude: >= 0, finite real
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=-0.1, seed=42)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=None, seed=42)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=float("nan"), seed=42)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=False, seed=42)

    # seed: explicit int required, never None or bool
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=10.0, seed=None)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=10.0, seed=True)
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=10.0, seed="42")
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=10.0, seed=1.5)

    # target: non-empty string
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=10.0, seed=42, target="")
    with pytest.raises(ValueError):
        BackgroundNoise(rate_hz=10.0, amplitude=10.0, seed=42, target="   ")

    # boundary amplitude 0.0 is accepted
    bn_zero = BackgroundNoise(rate_hz=10.0, amplitude=0.0, seed=42)
    assert bn_zero.amplitude == 0.0


def test_baseline_drive_validation_and_refusals():
    """Verify validation of NeuronalTensor.baseline_drive."""
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")

    # None is accepted (default drives retained)
    t_none = replace(tensor, baseline_drive=None)
    assert t_none.baseline_drive is None

    # Invalid types
    with pytest.raises(TypeError):
        replace(tensor, baseline_drive=[1.0, 2.0])
    with pytest.raises(ValueError):
        replace(tensor, baseline_drive={"E": None})
    with pytest.raises(ValueError):
        replace(tensor, baseline_drive={"E": True})
    with pytest.raises(ValueError):
        replace(tensor, baseline_drive={"E": "high"})
    with pytest.raises(ValueError):
        replace(tensor, baseline_drive={"E": float("inf")})
    with pytest.raises(ValueError):
        replace(tensor, baseline_drive={"": 2.5})
    with pytest.raises(ValueError):
        replace(tensor, baseline_drive={"NONEXISTENT_TYPE": 2.5})


def test_fields_perturbed_change_output():
    """Verify that perturbing each field realizes a change in output (stored-is-not-consumed)."""
    base_tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n-balanced")
    base_model = jtfne.construct_neuronal_tensor(base_tensor, duration_ms=200.0, dt_ms=0.5, seed=0)
    base_signals = jtfne.simulate(base_model, duration_ms=200.0, dt_ms=0.5, seed=0)
    base_spikes = np.asarray(base_signals.spikes)

    # 1. Perturb rate_hz
    t_rate = replace(
        base_tensor,
        background_noise=BackgroundNoise(rate_hz=20.0, amplitude=80.0, seed=42, target="all"),
    )
    m_rate = jtfne.construct_neuronal_tensor(t_rate, duration_ms=200.0, dt_ms=0.5, seed=0)
    sig_rate = jtfne.simulate(m_rate, duration_ms=200.0, dt_ms=0.5, seed=0)
    assert not np.array_equal(base_spikes, np.asarray(sig_rate.spikes))

    # 2. Perturb amplitude (including 0.0 boundary)
    t_amp0 = replace(
        base_tensor,
        background_noise=BackgroundNoise(rate_hz=10.0, amplitude=0.0, seed=42, target="all"),
    )
    m_amp0 = jtfne.construct_neuronal_tensor(t_amp0, duration_ms=200.0, dt_ms=0.5, seed=0)
    sig_amp0 = jtfne.simulate(m_amp0, duration_ms=200.0, dt_ms=0.5, seed=0)
    assert not np.array_equal(base_spikes, np.asarray(sig_amp0.spikes))

    # 3. Perturb seed
    t_seed = replace(
        base_tensor,
        background_noise=BackgroundNoise(rate_hz=10.0, amplitude=80.0, seed=43, target="all"),
    )
    m_seed = jtfne.construct_neuronal_tensor(t_seed, duration_ms=200.0, dt_ms=0.5, seed=0)
    sig_seed = jtfne.simulate(m_seed, duration_ms=200.0, dt_ms=0.5, seed=0)
    assert not np.array_equal(base_spikes, np.asarray(sig_seed.spikes))

    # 4. Perturb baseline_drive
    t_drive = replace(
        base_tensor,
        baseline_drive={"E": 2.5, "PV": 8.0, "SST": 6.0, "VIP": 2.5},
    )
    m_drive = jtfne.construct_neuronal_tensor(t_drive, duration_ms=200.0, dt_ms=0.5, seed=0)
    sig_drive = jtfne.simulate(m_drive, duration_ms=200.0, dt_ms=0.5, seed=0)
    assert not np.array_equal(base_spikes, np.asarray(sig_drive.spikes))


def test_caller_argument_overrides_preset():
    """Verify that caller-provided background_noise in simulate overrides preset noise."""
    tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n-balanced")
    model = jtfne.construct_neuronal_tensor(tensor, duration_ms=200.0, dt_ms=0.5, seed=0)

    # Preset run
    preset_signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0)

    # Caller override via Simulation
    override_noise = BackgroundNoise(rate_hz=5.0, amplitude=25.0, seed=123, target="all")
    sim = jtfne.Simulation(duration_ms=200.0, dt_ms=0.5, seed=0, background_noise=override_noise)
    override_signals = jtfne.simulate(model, sim)

    assert not np.array_equal(np.asarray(preset_signals.spikes), np.asarray(override_signals.spikes))
    assert override_signals.metadata["realized_noise"] == {
        "rate_hz": 5.0,
        "amplitude": 25.0,
        "seed": 123,
        "target": "all",
    }
    manifest = model.manifest(signals=override_signals)
    assert manifest["realized_noise"] == {
        "rate_hz": 5.0,
        "amplitude": 25.0,
        "seed": 123,
        "target": "all",
    }

    # Direct kwarg override in simulate(model, background_noise=...)
    kwarg_signals = jtfne.simulate(model, duration_ms=200.0, dt_ms=0.5, seed=0, background_noise=override_noise)
    assert np.array_equal(np.asarray(override_signals.spikes), np.asarray(kwarg_signals.spikes))


def test_canonical_bit_identical_pins():
    """Verify canonical-v1-column-1000n.json sha, config_hash, and probe numbers are strictly preserved."""
    # 1. File SHA256
    canonical_path = ROOT / "jaxfne" / "configs" / "canonical-v1-column-1000n.json"
    file_bytes = canonical_path.read_bytes()
    computed_sha = hashlib.sha256(file_bytes).hexdigest()
    assert computed_sha == CANONICAL_JSON_SHA, f"canonical JSON sha {computed_sha} != {CANONICAL_JSON_SHA}"

    # 2. Config hash
    tensor = jtfne.load(canonical_path)
    assert tensor.baseline_drive is None
    assert tensor.background_noise is None
    model = jtfne.construct_neuronal_tensor(tensor, duration_ms=1000.0, dt_ms=0.5, seed=0)
    assert model.summary()["config_hash"] == CANONICAL_CONFIG_HASH

    # 3. Probe observables pinned: E rate 11.00 Hz, CV 0.0585
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
    spikes = np.asarray(signals.spikes)
    if spikes.shape[0] != 2000 and spikes.shape[1] == 2000:
        spikes = spikes.T

    table = model.neuron_table()
    cell_types = np.array([r["cell_type"] for r in table])
    e_mask = (cell_types == "E")
    dt_s = 0.5 / 1000.0

    e_rate = float(np.mean(spikes[:, e_mask].sum(axis=0) / 1.0))
    assert abs(e_rate - 11.00) < 0.05, f"Canonical E rate {e_rate:.2f} != 11.00"

    # CV_ISI on neurons with at least 5 spikes (matching probe_canonical_dynamics.py)
    cv_list = []
    for i in range(spikes.shape[1]):
        st = np.where(spikes[:, i] > 0)[0] * dt_s
        if len(st) >= 5:
            isis = np.diff(st)
            if np.mean(isis) > 0:
                cv_list.append(float(np.std(isis) / np.mean(isis)))
    mean_cv = float(np.mean(cv_list))
    assert abs(mean_cv - 0.0585) < 0.005, f"Canonical CV {mean_cv:.4f} != 0.0585"

    # Absent fields produce clean metadata with no extraneous keys
    assert "realized_drive" not in signals.metadata
    assert "realized_noise" not in signals.metadata
    manifest = model.manifest(signals=signals)
    assert "realized_drive" not in manifest
    assert "realized_noise" not in manifest
