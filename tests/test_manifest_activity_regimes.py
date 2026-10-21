"""T-23 remainder: every run manifest carries the activity-regime block.

``io.manifest`` (and ``Model.manifest`` by delegation) adds a top-level
``activity_regimes`` key when the signals hold spikes and per-neuron
``cell_type`` metadata; the key is absent otherwise. ``run_receipt`` is
untouched, so receipt ids are stable.
"""

from __future__ import annotations

import dataclasses
import json

import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne.analysis import ACTIVITY_REGIME_DEFAULTS, activity_regimes
from jaxfne.io import json_safe

_DT_MS = 0.5
_NUMERIC_KEYS = (
    "rate_hz",
    "active_fraction",
    "kappa",
    "cv_isi",
    "rate_ratio",
    "spectral_peak_excess",
)


def _two_type_cfg():
    return (
        jtfne.build_laminar_column(
            n=60,
            ei_profile="flat",
            cell_type_fractions={"E": 0.5, "PV": 0.5},
            geometry="laminar",
            layers=["L4", "L5"],
        )
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )


def _two_type_run():
    model = jtfne.construct(_two_type_cfg())
    signals = model.simulate(jtfne.simulation(duration_ms=200.0, dt_ms=_DT_MS, seed=0))
    assert int(np.asarray(signals.spikes).sum()) > 0, "fixture must spike"
    return model, signals


def _populations(signals):
    table = signals.metadata["neuron_metadata"]
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(table):
        groups.setdefault(str(row["cell_type"]), []).append(i)
    return {name: np.asarray(idx, dtype=int) for name, idx in groups.items()}


def _grid_dt(signals):
    """Step of the signals' own time grid (mirrors ``io._activity_regime_block``)."""
    t = np.asarray(signals.time_ms)
    return round(float(t[1] - t[0]), 9)


def test_manifest_carries_activity_regimes_per_cell_type():
    model, signals = _two_type_run()
    spikes = np.asarray(signals.spikes)
    assert int(spikes.sum()) > 0, "fixture must spike"
    manifest = model.manifest(signals)
    block = manifest["activity_regimes"]
    populations = _populations(signals)
    assert set(block["populations"]) == set(populations)
    assert block["kept_window_ms"] == [float(signals.time_ms[0]), float(signals.time_ms[-1])]
    assert block["thresholds"] == dict(ACTIVITY_REGIME_DEFAULTS)
    want = activity_regimes(spikes, _grid_dt(signals), populations)
    for name in populations:
        got = block["populations"][name]
        assert got["label"] == want[name]["label"]
        assert got["thresholds"] == dict(ACTIVITY_REGIME_DEFAULTS)
        for key in _NUMERIC_KEYS:
            expected = want[name][key]
            if isinstance(expected, float) and np.isnan(expected):
                assert got[key] is None
            else:
                np.testing.assert_allclose(got[key], expected, rtol=1e-12)


def test_manifest_carries_silent_block_without_spikes():
    model, signals = _two_type_run()
    assert int(np.asarray(signals.spikes).sum()) > 0, "fixture must spike"
    silent = dataclasses.replace(signals, spikes=jnp.zeros_like(signals.spikes))
    assert int(np.asarray(silent.spikes).sum()) == 0, "fixture must be silent"
    block = model.manifest(silent)["activity_regimes"]
    assert set(block["populations"]) == set(_populations(silent))
    for name, pop in block["populations"].items():
        assert pop["label"] == "silent", name
        assert pop["rate_hz"] == 0.0
        assert pop["active_fraction"] == 0.0


def test_manifest_omits_block_without_cell_types():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12,
                 cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    signals = model.simulate(jtfne.simulation(duration_ms=10.0, dt_ms=0.1, seed=0))
    assert int(np.asarray(signals.spikes).sum()) > 0, "fixture must spike"
    assert signals.metadata.get("neuron_metadata") is None
    assert "activity_regimes" not in model.manifest(signals)


def test_manifest_activity_regimes_json_round_trip():
    model, signals = _two_type_run()
    assert int(np.asarray(signals.spikes).sum()) > 0, "fixture must spike"
    manifest = model.manifest(signals)
    assert json_safe(manifest) == manifest
    assert json.loads(json.dumps(manifest, allow_nan=False)) == manifest


def test_run_receipt_id_stable_across_manifest():
    model, signals = _two_type_run()
    assert int(np.asarray(signals.spikes).sum()) > 0, "fixture must spike"
    before = model.run_receipt(signals).receipt_id
    model.manifest(signals)
    after = model.run_receipt(signals).receipt_id
    assert before == after


def _downsampled(signals):
    """Every-second-sample pair; the declared dt stays at the original step."""
    spikes = np.asarray(signals.spikes)
    t = np.asarray(signals.time_ms)
    ds = dataclasses.replace(
        signals, spikes=jnp.asarray(spikes[::2]), time_ms=jnp.asarray(t[::2])
    )
    assert float(ds.metadata["dt_ms"]) == _DT_MS, "declared dt must stay original"
    assert _grid_dt(ds) == 2 * _DT_MS, "grid step must double"
    return ds


def test_manifest_downsampled_pair_uses_grid_dt():
    model, signals = _two_type_run()
    spikes = np.asarray(signals.spikes)
    assert int(spikes.sum()) > 0, "fixture must spike"
    ds = _downsampled(signals)
    block = model.manifest(ds)["activity_regimes"]
    ds_spikes = np.asarray(ds.spikes)
    grid_dt = _grid_dt(ds)
    duration_s = ds_spikes.shape[0] * grid_dt / 1000.0
    for name, idx in _populations(ds).items():
        count = int(ds_spikes[:, idx].sum())
        np.testing.assert_allclose(
            block["populations"][name]["rate_hz"],
            count / (len(idx) * duration_s),
            rtol=1e-12,
        )


def test_manifest_omits_block_for_3d_spikes_without_raising():
    model, signals = _two_type_run()
    spikes = np.asarray(signals.spikes)
    assert int(spikes.sum()) > 0, "fixture must spike"
    bad = jnp.asarray(np.broadcast_to(spikes[:, None, :], (spikes.shape[0], 1, spikes.shape[1])))
    weird = dataclasses.replace(signals, spikes=bad)
    assert weird.spikes.ndim == 3
    assert "activity_regimes" not in model.manifest(weird)


def test_manifest_rate_hz_matches_hand_count():
    model, signals = _two_type_run()
    spikes = np.asarray(signals.spikes)
    assert int(spikes.sum()) > 0, "fixture must spike"
    block = model.manifest(signals)["activity_regimes"]
    populations = _populations(signals)
    duration_s = spikes.shape[0] * _grid_dt(signals) / 1000.0
    name = "E"
    count = int(spikes[:, populations[name]].sum())
    np.testing.assert_allclose(
        block["populations"][name]["rate_hz"],
        count / (len(populations[name]) * duration_s),
        rtol=1e-12,
    )


def test_manifest_records_classifier_refusal():
    model, signals = _two_type_run()
    spikes = np.asarray(signals.spikes)
    assert int(spikes.sum()) > 0, "fixture must spike"
    t = np.asarray(signals.time_ms)
    off_grid = dataclasses.replace(
        signals, spikes=jnp.asarray(spikes[::3]), time_ms=jnp.asarray(t[::3])
    )
    assert _grid_dt(off_grid) == 3 * _DT_MS, "grid step must be 1.5 ms, off the 5 ms bin"
    block = model.manifest(off_grid)["activity_regimes"]
    assert set(block) == {"refused"}
    assert "multiple of dt_ms" in block["refused"]
