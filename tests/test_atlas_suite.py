"""Atlas suite: 6 fixed panels for any model, incl. N=1, 0 edges, no spikes, field/no-field, multi-area."""

import copy
import json
import pathlib
import numpy as np
import pytest

import jaxfne as J
from jaxfne.vis.atlas_suite import PANELS, OPTIONAL_FIELD_FILE, build_atlas

FIXED = [p[0] for p in PANELS]


def _build(kind: str, tmp: pathlib.Path, **kwargs):
    if kind == "single":
        cfg = J.suite2_single_neuron_config(seed=7, duration_ms=200.0, dt_ms=0.1)
    elif kind == "zero_edge":
        cfg = (
            J.Configuration()
            .runtime(seed=7, duration_ms=100.0, dt_ms=0.1)
            .column("V1", ["L2/3"], 10)
            .set_emitter("izhikevich", "cortical_eig")
            .connectivity(kind="empty")
            .probes(["spikes", "V_m"])
        )
    elif kind == "multi_area":
        cfg = J.suite2_v1_v4_config(duration_ms=150.0, dt_ms=0.1, seed=7)
    elif kind == "with_field":
        cfg = (
            J.Configuration()
            .runtime(seed=7, duration_ms=150.0, dt_ms=0.1)
            .column("V1", ["L2/3", "L4"], 16)
            .cell_types({"E": 0.75, "PV": 0.25})
            .connectivity(kind="laminar_signed_metadata", recurrent=True)
            .set_emitter("izhikevich", "cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probes(["spikes", "V_m", "source", "LFP-proxy", "CSD-proxy"], n_contacts=8)
        )
    else:
        cfg = J.suite2_net1_config(seed=7, n=10, duration_ms=200.0, dt_ms=0.1)
    model = J.construct(cfg)
    out = tmp / kind
    manifest = build_atlas(model, out_dir=str(out), duration_ms=kwargs.get("duration_ms", 200.0), dt_ms=0.1)
    return model, out, manifest


def test_atlas_single_neuron(tmp_path):
    """N=1 degradation test: all 6 fixed panels emitted, single neuron card."""
    _, out, manifest = _build("single", tmp_path)
    assert manifest["n_neurons"] == 1
    for f in FIXED:
        p = out / f
        assert p.exists(), f"missing fixed panel {f}"
        assert p.stat().st_size > 0
    assert (out / "index.html").exists()
    assert (out / "manifest.json").exists()
    disk = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert disk["n_neurons"] == 1 and len(disk["panels"]) >= 6


def test_atlas_small_network(tmp_path):
    """Ordinary small network test: all 6 fixed panels present."""
    _, out, manifest = _build("net10", tmp_path)
    assert manifest["n_neurons"] == 10
    for f in FIXED:
        p = out / f
        assert p.exists(), f"missing fixed panel {f}"
        assert p.stat().st_size > 0


def test_atlas_zero_edges(tmp_path):
    """Zero-edge network test: single neuron has exactly 0 edges, verifies empty-matrix connectivity card."""
    _, out, manifest = _build("single", tmp_path, duration_ms=100.0)
    assert manifest["n_edges"] == 0
    for f in FIXED:
        assert (out / f).exists()


def test_atlas_no_spikes(tmp_path):
    """Zero spikes test: signals with all-zero spikes array."""
    cfg = J.suite2_net1_config(seed=42, n=8, duration_ms=100.0, dt_ms=0.1)
    model = J.construct(cfg)
    sim = J.Simulation(duration_ms=100.0, dt_ms=0.1, seed=42)
    sig = J.simulate(model, sim)
    sig_quiet = J.Signals(
        time_ms=sig.time_ms,
        V_m=sig.V_m,
        spikes=np.zeros_like(sig.spikes),
        sources=sig.sources,
        field=sig.field,
        metadata=sig.metadata,
    )
    out = tmp_path / "quiet"
    manifest = build_atlas(model, sig_quiet, out_dir=str(out))
    for f in FIXED:
        assert (out / f).exists()


def test_atlas_short_run(tmp_path):
    """Short run (e.g. 5 ms): PSD handles gracefully without crashing."""
    cfg = J.suite2_single_neuron_config(duration_ms=5.0, dt_ms=0.1)
    model = J.construct(cfg)
    out = tmp_path / "short"
    manifest = build_atlas(model, out_dir=str(out), duration_ms=5.0, dt_ms=0.1)
    for f in FIXED:
        assert (out / f).exists()


def test_atlas_field_vs_no_field(tmp_path):
    """Field present emits optional field.html; no field skips it."""
    # With field:
    _, out_field, manifest_field = _build("with_field", tmp_path)
    assert (out_field / OPTIONAL_FIELD_FILE).exists()
    assert any(p["file"] == OPTIONAL_FIELD_FILE for p in manifest_field["panels"])

    # Without field (pass signals with field=None):
    cfg_nofield = J.suite2_net1_config(seed=7, n=10, duration_ms=100.0, dt_ms=0.1)
    model_nofield = J.construct(cfg_nofield)
    sim_nofield = J.Simulation(duration_ms=100.0, dt_ms=0.1, seed=7, record_fields=False)
    sig_nofield = J.simulate(model_nofield, sim_nofield)
    assert sig_nofield.field is None

    out_nofield = tmp_path / "nofield"
    manifest_nofield = build_atlas(model_nofield, sig_nofield, out_dir=str(out_nofield))
    assert not (out_nofield / OPTIONAL_FIELD_FILE).exists()
    assert all(p["file"] != OPTIONAL_FIELD_FILE for p in manifest_nofield["panels"])


def test_atlas_multi_area(tmp_path):
    """Multi-area hierarchical column configuration."""
    _, out, manifest = _build("multi_area", tmp_path, duration_ms=150.0)
    for f in FIXED:
        assert (out / f).exists()


def test_atlas_non_mutation_invariant(tmp_path):
    """Atlas generation must not mutate model, signals, or simulation state."""
    cfg = J.suite2_net1_config(seed=42, n=10, duration_ms=100.0, dt_ms=0.1)
    model = J.construct(cfg)
    sig = J.simulate(model, J.Simulation(duration_ms=100.0, dt_ms=0.1, seed=42))

    orig_spikes = np.array(sig.spikes, copy=True)
    orig_vm = np.array(sig.V_m, copy=True)
    orig_time = np.array(sig.time_ms, copy=True)
    orig_summary = dict(model.summary())

    out = tmp_path / "mutation_test"
    build_atlas(model, sig, out_dir=str(out))

    np.testing.assert_array_equal(sig.spikes, orig_spikes)
    np.testing.assert_array_equal(sig.V_m, orig_vm)
    np.testing.assert_array_equal(sig.time_ms, orig_time)
    assert dict(model.summary()) == orig_summary
