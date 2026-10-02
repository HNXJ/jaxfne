"""Pin the ``NeuronalTensor`` canonical data model and its bridge into the
existing ``Configuration``/``construct``/``simulate`` pipeline.

Covers: JSON roundtrip, construct, multi-area placement, cross-area/within-
area connection wiring (real edges, not just metadata), the ``PlasticParams.H``
-> HDP initial-state hook, the ``StaticParams.reversal_potentials_mV``
declared-metadata surfacing, and the backward bridge (``NeuronalTensor`` ->
``Configuration`` leaves the existing pipeline untouched).
"""
import os
import math
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt


def _single_area_tensor(h=0.0, with_inter_connection=True):
    layer = nt.Layer(
        name="L4",
        n_neurons=20,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )
    inter_connections = []
    if with_inter_connection:
        inter_connections.append(
            nt.InterConnection(
                source_layer="L4", source_neuron_type="E",
                target_layer="L4", target_neuron_type="PV",
                mechanism="AMPA",
                static=nt.StaticParams(
                    g_mech={"AMPA": 1.0},
                    reversal_potentials_mV={"AMPA": 0.0},
                    dT_ms=2.0,
                ),
                plastic=nt.PlasticParams(w_mech=2.0, H=h),
            )
        )
    area = nt.Area(name="V1", layers=[layer], inter_connections=inter_connections)
    return nt.NeuronalTensor(areas=[area], name="single_area_test")


def _two_area_tensor():
    l1 = nt.Layer(name="L4", n_neurons=15, neuron_types=[nt.NeuronType.make("E")])
    l2 = nt.Layer(name="L4", n_neurons=15, neuron_types=[nt.NeuronType.make("E")])
    area1 = nt.Area(name="V1", layers=[l1], pose=nt.Pose3D(translation=(0.0, 0.0, 0.0)))
    area2 = nt.Area(name="V4", layers=[l2], pose=nt.Pose3D(translation=(500.0, 0.0, 0.0)))
    ac = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V4", target_layer="L4", target_neuron_type="E",
        static=nt.StaticParams(dT_ms=3.0),
        plastic=nt.PlasticParams(w_mech=1.5, H=1.0),
    )
    return nt.NeuronalTensor(areas=[area1, area2], area_connections=[ac])


def test_roundtrip_json(tmp_path):
    tensor = _single_area_tensor(h=2.5)
    path = tmp_path / "tensor.json"
    written = nt.save_neuronal_tensor(tensor, path)
    assert str(written) == str(path)
    loaded = nt.load_neuronal_tensor(path)
    assert loaded.name == tensor.name
    assert len(loaded.areas) == len(tensor.areas)
    assert loaded.areas[0].inter_connections[0].plastic.H == pytest.approx(2.5)
    assert loaded.areas[0].inter_connections[0].static.reversal_potentials_mV["AMPA"] == pytest.approx(0.0)


def test_construct_produces_runnable_model():
    tensor = _single_area_tensor()
    model = nt.construct_neuronal_tensor(tensor, seed=0, duration_ms=5.0, dt_ms=0.5)
    assert model.params["emitter"].n_neurons == 20
    sig = jtfne.simulate(model, duration_ms=5.0, dt_ms=0.5, seed=0)
    assert bool(jnp.all(jnp.isfinite(sig.spikes)))
    assert bool(jnp.all(jnp.isfinite(sig.V_m)))


def test_multi_area_pose_placement():
    tensor = _two_area_tensor()
    model = nt.construct_neuronal_tensor(tensor, seed=1, duration_ms=2.0, dt_ms=0.5)
    rows = model.neuron_table()
    v1_idx = [i for i, r in enumerate(rows) if r["area"] == "V1"]
    v4_idx = [i for i, r in enumerate(rows) if r["area"] == "V4"]
    assert len(v1_idx) == 15 and len(v4_idx) == 15
    positions = model.params["positions"]
    v1_x = positions[v1_idx, 0]
    v4_x = positions[v4_idx, 0]
    assert float(v1_x.max()) < 100.0
    assert float(v4_x.min()) > 400.0


def test_cross_area_connections_compile_to_real_edges():
    tensor = _two_area_tensor()
    cfg = nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    assert model.cfg.metadata.get("recurrent_backend") == "edge_list"
    sig = jtfne.simulate(model, duration_ms=2.0, dt_ms=0.5, seed=0)
    assert bool(jnp.all(jnp.isfinite(sig.spikes)))


def test_within_area_inter_connection_compiles_to_real_edge():
    tensor = _single_area_tensor()
    cfg = nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    model = jtfne.construct(cfg)
    assert model.cfg.metadata.get("recurrent_backend") == "edge_list"


def test_h_override_seeds_hdp_initial_state():
    tensor = _single_area_tensor(h=3.5)
    model = nt.construct_neuronal_tensor(tensor, seed=0, duration_ms=5.0, dt_ms=0.5)
    H0 = model.params.get("hdp_initial_H")
    assert H0 is not None

    cfg2 = model.cfg.hdp(relative_baseline=1.2)
    model2 = jtfne.construct(cfg2).with_hdp_initial_state(H0=H0)
    jtfne.simulate(model2, duration_ms=5.0, dt_ms=0.5, seed=0)
    diag = model2._last_hdp_diag
    h_trace_0 = diag["H_trace"][0]

    rows = model.neuron_table()
    pv_idx = [i for i, r in enumerate(rows) if r["cell_type"] == "PV"]
    e_idx = [i for i, r in enumerate(rows) if r["cell_type"] == "E"]
    assert float(h_trace_0[pv_idx[0]]) == pytest.approx(3.5)
    assert float(h_trace_0[e_idx[0]]) == pytest.approx(1.0)


def test_h_override_absent_by_default():
    tensor = _single_area_tensor(h=0.0, with_inter_connection=False)
    model = nt.construct_neuronal_tensor(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    assert model.params.get("hdp_initial_H") is None


def test_reversal_metadata_surfaced_but_inert():
    tensor = _single_area_tensor()
    cfg = nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    mechanisms = cfg.metadata["circuit"]["mechanisms"]
    ampa_mech = next(m for m in mechanisms if m["kind"] == "AMPA")
    assert ampa_mech["params"]["reversal_mV"] == pytest.approx(0.0)

    model_a = jtfne.construct(cfg)
    model_b = jtfne.construct(cfg)
    sig_a = jtfne.simulate(model_a, duration_ms=2.0, dt_ms=0.5, seed=0)
    sig_b = jtfne.simulate(model_b, duration_ms=2.0, dt_ms=0.5, seed=0)
    assert bool(jnp.array_equal(sig_a.spikes, sig_b.spikes))


def test_geometry_collapses_to_2d_with_zero_range():
    layer = nt.Layer(
        name="L4", n_neurons=10,
        neuron_types=[nt.NeuronType.make("E")],
        geometry=nt.Geometry3D(z_range=(0.0, 0.0)),
    )
    area = nt.Area(name="V1", layers=[layer])
    tensor = nt.NeuronalTensor(areas=[area])
    model = nt.construct_neuronal_tensor(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    positions = model.params["positions"]
    # Column-relative (owner ruling 2026-09-30): a point range collapses the
    # axis at that fraction of the column, here the column's base.
    assert float(jnp.ptp(positions[:, 2])) == 0.0
    assert abs(float(positions[0, 2])) < 1e-6


def test_serialization_round_trips_geometry_and_pose():
    layer = nt.Layer(
        name="L4", n_neurons=8,
        neuron_types=[nt.NeuronType.make("E")],
        geometry=nt.Geometry3D(z_range=(0.1, 0.4)),
    )
    area = nt.Area(name="V1", layers=[layer], pose=nt.Pose3D(plane="xz", rotation_deg=45.0))
    tensor = nt.NeuronalTensor(areas=[area])
    d = tensor.to_dict()
    assert d["areas"][0]["pose"]["plane"] == "xz"
    assert d["areas"][0]["pose"]["rotation_deg"] == 45.0
    assert tuple(d["areas"][0]["layers"][0]["geometry"]["z_range"]) == (0.1, 0.4)


def test_backward_bridge_does_not_affect_existing_configuration_pipeline():
    cfg = (
        jtfne.build_laminar_column(n=40, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=4)
    )
    model = jtfne.construct(cfg)
    sig = jtfne.simulate(model, duration_ms=2.0, dt_ms=0.5, seed=0)
    assert bool(jnp.all(jnp.isfinite(sig.spikes)))


def test_neuron_type_fraction_overrides_even_split():
    """NeuronType.fraction (0.4.7 addition) lets a layer declare a
    non-uniform E:I composition instead of being flattened to 1/len(types)
    by the bridge -- e.g. a canonical deep-E-heavy layer (90% E, 10% I) must
    survive into Configuration.area_layer_cell_types as 0.9/0.1, not 0.5/0.5."""
    layer = nt.Layer(
        name="L6", n_neurons=100,
        neuron_types=[nt.NeuronType.make("E", fraction=0.9),
                      nt.NeuronType.make("PV", fraction=0.1)],
    )
    area = nt.Area(name="V1", layers=[layer])
    tensor = nt.NeuronalTensor(areas=[area], name="fraction_test")
    cfg = nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    layer_cell_types = cfg.metadata["area_layer_cell_types"]["V1"]["L6"]
    assert layer_cell_types["E"] == pytest.approx(0.9)
    assert layer_cell_types["PV"] == pytest.approx(0.1)
    model = jtfne.construct(cfg)
    labels = model.params["emitter"].labels
    e_count = sum(1 for l in labels if l == "E")
    assert e_count == 90


def test_neuron_type_fraction_absent_keeps_even_split():
    """Backward compatibility: a layer with no declared fractions (the
    pre-0.4.7 shape, and every existing canonical jaxfne/configs/*.json)
    must still bridge to an even split, unchanged."""
    layer = nt.Layer(
        name="L4", n_neurons=20,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )
    area = nt.Area(name="V1", layers=[layer])
    tensor = nt.NeuronalTensor(areas=[area], name="even_split_test")
    cfg = nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    layer_cell_types = cfg.metadata["area_layer_cell_types"]["V1"]["L4"]
    assert layer_cell_types["E"] == pytest.approx(0.5)
    assert layer_cell_types["PV"] == pytest.approx(0.5)


def test_default_relative_size():
    assert nt.default_relative_size("E") == pytest.approx(5.0)
    assert nt.default_relative_size("PV") == pytest.approx(1.0)
    assert nt.default_relative_size("SST") == pytest.approx(1.5)


def test_make_minimal_ei_tensor_default_matches_6e_2pv_shape():
    t = nt.make_minimal_ei_tensor()
    layer = t.areas[0].layers[0]
    assert layer.n_neurons == 8
    assert [ty.name for ty in layer.neuron_types] == ["E", "PV"]
    assert layer.neuron_types[0].fraction == pytest.approx(0.75)
    assert layer.neuron_types[1].fraction == pytest.approx(0.25)
    assert len(t.areas[0].inter_connections) == 4
    assert all(c.plastic.H == 1.0 for c in t.areas[0].inter_connections)


def test_make_minimal_ei_tensor_respects_overrides():
    t = nt.make_minimal_ei_tensor(n=4, e_fraction=0.5, layer_name="Lx", area_name="ax", h=2.0)
    layer = t.areas[0].layers[0]
    assert t.areas[0].name == "ax"
    assert layer.name == "Lx"
    assert layer.n_neurons == 4
    assert layer.neuron_types[0].fraction == pytest.approx(0.5)
    assert all(c.plastic.H == 2.0 for c in t.areas[0].inter_connections)


def test_merge_neuronal_tensors_flattens_and_renames_collisions():
    t1 = _single_area_tensor()
    t2 = _single_area_tensor()  # same area name "V1" -> collision
    merged = nt.merge_neuronal_tensors([t1, t2], name="merged")
    names = [a.name for a in merged.areas]
    assert names == ["V1", "V1_1"]


@pytest.mark.release
def test_canonical_json_config_library_loads_and_constructs():
    """jaxfne/configs/*.json (the canonical, package-shipped config library,
    see scripts/build_canonical_neuronal_tensor_configs.py) all load + construct
    and declare the current schema version."""
    config_dir = jtfne.configs_dir()
    json_files = sorted(config_dir.glob("*.json"))
    assert len(json_files) >= 4, f"Expected the canonical config library in {config_dir}"
    for path in json_files:
        tensor = nt.load_neuronal_tensor(path)
        model = nt.construct_neuronal_tensor(tensor, seed=0, duration_ms=10.0, dt_ms=0.5)
        sig = jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=0)
        assert bool(jnp.all(jnp.isfinite(sig.spikes))), f"{path.name} produced non-finite spikes"


def test_canonical_configs_declare_current_schema_version():
    import json
    for path in sorted(jtfne.configs_dir().glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw.get("schema_version") == nt.NEURONAL_TENSOR_SCHEMA_VERSION, (
            f"{path.name} missing/stale schema_version"
        )


def test_load_canonical_neuronal_tensor_by_name():
    names = jtfne.list_canonical_neuronal_tensors()
    assert "default-column" in names
    tensor = jtfne.load_canonical_neuronal_tensor("default-column")
    assert tensor.name == "default_column"
    # also accepts the .json suffix
    tensor2 = jtfne.load_canonical_neuronal_tensor("default-column.json")
    assert tensor2.name == tensor.name
    with pytest.raises(FileNotFoundError):
        jtfne.load_canonical_neuronal_tensor("does-not-exist")


def test_schema_version_mismatch_warns_not_raises(tmp_path):
    tensor = _single_area_tensor()
    path = tmp_path / "legacy.json"
    nt.save_neuronal_tensor(tensor, path)
    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["schema_version"] = "neuronal_tensor_v999_future"
    path.write_text(json.dumps(raw))
    with pytest.warns(UserWarning, match="schema_version"):
        reloaded = nt.load_neuronal_tensor(path)
    assert reloaded.name == tensor.name


def test_legacy_json_without_schema_version_loads_silently(tmp_path):
    tensor = _single_area_tensor()
    path = tmp_path / "legacy_no_version.json"
    nt.save_neuronal_tensor(tensor, path)
    import json
    raw = json.loads(path.read_text(encoding="utf-8"))
    del raw["schema_version"]
    path.write_text(json.dumps(raw))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reloaded = nt.load_neuronal_tensor(path)  # must NOT warn/raise
    assert reloaded.name == tensor.name


@pytest.mark.release
def test_example_08_neuronal_tensor_first_runs():
    """examples/08_neuronal_tensor_first.py (tensor-first workflow) runs to completion."""
    example = str(Path(__file__).parent.parent / "examples" / "08_neuronal_tensor_first.py")
    env = os.environ.copy()
    repo_root = str(Path(__file__).parent.parent)
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, example], capture_output=True, text=True, timeout=120, env=env,
    )
    assert result.returncode == 0, f"Example failed:\n{result.stderr}"
    assert "Saved + reloaded NeuronalTensor JSON" in result.stdout
    assert "recurrent_backend: edge_list" in result.stdout
    assert "spikes finite: True" in result.stdout


# -- 0.4.7 unified API surface: load() / RuntimeConfiguration / construct(tensor, runtime) / compute_fields() --

def test_load_is_canonical_and_load_neuronal_tensor_is_a_wrapper(tmp_path):
    tensor = _single_area_tensor()
    path = tmp_path / "t.json"
    nt.save_neuronal_tensor(tensor, path)
    via_load = jtfne.load(path)
    via_wrapper = jtfne.load_neuronal_tensor(path)
    assert via_load.name == via_wrapper.name == tensor.name


def test_load_rejects_non_tensor_json(tmp_path):
    path = tmp_path / "not_a_tensor.json"
    path.write_text('{"network": {}, "emitter": {}}')
    with pytest.raises(ValueError, match="areas"):
        jtfne.load(path)


def test_runtime_configuration_has_no_biology_fields():
    runtime = jtfne.RuntimeConfiguration()
    biology_terms = {"areas", "layers", "populations", "neuron_types", "mechanism", "connections"}
    field_names = {f.name for f in __import__("dataclasses").fields(runtime)}
    assert not (field_names & biology_terms), f"RuntimeConfiguration leaked biology fields: {field_names & biology_terms}"


def test_construct_tensor_runtime_matches_construct_neuronal_tensor():
    tensor = _single_area_tensor()
    runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=2.0, dt_ms=0.5)
    model_new = jtfne.construct(tensor, runtime)
    model_old = nt.construct_neuronal_tensor(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    sig_new = jtfne.simulate(model_new, duration_ms=2.0, dt_ms=0.5, seed=0)
    sig_old = jtfne.simulate(model_old, duration_ms=2.0, dt_ms=0.5, seed=0)
    assert bool(jnp.array_equal(sig_new.spikes, sig_old.spikes))


def test_tensor_entrance_adds_only_overlays_to_the_configuration_lowering():
    """`construct(tensor)` lowers through `neuronal_tensor_to_configuration`
    then `construct(cfg)`, and overlays H and pose afterwards. So this
    compares one lowering with itself plus overlays, not two independent
    entrances: it pins that the overlays leave every signal bit-identical.

    P-022 is closed: the bridge carries `Layer.geometry` (fractional
    `tfne_geometry` domains, so the default-geometry fixture declares
    nothing and both entrances sample the historical bounds), and the
    position overlay applies only the default (identity) `Pose3D` here --
    so positions and with them every field proxy are bit-identical.
    200 ms is the shortest round duration at which this fixture spikes
    reliably (59 spikes), so the activity comparison is not vacuous."""
    D = 200.0
    tensor = _single_area_tensor()
    runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=D, dt_ms=0.5)
    model_a = jtfne.construct(tensor, runtime)
    cfg_b = nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=D, dt_ms=0.5)
    model_b = jtfne.construct(cfg_b)
    sig_a = jtfne.simulate(model_a, duration_ms=D, dt_ms=0.5, seed=0)
    sig_b = jtfne.simulate(model_b, duration_ms=D, dt_ms=0.5, seed=0)
    assert int(sig_a.spikes.sum()) > 0, "fixture must spike for a non-vacuous comparison"
    assert bool(jnp.array_equal(sig_a.spikes, sig_b.spikes))
    assert bool(jnp.array_equal(sig_a.V_m, sig_b.V_m))
    assert bool(jnp.array_equal(sig_a.sources, sig_b.sources))
    # The overlays ran but are identity here (H stored-but-inert with HDP
    # disabled, default pose): positions and every field readout agree.
    assert bool(jnp.array_equal(model_a.params["positions"], model_b.params["positions"]))
    assert sig_a.field is not None and sig_b.field is not None
    assert sig_a.field.source_proxy.shape == sig_b.field.source_proxy.shape
    assert float(jnp.abs(sig_a.field.source_proxy).max()) > 0, "field must be non-trivial"
    assert bool(jnp.array_equal(sig_a.field.source_proxy, sig_b.field.source_proxy))
    assert bool(jnp.array_equal(sig_a.field.phi_e_proxy, sig_b.field.phi_e_proxy))
    assert bool(jnp.array_equal(sig_a.field.lfp_proxy, sig_b.field.lfp_proxy))
    assert bool(jnp.array_equal(sig_a.field.csd_proxy, sig_b.field.csd_proxy))
    assert bool(jnp.array_equal(sig_a.field.kernel, sig_b.field.kernel))


def test_hand_spelled_configuration_matches_tensor_entrance():
    """Two INDEPENDENT entrances, one lowering: a Configuration spelled by
    hand through the public Configuration builders must give the same
    Signals as the NeuronalTensor `_single_area_tensor()` fixture.

    Entrance A goes through `jtfne.construct(tensor, runtime)` (which lowers
    via the bridge internally). Entrance B below spells the SAME values
    literally -- population, cell-type fractions, mechanism name/kind/params,
    connection rule (selectors, probability, weight, sign, mechanism ref),
    emitter, probes, field, runtime seed/duration/dt/dtype -- without calling
    `nt.neuronal_tensor_to_configuration` or anything that calls it.

    Literal provenance (read from `neuronal_tensor_to_configuration` +
    `_wire_connection` + `_connection_edge_weight`, not executed):
    area V1 = 20 neurons in layer L4; no declared fractions -> even split
    E/PV = 0.5/0.5; mechanism name "AMPA__dt2__0" (kind + `:g`-formatted
    dT_ms + dedup index 0), params tau_ms=2.0 plus the surfaced-but-inert
    reversal_mV=0.0; rule "interconn_V1_0", full-bipartite probability 1.0,
    weight |2.0 * 1.0| / sqrt(20), sign from the E source -> excitatory, no
    delay (InterConnection.delay_ms defaults to None); connectivity_mode
    "explicit" (passing inter_connections=[...] to Area(...) marks that area
    explicit, so the compiled graph holds only the declared rule).
    Omitted as inert: `tensor_identity` (a provenance hash nothing in
    construct/simulate reads) and the H overlay (stored but inert with HDP
    disabled, per the overlay test above). The fixture declares default
    (full-range) geometry, which the bridge omits from `tfne_geometry`, so
    there is no geometry to spell here -- and positions plus every field
    proxy are compared bit-exact.

    D=200 ms is the shortest round duration at which this fixture spikes
    reliably, so the comparison is not vacuous."""
    D = 200.0
    tensor = _single_area_tensor()
    runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=D, dt_ms=0.5)
    model_a = jtfne.construct(tensor, runtime)
    sig_a = jtfne.simulate(model_a, duration_ms=D, dt_ms=0.5, seed=0)

    cfg_b = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=D, dt_ms=0.5, dtype="float32")
        .update_metadata(connectivity_mode="explicit")
        .population(20, neurons={"L4": 20}, name="V1", layers=["L4"])
        .area_layer_cell_types("V1", {"L4": {"E": 0.5, "PV": 0.5}})
        .cell_types({"E": 0.5, "PV": 0.5})
        .mechanisms(
            name="AMPA__dt2__0", kind="AMPA",
            params={"tau_ms": 2.0, "reversal_mV": 0.0},
        )
        .connections(
            name="interconn_V1_0",
            source={"area": "V1", "layer": "L4", "cell_type": "E"},
            target={"area": "V1", "layer": "L4", "cell_type": "PV"},
            probability=1.0,
            weight=2.0 / math.sqrt(20),
            sign="excitatory",
            mechanism="AMPA__dt2__0",
        )
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model_b = jtfne.construct(cfg_b)
    assert model_b.cfg.metadata.get("recurrent_backend") == "edge_list"
    sig_b = jtfne.simulate(model_b, duration_ms=D, dt_ms=0.5, seed=0)
    assert int(sig_a.spikes.sum()) > 0, "fixture must spike for a non-vacuous comparison"
    assert bool(jnp.array_equal(sig_a.spikes, sig_b.spikes))
    assert bool(jnp.array_equal(sig_a.V_m, sig_b.V_m))
    assert bool(jnp.array_equal(sig_a.sources, sig_b.sources))
    assert bool(jnp.array_equal(model_a.params["positions"], model_b.params["positions"]))
    assert sig_a.field is not None and sig_b.field is not None
    assert float(jnp.abs(sig_a.field.source_proxy).max()) > 0, "field must be non-trivial"
    assert bool(jnp.array_equal(sig_a.field.source_proxy, sig_b.field.source_proxy))
    assert bool(jnp.array_equal(sig_a.field.phi_e_proxy, sig_b.field.phi_e_proxy))
    assert bool(jnp.array_equal(sig_a.field.lfp_proxy, sig_b.field.lfp_proxy))
    assert bool(jnp.array_equal(sig_a.field.csd_proxy, sig_b.field.csd_proxy))
    assert bool(jnp.array_equal(sig_a.field.kernel, sig_b.field.kernel))


def test_hand_spelled_two_area_configuration_matches_tensor_entrance():
    """Two areas of DIFFERENT sizes through the same two entrances as the
    single-area test above -- this is the case that discriminates lowering
    details the single-area fixture cannot.

    The fixture has V1 with 20 neurons and V2 with 12 (total_n = 32, so area
    size != total size), one InterConnection inside V1 (as above), and one
    AreaConnection V1 -> V2 carrying a non-None delay_ms.

    Entrance A goes through `jtfne.construct(tensor, runtime)` (which lowers
    via the bridge internally). Entrance B below spells the SAME values
    literally without calling `nt.neuronal_tensor_to_configuration` or
    anything that calls it.

    Literal provenance (read from `neuronal_tensor_to_configuration` +
    `_wire_connection` + `_connection_edge_weight`, not executed): populations
    spelled per area in declaration order (V1 = 20 in L4, V2 = 12 in L4); no
    declared fractions -> even splits (V1/L4 E/PV = 0.5/0.5, V2/L4 E = 1.0);
    global cell_types from the neuron-count-weighted fallback (E = 10 + 12 =
    22, PV = 10, over 32); mechanism names "GABA_A__dt5__0" (kind +
    `:g`-formatted dT_ms + dedup index 0, params tau_ms=5.0 plus the
    surfaced-but-inert reversal_mV=-80.0) and "AMPA__dt2__1" (dedup index 1);
    rules "interconn_V1_0" and "areaconn_0", full-bipartite probability 1.0,
    weights |2.0 * 1.0| / sqrt(32) and |1.5 * 1.0| / sqrt(32) -- normalized by
    the TOTAL (32), not the area size -- signs from the E sources ->
    excitatory, and the AreaConnection's delay_ms=2.0 forwarded into its rule
    (the InterConnection declares no delay); connectivity_mode "explicit".
    Omitted as inert: `tensor_identity`, the H overlay (stored but inert with
    HDP disabled). Both layers declare default (full-range) geometry, which
    the bridge omits from `tfne_geometry`, so there is no geometry to spell
    here -- and positions plus every field proxy are compared bit-exact.

    D=200 ms spikes in both areas (57 V1 + 36 V2 spikes), so the per-area
    comparison is not vacuous."""
    D = 200.0
    layer_v1 = nt.Layer(
        name="L4",
        n_neurons=20,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
    )
    inter = nt.InterConnection(
        source_layer="L4", source_neuron_type="E",
        target_layer="L4", target_neuron_type="PV",
        mechanism="AMPA",
        static=nt.StaticParams(
            g_mech={"AMPA": 1.0},
            reversal_potentials_mV={"AMPA": 0.0},
            dT_ms=2.0,
        ),
        plastic=nt.PlasticParams(w_mech=2.0, H=0.0),
    )
    area_v1 = nt.Area(name="V1", layers=[layer_v1], inter_connections=[inter])
    layer_v2 = nt.Layer(
        name="L4", n_neurons=12,
        neuron_types=[nt.NeuronType.make("E")],
    )
    area_v2 = nt.Area(name="V2", layers=[layer_v2])
    cross = nt.AreaConnection(
        source_area="V1", source_layer="L4", source_neuron_type="E",
        target_area="V2", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA",
        static=nt.StaticParams(dT_ms=2.0),
        plastic=nt.PlasticParams(w_mech=1.5, H=0.0),
        delay_ms=2.0,
    )
    tensor = nt.NeuronalTensor(areas=[area_v1, area_v2], area_connections=[cross])
    runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=D, dt_ms=0.5)
    model_a = jtfne.construct(tensor, runtime)
    sig_a = jtfne.simulate(model_a, duration_ms=D, dt_ms=0.5, seed=0)

    cfg_b = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=D, dt_ms=0.5, dtype="float32")
        .update_metadata(connectivity_mode="explicit")
        .population(20, neurons={"L4": 20}, name="V1", layers=["L4"])
        .population(12, neurons={"L4": 12}, name="V2", layers=["L4"])
        .area_layer_cell_types("V1", {"L4": {"E": 0.5, "PV": 0.5}})
        .area_layer_cell_types("V2", {"L4": {"E": 1.0}})
        .cell_types({"E": 22.0 / 32.0, "PV": 10.0 / 32.0})
        .mechanisms(
            name="AMPA__dt2__0", kind="AMPA",
            params={"tau_ms": 2.0, "reversal_mV": 0.0},
        )
        .connections(
            name="interconn_V1_0",
            source={"area": "V1", "layer": "L4", "cell_type": "E"},
            target={"area": "V1", "layer": "L4", "cell_type": "PV"},
            probability=1.0,
            weight=2.0 / math.sqrt(32),
            sign="excitatory",
            mechanism="AMPA__dt2__0",
        )
        .mechanisms(
            name="AMPA__dt2__1", kind="AMPA",
            params={"tau_ms": 2.0},
        )
        .connections(
            name="areaconn_0",
            source={"area": "V1", "layer": "L4", "cell_type": "E"},
            target={"area": "V2", "layer": "L4", "cell_type": "E"},
            probability=1.0,
            weight=1.5 / math.sqrt(32),
            sign="excitatory",
            mechanism="AMPA__dt2__1",
            delay_ms=2.0,
        )
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model_b = jtfne.construct(cfg_b)
    assert model_b.cfg.metadata.get("recurrent_backend") == "edge_list"
    sig_b = jtfne.simulate(model_b, duration_ms=D, dt_ms=0.5, seed=0)
    rows = model_a.neuron_table()
    assert [r["area"] for r in model_b.neuron_table()] == [r["area"] for r in rows]
    v1_idx = [i for i, r in enumerate(rows) if r["area"] == "V1"]
    v2_idx = [i for i, r in enumerate(rows) if r["area"] == "V2"]
    assert len(v1_idx) == 20 and len(v2_idx) == 12
    assert int(sig_a.spikes[:, v1_idx].sum()) > 0, "V1 must spike for a non-vacuous comparison"
    assert int(sig_a.spikes[:, v2_idx].sum()) > 0, "V2 must spike for a non-vacuous comparison"
    assert bool(jnp.array_equal(sig_a.spikes, sig_b.spikes))
    assert bool(jnp.array_equal(sig_a.V_m, sig_b.V_m))
    assert bool(jnp.array_equal(sig_a.sources, sig_b.sources))
    assert bool(jnp.array_equal(model_a.params["positions"], model_b.params["positions"]))
    assert sig_a.field is not None and sig_b.field is not None
    assert float(jnp.abs(sig_a.field.source_proxy).max()) > 0, "field must be non-trivial"
    assert bool(jnp.array_equal(sig_a.field.source_proxy, sig_b.field.source_proxy))
    assert bool(jnp.array_equal(sig_a.field.phi_e_proxy, sig_b.field.phi_e_proxy))
    assert bool(jnp.array_equal(sig_a.field.lfp_proxy, sig_b.field.lfp_proxy))
    assert bool(jnp.array_equal(sig_a.field.csd_proxy, sig_b.field.csd_proxy))
    assert bool(jnp.array_equal(sig_a.field.kernel, sig_b.field.kernel))


def test_declared_layer_geometry_carried_through_bridge():
    """A non-default `Layer.geometry` reaches the Configuration the bridge
    emits, so all three entrances -- `construct(tensor)`, the bridged
    `Configuration`, and a hand spelling with the literal `tfne_geometry`
    domains -- realize bit-identical positions and field proxies.

    The fixture mirrors `_single_area_tensor()` with sub-range geometry on
    every axis; D=200 ms spikes reliably (59 spikes), so the field
    comparison is not vacuous."""
    D = 200.0
    layer = nt.Layer(
        name="L4",
        n_neurons=20,
        neuron_types=[nt.NeuronType.make("E"), nt.NeuronType.make("PV")],
        geometry=nt.Geometry3D(x_range=(0.25, 0.75), y_range=(0.1, 0.5), z_range=(0.2, 0.6)),
    )
    tensor = nt.NeuronalTensor(
        areas=[
            nt.Area(
                name="V1",
                layers=[layer],
                inter_connections=[
                    nt.InterConnection(
                        source_layer="L4", source_neuron_type="E",
                        target_layer="L4", target_neuron_type="PV",
                        mechanism="AMPA",
                        static=nt.StaticParams(
                            g_mech={"AMPA": 1.0},
                            reversal_potentials_mV={"AMPA": 0.0},
                            dT_ms=2.0,
                        ),
                        plastic=nt.PlasticParams(w_mech=2.0, H=0.0),
                    )
                ],
            )
        ],
        name="geometry_carried_test",
    )
    cfg = nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=D, dt_ms=0.5)
    domains = cfg.metadata["tfne_geometry"]["domains"]
    assert domains["V1"]["L4"]["x"] == [0.25, 0.75] and domains["V1"]["L4"]["y"] == [0.1, 0.5]

    # Column-relative frame (owner ruling 2026-09-30): z is 20-60 % of the
    # column depth, not of L4's own block; x, y are fractions of the column width.
    pos = jtfne.construct(cfg).params["positions"]
    r = float(cfg.metadata.get("column_radius_mm", 0.25))
    eps = 1e-6
    assert float(pos[:, 2].min()) >= 0.2 - eps and float(pos[:, 2].max()) <= 0.6 + eps
    assert float(pos[:, 0].min()) >= -r + 0.25 * 2 * r - eps and float(pos[:, 0].max()) <= -r + 0.75 * 2 * r + eps
    assert float(pos[:, 1].min()) >= -r + 0.1 * 2 * r - eps and float(pos[:, 1].max()) <= -r + 0.5 * 2 * r + eps

    runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=D, dt_ms=0.5)
    model_a = jtfne.construct(tensor, runtime)
    model_b = jtfne.construct(cfg)
    assert bool(jnp.array_equal(model_a.params["positions"], model_b.params["positions"]))
    sig_a = jtfne.simulate(model_a, duration_ms=D, dt_ms=0.5, seed=0)
    sig_b = jtfne.simulate(model_b, duration_ms=D, dt_ms=0.5, seed=0)
    assert int(sig_a.spikes.sum()) > 0, "fixture must spike for a non-vacuous comparison"
    assert bool(jnp.array_equal(sig_a.spikes, sig_b.spikes))
    assert bool(jnp.array_equal(sig_a.field.source_proxy, sig_b.field.source_proxy))
    assert bool(jnp.array_equal(sig_a.field.csd_proxy, sig_b.field.csd_proxy))

    cfg_h = (
        jtfne.Configuration()
        .runtime(seed=0, duration_ms=D, dt_ms=0.5, dtype="float32")
        .update_metadata(connectivity_mode="explicit")
        .population(20, neurons={"L4": 20}, name="V1", layers=["L4"])
        .area_layer_cell_types("V1", {"L4": {"E": 0.5, "PV": 0.5}})
        .cell_types({"E": 0.5, "PV": 0.5})
        .update_metadata(tfne_geometry={"value_tag": "relative", "domains": domains})
        .mechanisms(
            name="AMPA__dt2__0", kind="AMPA",
            params={"tau_ms": 2.0, "reversal_mV": 0.0},
        )
        .connections(
            name="interconn_V1_0",
            source={"area": "V1", "layer": "L4", "cell_type": "E"},
            target={"area": "V1", "layer": "L4", "cell_type": "PV"},
            probability=1.0,
            weight=2.0 / math.sqrt(20),
            sign="excitatory",
            mechanism="AMPA__dt2__0",
        )
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model_h = jtfne.construct(cfg_h)
    sig_h = jtfne.simulate(model_h, duration_ms=D, dt_ms=0.5, seed=0)
    assert bool(jnp.array_equal(model_a.params["positions"], model_h.params["positions"]))
    assert bool(jnp.array_equal(sig_a.spikes, sig_h.spikes))
    assert bool(jnp.array_equal(sig_a.field.source_proxy, sig_h.field.source_proxy))


def test_pose_translation_is_absolute_per_area():
    """Owner ruling 2026-09-30: positions use the Configuration frame; a
    non-default Pose3D applies about the area's own column origin, so V2's
    translation does not stack on construct's 2 mm per-area x offset, and a
    default-pose area keeps construct's positions exactly."""
    def area(name, pose):
        return nt.Area(
            name=name,
            layers=[nt.Layer(name="L4", n_neurons=12, neuron_types=[nt.NeuronType.make("E")])],
            pose=pose,
        )

    tensor = nt.NeuronalTensor(
        areas=[area("V1", nt.Pose3D()), area("V2", nt.Pose3D(translation=(5.0, 0.0, 0.0)))]
    )
    cfg = nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    base = jtfne.construct(cfg).params["positions"]
    model = nt.construct_neuronal_tensor(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)
    pos = model.params["positions"]
    assert bool(jnp.array_equal(pos[:12], base[:12]))  # default pose: unchanged
    r = float(cfg.metadata.get("column_radius_mm", 0.25))
    assert float(pos[12:, 0].min()) >= 5.0 - r - 1e-5 and float(pos[12:, 0].max()) <= 5.0 + r + 1e-5
    assert bool(jnp.allclose(pos[12:, 1:], base[12:, 1:]))


def test_non_uniform_geometry_distribution_refused():
    """A non-`uniform_random` distribution is refused by the bridge, where
    the declaration would otherwise be silently sampled as uniform."""
    layer = nt.Layer(
        name="L4", n_neurons=4,
        neuron_types=[nt.NeuronType.make("E")],
        geometry=nt.Geometry3D(distribution="grid"),
    )
    tensor = nt.NeuronalTensor(areas=[nt.Area(name="V1", layers=[layer])])
    with pytest.raises(NotImplementedError, match="uniform_random"):
        nt.neuronal_tensor_to_configuration(tensor, seed=0, duration_ms=2.0, dt_ms=0.5)


def test_construct_tensor_defaults_runtime_when_omitted():
    tensor = _single_area_tensor()
    model = jtfne.construct(tensor)
    assert model.params["emitter"].n_neurons == 20


def test_construct_tensor_with_wrong_runtime_type_raises():
    tensor = _single_area_tensor()
    with pytest.raises(TypeError, match="RuntimeConfiguration"):
        jtfne.construct(tensor, "not_a_runtime_config")


def test_construct_configuration_with_runtime_raises():
    cfg = jtfne.build_laminar_column(n=10, ei_profile="canonical").set_emitter("izhikevich", "cortical_eig")
    with pytest.raises(ValueError, match="runtime"):
        jtfne.construct(cfg, jtfne.RuntimeConfiguration())


def test_construct_configuration_path_unaffected():
    """The original Configuration-based construct(cfg) / construct(cfg, geometry=...) path is untouched."""
    cfg = (
        jtfne.build_laminar_column(n=30, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m"], n_contacts=4)
    )
    model = jtfne.construct(cfg)
    sig = jtfne.simulate(model, duration_ms=2.0, dt_ms=0.5, seed=0)
    assert bool(jnp.all(jnp.isfinite(sig.spikes)))


def test_compute_fields_returns_existing_field_unchanged():
    cfg = (
        jtfne.build_laminar_column(n=30, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP", "CSD"], n_contacts=8)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
    )
    model = jtfne.construct(cfg)
    signals = jtfne.simulate(model, duration_ms=2.0, dt_ms=0.5, seed=0)
    fields = jtfne.compute_fields(model, signals)
    assert fields is signals.field


def test_compute_fields_raises_when_field_absent():
    cfg = (
        jtfne.build_laminar_column(n=10, ei_profile="canonical")
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"], n_contacts=4)
    )
    model = jtfne.construct(cfg)
    fieldless_signals = jtfne.Signals(
        time_ms=jnp.zeros((1,)), V_m=jnp.zeros((1, 10)), spikes=jnp.zeros((1, 10)),
        sources=None, field=None, metadata={},
    )
    with pytest.raises(ValueError, match="signals.field is None"):
        jtfne.compute_fields(model, fieldless_signals)
