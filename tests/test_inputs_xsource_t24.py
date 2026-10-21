"""T-24: frozen ``XSource`` input-source records.

Construction compiles the existing drive declarations (``Configuration.drive()``,
per-type ``cell_params(drive=...)``, ``NeuronType.drive`` bridged as
``tensor_triple_drive`` metadata, emitter-table defaults, and
``Simulation(poisson_drive=...)``) into tuples of ``XSource``
(``jaxfne/inputs.py``) without changing numerics: the numeric apply paths read
their values back out of those entries in the same order and dtype as before.

Byte-identity pins below are sha256 over ``np.asarray(V_m).tobytes()`` and
``np.asarray(spikes).tobytes()``. Each hash was recorded on the baseline
implementation (e5ba1258) and reproduced bit-exactly on the refactored tree;
``np.array_equal`` on V and spikes is asserted against a repeated run.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne.inputs import (
    XSource,
    compile_poisson_sources,
    xsource_to_dict,
)

DT = 0.5
DUR = 20.0


def _hashes(signals):
    v = np.asarray(signals.V_m)
    s = np.asarray(signals.spikes)
    assert bool(np.isfinite(v).all()), "fixture must stay finite"
    return (
        hashlib.sha256(v.tobytes()).hexdigest(),
        hashlib.sha256(s.tobytes()).hexdigest(),
        int(s.sum()),
    )


def _nt(*args, **kwargs):
    return nt.NeuronType.make(*args, **kwargs)


def _tensor(types, name="t24"):
    return nt.NeuronalTensor(
        name=name,
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=6, neuron_types=list(types))],
            inter_connections=[nt.InterConnection(
                source_layer="L4", source_neuron_type="E",
                target_layer="L4", target_neuron_type="E",
                mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0),
                delay_ms=1.0,
            )],
        )],
    )


def _adex_tensor(types, name="t24-adex"):
    return nt.NeuronalTensor(
        name=name,
        areas=[nt.Area(
            name="V1",
            layers=[nt.Layer(name="L4", n_neurons=6, neuron_types=list(types))],
            inter_connections=[nt.InterConnection(
                source_layer="L4", source_neuron_type="E",
                target_layer="L4", target_neuron_type="E",
                mechanism="AMPA",
                static=nt.StaticParams(
                    dT_ms=5.0, reversal_potentials_mV={"AMPA": 0.0}),
                delay_ms=1.0,
            )],
        )],
    )


def _izhi_cfg():
    return jtfne.suite2_net1_config(seed=11, n=8, duration_ms=DUR, dt_ms=DT)


def _izhi_model():
    return jtfne.construct(_izhi_cfg())


def _lif_runtime(**kwargs):
    kwargs.setdefault("duration_ms", DUR)
    return nt.RuntimeConfiguration(seed=0, dt_ms=DT, emitter="lif", **kwargs)


def _lif_model(tensor=None):
    return jtfne.construct(
        _tensor([_nt("E"), _nt("PV")]) if tensor is None else tensor, _lif_runtime())


AD_N = 60
AD_DT = 0.1


def _adex_cfg():
    return (
        jtfne.Configuration()
        .network(name="adex60", n=AD_N, cell_types={"E": 0.8, "PV": 0.1, "SOM": 0.1})
        .emitter(family="adex", adex_decay="exp")
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": 2.0, "reversal_mV": 0.0})
        .mechanisms(name="gaba", kind="receptor",
                    params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=0.2, weight=0.5, mechanism="ampa", delay_ms=2.0,
                     sampling="bernoulli")
        .connections(name="e_pv", source={"cell_type": "E"},
                     target={"cell_type": "PV"},
                     probability=0.2, weight=1.0, mechanism="ampa", delay_ms=1.3,
                     sampling="bernoulli")
        .connections(name="pv_e", source={"cell_type": "PV"},
                     target={"cell_type": "E"},
                     probability=0.2, weight=-2.0, mechanism="gaba",
                     delay_ms=0.25, sampling="bernoulli")
        .connections(name="som_e", source={"cell_type": "SOM"},
                     target={"cell_type": "E"},
                     probability=0.2, weight=1.0, mechanism="gaba", delay_ms=0.8,
                     sampling="bernoulli")
        .connectivity(edge_seed=11)
    )


def _adex_sim(**kwargs):
    return jtfne.Simulation(duration_ms=50.0, dt_ms=AD_DT, **kwargs)


AD_DRIVE = {
    "n_sources": 560, "rate_hz": 4.0, "weight_nS": 2.5, "mechanism": "ampa",
    "target_cell_types": ["E", "PV"], "delay_ms": 2.0, "seed": 7,
}


# --- Byte-identity pins (baseline e5ba1258) ----------------------------------


def test_pin_izhi_default():
    # baseline V=880d49c4... spikes=fbcef97b...
    signals = jtfne.simulate(_izhi_model())
    v_hash, s_hash, n = _hashes(signals)
    assert n == 8, "fixture must spike"
    assert v_hash == "880d49c4291e85a5439f61b2d8153abd03c371a806dfc723aa5aa527f0251809"
    assert s_hash == "fbcef97b1ebed40f92f30a77ab67edb945f29a161e2a91d5a33b38701bdb8835"
    repeat = jtfne.simulate(_izhi_model())
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


def test_pin_izhi_pertype_drive():
    # baseline V=94b94796... spikes=77609ebb...
    model = jtfne.construct(
        _izhi_cfg().drive(baseline_drive_by_cell_type={"E": 9.0}))
    signals = jtfne.simulate(model)
    v_hash, s_hash, n = _hashes(signals)
    assert n == 9, "fixture must spike"
    assert v_hash == "94b94796b7de3a5d6323e0543f553e6a7a05f496bec6a1d4e455b65cbb5ecb73"
    assert s_hash == "77609ebb81652ea7dd04eae09ec8c72d6572973697b6cf04a0d26918814f73e2"
    repeat = jtfne.simulate(model)
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


def test_pin_izhi_poisson_drive():
    # baseline V=d276e1ce... spikes=b71a9523...
    model = _izhi_model()
    kw = dict(
        duration_ms=12.0, dt_ms=DT, seed=17,
        runtime=jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list"),
        record_sources=True, record_fields=False,
        poisson_drive={"rate_hz": 200.0, "amplitude": 4.0, "seed": 11},
    )
    signals = jtfne.simulate(model, **kw)
    v_hash, s_hash, n = _hashes(signals)
    assert n == 7, "fixture must spike"
    assert v_hash == "d276e1cebec2653c4f13da3c8c64b98711b548331fbcb0bf262a7de22bdcfa43"
    assert s_hash == "b71a95234d6203e73011b12444e5a05a83bc5a999edfb4b774750aa6727c8f5a"
    repeat = jtfne.simulate(model, **kw)
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


def test_pin_lif_default():
    # baseline V=6bd3fdb4... spikes=f555501d...
    signals = jtfne.simulate(_lif_model())
    v_hash, s_hash, n = _hashes(signals)
    assert n == 36, "fixture must spike"
    assert v_hash == "6bd3fdb401fc161f6aefe1f10d3e4665f9c74d46aac0082e474ddee0c5304727"
    assert s_hash == "f555501deeaab2c885bcb6e0a18320815b331ef919021af47016ab839ccfcc03"
    repeat = jtfne.simulate(_lif_model())
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


def test_pin_lif_pertype_drive():
    # baseline V=1ecb4d57... spikes=5acb822b...
    tensor = _tensor([_nt("E", drive=9.0), _nt("PV")])
    signals = jtfne.simulate(jtfne.construct(tensor, _lif_runtime()))
    v_hash, s_hash, n = _hashes(signals)
    assert n == 48, "fixture must spike"
    assert v_hash == "1ecb4d57926a3c8eef31c0f05e12d0f0b5be023e77ca46a33d160c807aa68d55"
    assert s_hash == "5acb822b48a1330b10f7fdd06aca1b0731a79fb865e39b3961e7c205ceb12605"
    repeat = jtfne.simulate(jtfne.construct(tensor, _lif_runtime()))
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


def test_pin_lif_poisson_drive():
    # baseline V=9e5f29c6... spikes=6688b871...
    model = _lif_model()
    sim = jtfne.Simulation(
        duration_ms=DUR, dt_ms=DT, seed=1,
        poisson_drive={"rate_hz": 50.0, "amplitude": 100.0},
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"))
    signals = model.simulate(sim)
    v_hash, s_hash, n = _hashes(signals)
    assert n == 37, "fixture must spike"
    assert v_hash == "9e5f29c69115ee2932ddff333c203a9e9e8ae52561f0c5fb8530eaf98c1f2c23"
    assert s_hash == "6688b871d6ece9e0d26313060090409929eee9fc159d291ad5d5fac9d6db8380"
    repeat = model.simulate(sim)
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


def test_pin_adex_default():
    # baseline V=1899ffcb... spikes=37558623... (the fixture runs quiet by
    # design; the V pin still guards the no-drive path bit-exactly).
    model = jtfne.construct(_adex_cfg())
    signals = model.simulate(_adex_sim())
    v_hash, s_hash, n = _hashes(signals)
    assert n == 0, "fixture runs quiet with no drive"
    assert v_hash == "1899ffcbdebfd5aeab43a03a8343532f8bbb68e5bcf2bbd732404e3e77068983"
    assert s_hash == "3755862355e2e7d0e0dc0f6b98a89978c0710890982862dd17975829e35be6b4"
    repeat = model.simulate(_adex_sim())
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


def test_pin_adex_pertype_drive():
    # baseline V=fd9d43aa... spikes=31bf2217... (300 pA triple drive).
    tensor = _adex_tensor([_nt("E", emitter="adex", drive=300.0),
                           _nt("PV", emitter="adex")])
    runtime = nt.RuntimeConfiguration(
        seed=0, duration_ms=50.0, dt_ms=AD_DT, emitter="adex")
    signals = jtfne.construct(tensor, runtime).simulate(_adex_sim())
    v_hash, s_hash, n = _hashes(signals)
    assert n == 3, "fixture must spike"
    assert v_hash == "fd9d43aa57c48c303f3ca90061c6ce52286fd74b257db483ed7ad26e0e26e59d"
    assert s_hash == "31bf2217574b60f4f235a2a2d4f683deeb9f4268bed098deb3ac9d8f2762c5a1"
    repeat = jtfne.construct(tensor, runtime).simulate(_adex_sim())
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


def test_pin_adex_poisson_drive():
    # baseline V=6b4c5f4f... spikes=79fdc387...
    model = jtfne.construct(_adex_cfg())
    signals = model.simulate(_adex_sim(poisson_drive=dict(AD_DRIVE)))
    v_hash, s_hash, n = _hashes(signals)
    assert n == 166, "the drive must spike the fixture"
    assert v_hash == "6b4c5f4fbb2c047325d68223e9a9f11732187c129fa4a6661a9f53dd1267ecbe"
    assert s_hash == "79fdc3877395929a912772756888359c0d385dd7fd12f81b33e366874983dd7c"
    repeat = model.simulate(_adex_sim(poisson_drive=dict(AD_DRIVE)))
    assert np.array_equal(np.asarray(repeat.V_m), np.asarray(signals.V_m))
    assert np.array_equal(np.asarray(repeat.spikes), np.asarray(signals.spikes))


# --- Entry contents -----------------------------------------------------------


def test_xsource_is_an_advanced_root_attr():
    assert jtfne.XSource is XSource
    assert "XSource" not in jtfne.__all__
    from jaxfne.public_surface import ADVANCED_NAMESPACE

    assert ADVANCED_NAMESPACE["XSource"] == "jaxfne.inputs"


def _drive_by_triple(model):
    """Realized per-neuron drive keyed by (area, layer, cell_type), in order."""
    rows = model.neuron_table()
    emitter = model.params.get("emitter")
    adex = (model.static or {}).get("adex")
    if adex is not None:
        arr = adex.get("triple_drive_pA", None)
        assert arr is not None
        values = [float(v) for v in np.asarray(arr, dtype=float).tolist()]
    else:
        values = [float(v) for v in np.asarray(emitter.drive, dtype=float).tolist()]
    assert len(rows) == len(values)
    return [(row["area"], row["layer"], row["cell_type"]) for row in rows], values


def test_model_x_sources_derive_from_realized_drive():
    model = _izhi_model()
    assert isinstance(model.x_sources, tuple)
    assert all(isinstance(s, XSource) for s in model.x_sources)
    assert all(s.kind == "constant" and s.unit == "native_current" for s in model.x_sources)
    assert all(s.schedule is None and s.seed is None for s in model.x_sources)
    keys, values = _drive_by_triple(model)
    # One entry per realized (area, layer, cell_type) group: n=8 builds no
    # VIP, so unlike a defaults-filled declaration no VIP entry is reported.
    assert {s.targets for s in model.x_sources} == {tuple(k) for k in keys}
    for source in model.x_sources:
        members = [v for k, v in zip(keys, values) if tuple(k) == source.targets]
        assert members and all(v == source.value for v in members)


def test_derived_entries_carry_neuron_counts():
    model = _izhi_model()
    n = int(np.asarray(model.params["emitter"].drive).shape[0])
    assert n == 8
    assert all(isinstance(s.n_neurons, int) and s.n_neurons >= 1
               for s in model.x_sources)
    assert sum(s.n_neurons for s in model.x_sources) == n
    # The per-neuron drive reconstructs as each value repeated n_neurons.
    keys, values = _drive_by_triple(model)
    assert sorted(v for s in model.x_sources for v in [s.value] * s.n_neurons) == \
        sorted(values)
    row = xsource_to_dict(model.x_sources[0])
    assert row["n_neurons"] == model.x_sources[0].n_neurons


def test_heterogeneous_drive_per_neuron_splits_counts():
    import jax.numpy as jnp

    model = _izhi_model()
    n = int(np.asarray(model.params["emitter"].drive).shape[0])
    hetero = model.with_emitter_parameters(
        drive_per_neuron=jnp.arange(1, n + 1, dtype=jnp.float32))
    drive = [float(v) for v in np.asarray(
        hetero.params["emitter"].drive, dtype=float).tolist()]
    assert len(set(drive)) == n, "fixture must be fully heterogeneous"
    assert sum(s.n_neurons for s in hetero.x_sources) == n
    assert sorted(v for s in hetero.x_sources for v in [s.value] * s.n_neurons) == \
        sorted(drive)


def test_pertype_drive_entry_overrides_the_default_value():
    model = jtfne.construct(
        _izhi_cfg().drive(baseline_drive_by_cell_type={"E": 9.0}))
    by_target = {s.targets: s.value for s in model.x_sources}
    assert by_target[("net1", "uniform_3d", "E")] == 9.0
    # Partial override: undeclared types keep the emitter-table value (3.0
    # for PV), not the suite2_net1 full-map value (2.0).
    assert by_target[("net1", "uniform_3d", "PV")] == 3.0


def test_triple_drive_entries_carry_area_layer_type_targets():
    tensor = _tensor([_nt("E", drive=9.0), _nt("PV")])
    model = jtfne.construct(tensor, _lif_runtime())
    triples = [s for s in model.x_sources if len(s.targets) == 3]
    assert ("V1", "L4", "E") in [t.targets for t in triples]
    hit = [t for t in triples if t.targets == ("V1", "L4", "E")][0]
    assert (hit.kind, hit.value, hit.unit) == ("constant", 9.0, "native_current")


def test_cell_params_drive_entries_follow_declaration_order():
    tensor = _tensor([_nt("E", drive=7.0)], name="t24-cell")
    cfg = nt.neuronal_tensor_to_configuration(
        tensor, seed=0, duration_ms=DUR, dt_ms=DT).cell_params(
            {"cell_type": "E"}, {"drive": 2.0})
    model = jtfne.construct(cfg)
    # Derived entries name the realized triple, not the declaration syntax.
    by_target = {s.targets: s.value for s in model.x_sources}
    assert by_target[("V1", "L4", "E")] == 2.0


def test_adex_default_model_has_no_constant_entries():
    # No declared triple drive: the route carries no drive channel
    # (triple_drive_pA None), so the derived record is empty.
    assert jtfne.construct(_adex_cfg()).x_sources == ()


def test_adex_constant_entries_read_pA():
    tensor = _adex_tensor([_nt("E", emitter="adex", drive=300.0),
                           _nt("PV", emitter="adex")])
    runtime = nt.RuntimeConfiguration(
        seed=0, duration_ms=50.0, dt_ms=AD_DT, emitter="adex")
    model = jtfne.construct(tensor, runtime)
    units = {s.unit for s in model.x_sources}
    assert units == {"pA"}
    hit = [s for s in model.x_sources if s.targets == ("V1", "L4", "E")][0]
    assert hit.value == 300.0


def test_poisson_manifest_entry_current_route():
    model = _izhi_model()
    signals = jtfne.simulate(
        model, duration_ms=12.0, dt_ms=DT, seed=17,
        runtime=jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list"),
        record_sources=True, record_fields=False,
        poisson_drive={"rate_hz": 200.0, "amplitude": 4.0, "seed": 11},
    )
    manifest = model.manifest(signals)
    poisson = [row for row in manifest["x_sources"] if row["kind"] == "poisson"]
    assert len(poisson) == 1
    row = poisson[0]
    assert (row["value"], row["unit"]) == (200.0, "Hz")
    assert row["targets"] == ["all"]
    assert row["schedule"] == {"amplitude": 4.0}
    assert row["seed"] == 11
    constant = [r for r in manifest["x_sources"] if r["kind"] == "constant"]
    assert len(constant) == len(model.x_sources)


def test_poisson_manifest_entry_adex_route():
    # Model.manifest() is refused on the adex route by design (it reads
    # params['edge_list']); the io.manifest path carries the same x_sources.
    cfg = _adex_cfg()
    model = jtfne.construct(cfg)
    signals = model.simulate(_adex_sim(poisson_drive=dict(AD_DRIVE)))
    manifest = jtfne.io.manifest(cfg, signals)
    poisson = [row for row in manifest["x_sources"] if row["kind"] == "poisson"]
    assert len(poisson) == 1
    row = poisson[0]
    assert (row["value"], row["unit"]) == (4.0, "Hz")
    assert row["targets"] == ["E", "PV"]
    assert row["schedule"] == {
        "n_sources": 560, "weight_nS": 2.5, "mechanism": "ampa",
        "delay_ms": 2.0,
    }
    assert row["seed"] == 7
    # Without a model io.manifest emits no constant rows (a bare
    # configuration cannot name the realized population); the model's own
    # derived entries are empty here too (no declared triple drive).
    # Model.manifest() stays refused on the AdEx route by design.
    constant = [r for r in manifest["x_sources"] if r["kind"] == "constant"]
    assert constant == []
    assert model.x_sources == ()


def test_lif_entries_name_final_population_types():
    # Lif-only "complex" rides the bridge under an aliased Izhikevich name;
    # the derived entries read the restored labels + LIF drive channel.
    tensor = nt.NeuronalTensor(
        name="lif-two",
        areas=[nt.Area(
            name="V1",
            layers=[
                nt.Layer(name="LS", n_neurons=1,
                         neuron_types=[_nt("E", drive=1.0)],
                         geometry=nt.Geometry3D(
                             x_range=(0.0, 0.0), y_range=(0.0, 0.0))),
                nt.Layer(name="LC", n_neurons=1,
                         neuron_types=[_nt("complex", drive=2.0)],
                         geometry=nt.Geometry3D(
                             x_range=(5.0, 5.0), y_range=(0.0, 0.0))),
            ],
            inter_connections=[nt.InterConnection(
                source_layer="LS", source_neuron_type="E",
                target_layer="LC", target_neuron_type="complex",
                mechanism="AMPA", static=nt.StaticParams(dT_ms=5.0))])],
    )
    model = jtfne.construct(tensor, _lif_runtime())
    final_types = {r["cell_type"] for r in model.neuron_table()}
    assert final_types == {"E", "complex"}
    by_target = {s.targets: s.value for s in model.x_sources}
    assert by_target == {("V1", "LS", "E"): 1.0, ("V1", "LC", "complex"): 2.0}
    assert {t[2] for t in by_target} <= final_types


def test_poisson_window_maps_to_schedule():
    (entry,) = compile_poisson_sources(
        dict(AD_DRIVE, t_start_ms=5.0, t_stop_ms=40.0), route="adex")
    assert entry.schedule[-2:] == (("t_start_ms", 5.0), ("t_stop_ms", 40.0))
    row = xsource_to_dict(entry)
    assert row["schedule"]["t_start_ms"] == 5.0
    assert row["schedule"]["t_stop_ms"] == 40.0
    assert set(row) == {"kind", "targets", "value", "unit", "schedule", "seed", "n_neurons"}


def test_io_manifest_without_model_emits_no_constant_rows():
    # A bare configuration cannot name the realized population: without a
    # model there are no constant rows, only the run's Poisson rows.
    cfg = _izhi_cfg()
    signals = jtfne.simulate(_izhi_model())
    manifest = jtfne.io.manifest(cfg, signals)
    assert manifest["x_sources"] == []


def test_both_manifests_emit_the_same_x_sources():
    # One Izhikevich run: io.manifest with the model and Model.manifest
    # agree row for row (derived triples plus Poisson rows).
    model = _izhi_model()
    kw = dict(
        duration_ms=12.0, dt_ms=DT, seed=17,
        runtime=jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list"),
        record_sources=True, record_fields=False,
        poisson_drive={"rate_hz": 200.0, "amplitude": 4.0, "seed": 11},
    )
    signals = jtfne.simulate(model, **kw)
    cfg = _izhi_cfg()
    assert (jtfne.io.manifest(cfg, signals, model=model)["x_sources"]
            == model.manifest(signals)["x_sources"])
    assert len(model.manifest(signals)["x_sources"]) == len(model.x_sources) + 1


def test_connect_entries_name_merged_areas():
    first = _izhi_model()
    second = _izhi_model()
    combo = jtfne.connect(first, second, namespace=("A", "B"))
    areas = {r["area"] for r in combo.neuron_table()}
    assert areas == {"A/net1", "B/net1"}
    assert {s.targets[0] for s in combo.x_sources} == areas
    for source in combo.x_sources:
        assert source.targets[1:] == ("uniform_3d", source.targets[2])
    before = {(s.targets[1:], s.value) for s in first.x_sources}
    after_a = {(s.targets[1:], s.value) for s in combo.x_sources
               if s.targets[0] == "A/net1"}
    after_b = {(s.targets[1:], s.value) for s in combo.x_sources
               if s.targets[0] == "B/net1"}
    assert after_a == before and after_b == before


def test_set_parameter_on_drive_changes_x_sources():
    model = _izhi_model()
    before = {s.targets: s.value for s in model.x_sources}
    tuned = jtfne.set_parameter(model, "drive.E.rate", 9.0)
    after = {s.targets: s.value for s in tuned.x_sources}
    assert set(after) == set(before)
    for targets, value in after.items():
        if targets[2] == "E":
            assert value == 9.0
        else:
            assert value == before[targets]
    # Functional: the input model is unchanged.
    assert {s.targets: s.value for s in model.x_sources} == before


def test_tune_route_drive_scale_changes_x_sources():
    model = _izhi_model()
    before = {s.targets: s.value for s in model.x_sources}
    scaled = model.with_emitter_parameters(drive_scale=2.0)
    after = {s.targets: s.value for s in scaled.x_sources}
    assert set(after) == set(before)
    for targets, value in after.items():
        assert value == before[targets] * 2.0


def test_x_sources_assignment_is_refused():
    import dataclasses

    model = _izhi_model()
    with pytest.raises(AttributeError):
        model.x_sources = ()
    with pytest.raises(TypeError):
        dataclasses.replace(model, x_sources=())


# --- Refusals -----------------------------------------------------------------


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown XSource kind"):
        XSource(kind="burst", targets=("E",), value=1.0, unit="native_current")


def test_negative_poisson_rate_is_refused():
    with pytest.raises(ValueError, match="must be >= 0"):
        XSource(kind="poisson", targets=("all",), value=-1.0, unit="Hz",
                schedule=(("amplitude", 0.5),), seed=0)


def test_inverted_window_is_refused():
    with pytest.raises(ValueError, match="must be below"):
        XSource(kind="poisson", targets=("E",), value=4.0, unit="Hz",
                schedule=(("n_sources", 10), ("weight_nS", 2.0),
                          ("mechanism", "ampa"), ("delay_ms", 2.0),
                          ("t_start_ms", 40.0), ("t_stop_ms", 5.0)),
                seed=7)


def test_constant_takes_no_schedule_or_seed():
    with pytest.raises(ValueError, match="takes no schedule"):
        XSource(kind="constant", targets=("E",), value=5.0,
                unit="native_current", schedule=(("amplitude", 1.0),))
    with pytest.raises(ValueError, match="takes no seed"):
        XSource(kind="constant", targets=("E",), value=5.0,
                unit="native_current", seed=0)


def test_constant_unit_vocabulary_is_refused():
    with pytest.raises(ValueError, match="not realized"):
        XSource(kind="constant", targets=("E",), value=5.0, unit="Hz")


def test_poisson_unit_is_refused():
    with pytest.raises(ValueError, match="not realized"):
        XSource(kind="poisson", targets=("all",), value=2.0,
                unit="native_current", schedule=(("amplitude", 0.5),), seed=0)


def test_poisson_seed_takes_an_int():
    with pytest.raises(ValueError, match="seed must be an int"):
        XSource(kind="poisson", targets=("all",), value=2.0, unit="Hz",
                schedule=(("amplitude", 0.5),), seed=1.5)


def test_poisson_schedule_exact_mappings_only():
    with pytest.raises(ValueError, match="alone"):
        XSource(kind="poisson", targets=("all",), value=2.0, unit="Hz",
                schedule=(("amplitude", 0.5), ("delay_ms", 1.0)), seed=0)
    with pytest.raises(ValueError, match="exactly"):
        XSource(kind="poisson", targets=("E",), value=4.0, unit="Hz",
                schedule=(("n_sources", 10), ("weight_nS", 2.0)), seed=7)
    with pytest.raises(ValueError, match="together"):
        XSource(kind="poisson", targets=("E",), value=4.0, unit="Hz",
                schedule=(("n_sources", 10), ("weight_nS", 2.0),
                          ("mechanism", "ampa"), ("delay_ms", 2.0),
                          ("t_start_ms", 5.0)),
                seed=7)
    with pytest.raises(ValueError, match="unknown keys"):
        XSource(kind="poisson", targets=("all",), value=2.0, unit="Hz",
                schedule=(("gain", 3.0),), seed=0)


# --- End-to-end refusals through the public entry points ----------------------
#
# No public entry point accepts an XSource object (Configuration.drive(),
# cell_params(), NeuronType(drive=) and Simulation(poisson_drive=...) take
# plain mappings; asserted below), so the XSource-level refusals above are
# unit-tested. The entry points that consume the same declarations refuse
# the same bad values end to end:


def test_no_public_entry_point_accepts_xsource():
    import inspect

    for owner, name in (
        (jtfne.Configuration, "drive"),
        (jtfne.Configuration, "cell_params"),
        (jtfne.Simulation, "__init__"),
    ):
        params = inspect.signature(getattr(owner, name)).parameters
        assert not any("xsource" in p.lower() for p in params), (owner, name)


def test_negative_poisson_rate_refused_end_to_end():
    model = _izhi_model()
    with pytest.raises(ValueError, match="must be >= 0"):
        jtfne.simulate(
            model, duration_ms=12.0, dt_ms=DT, seed=17,
            runtime=jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list"),
            record_sources=True, record_fields=False,
            poisson_drive={"rate_hz": -1.0, "amplitude": 4.0, "seed": 11},
        )


def test_non_all_poisson_target_refused_end_to_end():
    model = _izhi_model()
    with pytest.raises(ValueError, match="only 'all'"):
        jtfne.simulate(
            model, duration_ms=12.0, dt_ms=DT, seed=17,
            runtime=jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list"),
            record_sources=True, record_fields=False,
            poisson_drive={"rate_hz": 200.0, "amplitude": 4.0, "seed": 11,
                           "target": "E"},
        )


def _adex_bad_drive(drive):
    model = jtfne.construct(_adex_cfg())
    with pytest.raises(ValueError, match="poisson_drive"):
        model.simulate(_adex_sim(poisson_drive=drive))


def test_adex_unknown_poisson_keys_refused_end_to_end():
    _adex_bad_drive(dict(AD_DRIVE, gain=3.0))


def test_adex_half_window_refused_end_to_end():
    _adex_bad_drive(dict(AD_DRIVE, t_start_ms=5.0))


def test_adex_inverted_window_refused_end_to_end():
    with pytest.raises(ValueError, match="must be below"):
        jtfne.construct(_adex_cfg()).simulate(
            _adex_sim(poisson_drive=dict(AD_DRIVE, t_start_ms=40.0, t_stop_ms=5.0)))


def test_adex_non_int_seed_refused_end_to_end():
    _adex_bad_drive(dict(AD_DRIVE, seed=1.5))


def _current_poisson_kw(**drive):
    return dict(
        duration_ms=12.0, dt_ms=DT, seed=17,
        runtime=jtfne.RuntimeConfig(dtype="float32", recurrent_backend="edge_list"),
        record_sources=True, record_fields=False,
        poisson_drive=drive,
    )


def test_current_route_window_refused_end_to_end():
    # The current route consumes no drive window (only the AdEx route
    # does): a windowed declaration fails at the public entry point,
    # naming both routes' consumed keys.
    with pytest.raises(ValueError, match="does not consume"):
        jtfne.simulate(
            _izhi_model(),
            **_current_poisson_kw(rate_hz=200.0, amplitude=4.0, seed=11,
                                 t_start_ms=2.0, t_stop_ms=8.0),
        )


def test_current_route_window_refused_on_lif_end_to_end():
    with pytest.raises(ValueError, match="does not consume"):
        jtfne.simulate(
            _lif_model(),
            **_current_poisson_kw(rate_hz=50.0, amplitude=100.0, seed=11,
                                 t_start_ms=2.0, t_stop_ms=8.0),
        )


@pytest.mark.parametrize("seed", [1.5, 2.0, "3"])
def test_current_route_non_int_seed_refused_end_to_end(seed):
    # A float seed was truncated silently (1.5 ran as 1): refused instead.
    with pytest.raises(ValueError, match="must be an int"):
        jtfne.simulate(
            _izhi_model(),
            **_current_poisson_kw(rate_hz=200.0, amplitude=4.0, seed=seed),
        )


def test_current_route_adex_keys_refused_end_to_end():
    # An AdEx-only schedule key on a mapping without target_cell_types is
    # unconsumed on the current route: refused, not silently dropped.
    with pytest.raises(ValueError, match="does not consume"):
        jtfne.simulate(
            _izhi_model(),
            **_current_poisson_kw(rate_hz=200.0, amplitude=4.0, seed=11,
                                 n_sources=10),
        )


def test_current_route_window_refused_in_compiler():
    with pytest.raises(ValueError, match="does not consume"):
        compile_poisson_sources(
            {"rate_hz": 2.0, "t_start_ms": 1.0, "t_stop_ms": 5.0},
            route="current",
        )
