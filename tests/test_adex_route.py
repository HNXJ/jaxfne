"""Tests for the AdEx emitter family's route: Configuration -> construct -> simulate
(jaxfne G-2 F2-F4, part B). Kernel numerics are covered by tests/test_emitters_adex.py.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pytest

import jaxfne as jtfne
import jaxfne._model_simulate as route_simulate
from jaxfne._construct_adex import ADEX_CONSUMED, _runtime_defaults
from jaxfne._model import Model
from jaxfne._runtime_config import RuntimeConfig
from jaxfne._signals import ObjectiveReport, Signals, Simulation
from jaxfne.connectivity import compile_connection_rules
from jaxfne.emitters_adex import make_adex_params, simulate_adex
from jaxfne.paradigm import ParadigmCondition

N = 60
DT_MS = 0.1
N_STEPS = 500  # 50 ms at dt 0.1 ms
CELL_TYPES = {"E": 0.8, "PV": 0.1, "SOM": 0.1}


def _kernel_args(model, dt_ms):
    from jaxfne._construct_adex import adex_kernel_args

    return adex_kernel_args(model, dt_ms)


def _plumbing_cfg(*, decay="exp", step_order="default", edge_seed=11, probability=0.2,
                  sampling="bernoulli", e_e_weight=0.5, e_e_delay=2.0,
                  ampa_tau=2.0, ampa_rev=0.0):
    return (
        jtfne.Configuration()
        .network(name="adex60", n=N, cell_types=CELL_TYPES)
        .emitter(family="adex", adex_decay=decay, adex_step_order=step_order)
        .mechanisms(name="ampa", kind="receptor",
                    params={"tau_ms": ampa_tau, "reversal_mV": ampa_rev})
        .mechanisms(name="gaba", kind="receptor", params={"tau_ms": 8.0, "reversal_mV": -80.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=probability, weight=e_e_weight, mechanism="ampa",
                     delay_ms=e_e_delay, sampling=sampling)
        .connections(name="e_pv", source={"cell_type": "E"}, target={"cell_type": "PV"},
                     probability=probability, weight=1.0, mechanism="ampa", delay_ms=1.3,
                     sampling=sampling)
        .connections(name="pv_e", source={"cell_type": "PV"}, target={"cell_type": "E"},
                     probability=probability, weight=-2.0, mechanism="gaba", delay_ms=0.25,
                     sampling=sampling)
        .connections(name="som_e", source={"cell_type": "SOM"}, target={"cell_type": "E"},
                     probability=probability, weight=1.0, mechanism="gaba", delay_ms=0.8,
                     sampling=sampling)
        .connectivity(edge_seed=edge_seed)
    )


def _base():
    return _plumbing_cfg()


def _fresh():
    return jtfne.Configuration().emitter(family="adex", adex_decay="exp")


def _network_cfg(**extra):
    return (
        jtfne.Configuration()
        .network(name="adex60", n=N, cell_types=CELL_TYPES, **extra)
        .emitter(family="adex", adex_decay="exp")
    )


def _sim(**kwargs):
    return jtfne.Simulation(duration_ms=50.0, dt_ms=DT_MS, **kwargs)


def _edges(cfg):
    st = jtfne.construct(cfg).static["adex"]
    return sorted(zip(st["pre"].tolist(), st["post"].tolist(), st["receptor"].tolist()))


# Independent expectation for the plumbing fixture, written out here rather than read from the
# route: (source type, target type) -> (receptor, weight in nS, delay in ms).
RULE_OF_PAIR = {
    ("E", "E"): ("ampa", 0.5, 2.0),
    ("E", "PV"): ("ampa", 1.0, 1.3),
    ("PV", "E"): ("gaba", -2.0, 0.25),
    ("SOM", "E"): ("gaba", 1.0, 0.8),
}
RECEPTOR_ORDER = ("ampa", "gaba")  # mechanisms() declaration order
EXPECTED_COUNTS = {"E": 48, "PV": 6, "SOM": 6}  # fractions 0.8, 0.1, 0.1 of N = 60


def _half_up_steps(delay_ms):
    """delay/dt rounded half up in exact decimal arithmetic (independent of the float helper)."""
    ratio = Decimal(repr(delay_ms)) / Decimal(repr(DT_MS))
    return int(ratio.quantize(Decimal(1), rounding=ROUND_HALF_UP))


@pytest.fixture
def kernel_calls(monkeypatch):
    """Route simulate() through a wrapper that records the kernel kwargs and outputs."""
    calls = []
    real = route_simulate.simulate_adex

    def recording(**kwargs):
        out = real(**kwargs)
        calls.append((kwargs, out))
        return out

    monkeypatch.setattr(route_simulate, "simulate_adex", recording)
    return calls


@pytest.mark.parametrize("decay", ["exp", "euler"])
@pytest.mark.parametrize("step_order", ["default", "brian2"])
def test_route_simulate_passes_declared_edges_and_cells(decay, step_order, kernel_calls):
    model = jtfne.construct(_plumbing_cfg(decay=decay, step_order=step_order))
    signals = model.simulate(_sim())
    assert len(kernel_calls) == 1, "simulate must run the kernel once"
    kw, out = kernel_calls[0]
    assert kw["n_steps"] == N_STEPS and kw["dt_ms"] == DT_MS
    assert kw["decay"] == decay and kw["step_order"] == step_order
    np.testing.assert_array_equal(np.asarray(signals.spikes), np.asarray(out["spikes"]))
    np.testing.assert_array_equal(np.asarray(signals.V_m), np.asarray(out["V_mV"]))

    labels = tuple(kw["params"].labels)
    assert {t: labels.count(t) for t in EXPECTED_COUNTS} == EXPECTED_COUNTS
    expected = make_adex_params(labels)
    for field in kw["params"]._fields:
        if field == "labels":
            continue
        want = np.asarray(getattr(expected, field), dtype=np.float32).astype(np.float64)
        got = np.asarray(getattr(kw["params"], field), dtype=np.float64)
        np.testing.assert_allclose(got, want, rtol=1e-12)

    pre, post = np.asarray(kw["edge_pre"]).tolist(), np.asarray(kw["edge_post"]).tolist()
    pairs = [(labels[a], labels[b]) for a, b in zip(pre, post)]
    assert set(pairs) <= set(RULE_OF_PAIR), f"unexpected edge pairs {set(pairs) - set(RULE_OF_PAIR)}"
    assert set(pairs) == set(RULE_OF_PAIR), "every declared rule must wire at least one edge"
    want_receptor = [RECEPTOR_ORDER.index(RULE_OF_PAIR[p][0]) for p in pairs]
    want_weight = [RULE_OF_PAIR[p][1] for p in pairs]
    want_delay = [_half_up_steps(RULE_OF_PAIR[p][2]) for p in pairs]
    np.testing.assert_array_equal(np.asarray(kw["edge_receptor"]), want_receptor)
    np.testing.assert_allclose(np.asarray(kw["edge_weight_nS"]), want_weight, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(kw["edge_delay_steps"]), want_delay)
    assert tuple(kw["receptor_reversal_mV"]) == (0.0, -80.0)
    assert tuple(kw["receptor_tau_ms"]) == (2.0, 8.0)


def test_half_step_delays_round_half_up():
    from jaxfne._construct_adex import _rule_delay_steps

    # 2.5 steps and 13.5 steps round half up; floor would give 2 and 13.
    assert _rule_delay_steps("half", 0.25, DT_MS) == 3
    assert _rule_delay_steps("half", 1.35, DT_MS) == 14


def test_model_methods_on_adex_are_refused_or_correct():
    model = jtfne.construct(_base())
    with pytest.raises(NotImplementedError, match="edge_table"):
        model.edge_table()
    with pytest.raises(NotImplementedError, match="compile_connections"):
        model.compile_connections(seed=1)
    signals = model.simulate(_sim())
    out = model.probe(signals, ["spikes", "V_m"])
    np.testing.assert_array_equal(np.asarray(out["spikes"]), np.asarray(signals.spikes))
    np.testing.assert_array_equal(np.asarray(out["V_m"]), np.asarray(signals.V_m))


def test_v0_drive_spikes_and_reaches_post_cells():
    model = jtfne.construct(_plumbing_cfg())
    assert tuple(model.params["emitter"].labels[:5]) == ("E",) * 5, (
        "fixture: the first five cells must be E"
    )
    V0 = np.full(N, -65.0)
    V0[:5] = -40.0
    out = simulate_adex(
        **_kernel_args(model, DT_MS), n_steps=N_STEPS, dt_ms=DT_MS, record_v=True, V0_mV=V0
    )
    assert int(np.sum(out["spikes"])) > 0, "fixture must spike"
    assert float(np.max(np.abs(np.asarray(out["g"])))) > 0.0, (
        "spikes must reach a post cell's conductance"
    )


def test_compiled_edges_equal_compile_connection_rules():
    cfg = _plumbing_cfg(edge_seed=11)
    model = jtfne.construct(cfg)
    ref = compile_connection_rules(
        model.static["neuron_metadata"],
        cfg.metadata["circuit"]["connections"],
        cfg.metadata["circuit"]["mechanisms"],
        seed=11,
    )
    st = model.static["adex"]
    got = sorted(zip(
        st["pre"].tolist(), st["post"].tolist(), st["receptor"].tolist(),
        st["weight_nS"].tolist(), st["delay_ms"].tolist(),
    ))
    want = sorted(zip(
        np.asarray(ref.edge_pre).tolist(), np.asarray(ref.edge_post).tolist(),
        np.asarray(ref.edge_mechanism).tolist(), np.asarray(ref.edge_weight).tolist(),
        np.asarray(ref.edge_delay_ms).tolist(),
    ))
    assert len(got) > 0, "fixture must wire edges"
    assert got == want


def test_edge_seed_selects_the_edge_set():
    a = jtfne.construct(_plumbing_cfg(edge_seed=1)).static["adex"]
    b = jtfne.construct(_plumbing_cfg(edge_seed=2)).static["adex"]
    a_again = jtfne.construct(_plumbing_cfg(edge_seed=1)).static["adex"]
    assert a["pre"].size > 0 and b["pre"].size > 0, "fixture must wire edges"
    assert set(zip(a["pre"].tolist(), a["post"].tolist())) != set(
        zip(b["pre"].tolist(), b["post"].tolist())
    )
    for key in ("pre", "post", "receptor", "weight_nS", "delay_ms"):
        np.testing.assert_array_equal(a[key], a_again[key])


def test_runtime_seed_draws_the_edges_when_edge_seed_is_unset():
    one = _edges(_plumbing_cfg(edge_seed=None).runtime(seed=1))
    two = _edges(_plumbing_cfg(edge_seed=None).runtime(seed=2))
    assert len(one) > 0 and one != two


def test_probability_and_sampling_change_the_edges():
    assert _edges(_plumbing_cfg(probability=0.2)) != _edges(_plumbing_cfg(probability=0.05))
    assert _edges(_plumbing_cfg(sampling="bernoulli")) != _edges(_plumbing_cfg(sampling="density"))


def test_network_and_rule_names_reach_the_signals_metadata():
    signals = jtfne.construct(_base()).simulate(_sim())
    assert signals.metadata["network"] == "adex60"
    assert signals.metadata["receptors"] == ["ampa", "gaba"]
    assert signals.metadata["rules"] == ["e_e", "e_pv", "pv_e", "som_e"]


def test_record_flags_at_default_are_recorded_as_not_recorded():
    signals = jtfne.construct(_base()).simulate(_sim())
    assert signals.sources is None and signals.field is None
    assert signals.metadata["not_recorded"] == ["sources", "field"]


def test_runtime_dt_and_duration_must_match_the_simulation():
    with pytest.raises(ValueError, match="dt_ms"):
        jtfne.construct(_base().runtime(dt_ms=0.2)).simulate(_sim())
    with pytest.raises(ValueError, match="duration_ms"):
        jtfne.construct(_base().runtime(duration_ms=99.0)).simulate(_sim())
    jtfne.construct(_base().runtime(dt_ms=DT_MS, duration_ms=50.0)).simulate(_sim())


def test_jit_default_accepted_and_other_values_refused_at_both_stages():
    jtfne.construct(_base().runtime(jit="auto"))
    with pytest.raises(ValueError, match="jit"):
        jtfne.construct(_base().runtime(jit=False))
    model = jtfne.construct(_base())
    model.simulate(_sim(runtime=RuntimeConfig(recurrent_backend="edge_list", jit="auto")))
    with pytest.raises(ValueError, match="jit"):
        model.simulate(_sim(runtime=RuntimeConfig(recurrent_backend="edge_list", jit=False)))


def test_default_runtime_config_is_refused_for_its_dense_recurrence():
    model = jtfne.construct(_base())
    with pytest.raises(ValueError, match="recurrent_backend"):
        model.simulate(_sim(runtime=RuntimeConfig()))


def _non_default(name, default):
    """A value that differs from the field's default (a value the core rejects is also a refusal)."""
    table = {
        "poisson_drive": {"rate_hz": 2.0},
        "ablation": "E_silence",
        "device_type": "gpu",
        "dtype_primary": "float64",
        "x64_enabled": True,
        "synaptic_kernel": "receptor_exponential",
    }
    if name in table:
        return table[name]
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 1
    if isinstance(default, float):
        return default + 0.5
    if isinstance(default, str):
        return default + "_x"
    if isinstance(default, dict):
        return {"k": 1.0}
    raise AssertionError(f"no non-default value is defined for {name!r}")


UNCONSUMED_RUNTIME = [
    f.name for f in dataclasses.fields(RuntimeConfig) if ("RuntimeConfig", f.name) not in ADEX_CONSUMED
]
UNCONSUMED_SIMULATION = [
    f.name for f in dataclasses.fields(Simulation) if ("Simulation", f.name) not in ADEX_CONSUMED
]


@pytest.mark.parametrize("name", UNCONSUMED_SIMULATION)
def test_unconsumed_simulation_field_is_refused_by_name(name):
    model = jtfne.construct(_base())
    value = _non_default(name, getattr(Simulation(duration_ms=50.0, dt_ms=DT_MS), name))
    with pytest.raises(ValueError, match=name):
        model.simulate(_sim(**{name: value}))


@pytest.mark.parametrize("name", UNCONSUMED_RUNTIME)
def test_unconsumed_runtime_field_is_refused_by_name(name):
    model = jtfne.construct(_base())
    value = _non_default(name, _runtime_defaults()[name])
    kwargs = {"recurrent_backend": "edge_list", name: value}
    with pytest.raises(ValueError, match=name):
        model.simulate(_sim(runtime=RuntimeConfig(**kwargs)))


# Each case declares one refused input and must raise at construct. The needle is a field name
# the message must carry; a tuple accepts any of several (the first refused field is reported).
CONSTRUCT_REFUSALS = [
    ("columns", lambda: _fresh().column("V1", layers=["L2"], n=N), ("networks", "column")),
    ("areas", lambda: _fresh().column("V1", layers=["L2"], n=N).areas(["V1"]), ("networks", "areas")),
    ("layers", lambda: jtfne.Configuration().network(name="x", n=N, layers=["L2"]), "layers"),
    ("uniform3d", lambda: _base().uniform3d(), ("uniform3d", "geometry_mode")),
    ("population", lambda: _fresh().population(N, {"L2": 1.0}), ("networks", "population")),
    ("layer_fractions", lambda: _base().geometry(), "layer_fractions"),
    ("drive", lambda: _base().drive(), "drive"),
    ("cell_params", lambda: _base().cell_params({"cell_type": "E"}, {"a": 0.1}), "cell_params"),
    ("canonical_biophysics", lambda: _base().runtime(canonical_biophysics=True),
     "canonical_biophysics"),
    ("random_v0", lambda: _base().runtime(random_v0=True), "random_v0"),
    ("recurrent_backend", lambda: _base().runtime(recurrent_backend="dense"),
     "recurrent_backend"),
    ("synaptic_kernel", lambda: _base().runtime(synaptic_kernel="receptor_exponential"),
     "synaptic_kernel"),
    ("runtime backend", lambda: _base().runtime(backend="cpu"), "backend"),
    ("runtime device_type", lambda: _base().runtime(device_type="gpu"), "device_type"),
    ("runtime precision", lambda: _base().runtime(precision="high"), "precision"),
    ("runtime vmap", lambda: _base().runtime(vmap=True), "vmap"),
    ("runtime enable_hdp", lambda: _base().runtime(enable_hdp=True), "enable_hdp"),
    ("runtime enable_homeostasis", lambda: _base().runtime(enable_homeostasis=True),
     "enable_homeostasis"),
    ("field", lambda: _base().field(), "field"),
    ("probe", lambda: _base().probe(modes=["vm"]), "probe"),
    ("probe with no arguments", lambda: _base().probe(), "probes"),
    ("homeostasis", lambda: _base().homeostasis(relative_baseline=0.5), "homeostasis"),
    ("hdp", lambda: _base().hdp(relative_baseline=0.5), "hdp"),
    ("plasticity", lambda: _base().plasticity(), "plasticity"),
    ("p_connect", lambda: _base().connectivity(p_connect=0.5), "p_connect"),
    ("network p_connect", lambda: _network_cfg(p_connect=0.5), "p_connect"),
    ("lesions", lambda: _base().lesions(name="l", selector={"cell_type": "E"}, effect="silence"),
     "lesions"),
    ("preset", lambda: jtfne.Configuration().network(name="x", n=N, cell_types=CELL_TYPES)
     .emitter(family="adex", adex_decay="exp", preset="cortical_eig"), "preset"),
    ("geometry", lambda: jtfne.construct(_base(), geometry=object()), "geometry"),
    ("sign", lambda: _base().connections(
        name="x", source={"cell_type": "E"}, target={"cell_type": "E"}, probability=0.1,
        weight=1.0, mechanism="ampa", delay_ms=2.0, sign="excitatory"), "sign"),
    ("max_in_degree", lambda: _base().connections(
        name="x", source={"cell_type": "E"}, target={"cell_type": "E"}, weight=1.0,
        mechanism="ampa", delay_ms=2.0, max_in_degree=3), "max_in_degree"),
    ("spatial_sigma", lambda: _base().connections(
        name="x", source={"cell_type": "E"}, target={"cell_type": "E"}, probability=0.1,
        weight=1.0, mechanism="ampa", delay_ms=2.0, spatial_sigma=0.1), "spatial_sigma"),
    ("weight", lambda: _base().connections(
        name="x", source={"cell_type": "E"}, target={"cell_type": "E"}, probability=0.1,
        mechanism="ampa", delay_ms=2.0), "weight"),
    ("delay_ms", lambda: _base().connections(
        name="x", source={"cell_type": "E"}, target={"cell_type": "E"}, probability=0.1,
        weight=1.0, mechanism="ampa"), "delay_ms"),
    ("mechanism", lambda: _base().connections(
        name="x", source={"cell_type": "E"}, target={"cell_type": "E"}, probability=0.1,
        weight=1.0, delay_ms=2.0), "mechanism"),
    ("mechanism params", lambda: _base().mechanisms(name="nmda", kind="receptor"), "nmda"),
    ("mechanism tau", lambda: _base().mechanisms(
        name="bad", kind="receptor", params={"tau_ms": 0.0, "reversal_mV": 0.0}), "bad"),
    ("mechanism kind", lambda: _base().mechanisms(
        name="kx", kind="ion", params={"tau_ms": 2.0, "reversal_mV": 0.0}), "kind"),
    ("adex_decay", lambda: jtfne.Configuration().network(
        name="x", n=N, cell_types=CELL_TYPES).emitter(family="adex"), "adex_decay"),
    ("trainables", lambda: _base().trainables(name="w", path="x"), ("training", "trainables")),
    ("objective_outputs", lambda: _base().objective_outputs(name="o"),
     ("training", "objective_outputs")),
    ("objective", lambda: _base().objective(), "objective"),
    ("optimizer", lambda: _base().optimizer(), "optimizer"),
    ("suite2_interarea", lambda: _base().suite2_interarea(True), "suite2_interarea"),
    ("inter_column_connectivity", lambda: _base().inter_column_connectivity(),
     "inter_column_connectivity"),
    ("runtime dtype", lambda: _base().runtime(dtype="float64"), "dtype"),
    ("runtime jit", lambda: _base().runtime(jit=False), "jit"),
]


@pytest.mark.parametrize(
    "label, build, needle", CONSTRUCT_REFUSALS, ids=[c[0] for c in CONSTRUCT_REFUSALS]
)
def test_construct_refusals_name_the_argument(label, build, needle):
    needles = (needle,) if isinstance(needle, str) else needle
    with pytest.raises(ValueError) as info:
        cfg = build()
        jtfne.construct(cfg)
    assert any(n in str(info.value) for n in needles), str(info.value)


SIMULATE_REFUSALS = [
    ("continuation", lambda m: m.simulate(_sim(), continuation=object()), "continuation"),
    ("return_state", lambda m: m.simulate(_sim(), return_state=True), "return_state"),
    ("paradigm", lambda m: m.simulate(_sim(), paradigm=object()), "paradigm"),
    ("poisson_drive", lambda m: m.simulate(_sim(poisson_drive={"rate_hz": 2.0})),
     "poisson_drive"),
    ("ablation", lambda m: m.simulate(_sim(ablation="E_silence")), "ablation"),
    ("simulate_batch", lambda m: m.simulate_batch(_sim(), n_seeds=2), "simulate_batch"),
    ("simulate_condition", lambda m: m.simulate_condition(_sim(), condition=object()),
     "simulate_condition"),
    ("plasticity", lambda m: m.simulate(_sim(plasticity=0.5)), "plasticity"),
    ("seed", lambda m: m.simulate(_sim(seed=3)), "seed"),
    ("record_sources", lambda m: m.simulate(_sim(record_sources=False)), "record_sources"),
    ("record_fields", lambda m: m.simulate(_sim(record_fields=False)), "record_fields"),
    ("runtime dtype", lambda m: m.simulate(
        _sim(runtime=RuntimeConfig(recurrent_backend="edge_list", dtype="float64"))), "dtype"),
    ("runtime jit", lambda m: m.simulate(
        _sim(runtime=RuntimeConfig(recurrent_backend="edge_list", jit=True))), "jit"),
]


@pytest.mark.parametrize(
    "label, act, needle", SIMULATE_REFUSALS, ids=[c[0] for c in SIMULATE_REFUSALS]
)
def test_simulate_refusals_name_the_argument_and_family(label, act, needle):
    model = jtfne.construct(_base())
    with pytest.raises((ValueError, NotImplementedError)) as info:
        act(model)
    msg = str(info.value)
    assert needle in msg
    assert "adex" in msg.lower()


@pytest.mark.parametrize("family", ["izhikevich", "homeostatic_ei"])
def test_adex_keys_refused_for_other_families(family):
    # Match the family-specific message: the generic "has no consumer" refusal
    # for unknown keys must not satisfy this test.
    with pytest.raises(ValueError, match="construct reads it for family='adex' only"):
        jtfne.Configuration().emitter(family=family, adex_decay="exp")
    with pytest.raises(ValueError, match="construct reads it for family='adex' only"):
        jtfne.Configuration().emitter(family=family, adex_step_order="brian2")


@pytest.mark.parametrize(
    "kwargs, needle",
    [
        ({"adex_decay": "rk4"}, "realizes adex_decay in"),
        ({"adex_step_order": "fast"}, "realizes adex_step_order in"),
    ],
)
def test_adex_key_values_refused(kwargs, needle):
    with pytest.raises(ValueError, match=needle):
        jtfne.Configuration().emitter(family="adex", **kwargs)


def test_delay_below_one_step_refused_at_simulate():
    # 0.04 ms at dt 0.1 ms rounds to 0 steps: construct cannot see dt, so simulate refuses.
    cfg = _base().connections(
        name="fast", source={"cell_type": "E"}, target={"cell_type": "E"}, probability=0.1,
        weight=1.0, mechanism="ampa", delay_ms=0.04,
    )
    model = jtfne.construct(cfg)
    with pytest.raises(ValueError) as info:
        model.simulate(_sim())
    msg = str(info.value)
    assert "fast" in msg and "0.04" in msg and "0.1" in msg


def test_unknown_cell_type_refused_at_construct():
    cfg = (
        jtfne.Configuration()
        .network(name="x", n=N, cell_types={"E": 0.8, "SST": 0.2})
        .emitter(family="adex", adex_decay="exp")
    )
    with pytest.raises(ValueError, match="SST"):
        jtfne.construct(cfg)


@pytest.mark.parametrize(
    "key, value",
    [("foo", False), ("foo", None), ("circuit_declared", False), ("circuit_compiled", True)],
)
def test_metadata_key_absent_from_defaults_refused_whatever_its_value(key, value):
    with pytest.raises(ValueError, match=f"'{key}'"):
        jtfne.construct(_base().update_metadata(**{key: value}))


def test_builder_flags_accepted_at_the_value_the_builder_writes():
    jtfne.construct(_base().update_metadata(circuit_declared=True, circuit_compiled=False))
    with pytest.raises(ValueError, match="'circuit_declared'"):
        jtfne.construct(_network_cfg().update_metadata(circuit_declared=True))


def test_refusal_text_says_does_not_consume_once():
    with pytest.raises(ValueError) as info:
        jtfne.construct(_base().update_metadata(foo=False))
    assert str(info.value).count("does not consume") == 1, str(info.value)


# Walk of every public Model method on an adex model. Each method either returns a result that
# _check_walk_result verifies, or raises the named error in WALK_NAMED (the message must name the
# method), or restore's missing-file error. The walk is keyed by dir(Model): a method with no entry
# fails the walk instead of passing unseen.
PUBLIC_MODEL_METHODS = sorted(name for name in dir(Model) if not name.startswith("_"))

WALK_NAMED = {
    "checkpoint": NotImplementedError,
    "compile_connections": NotImplementedError,
    "connectivity_summary": NotImplementedError,
    "dense_recurrent_weights": NotImplementedError,
    "edge_table": NotImplementedError,
    "manifest": NotImplementedError,
    "run_receipt": NotImplementedError,
    "simulate_batch": NotImplementedError,
    "simulate_condition": ValueError,
    "summary": NotImplementedError,
    "tune": NotImplementedError,
    "with_emitter_parameters": NotImplementedError,
    "with_hdp_initial_state": NotImplementedError,
    "with_recurrent_coupling": NotImplementedError,
}

# Arguments are minimal valid ones: m is the adex model, s its signals, p a scratch directory.
WALK_CALLS = {
    "checkpoint": lambda m, s, p: m.checkpoint(str(p / "walk")),
    "compile_connections": lambda m, s, p: m.compile_connections(seed=1),
    "compute_readout": lambda m, s, p: m.compute_readout(
        s, [jtfne.ReadoutSpec(name="rate", metric="spike_rate_hz")]
    ),
    "connectivity_summary": lambda m, s, p: m.connectivity_summary(),
    "dense_recurrent_weights": lambda m, s, p: m.dense_recurrent_weights(),
    "edge_table": lambda m, s, p: m.edge_table(),
    "evaluate": lambda m, s, p: m.evaluate(s, jtfne.Objective(name="walk")),
    "evaluate_report": lambda m, s, p: m.evaluate_report(s, jtfne.Objective(name="walk")),
    "last_hdp_diagnostics": lambda m, s, p: m.last_hdp_diagnostics(),
    "last_homeostasis_diagnostics": lambda m, s, p: m.last_homeostasis_diagnostics(),
    "manifest": lambda m, s, p: m.manifest(s),
    "neuron_table": lambda m, s, p: m.neuron_table(),
    "probe": lambda m, s, p: m.probe(s, ["spikes", "V_m"]),
    "record": lambda m, s, p: m.record(s, ["spikes"]),
    "restore": lambda m, s, p: m.restore(str(p / "missing"), m.cfg),
    "run_receipt": lambda m, s, p: m.run_receipt(s),
    "run_trials": lambda m, s, p: m.run_trials(
        jtfne.TrialBatch(trials=(jtfne.TrialSpec("a", seed=0),)), _sim()
    ),
    "select": lambda m, s, p: m.select(cell_type="E"),
    "simulate": lambda m, s, p: m.simulate(_sim()),
    "simulate_batch": lambda m, s, p: m.simulate_batch(_sim(), n_seeds=2),
    "simulate_condition": lambda m, s, p: m.simulate_condition(
        _sim(), ParadigmCondition(name="walk", sequence=("A",))
    ),
    "summary": lambda m, s, p: m.summary(),
    "tune": lambda m, s, p: m.tune(objective=jtfne.Objective(name="walk")),
    "with_emitter_parameters": lambda m, s, p: m.with_emitter_parameters(a=0.02),
    "with_hdp_initial_state": lambda m, s, p: m.with_hdp_initial_state(H0=np.ones(N)),
    "with_recurrent_coupling": lambda m, s, p: m.with_recurrent_coupling(),
}


def test_run_trials_refuses_a_trial_seed_other_than_the_route_default(walk_model):
    # run_trials sets Simulation.seed to the trial seed; the adex route consumes no Simulation.seed.
    batch = jtfne.TrialBatch(trials=(jtfne.TrialSpec("a", seed=1),))
    with pytest.raises(ValueError, match="'seed'"):
        walk_model.run_trials(batch, _sim())


def _check_walk_result(name, result, model, signals):
    """Verify the result of a method the walk expects to return (not refuse)."""
    if name == "compute_readout":
        (rate,) = result
        assert rate.status == "computed", rate
        want = float(np.mean(np.asarray(signals.spikes))) * 1000.0 / DT_MS
        assert rate.value == pytest.approx(want, rel=1e-5)
    elif name == "evaluate":
        assert result["objective_name"] == "walk" and result["total_loss"] is None, result
        assert result["all_gates_pass"] is True
    elif name == "evaluate_report":
        assert isinstance(result, ObjectiveReport) and result.objective_name == "walk"
    elif name in ("last_hdp_diagnostics", "last_homeostasis_diagnostics"):
        assert result is None, "the route never enables HDP or homeostasis"
    elif name == "neuron_table":
        assert len(result) == N
        assert Counter(row["cell_type"] for row in result) == EXPECTED_COUNTS
    elif name == "probe":
        assert result["spikes"] is signals.spikes and result["V_m"] is signals.V_m
    elif name == "record":
        assert result["spikes"] is signals.spikes
    elif name == "run_trials":
        (trial,) = result.results
        assert trial.success and trial.signals is not None, trial
        assert np.asarray(trial.signals.spikes).shape == np.asarray(signals.spikes).shape
    elif name == "select":
        rows = model.neuron_table()
        idx = np.asarray(result).tolist()
        assert len(idx) == EXPECTED_COUNTS["E"]
        assert {rows[i]["cell_type"] for i in idx} == {"E"}
    elif name == "simulate":
        assert isinstance(result, Signals)
        assert np.asarray(result.spikes).shape == (N_STEPS, N)
    else:
        raise AssertionError(f"no check for Model.{name}(): add one with its walk entry")


@pytest.fixture(scope="module")
def walk_model():
    return jtfne.construct(_base())


@pytest.fixture(scope="module")
def walk_signals(walk_model):
    return walk_model.simulate(_sim())


def test_walk_table_names_every_public_method():
    assert sorted(WALK_CALLS) == PUBLIC_MODEL_METHODS


@pytest.mark.parametrize("name", PUBLIC_MODEL_METHODS)
def test_public_model_method_on_adex_returns_checked_or_refuses_by_name(
    name, walk_model, walk_signals, tmp_path
):
    call = WALK_CALLS[name]
    if name == "restore":
        # A checkpoint cannot be written on this route, so restore reads a file that is absent.
        with pytest.raises(FileNotFoundError):
            call(walk_model, walk_signals, tmp_path)
    elif name in WALK_NAMED:
        with pytest.raises(WALK_NAMED[name]) as info:
            call(walk_model, walk_signals, tmp_path)
        assert f"{name}(" in str(info.value), str(info.value)
    else:
        result = call(walk_model, walk_signals, tmp_path)
        _check_walk_result(name, result, walk_model, walk_signals)


# The CAMINOS external drive on the route: Simulation(poisson_drive=...) reaches the kernel
# with the reference scaling, and the run records the drive in its metadata.
DRIVE = {
    "n_sources": 560, "rate_hz": 4.0, "weight_nS": 2.5, "mechanism": "ampa",
    "target_cell_types": ["E", "PV"], "delay_ms": 2.0, "seed": 7,
}


def _drive_sim(**kwargs):
    drive = dict(DRIVE)
    drive.update(kwargs)
    return _sim(poisson_drive=drive)


def test_route_passes_drive_kwargs_to_the_kernel(kernel_calls):
    model = jtfne.construct(_base())
    model.simulate(_drive_sim())
    assert len(kernel_calls) == 1, "simulate must run the kernel once"
    kw = kernel_calls[0][0]
    np.testing.assert_allclose(kw["ext_lambda"], 560 * 4.0 * (DT_MS / 1000.0), rtol=1e-12)
    assert kw["ext_weight_nS"] == pytest.approx(2.5)
    assert kw["ext_receptor"] == 0, "ampa is the first declared mechanism"
    assert kw["ext_delay_steps"] == 20, "2 ms at dt 0.1 ms"
    labels = tuple(model.params["emitter"].labels)
    want = [i for i, lab in enumerate(labels) if lab in ("E", "PV")]
    assert np.asarray(kw["ext_targets"]).tolist() == want
    assert "key" in kw


def test_route_drive_spikes_and_is_recorded_in_metadata():
    model = jtfne.construct(_base())
    quiet = model.simulate(_sim())
    assert int(np.sum(np.asarray(quiet.spikes))) == 0, "fixture runs quiet with no drive"
    assert "poisson_drive" not in quiet.metadata
    signals = model.simulate(_drive_sim())
    assert int(np.sum(np.asarray(signals.spikes))) > 0, "the drive must spike the fixture"
    recorded = signals.metadata["poisson_drive"]
    assert recorded["status"] == "stochastic_drive_applied"
    assert recorded["ext_lambda"] == pytest.approx(560 * 4.0 * (DT_MS / 1000.0), rel=1e-12)
    for key in DRIVE:
        assert recorded[key] == DRIVE[key]


@pytest.mark.parametrize(
    "key, value",
    [
        ("n_sources", 100), ("rate_hz", 1.0), ("weight_nS", 0.5), ("mechanism", "gaba"),
        ("target_cell_types", ["E"]), ("delay_ms", 1.0), ("seed", 8),
    ],
)
def test_route_drive_field_perturbation_changes_spikes(key, value):
    model = jtfne.construct(_base())
    ref = np.asarray(model.simulate(_drive_sim()).spikes)
    assert ref.sum() > 0, "fixture must spike"
    got = np.asarray(model.simulate(_drive_sim(**{key: value})).spikes)
    assert not np.array_equal(ref, got), f"changing {key} must change the spikes"


DRIVE_SCHEMA_REFUSALS = [
    ({}, "missing"),
    ({"extra": 1}, "unknown keys"),
    ({"n_sources": 0}, "n_sources"),
    ({"rate_hz": 0.0}, "rate_hz"),
    ({"weight_nS": -1.0}, "weight_nS"),
    ({"mechanism": "nmda"}, "mechanism"),
    ({"target_cell_types": ["E", "SST"]}, "target_cell_types"),
    ({"delay_ms": 0.04}, "delay_ms"),
    ({"seed": "x"}, "seed"),
]


@pytest.mark.parametrize("override, needle", DRIVE_SCHEMA_REFUSALS)
def test_route_drive_schema_refusals_name_poisson_drive_and_key(override, needle):
    model = jtfne.construct(_base())
    drive = dict(DRIVE)
    if needle == "missing":
        del drive["seed"]
    else:
        drive.update(override)
    with pytest.raises(ValueError) as info:
        model.simulate(_sim(poisson_drive=drive))
    msg = str(info.value)
    assert "poisson_drive" in msg, msg
    assert needle in msg, msg


def _drive_run(cfg):
    return np.asarray(jtfne.construct(cfg).simulate(_drive_sim()).spikes)


# Packet test 6: with the drive on, every consumed configuration field moves the spikes.
# Each case sets one field to a non-default value through the same builders the tests above
# use; the comparison is on Signals.spikes, never on kernel kwargs.
CONFIG_FIELD_VARIANTS = [
    ("adex_decay", {"decay": "euler"}),
    ("adex_step_order", {"step_order": "brian2"}),
    ("rule weight", {"e_e_weight": 2.0}),
    ("rule delay", {"e_e_delay": 1.0}),
    ("mechanism tau", {"ampa_tau": 4.0}),
    ("mechanism reversal", {"ampa_rev": 20.0}),
    ("rule probability", {"probability": 0.05}),
    ("edge seed", {"edge_seed": 12}),
]


@pytest.mark.parametrize(
    "label, override", CONFIG_FIELD_VARIANTS, ids=[v[0] for v in CONFIG_FIELD_VARIANTS]
)
def test_consumed_configuration_field_changes_driven_spikes(label, override):
    ref = _drive_run(_base())
    assert ref.sum() > 0, "the drive must spike the fixture"
    got = _drive_run(_plumbing_cfg(**override))
    assert not np.array_equal(ref, got), (
        f"{label} is consumed: a non-default value must change the driven spikes"
    )


@pytest.mark.parametrize("key, value", [("amplitude", 0.5), ("target", "all")])
def test_drive_current_schema_keys_refused_by_name(key, value):
    model = jtfne.construct(_base())
    with pytest.raises(ValueError) as info:
        model.simulate(_sim(poisson_drive=dict(DRIVE, **{key: value})))
    msg = str(info.value)
    assert "poisson_drive" in msg, msg
    assert key in msg, msg
