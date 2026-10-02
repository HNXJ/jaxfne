"""H1 consumption gate: every public Simulation / RuntimeConfig field either changes
the realized run or is refused (class of P-014, P-015).

Each field has one case: a baseline and a perturbed setting that differ only in
that field, and the expected outcome:

- ``output``: V_m, spikes, sources or field presence differ.
- ``refuse``: the perturbed setting raises ValueError.
- ``label``: provenance label; the output is bit-identical and the label is recorded.
- ``policy``: execution policy whose contract is numerics-invariance; the output
  is bit-identical (a change here would itself be a defect).

A new field without a case fails ``test_every_field_has_a_case``.
"""

from __future__ import annotations

import dataclasses

import jax
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._runtime_config import RuntimeConfig

_BASE = dict(duration_ms=20.0, dt_ms=0.1, seed=0)
_POIS = {"rate_hz": 200, "amplitude": 4, "target": "all", "seed": 11}
_HDP_ON = {**RuntimeConfig().hdp_params, "K_HDP": 0.5, "K_ctrl": 0.2, "alpha": 0.1}
_HOMEO = RuntimeConfig().homeostasis_params


def _rt(**kw):
    return {"runtime": RuntimeConfig(**kw)}


# field -> (baseline Simulation kwargs, perturbed Simulation kwargs, outcome)
SIM_CASES = {
    "duration_ms": ({}, {"duration_ms": 30.0}, "output"),
    "dt_ms": ({}, {"dt_ms": 0.05}, "output"),
    "plasticity": ({}, {"plasticity": 0.7}, "label"),
    "seed": ({}, {"seed": 5}, "output"),
    "record_sources": ({}, {"record_sources": False}, "output"),
    "record_fields": ({}, {"record_fields": False}, "output"),
    "poisson_drive": ({}, {"poisson_drive": _POIS}, "output"),
    "ablation": ({}, {"ablation": "disconnected_null"}, "output"),
    "runtime": ({}, _rt(recurrent_backend="edge_list"), "output"),
}
# A "policy" case names its consumer; a field with no consumer is not policy.
RT_CASES = {
    # consumer: _runtime_config._device_scope via selected_backend
    "backend": ({}, {"backend": "cpu"}, "policy"),
    "dtype": ({}, {"dtype": "bfloat16"}, "output"),
    # consumer: RuntimeConfig.resolve_jit in _model_simulate dispatch
    "jit": ({"jit": False}, {"jit": True}, "policy"),
    # consumer: RuntimeConfig.resolve_vmap in simulate_batch (the only reader)
    "vmap": ({"vmap": False}, {"vmap": True}, "batch_mode"),
    "precision": ({}, {"precision": "high"}, "refuse"),
    "seed": ({}, {"seed": 5}, "refuse"),
    "n_steps": ({}, {"n_steps": 7}, "refuse"),
    "recurrent_backend": ({}, {"recurrent_backend": "edge_list"}, "output"),
    "synaptic_kernel": ({"recurrent_backend": "edge_list"},
                        {"recurrent_backend": "edge_list", "synaptic_kernel": "receptor_exponential"},
                        "output"),
    # consumer: validation.make_recompilation_guard on the jit path
    "recompilation_guard": ({"jit": True}, {"jit": True, "recompilation_guard": "off"}, "policy"),
    "enable_homeostasis": ({}, {"enable_homeostasis": True}, "output"),
    "homeostasis_params": ({"enable_homeostasis": True},
                           {"enable_homeostasis": True, "homeostasis_params": {**_HOMEO, "r_star": 0.5}},
                           "output"),
    "enable_hdp": ({"hdp_params": _HDP_ON}, {"enable_hdp": True, "hdp_params": _HDP_ON}, "output"),
    "hdp_params": ({"enable_hdp": True, "hdp_params": _HDP_ON},
                   {"enable_hdp": True, "hdp_params": {**_HDP_ON, "K_ctrl": 0.0, "alpha": 0.0}},
                   "output"),
    # consumer: RuntimeConfig.selected_backend (compatibility alias of backend)
    "device_type": ({}, {"device_type": "cpu"}, "policy"),
    "dtype_primary": ({}, {"dtype_primary": "bfloat16"}, "output"),
    "x64_enabled": ({}, {"x64_enabled": True}, "refuse"),
}


@pytest.fixture(scope="module")
def model():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12, cell_types={"E": 0.8, "PV": 0.1, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    return jtfne.construct(cfg)


def _run(model, sim_kwargs):
    return model.simulate(jtfne.Simulation(**{**_BASE, **sim_kwargs}))


def _output_differs(a, b) -> bool:
    for key in ("V_m", "spikes", "sources"):
        x, y = getattr(a, key), getattr(b, key)
        if (x is None) != (y is None):
            return True
        if x is not None and (x.shape != y.shape or x.dtype != y.dtype
                              or not np.array_equal(np.asarray(x), np.asarray(y))):
            return True
    return (a.field is None) != (b.field is None)


def _check(model, base_kw, pert_kw, outcome, field):
    if outcome == "refuse":
        with pytest.raises(ValueError):
            _run(model, pert_kw)
        _run(model, base_kw)  # the baseline itself is valid
        return
    base, pert = _run(model, base_kw), _run(model, pert_kw)
    if outcome == "output":
        assert _output_differs(base, pert), f"{field}: accepted but the run did not change"
    else:
        assert not _output_differs(base, pert), f"{field}: {outcome} field changed numerics"
        if outcome == "label":
            assert base.metadata != pert.metadata, f"{field}: label not recorded"


def test_every_field_has_a_case():
    assert {f.name for f in dataclasses.fields(jtfne.Simulation)} == set(SIM_CASES)
    assert {f.name for f in dataclasses.fields(RuntimeConfig)} == set(RT_CASES)


@pytest.mark.parametrize("field", sorted(SIM_CASES))
def test_simulation_field_is_consumed_or_refused(model, field):
    base_kw, pert_kw, outcome = SIM_CASES[field]
    _check(model, base_kw, pert_kw, outcome, field)


@pytest.mark.parametrize("field", sorted(RT_CASES))
def test_runtime_field_is_consumed_or_refused(model, field):
    base_kw, pert_kw, outcome = RT_CASES[field]
    if field == "x64_enabled" and jax.config.read("jax_enable_x64"):
        pytest.skip("x64 already enabled in this process; x64_enabled=True is truthful")
    if outcome == "batch_mode":
        modes = [
            model.simulate_batch(jtfne.Simulation(**_BASE, **_rt(**kw)), n_seeds=2)["metadata"][
                "batch_execution_mode"
            ]
            for kw in (base_kw, pert_kw)
        ]
        assert modes[0] != modes[1], f"{field}: accepted but the batch mode did not change"
        return
    if outcome == "refuse":  # may raise at RuntimeConfig construction or at simulate
        with pytest.raises(ValueError):
            _run(model, _rt(**pert_kw))
        _run(model, _rt(**base_kw))
        return
    _check(model, _rt(**base_kw), _rt(**pert_kw), outcome, field)


@pytest.mark.parametrize(
    "sim_kwargs, match",
    [
        ({"ablation": "no_recurrence"}, "ablation must be"),
        ({"ablation": "shuffled_timing"}, "shuffles a time-varying drive"),
        (_rt(synaptic_kernel="receptor_exponential"), "runs only on"),
    ],
)
def test_unconsumed_values_are_refused(model, sim_kwargs, match):
    with pytest.raises(ValueError, match=match):
        _run(model, sim_kwargs)


@pytest.mark.parametrize(
    "sim_kwargs",
    [{"poisson_drive": _POIS}, {"ablation": "E_silence"},
     {"record_sources": False}],
)
def test_simulate_batch_refuses_fields_it_does_not_consume(model, sim_kwargs):
    with pytest.raises(ValueError, match="simulate_batch does not consume"):
        model.simulate_batch(jtfne.Simulation(**{**_BASE, **sim_kwargs}), n_seeds=2)


# RuntimeConfiguration (tensor path): field -> (perturbed kwargs, outcome, refusal match).
# "policy": jit/device are numerics-invariant.
# The dtype and emitter refusals predate H1; they keep the table complete.
_TR_BASE = dict(duration_ms=20.0, dt_ms=0.5, seed=0)
TENSOR_RT_CASES = {
    "duration_ms": ({"duration_ms": 30.0}, "output", None),
    "dt_ms": ({"dt_ms": 0.25}, "output", None),
    "seed": ({"seed": 3}, "output", None),
    "dtype": ({"dtype": "float16"}, "refuse", "dtype"),
    "emitter": ({"emitter": "lif"}, "refuse", "Unsupported emitter family"),
    "device": ({"device": "cpu"}, "policy", None),
    "jit": ({"jit": True}, "policy", None),
    "vmap": ({"vmap": True}, "refuse", "vmap is not consumed"),
    "n_contacts": ({"n_contacts": 8}, "output", None),
    "solver": ({"solver": "euler"}, "refuse", "reserved"),
    "probes": ({"probes": ["LFP"]}, "refuse", "reserved"),
    "outputs": ({"outputs": {"lfp": True}}, "refuse", "reserved"),
    "optimizer": ({"optimizer": "AGSDR"}, "refuse", "reserved"),
}


def _tensor_model(**kw):
    from jaxfne.neuronal_tensor import make_minimal_ei_tensor

    return jtfne.construct(make_minimal_ei_tensor(), jtfne.RuntimeConfiguration(**{**_TR_BASE, **kw}))


def _tensor_run(**kw):
    return jtfne.simulate(_tensor_model(**kw))


def _tensor_output(sig):
    lfp = sig.get("lfp_proxy")
    return [np.asarray(sig.V_m), np.asarray(sig.spikes), None if lfp is None else np.asarray(lfp)]


def test_every_tensor_runtime_field_has_a_case():
    assert {f.name for f in dataclasses.fields(jtfne.RuntimeConfiguration)} == set(TENSOR_RT_CASES)


@pytest.mark.parametrize("field", sorted(TENSOR_RT_CASES))
def test_tensor_runtime_field_is_consumed_or_refused(field):
    pert_kw, outcome, match = TENSOR_RT_CASES[field]
    if outcome == "refuse":
        with pytest.raises(ValueError, match=match):
            _tensor_run(**pert_kw)
        return
    base, pert = _tensor_output(_tensor_run()), _tensor_output(_tensor_run(**pert_kw))
    same = all(
        (x is None and y is None)
        or (x is not None and y is not None and x.shape == y.shape and np.array_equal(x, y))
        for x, y in zip(base, pert, strict=True)
    )
    assert same == (outcome == "policy"), f"{field}: outcome {outcome} not realized"


@pytest.mark.parametrize("n", [1, 16.0, "16", True])
def test_tensor_runtime_n_contacts_refuses_non_int(n):
    from jaxfne.neuronal_tensor import make_minimal_ei_tensor, neuronal_tensor_to_configuration

    with pytest.raises(ValueError, match="n_contacts"):
        jtfne.RuntimeConfiguration(n_contacts=n)
    with pytest.raises(ValueError, match="n_contacts"):
        neuronal_tensor_to_configuration(make_minimal_ei_tensor(), n_contacts=n)


def test_probes_refuse_a_non_int_or_disagreeing_n_contacts():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=12)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
    )
    with pytest.raises(ValueError, match="n_contacts"):
        jtfne.construct(cfg.probe(name="a", modes=["LFP"], n_contacts="x"))
    with pytest.raises(ValueError, match="n_contacts"):
        jtfne.construct(cfg.probe(name="a", modes=["LFP"], n_contacts=8).probe(name="b", modes=["CSD"], n_contacts=4))


def test_set_emitter_refuses_rule_kwargs_its_family_drops():
    with pytest.raises(ValueError, match="homeostatic_ei"):
        jtfne.configuration().set_emitter("izhikevich", activation_rule="linear")
    jtfne.configuration().set_emitter("homeostatic_ei", activation_rule="linear")


@pytest.mark.parametrize("selector, params", [({"area": "V1"}, {"a": 0.02}), ({"cell_type": "E"}, {"tau_ms": 5.0})])
def test_cell_params_refuses_keys_the_applier_ignores(selector, params):
    with pytest.raises(ValueError, match="cell_params"):
        jtfne.configuration().cell_params(selector, params)


# Declarative Configuration keys accept only the value the implementation realizes
# (human decision 2026-09-27); each case is one unrealized value.
_C = jtfne.configuration
UNREALIZED_DECLARATIONS = {
    "field.domain": lambda: _C().field(domain="point"),
    "field.conductivity": lambda: _C().field(conductivity="anisotropic"),
    "field.boundary": lambda: _C().field(boundary="dirichlet"),
    "field.gauge": lambda: _C().field(gauge="reference_electrode"),
    "field.unknown_key": lambda: _C().field(resolution_um=10),
    "field.solver": lambda: _C().field(solver="fem"),
    "probe.width": lambda: _C().probe(name="p", width=0.3),
    "probe.contact_depths": lambda: _C().probe(name="p", n_contacts=4, contact_depths=[0.0, 0.2, 0.4, 0.6]),
    "probe.reference": lambda: _C().probe(name="p", reference="common_average"),
    "probe.filter_spec": lambda: _C().probe(name="p", filter_spec={"kind": "bandpass"}),
    "probe.unknown_key": lambda: _C().probe(name="p", impedance_kohm=500),
    "drive.noise_policy": lambda: _C().drive(noise_policy="none"),
    "inter_column.mode": lambda: _C().inter_column_connectivity(mode="all_to_all"),
    "inter_column.sign_policy": lambda: _C().inter_column_connectivity(sign_policy="declared"),
    "inter_column.delay": lambda: _C().inter_column_connectivity(delay_ms_or_status=2.0),
    "inter_column.cell_type_map": lambda: _C().inter_column_connectivity(cell_type_to_cell_type_map={"E": "PV"}),
    "connections.plasticity": lambda: _C().connections(name="c", source={}, target={}, plasticity={"rule": "stdp"}),
    "connections.control_key": lambda: _C().connections(name="c", source={}, target={}, control_key="k"),
    "connectivity.within_area": lambda: _C().connectivity(within_area="small_world"),
    "connectivity.recurrent": lambda: _C().connectivity(recurrent=False),
    "emitter.preset": lambda: _C().emitter(family="izhikevich", preset="regular_spiking"),
    "emitter.unknown_key": lambda: _C().emitter(kind="izhikevich"),
    # construct builds emitters[0], networks[0] and the first Poisson field only.
    "emitter.after_default": lambda: _C().network(n=6).probes(["spikes"]).set_emitter("homeostatic_ei"),
    "emitter.conflict_at_construct": lambda: jtfne.construct(dataclasses.replace(
        _C().network(n=6).field().probe(name="p"), emitters=[{"family": "izhikevich"}, {"family": "homeostatic_ei"}])),
    "network.second": lambda: _C().network(n=6).network(n=12),
    "network.after_cell_types": lambda: _C().cell_types({"E": 0.8, "PV": 0.2}).network(n=6),
    "column.after_network": lambda: _C().network(n=6).column("V1", layers=["L4"], n=6),
    "field.second_poisson": lambda: jtfne.construct(
        _C().network(n=6).emitter(family="izhikevich").field(solver="experimental_poisson_1d", n_bins=8)
        .field(solver="experimental_poisson_1d").probe(name="p")),  # n_bins defaults to n_contacts (16)
    "field.direct_at_construct": lambda: jtfne.construct(dataclasses.replace(
        _C().network(n=6).emitter(family="izhikevich").field().probe(name="p"), fields=[{"domain": "point"}])),
    "probe.direct_at_construct": lambda: jtfne.construct(dataclasses.replace(
        _C().network(n=6).emitter(family="izhikevich").field().probe(name="p"),
        probes=[{"name": "p", "reference": "common_average"}])),
    "connectivity.p_connect_range": lambda: jtfne.construct(
        _C().network(n=6).uniform3d().connectivity(p_connect=1.5).emitter(family="izhikevich").field()
        .probe(name="p")),
    "network.layers": lambda: jtfne.construct(
        _C().network(n=6, layers=["L4"]).emitter(family="izhikevich").field().probe(name="p")),
    "runtime.unknown_key": lambda: _C().runtime(noise_scale=0.0),
    "runtime.vmap": lambda: jtfne.construct(
        _C().network(n=6).emitter(family="izhikevich").field().probe(name="p").runtime(vmap=True)),
    "areas.not_the_columns": lambda: jtfne.construct(
        _C().areas(["V1", "V4"]).column("V1", layers=["L4"], n=6).emitter(family="izhikevich").field()
        .probe(name="p")),
    "plasticity.relative_baseline": lambda: _C().plasticity(relative_baseline=0.5),
    "plasticity.unknown_key": lambda: _C().plasticity(rule="stdp"),
    "plasticity.bool_baseline": lambda: _C().plasticity(relative_baseline=True),
    "homeostasis.bool_baseline": lambda: _C().homeostasis(relative_baseline=True),
    "hdp.bool_baseline": lambda: _C().hdp(relative_baseline=True),
    "homeostasis.nan_baseline": lambda: _C().homeostasis(relative_baseline=float("nan")),
    "hdp.inf_baseline": lambda: _C().hdp(relative_baseline=float("inf")),
    "network.unknown_key": lambda: _C().network(n=2, connectivity={"E→PV": 0.2}),
    "network.built_kind": lambda: _C().network(n=9999, kind="multi_column").column("V1", layers=["L4"], n=6),
    "network.second_at_construct": lambda: jtfne.construct(dataclasses.replace(
        _C().network(n=6).emitter(family="izhikevich").field().probe(name="p"), networks=[{"n": 6}, {"n": 12}])),
    "network.p_connect_population_route": lambda: jtfne.construct(
        _C().network(n=6, p_connect=0.5).uniform3d().emitter(family="izhikevich").field().probe(name="p")),
    "network.layers_population_route": lambda: jtfne.construct(dataclasses.replace(
        _C().network(n=6).uniform3d().emitter(family="izhikevich").field().probe(name="p"),
        networks=[{"n": 6, "layers": ["L4"]}])),
    "emitter.rule_key_for_izhikevich": lambda: _C().emitter(family="izhikevich", homeostatic_ei_bound_mode="stable"),
    "cell_params.layer_on_plain_route": lambda: jtfne.construct(
        _C().network(n=4).cell_params({"layer": "L4"}, {"a": 0.05}).emitter(family="izhikevich").field()
        .probe(name="p")),
}


@pytest.mark.parametrize("case", sorted(UNREALIZED_DECLARATIONS))
def test_unrealized_declarations_are_refused(case):
    with pytest.raises(ValueError, match="no consumer" if case.endswith("unknown_key") else "not realized"):
        UNREALIZED_DECLARATIONS[case]()


# Homeostatic_ei route: _construct_homeostatic_ei_model reads only
# networks[0]["n"], the emitter's homeostatic_ei_rules/homeostatic_ei_bound_mode,
# metadata["dtype"] and the fields/probes _construct_build_static consumes.
# Every other declaration is dropped there, so a non-neutral value is refused
# at construct; each case builds the minimal homeostatic_ei config plus one
# declaration and constructs it.
def _hei_base():
    return _C().network(n=4).emitter(family="homeostatic_ei").field().probe(name="p")


HOMEOSTATIC_EI_UNREALIZED = {
    "hei.network.cell_types": lambda: jtfne.construct(
        _C().network(n=4, cell_types={"E": 0.8, "I": 0.2}).emitter(family="homeostatic_ei").field()
        .probe(name="p")),
    "hei.network.p_connect": lambda: jtfne.construct(
        _C().network(n=4, p_connect=0.5).emitter(family="homeostatic_ei").field().probe(name="p")),
    "hei.network.p_connect_range": lambda: jtfne.construct(
        _C().network(n=4, p_connect=1.5).emitter(family="homeostatic_ei").field().probe(name="p")),
    "hei.network.layers": lambda: jtfne.construct(dataclasses.replace(
        _hei_base(), networks=[{"n": 4, "layers": ["L4"]}])),
    "hei.cell_types": lambda: jtfne.construct(
        _hei_base().cell_types({"E": 0.5, "I": 0.5})),
    "hei.column": lambda: jtfne.construct(
        _C().column("V1", layers=["L4"], n=4).emitter(family="homeostatic_ei").field().probe(name="p")),
    "hei.areas": lambda: jtfne.construct(
        _C().column("V1", layers=["L4"], n=4).areas(["V1"]).emitter(family="homeostatic_ei").field()
        .probe(name="p")),
    "hei.uniform3d": lambda: jtfne.construct(_hei_base().uniform3d()),
    "hei.layer_fractions": lambda: jtfne.construct(_hei_base().layer_fractions()),
    "hei.area_layer_cell_types": lambda: jtfne.construct(
        _hei_base().area_layer_cell_types("V1", {"L4": {"E": 1.0}})),
    "hei.connectivity.p_connect": lambda: jtfne.construct(_hei_base().connectivity(p_connect=0.5)),
    "hei.connectivity.p_connect_range": lambda: jtfne.construct(
        _hei_base().connectivity(p_connect=1.5)),
    "hei.connectivity.within_gain": lambda: jtfne.construct(
        _hei_base().connectivity(within_gain=0.9)),
    "hei.connectivity.feedforward_gain": lambda: jtfne.construct(
        _hei_base().connectivity(feedforward_gain=1.0)),
    "hei.connectivity.tcm": lambda: jtfne.construct(_hei_base().connectivity(tcm_v1_6pop=True)),
    "hei.suite2_interarea": lambda: jtfne.construct(_hei_base().suite2_interarea()),
    "hei.inter_column": lambda: jtfne.construct(_hei_base().inter_column_connectivity()),
    "hei.drive": lambda: jtfne.construct(_hei_base().drive()),
    "hei.drive.baseline": lambda: jtfne.construct(
        _hei_base().drive(baseline_drive_by_cell_type={"E": 1.0})),
    "hei.connections": lambda: jtfne.construct(
        _hei_base().connections(name="c", source={}, target={})),
    "hei.cell_params": lambda: jtfne.construct(
        _hei_base().cell_params({"cell_type": "E"}, {"drive": 1.0})),
    "hei.emitter.preset": lambda: jtfne.construct(
        _C().network(n=4).emitter(family="homeostatic_ei", preset="cortical_eig").field()
        .probe(name="p")),
    "hei.runtime.canonical_biophysics": lambda: jtfne.construct(
        _hei_base().runtime(canonical_biophysics=True)),
    "hei.runtime.random_v0": lambda: jtfne.construct(_hei_base().runtime(random_v0=True)),
    "hei.runtime.recurrent_backend": lambda: jtfne.construct(
        _hei_base().runtime(recurrent_backend="edge_list")),
    "hei.runtime.synaptic_kernel": lambda: jtfne.construct(
        _hei_base().runtime(synaptic_kernel="receptor_exponential")),
    "hei.runtime.enable_hdp": lambda: jtfne.construct(_hei_base().runtime(enable_hdp=True)),
    "hei.runtime.hdp_params": lambda: jtfne.construct(_hei_base().runtime(hdp_params={"K_HDP": 0.5})),
    "hei.runtime.enable_homeostasis": lambda: jtfne.construct(
        _hei_base().runtime(enable_homeostasis=True)),
    "hei.runtime.homeostasis_params": lambda: jtfne.construct(
        _hei_base().runtime(homeostasis_params={"k_gain": 0.1})),
    "hei.homeostasis": lambda: jtfne.construct(_hei_base().homeostasis(relative_baseline=1.5)),
    "hei.hdp": lambda: jtfne.construct(_hei_base().hdp(relative_baseline=1.5)),
    "hei.field.solver": lambda: jtfne.construct(
        _C().network(n=4).emitter(family="homeostatic_ei").field(solver="experimental_poisson_1d")
        .probe(name="p")),
    "hei.geometry": lambda: jtfne.construct(_hei_base(), geometry=object()),
}


@pytest.mark.parametrize("case", sorted(HOMEOSTATIC_EI_UNREALIZED))
def test_homeostatic_ei_dropped_declarations_are_refused(case):
    with pytest.raises(ValueError, match="not realized"):
        HOMEOSTATIC_EI_UNREALIZED[case]()


def _hei_emitter_params_equal(a, b):
    assert type(a) is type(b)
    for f in dataclasses.fields(a):
        assert np.array_equal(np.asarray(getattr(a, f.name)), np.asarray(getattr(b, f.name))), f.name


def test_homeostatic_ei_route_accepts_neutral_values():
    cfg = (
        _C().network(n=4, name="ei", kind="cortical_column", p_connect=1.0)
        .emitter(family="homeostatic_ei")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann",
               gauge="mean_zero")
        .probe(name="p")
        .connectivity(within_area="all_to_all_uniform_random", recurrent=True, p_connect=1.0,
                      feedforward=("V1", "V4"), feedback=("V4", "V1"), mode="sparse")
        .runtime(seed=0, duration_ms=20.0, dt_ms=0.5, random_v0=False,
                 enable_hdp=False, enable_homeostasis=False, jit=False)
        .plasticity().homeostasis().hdp()
    )
    model = jtfne.construct(cfg)
    assert model.static["n_contacts"] == 16
    # Neutral declarations change nothing: the emitter equals the bare base
    # built with the same runtime.
    ref = jtfne.construct(_hei_base().runtime(
        seed=0, duration_ms=20.0, dt_ms=0.5, random_v0=False,
        enable_hdp=False, enable_homeostasis=False, jit=False))
    _hei_emitter_params_equal(model.params["emitter"], ref.params["emitter"])


@pytest.mark.parametrize("kwargs", [
    {"conductivity": "proxy"}, {"conductivity": 0.0}, {"conductivity": float("nan")}, {"conductivity": True},
    {"n_bins": 1}, {"n_bins": 8.5}, {"n_bins": "8"},
])
def test_poisson_diagnostic_refuses_inputs_simulate_would_swallow(kwargs):
    with pytest.raises(ValueError, match="Poisson diagnostic needs"):
        _C().field(solver="experimental_poisson_1d", **kwargs)


def test_plain_route_checks_both_p_connect_spellings():
    cfg = (_C().network(n=6, p_connect=0.5).connectivity(p_connect=1.0).emitter(family="izhikevich").field()
           .probe(name="p"))
    with pytest.raises(ValueError, match=r"network\(p_connect=0.5\) cannot be honoured"):
        jtfne.construct(cfg)


def test_realized_declarations_are_accepted():
    cfg = (
        _C()
        .network(n=6)
        .network(n=6)  # an identical repeat is realized
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .field(solver=None)
        .field(solver="experimental_poisson_1d", conductivity=1.0, n_bins=4)
        .field(solver="experimental_poisson_1d", n_bins=4)  # conductivity defaults to 1.0
        .field(solver="experimental_poisson_1d")  # n_bins defaults to n_contacts (4)
        .probe(name="p", n_contacts=4, width=0.1, contact_depths=[0.0, 1 / 3, 2 / 3, 1.0], reference=None)
        .set_emitter("izhikevich", "cortical_eig")  # a repeat of the first emitter is realized
        .drive(noise_policy="additive_poisson")
        .connectivity(within_area="all_to_all_uniform_random", recurrent=True, p_connect=1.0)
        .runtime(vmap=False)
        .plasticity()
    )
    assert jtfne.construct(cfg).static["n_contacts"] == 4
    areas = (_C().areas(["V2", "V1"]).column("V1", layers=["L4"], n=4).column("V2", layers=["L4"], n=4)
             .emitter(family="izhikevich").field().probe(name="p", n_contacts=4))
    assert jtfne.construct(areas).static["n_contacts"] == 4


def test_reseeding_keeps_a_matching_runtime_seed_consistent(model):
    """run_trials reseeds a Simulation whose runtime pins the same seed."""
    sim = jtfne.Simulation(**{**_BASE, "seed": 5}, runtime=RuntimeConfig(seed=5))
    batch = jtfne.TrialBatch(trials=(jtfne.TrialSpec("a", seed=6), jtfne.TrialSpec("b", seed=7)))
    result = model.run_trials(batch, sim)
    a, b = (r.signals for r in result.results)
    assert _output_differs(a, b)


def test_plain_route_normalizes_fractions_above_unit_mass():
    cfg = (
        _C()
        .network(n=2, cell_types={"E": 1, "PV": 1})
        .emitter(family="izhikevich")
        .field()
        .probe(name="p")
    )
    assert tuple(jtfne.construct(cfg).params["emitter"].labels) == ("E", "PV")


def test_plain_route_normalizes_fractions_below_unit_mass():
    cfg = (
        _C()
        .network(n=10, cell_types={"E": 0.4, "PV": 0.4})
        .emitter(family="izhikevich")
        .field()
        .probe(name="p")
    )
    labels = tuple(jtfne.construct(cfg).params["emitter"].labels)
    assert sorted(labels) == ["E"] * 5 + ["PV"] * 5


@pytest.mark.parametrize("uniform3d", [False, True], ids=["plain", "population"])
@pytest.mark.parametrize(
    "cell_types",
    [{"E": 1.0, "PV": -0.1}, {"E": float("nan")}, {"E": 0.0, "PV": 0.0}, {"E": True}, {"E": None}, {"E": "0.5"}],
)
def test_construct_refuses_invalid_fractions(cell_types, uniform3d):
    cfg = _C().network(n=6, cell_types=cell_types)
    if uniform3d:
        cfg = cfg.uniform3d()
    cfg = cfg.emitter(family="izhikevich").field().probe(name="p")
    with pytest.raises(ValueError, match="cell type fraction"):
        jtfne.construct(cfg)


@pytest.mark.parametrize("uniform3d", [False, True], ids=["plain", "population"])
def test_drive_and_cell_params_reach_the_emitter(uniform3d):
    cfg = _C().network(n=4, cell_types={"E": 0.5, "PV": 0.5})
    if uniform3d:
        cfg = cfg.uniform3d()
    cfg = (cfg.drive(baseline_drive_by_cell_type={"E": 7.0})
           .cell_params({"cell_type": "PV"}, {"a": 0.05, "drive": 2.5})
           .emitter(family="izhikevich").field().probe(name="p"))
    p = jtfne.construct(cfg).params["emitter"]
    got = {label: (float(a), float(d)) for label, a, d in zip(p.labels, p.a, p.drive)}
    assert got["E"] == pytest.approx((0.02, 7.0))
    assert got["PV"] == pytest.approx((0.05, 2.5))


def test_make_eig_network_refuses_a_bool_fraction():
    from jaxfne.emitters import make_eig_network

    with pytest.raises(ValueError, match="must be finite and non-negative"):
        make_eig_network(n=2, cell_type_fractions={"E": True})


_ABSENT = object()


def _edge_seed_column_cfg(*, seed, edge_seed=_ABSENT, p_connect=None):
    """Population-route column: edge_seed rides in .connectivity()."""
    cfg = (
        _C().column("V1", layers=["L4"], n=12)
        .emitter(family="izhikevich").field().probe(name="p")
        .runtime(seed=seed)
    )
    conn = {} if p_connect is None else {"p_connect": p_connect}
    if edge_seed is not _ABSENT:
        conn["edge_seed"] = edge_seed
    return cfg.connectivity(**conn)


@pytest.mark.parametrize("p_connect", [None, 0.5], ids=["dense", "sparse"])
def test_edge_seed_reseeds_population_route_edges_not_positions(p_connect):
    base = jtfne.construct(_edge_seed_column_cfg(seed=7, p_connect=p_connect))
    reseeded = jtfne.construct(_edge_seed_column_cfg(seed=7, edge_seed=8, p_connect=p_connect))
    same = jtfne.construct(_edge_seed_column_cfg(seed=7, edge_seed=7, p_connect=p_connect))
    w_base = np.asarray(base.params["emitter"].W)
    w_reseeded = np.asarray(reseeded.params["emitter"].W)
    w_same = np.asarray(same.params["emitter"].W)
    assert w_base.shape == (12, 12) and w_reseeded.shape == (12, 12)
    assert not np.array_equal(w_base, w_reseeded), "edge_seed=8 left the realized weights unchanged"
    assert np.array_equal(w_base, w_same), "edge_seed equal to the runtime seed changed the weights"
    for tag, model in (("reseeded", reseeded), ("same", same)):
        assert np.array_equal(
            np.asarray(base.params["positions"]), np.asarray(model.params["positions"])
        ), f"edge_seed moved positions ({tag})"


@pytest.mark.parametrize("edge_seed", [1.5, "8", True, False])
def test_connectivity_refuses_a_non_int_edge_seed(edge_seed):
    with pytest.raises(ValueError, match="edge_seed"):
        _C().connectivity(edge_seed=edge_seed)


def test_edge_seed_refused_on_the_plain_route():
    cfg = (
        _C().network(n=6).emitter(family="izhikevich").field().probe(name="p")
        .connectivity(edge_seed=8)
    )
    with pytest.raises(ValueError, match="is not realized"):
        jtfne.construct(cfg)


def test_edge_seed_reseeds_connection_rules_on_the_plain_route():
    def edges(**conn):
        cfg = (_C().runtime(seed=7).network(n=12).emitter(family="izhikevich").field().probe(name="p")
               .connections(name="ee", source={"cell_type": "E"}, target={"cell_type": "E"}, probability=0.5)
               .connectivity(**conn))
        e = jtfne.construct(cfg).params["edge_list"]
        return np.asarray(e.pre), np.asarray(e.post)

    base, same, reseeded = edges(), edges(edge_seed=7), edges(edge_seed=8)
    assert all(np.array_equal(a, b) for a, b in zip(base, same))
    assert not all(np.array_equal(a, b) for a, b in zip(base, reseeded))


def test_edge_seed_refused_on_the_homeostatic_ei_route():
    cfg = (
        _C().network(n=6).set_emitter("homeostatic_ei").field().probe(name="p")
        .connectivity(edge_seed=8)
    )
    with pytest.raises(ValueError, match="is not realized"):
        jtfne.construct(cfg)


# Review fixes (agent decisions 2026-09-27/28): a public parameter either changes
# the realized object or is refused. One test per fixed behaviour below.


def test_connectivity_refuses_keys_nothing_reads():
    with pytest.raises(ValueError, match="not realized"):
        _C().connectivity(foo=1.0)
    with pytest.raises(ValueError, match="connectivity_mode is top-level metadata"):
        _C().connectivity(connectivity_mode="explicit")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "5"])
def test_drive_refuses_a_non_finite_real_baseline_drive(value):
    with pytest.raises(ValueError, match="not realized"):
        _C().drive(baseline_drive_by_cell_type={"E": value})


def _epv_base(*, population_route):
    cfg = _C().network(n=4, cell_types={"E": 0.5, "PV": 0.5})
    if population_route:
        cfg = cfg.uniform3d()
    return cfg


@pytest.mark.parametrize("population_route", [False, True], ids=["plain", "population"])
@pytest.mark.parametrize("entry", [{"EX": 7.0}, {"SST": 9.0}])
def test_drive_refuses_entries_naming_no_neuron(population_route, entry):
    cfg = (_epv_base(population_route=population_route)
           .drive(baseline_drive_by_cell_type=entry)
           .emitter(family="izhikevich").field().probe(name="p"))
    with pytest.raises(ValueError, match="no neuron"):
        jtfne.construct(cfg)


@pytest.mark.parametrize("population_route", [False, True], ids=["plain", "population"])
def test_default_drive_matches_no_drive_call(population_route):
    plain = _epv_base(population_route=population_route).emitter(
        family="izhikevich").field().probe(name="p")
    # The drive() defaults name all four types; SST/VIP name no neuron of the
    # E/PV population but hold their emitter defaults, so the call is accepted.
    defaulted = _epv_base(population_route=population_route).drive().emitter(
        family="izhikevich").field().probe(name="p")
    p_plain = jtfne.construct(plain).params["emitter"]
    p_default = jtfne.construct(defaulted).params["emitter"]
    assert np.array_equal(np.asarray(p_plain.drive), np.asarray(p_default.drive))


def test_column_drive_refuses_only_what_nothing_declares():
    # column("V1", layers=["L4"], n=3) builds (E, PV, VIP). EX is declared
    # nowhere, so it is refused ...
    base = _C().column("V1", layers=["L4"], n=3)
    declined = jtfne.construct(base.emitter(family="izhikevich").field().probe(name="p"))
    assert "SST" not in set(map(str, declined.params["emitter"].labels))
    with pytest.raises(ValueError, match="no neuron"):
        jtfne.construct(base.drive(baseline_drive_by_cell_type={"EX": 7.0}).emitter(
            family="izhikevich").field().probe(name="p"))
    # ... while SST is in column()'s default-declared composition
    # (networks[0]["cell_types"] = E/PV/SST) but rounds away at n=3, so a
    # non-default SST entry is accepted and applies to no neuron.
    model = jtfne.construct(base.drive(baseline_drive_by_cell_type={"SST": 9.0}).emitter(
        family="izhikevich").field().probe(name="p"))
    assert np.array_equal(np.asarray(model.params["emitter"].drive),
                          np.asarray(declined.params["emitter"].drive))


def test_suite2_preset_accepts_a_declared_but_rounded_away_drive():
    # suite2_net1_config(n=6) declares SST (cell_types E/PV/SST/VIP) but builds
    # (E, PV, VIP): the SST entry is a count artifact, not a dropped
    # declaration, so it is accepted ...
    cfg = jtfne.suite2_net1_config(seed=7, n=6, duration_ms=20.0, dt_ms=1.0,
                                    drives={"E": 4.0, "PV": 2.0, "SST": 9.0, "VIP": 2.0})
    model = jtfne.construct(cfg)
    assert "SST" not in set(map(str, model.params["emitter"].labels))
    # ... while a type nothing declares is refused on the same population.
    with pytest.raises(ValueError, match="no neuron"):
        jtfne.construct(jtfne.suite2_net1_config(
            seed=7, n=6, duration_ms=20.0, dt_ms=1.0, drives={"EX": 7.0}))


def test_column_default_drive_matches_no_drive_call():
    base = _C().column("V1", layers=["L4"], n=3)
    p_plain = jtfne.construct(base.emitter(family="izhikevich").field().probe(name="p")).params["emitter"]
    p_default = jtfne.construct(
        base.drive().emitter(family="izhikevich").field().probe(name="p")).params["emitter"]
    assert np.array_equal(np.asarray(p_plain.drive), np.asarray(p_default.drive))


@pytest.mark.parametrize("population_route", [False, True], ids=["plain", "population"])
def test_cell_params_refuses_a_selector_matching_no_neuron(population_route):
    cfg = (_epv_base(population_route=population_route)
           .cell_params({"cell_type": "SST"}, {"a": 0.05})
           .emitter(family="izhikevich").field().probe(name="p"))
    with pytest.raises(ValueError, match="matches no neuron"):
        jtfne.construct(cfg)


def test_cell_params_refuses_a_layer_selector_on_an_unlayered_population():
    # The uniform3d population labels every layer "uniform_3d", so L4 matches nothing.
    cfg = (_C().network(n=4).uniform3d().cell_params({"layer": "L4"}, {"a": 0.05})
           .emitter(family="izhikevich").field().probe(name="p"))
    with pytest.raises(ValueError, match="matches no neuron"):
        jtfne.construct(cfg)


def test_cell_params_accepts_a_declared_but_rounded_away_selector():
    # suite2_net1_config(n=6) declares SST but builds no SST neuron: the
    # selector is accepted and applies to no neuron ...
    base = jtfne.suite2_net1_config(seed=7, n=6, duration_ms=20.0, dt_ms=1.0)
    jtfne.construct(base.cell_params({"cell_type": "SST"}, {"a": 0.05}))
    # ... while a type nothing declares is refused on the same population.
    with pytest.raises(ValueError, match="matches no neuron"):
        jtfne.construct(base.cell_params({"cell_type": "EX"}, {"a": 0.05}))


def test_plain_route_canonical_biophysics_needs_no_layers():
    # The plain route builds one unlayered population (geometry_meta None);
    # canonical_biophysics must not crash there. Deep-E grading is laminar
    # only, so E keeps its default a.
    model = jtfne.construct(
        _C().network(n=4).emitter(family="izhikevich").field().probe(name="p")
        .runtime(seed=0, canonical_biophysics=True))
    p = model.params["emitter"]
    lab = np.asarray([str(x) for x in p.labels])
    assert (lab == "E").any()
    assert np.allclose(np.asarray(p.a)[lab == "E"], 0.02)


def test_construct_refuses_a_surgically_set_non_real_drive():
    # A drive value that bypassed drive() validation (metadata surgery) must
    # fail as ValueError at construct, not TypeError.
    cfg = (_C().network(n=4, cell_types={"E": 0.5, "PV": 0.5}).emitter(family="izhikevich")
           .field().probe(name="p"))
    metadata = dict(cfg.metadata)
    metadata["drive"] = {"baseline_drive_by_cell_type": {"SST": None}}
    with pytest.raises(ValueError, match="finite real"):
        jtfne.construct(dataclasses.replace(cfg, metadata=metadata))


def test_canonical_biophysics_cell_params_override_deep_e_grading():
    base = (jtfne.build_laminar_column(ei_profile="canonical", n=60)
            .emitter(family="izhikevich").field().probe(name="p")
            .runtime(seed=0, canonical_biophysics=True))
    graded = jtfne.construct(base).params["emitter"]
    lab = np.asarray([str(x) for x in graded.labels])
    assert (lab == "E").any()
    # Without cell_params the deep-E grading spreads a and d with depth.
    assert np.unique(np.asarray(graded.a)[lab == "E"]).size > 1
    assert np.unique(np.asarray(graded.d)[lab == "E"]).size > 1
    over = jtfne.construct(base.cell_params({"cell_type": "E"}, {"a": 0.09, "d": 12.0})).params["emitter"]
    lab_over = np.asarray([str(x) for x in over.labels])
    assert np.allclose(np.asarray(over.a)[lab_over == "E"], 0.09)
    assert np.allclose(np.asarray(over.d)[lab_over == "E"], 12.0)


def test_metadata_edge_seed_bypassing_connectivity_is_refused():
    cfg = (_C().column("V1", layers=["L4"], n=12).emitter(family="izhikevich").field()
           .probe(name="p").runtime(seed=7))
    metadata = dict(cfg.metadata)
    metadata["connectivity"] = {**metadata.get("connectivity", {}), "edge_seed": 1.5}
    with pytest.raises(ValueError, match="edge_seed"):
        jtfne.construct(dataclasses.replace(cfg, metadata=metadata))


def _uniform_base(layers):
    return (_C().column("V1", layers=layers, n=6).emitter(family="izhikevich").field()
            .probe(name="p").runtime(seed=0))


_UNIFORM_DROPS = {
    "uniform3d.column_layers": (["L2/3", "L4"], lambda c: c.uniform3d()),
    "uniform3d.layer_fractions": (["uniform_3d"], lambda c: c.uniform3d().layer_fractions()),
    "uniform_column.layer_fractions": (["uniform_3d"], lambda c: c.layer_fractions()),
    "uniform_column.area_table": (["uniform_3d"], lambda c: c.area_layer_cell_types("V1", {"L4": {"E": 1.0}})),
    "area_table.unknown_area": (["L4"], lambda c: c.area_layer_cell_types("V9", {"L4": {"E": 1.0}})),
}


@pytest.mark.parametrize("case", sorted(_UNIFORM_DROPS))
def test_construct_refuses_layer_declarations_the_route_drops(case):
    layers, declare = _UNIFORM_DROPS[case]
    jtfne.construct(_uniform_base(layers))  # the base builds
    with pytest.raises(ValueError, match="not realized"):
        jtfne.construct(declare(_uniform_base(layers)))


def test_default_builders_declare_only_what_they_build():
    with pytest.raises(ValueError, match="not realized"):
        jtfne.default_cortical_column_config(layers=["L4"])
    with pytest.raises(ValueError, match="uniform3d"):
        jtfne.build_laminar_column(geometry="uniform3d", layers=["L4"])
    meta = jtfne.default_cortical_column_config(n=20).metadata
    assert meta["columns"][0]["layers"] == ["uniform_3d"] and "layer_fractions" not in meta
    # default_complete_configuration is laminar, so its L2/3 -> core map builds edges.
    model = jtfne.construct(jtfne.default_complete_configuration(n_column=40, n_nucleus=20))
    areas = np.asarray([row["area"] for row in model.neuron_table()])
    edges = model.params["edge_list"]
    pre, post = np.asarray(edges.pre), np.asarray(edges.post)
    assert (areas[pre] != areas[post]).sum() > 0


def test_homeostatic_ei_route_accepts_unspecified_connectivity_mode():
    assert jtfne.construct(
        _hei_base().update_metadata(connectivity_mode="unspecified")).static["n_contacts"] == 16
    with pytest.raises(ValueError, match="not realized"):
        jtfne.construct(_hei_base().update_metadata(connectivity_mode="explicit"))
