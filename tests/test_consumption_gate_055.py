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
    "field.second_poisson": lambda: _C().field(solver="experimental_poisson_1d", n_bins=8).field(
        solver="experimental_poisson_1d", n_bins=16),
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
}


@pytest.mark.parametrize("case", sorted(UNREALIZED_DECLARATIONS))
def test_unrealized_declarations_are_refused(case):
    with pytest.raises(ValueError, match="no consumer" if case.endswith("unknown_key") else "not realized"):
        UNREALIZED_DECLARATIONS[case]()


@pytest.mark.parametrize("kwargs", [
    {"conductivity": "proxy"}, {"conductivity": 0.0}, {"conductivity": float("nan")}, {"conductivity": True},
    {"n_bins": 1}, {"n_bins": 8.5}, {"n_bins": "8"},
])
def test_poisson_diagnostic_refuses_inputs_simulate_would_swallow(kwargs):
    with pytest.raises(ValueError, match="Poisson diagnostic needs"):
        _C().field(solver="experimental_poisson_1d", **kwargs)


def test_realized_declarations_are_accepted():
    cfg = (
        _C()
        .network(n=6)
        .network(n=6)  # an identical repeat is realized
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .field(solver=None)
        .field(solver="experimental_poisson_1d", conductivity=1.0, n_bins=8)
        .field(solver="experimental_poisson_1d", n_bins=8)  # conductivity defaults to 1.0
        .probe(name="p", n_contacts=4, width=0.1, contact_depths=[0.0, 1 / 3, 2 / 3, 1.0], reference=None)
        .set_emitter("izhikevich", "cortical_eig")  # a repeat of the first emitter is realized
        .drive(noise_policy="additive_poisson")
        .connectivity(within_area="all_to_all_uniform_random", recurrent=True)
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
