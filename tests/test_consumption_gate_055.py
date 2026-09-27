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


def test_reseeding_keeps_a_matching_runtime_seed_consistent(model):
    """run_trials reseeds a Simulation whose runtime pins the same seed."""
    sim = jtfne.Simulation(**{**_BASE, "seed": 5}, runtime=RuntimeConfig(seed=5))
    batch = jtfne.TrialBatch(trials=(jtfne.TrialSpec("a", seed=6), jtfne.TrialSpec("b", seed=7)))
    result = model.run_trials(batch, sim)
    a, b = (r.signals for r in result.results)
    assert _output_differs(a, b)
