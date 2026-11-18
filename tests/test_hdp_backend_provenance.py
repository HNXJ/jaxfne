"""P-062 A, step 1: runs whose controller forces the edge-list kernel (an engaged HDP rule,
homeostasis) record ``edge_list`` as the executed backend, keep the requested value in the
runtime report, and refuse an explicit model-level ``recurrent_backend="dense"`` instead of
replacing it. ``simulate()`` and ``simulate_batch()`` alike.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne._model import Model

DURATION_MS = 20.0
DT_MS = 0.5
RULE = dict(enable_hdp=True, hdp_params={"hdp_rule": "synthetic_presyn_gain"})
HOMEOSTASIS = dict(enable_homeostasis=True)


def _model():
    layer = nt.Layer(
        name="L4",
        n_neurons=40,
        neuron_types=[nt.NeuronType.make("E", fraction=0.9), nt.NeuronType.make("PV", fraction=0.1)],
    )
    tensor = nt.NeuronalTensor(areas=[nt.Area(name="V1", layers=[layer])], name="p062_provenance")
    return jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=DURATION_MS, dt_ms=DT_MS))


def _explicit_dense(model):
    return Model(
        cfg=replace(model.cfg, metadata={**model.cfg.metadata, "recurrent_backend": "dense"}),
        params=model.params,
        static=model.static,
    )


def _run(model, **runtime):
    return jtfne.simulate(model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=0, runtime=jtfne.RuntimeConfig(**runtime))


def _batch(model, **runtime):
    sim = jtfne.simulation(duration_ms=DURATION_MS, dt_ms=DT_MS, seed=0, runtime=jtfne.RuntimeConfig(**runtime))
    return model.simulate_batch(sim, n_seeds=2)


@pytest.mark.parametrize("controller", [RULE, HOMEOSTASIS], ids=["hdp", "homeostasis"])
def test_simulate_records_executed_and_requested_backend(controller):
    model = _model()
    sig = _run(model, **controller)
    assert sig.metadata["recurrent_backend"] == "edge_list"  # executed
    assert sig.metadata["runtime"]["recurrent_backend"] == "dense"  # requested (the default)
    assert model.manifest(signals=sig)["backend_metadata"]["used_recurrent_backend"] == "edge_list"


@pytest.mark.parametrize("controller", [RULE, HOMEOSTASIS], ids=["hdp", "homeostasis"])
def test_simulate_batch_records_executed_and_requested_backend(controller):
    meta = _batch(_model(), **controller)["metadata"]
    assert meta["recurrent_backend"] == "edge_list"
    assert meta["runtime"]["recurrent_backend"] == "dense"


def test_rule_free_default_still_records_dense():
    model = _model()
    assert _run(model).metadata["recurrent_backend"] == "dense"
    assert _batch(model)["metadata"]["recurrent_backend"] == "dense"


@pytest.mark.parametrize("controller", [RULE, HOMEOSTASIS], ids=["hdp", "homeostasis"])
def test_explicit_model_dense_with_edge_only_controller_is_refused(controller):
    dense = _explicit_dense(_model())
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _run(dense, **controller)
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _batch(dense, **controller)


def test_enable_hdp_without_engaged_rule():
    # simulate() runs the dense kernel when no rule is engaged; simulate_batch runs the HDP
    # edge kernel for any enable_hdp, so it records edge_list and refuses an explicit dense.
    no_rule = dict(enable_hdp=True, hdp_params={})
    model = _model()
    assert _run(model, **no_rule).metadata["recurrent_backend"] == "dense"
    assert _batch(model, **no_rule)["metadata"]["recurrent_backend"] == "edge_list"
    assert _run(_explicit_dense(model), **no_rule).metadata["recurrent_backend"] == "dense"
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _batch(_explicit_dense(model), **no_rule)


def test_placeholder_w_model_records_edge_list_with_explicit_runtime():
    # Edges live only in params["edge_list"]; the dense kernel is never used for this model.
    from scripts.perf.w10_allocation_map import build_config

    cfg = build_config(n=200, p_connect=0.0, max_in_degree=20, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=1)
    model = jtfne.construct(cfg)
    assert model.params["emitter"].W.shape == (0, 0), "fixture must carry a placeholder W"
    for runtime in ({}, dict(enable_hdp=True, hdp_params={})):
        assert _run(model, **runtime).metadata["recurrent_backend"] == "edge_list"
        assert _batch(model, **runtime)["metadata"]["recurrent_backend"] == "edge_list"
