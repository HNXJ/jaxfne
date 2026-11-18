"""P-062 A, step 1: an engaged HDP rule runs the edge-list kernel, so the run records
``edge_list`` as its backend, and an explicit model-level ``recurrent_backend="dense"``
is refused instead of being silently replaced.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne._model import Model

DURATION_MS = 20.0
DT_MS = 0.5
RULE_RUNTIME = dict(enable_hdp=True, hdp_params={"hdp_rule": "synthetic_presyn_gain"})


def _model():
    layer = nt.Layer(
        name="L4",
        n_neurons=40,
        neuron_types=[nt.NeuronType.make("E", fraction=0.9), nt.NeuronType.make("PV", fraction=0.1)],
    )
    tensor = nt.NeuronalTensor(areas=[nt.Area(name="V1", layers=[layer])], name="p062_provenance")
    return jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=DURATION_MS, dt_ms=DT_MS))


def _run(model, **runtime):
    return jtfne.simulate(model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=0, runtime=jtfne.RuntimeConfig(**runtime))


def test_hdp_run_records_the_edge_list_backend_it_ran():
    model = _model()
    sig = _run(model, **RULE_RUNTIME)
    assert sig.metadata["recurrent_backend"] == "edge_list"
    assert sig.metadata["runtime"]["recurrent_backend"] == "edge_list"
    assert model.manifest(signals=sig)["backend_metadata"]["used_recurrent_backend"] == "edge_list"


def test_rule_free_default_still_records_dense():
    sig = _run(_model())
    assert sig.metadata["recurrent_backend"] == "dense"


def test_explicit_model_dense_with_engaged_rule_is_refused():
    model = _model()
    dense = Model(
        cfg=replace(model.cfg, metadata={**model.cfg.metadata, "recurrent_backend": "dense"}),
        params=model.params,
        static=model.static,
    )
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _run(dense, **RULE_RUNTIME)
