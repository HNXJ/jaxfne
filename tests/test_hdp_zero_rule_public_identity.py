"""HDP through the public ``simulate()`` and ``simulate_batch()``: identity and backend provenance.

P-057 closure: a registered HDP rule at zero gain reproduces the rule-free run on the same
backend. An engaged HDP rule always runs the edge-list kernel, so the rule-free reference
uses ``recurrent_backend="edge_list"``; against the default dense backend the runs differ for
backend reasons, not rule reasons (ISSUE_LOG P-057, P-062).

P-062 step 1: runs whose controller forces the edge-list kernel (an engaged HDP rule,
homeostasis, a placeholder dense W) record ``edge_list`` as the executed backend, keep the
requested value in the runtime report, and refuse an explicit model-level
``recurrent_backend="dense"`` instead of replacing it.

P-062 step 2: ``recurrent_backend`` takes ``auto`` (the default; selects the backend the
engaged model rules need), ``dense`` or ``edge_list`` (executed or refused, never replaced);
any other value is rejected where the configuration is built. ``auto`` leaves the numbers of
the old default unchanged.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne import neuronal_tensor as nt
from jaxfne._model import Model
from jaxfne.hdp_rules_ephaptic import EPHAPTIC_RULE_NAME, ephaptic_kernel, ephaptic_rule_params

N = 100
DURATION_MS = 100.0
DT_MS = 0.5
SEED = 0


def _model(n=N, duration_ms=DURATION_MS):
    layer = nt.Layer(
        name="L4",
        n_neurons=n,
        neuron_types=[nt.NeuronType.make("E", fraction=0.9), nt.NeuronType.make("PV", fraction=0.1)],
    )
    tensor = nt.NeuronalTensor(areas=[nt.Area(name="V1", layers=[layer])], name="hdp_public")
    return jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=SEED, duration_ms=duration_ms, dt_ms=DT_MS))


def _zero_rule_params(rule):
    if rule == "synthetic_presyn_gain":
        return {"hdp_rule": rule}  # every gain defaults to 0
    pos = np.random.default_rng(1).uniform(0.0, 100.0, size=(N, 2))
    k = ephaptic_kernel(pos, sigma=15.0, cutoff=40.0)
    return {"hdp_rule": rule, "hdp_rule_params": ephaptic_rule_params(k, alpha=0.0, v_rest=-65.0)}


@pytest.mark.parametrize("rule", ["synthetic_presyn_gain", EPHAPTIC_RULE_NAME])
def test_zero_gain_rule_equals_rule_free_on_edge_list(rule):
    model = _model()
    free = jtfne.simulate(
        model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED,
        runtime=jtfne.RuntimeConfig(recurrent_backend="edge_list"),
    )
    zero = jtfne.simulate(
        model, duration_ms=DURATION_MS, dt_ms=DT_MS, seed=SEED,
        runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=_zero_rule_params(rule), recurrent_backend="edge_list"),
    )
    s_free, s_zero = np.asarray(free.spikes), np.asarray(zero.spikes)
    assert (s_free > 0.5).sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(s_zero, s_free)
    np.testing.assert_array_equal(np.asarray(zero.V_m), np.asarray(free.V_m))


# --- P-062 step 1: executed vs requested backend -------------------------------------------

PROV_DURATION_MS = 20.0
RULE = dict(enable_hdp=True, hdp_params={"hdp_rule": "synthetic_presyn_gain"})
HOMEOSTASIS = dict(enable_homeostasis=True)


def _small_model():
    return _model(n=40, duration_ms=PROV_DURATION_MS)


def _explicit_dense(model, backend="dense"):
    return Model(
        cfg=replace(model.cfg, metadata={**model.cfg.metadata, "recurrent_backend": backend}),
        params=model.params,
        static=model.static,
    )


def _run(model, **runtime):
    return jtfne.simulate(model, duration_ms=PROV_DURATION_MS, dt_ms=DT_MS, seed=SEED,
                          runtime=jtfne.RuntimeConfig(**runtime))


def _batch(model, **runtime):
    sim = jtfne.simulation(duration_ms=PROV_DURATION_MS, dt_ms=DT_MS, seed=SEED, runtime=jtfne.RuntimeConfig(**runtime))
    return model.simulate_batch(sim, n_seeds=2)


@pytest.mark.parametrize("controller", [RULE, HOMEOSTASIS], ids=["hdp", "homeostasis"])
def test_simulate_records_executed_and_requested_backend(controller):
    model = _small_model()
    sig = _run(model, **controller)
    assert sig.metadata["recurrent_backend"] == "edge_list"  # executed
    assert sig.metadata["runtime"]["recurrent_backend"] == "auto"  # requested (the default)
    assert model.manifest(signals=sig)["backend_metadata"]["used_recurrent_backend"] == "edge_list"


@pytest.mark.parametrize("controller", [RULE, HOMEOSTASIS], ids=["hdp", "homeostasis"])
def test_simulate_batch_records_executed_and_requested_backend(controller):
    meta = _batch(_small_model(), **controller)["metadata"]
    assert meta["recurrent_backend"] == "edge_list"
    assert meta["runtime"]["recurrent_backend"] == "auto"


def test_rule_free_default_still_records_dense():
    model = _small_model()
    assert _run(model).metadata["recurrent_backend"] == "dense"
    assert _batch(model)["metadata"]["recurrent_backend"] == "dense"


@pytest.mark.parametrize("controller", [RULE, HOMEOSTASIS], ids=["hdp", "homeostasis"])
def test_explicit_model_dense_with_edge_only_controller_is_refused(controller):
    dense = _explicit_dense(_small_model())
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _run(dense, **controller)
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _batch(dense, **controller)


def test_enable_hdp_without_engaged_rule():
    # simulate() runs the dense kernel when no rule is engaged; simulate_batch runs the HDP
    # edge kernel for any enable_hdp, so it records edge_list and refuses an explicit dense.
    no_rule = dict(enable_hdp=True, hdp_params={})
    model = _small_model()
    assert _run(model, **no_rule).metadata["recurrent_backend"] == "dense"
    assert _batch(model, **no_rule)["metadata"]["recurrent_backend"] == "edge_list"
    assert _run(_explicit_dense(model), **no_rule).metadata["recurrent_backend"] == "dense"
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _batch(_explicit_dense(model), **no_rule)


def test_placeholder_w_model_records_edge_list_with_explicit_runtime():
    # Edges live only in params["edge_list"]; the dense kernel is never used for this model.
    model = _placeholder_w_model()
    for runtime in ({}, dict(enable_hdp=True, hdp_params={})):
        assert _run(model, **runtime).metadata["recurrent_backend"] == "edge_list"
        assert _batch(model, **runtime)["metadata"]["recurrent_backend"] == "edge_list"


def test_simulate_batch_applies_hdp_rule_params():
    """P-064: simulate_batch passed {} for hdp_rule_params, so a rule with gains ran at zero gain."""
    gain = {"hdp_rule": "synthetic_presyn_gain", "hdp_rule_params": {"gamma": 5.0, "k_h": 2.0, "k_w": 2.0}}
    zero = _batch(_small_model(), enable_hdp=True, hdp_params={"hdp_rule": "synthetic_presyn_gain"})
    with_gain = _batch(_small_model(), enable_hdp=True, hdp_params=gain)
    one = _run(_small_model(), enable_hdp=True, hdp_params=gain)
    assert not np.array_equal(np.asarray(one.V_m), np.asarray(_run(_small_model(), **RULE).V_m)), (
        "fixture: the gain must act in simulate()"
    )
    assert not np.array_equal(np.asarray(zero["V_m"]), np.asarray(with_gain["V_m"]))


def test_simulate_batch_runs_ephaptic_field():
    pos = np.random.default_rng(1).uniform(0.0, 100.0, size=(40, 2))
    params = ephaptic_rule_params(ephaptic_kernel(pos, sigma=15.0, cutoff=40.0), alpha=0.5, v_rest=-65.0)
    out = _batch(_small_model(), enable_hdp=True, hdp_params={"hdp_rule": EPHAPTIC_RULE_NAME, "hdp_rule_params": params})
    assert out["metadata"]["recurrent_backend"] == "edge_list"


def _placeholder_w_model():
    from scripts.perf.w10_allocation_map import build_config

    cfg = build_config(n=200, p_connect=0.0, max_in_degree=20, duration_ms=PROV_DURATION_MS, dt_ms=DT_MS, seed=1)
    model = jtfne.construct(cfg)
    assert model.params["emitter"].W.shape == (0, 0), "fixture must carry a placeholder W"
    return model


# --- P-062 step 2: auto | dense | edge_list, validated -------------------------------------


@pytest.mark.parametrize("value", ["edn_list", "Dense", ""])
def test_unknown_backend_is_rejected_where_the_configuration_is_built(value):
    with pytest.raises(ValueError, match="recurrent_backend"):
        jtfne.RuntimeConfig(recurrent_backend=value)
    with pytest.raises(ValueError, match="recurrent_backend"):
        jtfne.Configuration().runtime(recurrent_backend=value)


def test_default_backend_is_auto():
    assert jtfne.RuntimeConfig().recurrent_backend == "auto"
    assert jtfne.runtime().recurrent_backend == "auto"
    jtfne.RuntimeConfig(recurrent_backend="auto")  # accepted explicitly too


@pytest.mark.parametrize("controller", [RULE, HOMEOSTASIS], ids=["hdp", "homeostasis"])
def test_explicit_runtime_dense_with_edge_only_controller_is_refused(controller):
    model = _small_model()
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _run(model, recurrent_backend="dense", **controller)
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _batch(model, recurrent_backend="dense", **controller)


def test_explicit_runtime_dense_on_placeholder_w_is_refused():
    model = _placeholder_w_model()
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _run(model, recurrent_backend="dense")
    with pytest.raises(ValueError, match="recurrent_backend='dense'"):
        _batch(model, recurrent_backend="dense")


def test_auto_selects_edge_list_for_the_receptor_kernel():
    sig = _run(_small_model(), synaptic_kernel="receptor_exponential")
    assert sig.metadata["recurrent_backend"] == "edge_list"
    assert sig.metadata["runtime"]["recurrent_backend"] == "auto"


def test_auto_runtime_does_not_take_a_model_level_edge_list():
    # Before step 2 an explicit RuntimeConfig ran dense whatever cfg.metadata said; "auto" keeps that.
    model = _explicit_dense(_small_model(), backend="edge_list")
    auto, dense = _run(model), _run(model, recurrent_backend="dense")
    assert auto.metadata["recurrent_backend"] == "dense"
    _assert_same_run(auto, dense)


def test_auto_rule_free_return_state_runs_edge_list():
    model = _small_model()
    sig, state = jtfne.simulate(model, duration_ms=PROV_DURATION_MS, dt_ms=DT_MS, seed=SEED,
                                return_state=True, runtime=jtfne.RuntimeConfig())
    edge = _run(model, recurrent_backend="edge_list")
    assert sig.metadata["recurrent_backend"] == "edge_list"
    _assert_same_run(sig, edge)


def _assert_same_run(a, b):
    assert (np.asarray(a.spikes) > 0.5).sum() > 0, "fixture must spike"
    np.testing.assert_array_equal(np.asarray(a.spikes), np.asarray(b.spikes))
    np.testing.assert_array_equal(np.asarray(a.V_m), np.asarray(b.V_m))


def test_auto_rule_free_equals_explicit_dense():
    model = _model(n=60, duration_ms=60.0)
    run = lambda **rt: jtfne.simulate(model, duration_ms=60.0, dt_ms=DT_MS, seed=SEED, runtime=jtfne.RuntimeConfig(**rt))
    auto, dense = run(), run(recurrent_backend="dense")
    assert auto.metadata["recurrent_backend"] == dense.metadata["recurrent_backend"] == "dense"
    _assert_same_run(auto, dense)


@pytest.mark.parametrize("rule", ["synthetic_presyn_gain", EPHAPTIC_RULE_NAME])
def test_auto_hdp_equals_explicit_edge_list_including_state(rule):
    model = _model(n=N, duration_ms=60.0)
    params = _zero_rule_params(rule)
    if rule == "synthetic_presyn_gain":
        params = {**params, "hdp_rule_params": {"gamma": 0.3, "k_h": 0.2, "k_w": 0.1}}
    else:
        params["hdp_rule_params"]["alpha"] = 0.5

    def run(**rt):
        return jtfne.simulate(model, duration_ms=60.0, dt_ms=DT_MS, seed=SEED, return_state=True,
                              runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=params, **rt))

    (auto, s_auto), (edge, s_edge) = run(), run(recurrent_backend="edge_list")
    assert auto.metadata["recurrent_backend"] == "edge_list"
    _assert_same_run(auto, edge)
    import jax

    leaves_auto, leaves_edge = jax.tree_util.tree_leaves(s_auto), jax.tree_util.tree_leaves(s_edge)
    assert len(leaves_auto) == len(leaves_edge) > 0, "fixture must return state"
    for x, y in zip(leaves_auto, leaves_edge):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
