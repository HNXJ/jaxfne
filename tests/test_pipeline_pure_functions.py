"""Tests for jaxfne/_pipeline.py -- the internal pure-function layer.

Each wrapper is checked for behavioral equivalence against the existing
public call it delegates to, not just import success.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jaxfne import _pipeline
from jaxfne.core import Configuration, Model, Signals
from jaxfne.neuronal_tensor import (
    NeuronalTensor,
    RuntimeConfiguration,
    load as tensor_load,
    neuronal_tensor_to_configuration,
)

CONFIG_DIR = Path(__file__).parent.parent / "jaxfne" / "configs"
SMALL_TENSOR_PATH = CONFIG_DIR / "default-column.json"


def test_load_tensor_matches_direct_load():
    via_pipeline = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    via_direct = tensor_load(SMALL_TENSOR_PATH)
    assert isinstance(via_pipeline, NeuronalTensor)
    assert via_pipeline.name == via_direct.name
    assert len(via_pipeline.areas) == len(via_direct.areas)


def test_save_tensor_roundtrip(tmp_path):
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    out_path = tmp_path / "roundtrip.json"
    written = _pipeline.save_tensor(tensor, out_path)
    assert Path(written).exists()
    reloaded = _pipeline.load_tensor(out_path)
    assert reloaded.name == tensor.name
    assert len(reloaded.areas) == len(tensor.areas)


def test_tensor_to_configuration_matches_direct_call():
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    via_pipeline = _pipeline.tensor_to_configuration(
        tensor, seed=3, duration_ms=200.0, dt_ms=0.5,
    )
    via_direct = neuronal_tensor_to_configuration(
        tensor, seed=3, duration_ms=200.0, dt_ms=0.5,
    )
    assert isinstance(via_pipeline, Configuration)
    assert via_pipeline.networks == via_direct.networks
    assert via_pipeline.emitters == via_direct.emitters


def test_build_network_configuration_path():
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    cfg = _pipeline.tensor_to_configuration(
        tensor, seed=1, duration_ms=100.0, dt_ms=0.5,
    )
    model = _pipeline.build_network(cfg)
    assert isinstance(model, Model)


def test_build_network_tensor_path():
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    runtime = RuntimeConfiguration(seed=2, duration_ms=100.0, dt_ms=0.5)
    model = _pipeline.build_network(tensor, runtime)
    assert isinstance(model, Model)


def test_run_network_matches_direct_simulate():
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    cfg = _pipeline.tensor_to_configuration(
        tensor, seed=5, duration_ms=50.0, dt_ms=0.5,
    )
    model = _pipeline.build_network(cfg)
    sig = _pipeline.run_network(model, duration_ms=50.0, dt_ms=0.5, seed=5)
    assert isinstance(sig, Signals)
    assert sig.spikes.shape[0] > 0


def _small_model() -> Model:
    tensor = _pipeline.load_tensor(SMALL_TENSOR_PATH)
    cfg = _pipeline.tensor_to_configuration(tensor, seed=1, duration_ms=50.0, dt_ms=0.5)
    return _pipeline.build_network(cfg)


def test_checkpoint_restore_roundtrip(tmp_path):
    model = _small_model()
    ckpt_path = tmp_path / "ckpt"
    written = _pipeline.checkpoint_state(model, ckpt_path)
    assert written.with_suffix(".npz").exists()
    assert written.with_suffix(".json").exists()

    leaves, _static = _pipeline.restore_state(ckpt_path)
    original_leaves = jax.tree_util.tree_leaves(model.params)
    assert len(leaves) == len(original_leaves)
    for restored, original in zip(leaves, original_leaves):
        assert jnp.allclose(jnp.asarray(restored), jnp.asarray(original))


def test_initialize_static_state_matches_model_static():
    model = _small_model()
    static_copy = _pipeline.initialize_static_state(model)
    assert set(static_copy.keys()) == set(model.static.keys())
    for key, value in model.static.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            assert static_copy[key] == value


def test_initialize_dynamic_state_is_independent_copy():
    model = _small_model()
    dynamic = _pipeline.initialize_dynamic_state(model)

    # Independent container: mutating the returned dict (e.g. adding a key,
    # or rebinding "emitter") must not affect model.params.
    assert dynamic is not model.params
    dynamic["emitter"] = "overwritten"
    assert "emitter" in model.params and model.params["emitter"] != "overwritten"

    # Leaf values are unaffected by the rebinding above and still match the
    # original model's emitter array values (JAX arrays are immutable, so
    # leaf-level identity is not the independence question -- container
    # rebinding is).
    fresh = _pipeline.initialize_dynamic_state(model)
    assert jnp.allclose(fresh["emitter"].v0, model.params["emitter"].v0)


def test_dynamic_state_from_model_shapes():
    model = _small_model()
    state = _pipeline.dynamic_state_from_model(model)
    n_neurons = model.params["emitter"].n_neurons
    n_edges = model.params["edge_list"].n_edges
    assert isinstance(state, _pipeline.DynamicState)
    assert state.v.shape == (n_neurons,)
    assert state.u.shape == (n_neurons,)
    assert state.prev_spikes.shape == (n_neurons,)
    assert state.H.shape == (n_neurons,)
    assert state.syn_state.shape == (n_edges,)
    assert state.w.shape == (n_edges,)


def test_compile_step_fn_shape_via_eval_shape():
    model = _small_model()
    step_fn, init = _pipeline.compile_step_fn(model, dt_ms=0.5)
    assert isinstance(init, _pipeline.DynamicState)
    n_neurons = model.params["emitter"].n_neurons

    sched_t = jnp.zeros((n_neurons,), dtype=init.v.dtype)
    key_t = jax.random.PRNGKey(0)
    out_carry, out_outputs = jax.eval_shape(step_fn, init, (sched_t, key_t))

    assert isinstance(out_carry, _pipeline.DynamicState)
    assert out_carry.v.shape == init.v.shape
    assert out_carry.H.shape == init.H.shape
    assert out_carry.w.shape == init.w.shape
    assert len(out_outputs) == 5  # (v, spikes, sources, H_trace, w_trace) base arity


def test_scan_network_shape_via_eval_shape():
    model = _small_model()
    step_fn, init = _pipeline.compile_step_fn(model, dt_ms=0.5)
    n_neurons = model.params["emitter"].n_neurons
    n_steps = 10

    sched = jnp.zeros((n_steps, n_neurons), dtype=init.v.dtype)
    keys = jax.random.split(jax.random.PRNGKey(0), n_steps)

    final_carry, outputs = jax.eval_shape(
        lambda i, s, k: _pipeline.scan_network(step_fn, i, s, k), init, sched, keys
    )

    assert isinstance(final_carry, _pipeline.DynamicState)
    assert final_carry.v.shape == init.v.shape
    assert final_carry.w.shape == init.w.shape
    for leaf in outputs:
        assert leaf.shape[0] == n_steps
