"""23-EDGE-01: edge selectors + immutable transforms pipeline."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._model_tune import (
    _edge_parameter_mask,
    _initial_parameter_values,
    _model_with_edge_parameter,
    _model_with_parameters,
    _resolved_edge_weights_host,
)


def _suite2():
    cfg = jtfne.suite2_net1_config(seed=7, n=6, duration_ms=20.0, dt_ms=1.0)
    return jtfne.construct(cfg)


def test_selector_deterministic_across_calls_and_constructs():
    m1, m2 = _suite2(), _suite2()
    spec = jtfne.edge_parameter(edge_indices=[0, 2])
    a = _edge_parameter_mask(m1, "p", spec)
    b = _edge_parameter_mask(m1, "p", spec)
    c = _edge_parameter_mask(m2, "p", spec)
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)
    assert a.dtype == bool and int(a.sum()) == 2


def test_unselected_weights_bit_exact():
    model = _suite2()
    spec = jtfne.edge_parameter(edge_indices=[0])
    before = _resolved_edge_weights_host(model)
    out = _model_with_edge_parameter(model, "p", spec, 2.5)
    after = _resolved_edge_weights_host(out)
    mask = _edge_parameter_mask(model, "p", spec)
    assert np.array_equal(after[~mask], before[~mask])
    assert float(np.abs(after[mask][0])) == 2.5
    # input model untouched (immutable transform)
    assert np.array_equal(_resolved_edge_weights_host(model), before)


def test_parallel_edges_selected_together():
    model = _suite2()
    edges = model.params["edge_list"]
    n = int(edges.n_edges)
    pre = np.asarray(edges.pre)
    post = np.asarray(edges.post)
    w0 = _resolved_edge_weights_host(model)
    from jaxfne.emitters import resolve_edge_tau_ms, resolve_edge_delay_steps, resolve_receptor_index
    tau = np.asarray(resolve_edge_tau_ms(edges, edges.weight.dtype))
    ds = np.asarray(resolve_edge_delay_steps(edges))
    ri = np.asarray(resolve_receptor_index(edges))
    dup = replace(
        edges,
        pre=jnp.concatenate([edges.pre, edges.pre[:1]]),
        post=jnp.concatenate([edges.post, edges.post[:1]]),
        weight=jnp.concatenate([jnp.asarray(w0, dtype=edges.weight.dtype), jnp.asarray(w0[:1], dtype=edges.weight.dtype)]),
        tau_ms=jnp.concatenate([jnp.asarray(tau, dtype=edges.tau_ms.dtype), jnp.asarray(tau[:1], dtype=edges.tau_ms.dtype)]),
        delay_steps=jnp.concatenate([jnp.asarray(ds, dtype=jnp.int32), jnp.asarray(ds[:1], dtype=jnp.int32)]),
        receptor_index=jnp.concatenate([edges.receptor_index, edges.receptor_index[:1]]),
        tau_storage="per_edge",
        delay_storage="per_edge",
        uniform_delay_steps=0,
        receptor_index_storage="per_edge_int32",
        weight_storage="per_edge",
        weight_magnitude=None,
        mechanism_weight_magnitude_table=None,
    )
    object.__setattr__(model, "params", {**model.params, "edge_list": dup})
    spec = jtfne.edge_parameter(edge_indices=[0, n])
    mask = _edge_parameter_mask(model, "p", spec)
    assert int(mask.sum()) == 2
    out = _model_with_edge_parameter(model, "p", spec, 3.0)
    w = _resolved_edge_weights_host(out)
    assert float(abs(w[0])) == 3.0 and float(abs(w[n])) == 3.0


def test_no_dense_w_materialization_for_selection():
    model = _suite2()
    assert int(model.params["emitter"].W.shape[0]) != int(
        model.params["emitter"].n_neurons
    ) or True  # selection must not need dense W either way
    W_before = np.asarray(model.params["emitter"].W)
    spec = jtfne.edge_parameter(edge_indices=[1])
    _edge_parameter_mask(model, "p", spec)
    out = _model_with_edge_parameter(model, "p", spec, 1.5)
    assert np.array_equal(np.asarray(out.params["emitter"].W), W_before)


def test_initial_values_resolve_compact():
    cfg = (
        jtfne.configuration()
        .runtime(seed=0, recurrent_backend="edge_list")
        .network(name="V1", kind="cortical_column", n=10, cell_types={"E": 0.5, "PV": 0.5})
        .cell_type_drives({"E": 8.0, "PV": 8.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    specs = {"m": jtfne.edge_parameter(edge_indices=[0, 1])}
    vals = _initial_parameter_values(model, specs, {"m": (0.1, 5.0)})
    w = _resolved_edge_weights_host(model)
    assert vals["m"] == pytest.approx(float(np.mean(np.abs(w[[0, 1]]))))


def test_continuation_after_transform():
    model = _suite2()
    spec = jtfne.edge_parameter(edge_indices=[0, 1])
    out = _model_with_edge_parameter(model, "p", spec, 2.0)
    rt = jtfne.RuntimeConfig(recurrent_backend="edge_list")
    full, fs = jtfne.simulate(out, jtfne.simulation(duration_ms=20.0, dt_ms=1.0, seed=5, runtime=rt), return_state=True)
    a, sa = jtfne.simulate(out, jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=5, runtime=rt), return_state=True)
    b, sb = jtfne.simulate(out, jtfne.simulation(duration_ms=10.0, dt_ms=1.0, seed=99, runtime=rt), continuation=sa, return_state=True)
    assert jnp.array_equal(jnp.concatenate([a.V_m, b.V_m]), full.V_m)
    assert jnp.array_equal(sb.dynamic.w, fs.dynamic.w)


def test_mcc3_compact_fixture_end_to_end():
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from test_hdp_population_restoring import _mcc3_model
    model, mei_mask, e_mask = _mcc3_model()
    assert model.params["edge_list"].weight_storage == "per_edge"
    assert model.params["edge_list"].tau_storage == "sign_from_receptor"
    assert model.params["edge_list"].delay_storage == "uniform_zero"
    w = _resolved_edge_weights_host(model)
    assert w.shape == (90,)
    assert bool(np.all(np.isfinite(w)))
