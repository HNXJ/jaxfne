"""23-PARAM-01: class-shared edge weight storage (derivable only)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne.emitters import resolve_edge_weight
from scripts.perf.w10_allocation_map import build_config, total_bytes, walk_arrays


def _bounded_model(n=200, seed=1):
    cfg = build_config(
        n=n, p_connect=0.0, max_in_degree=100, duration_ms=10.0, dt_ms=0.5, seed=seed
    )
    return cfg, jtfne.construct(cfg)


def test_construct_compacts_weight_via_presynaptic_sign_magnitude():
    _, model = _bounded_model()
    el = model.params["edge_list"]
    assert el.weight_storage == "magnitude_times_presynaptic_sign"
    assert tuple(np.asarray(el.weight).shape) == (0,)
    table = np.asarray(el.mechanism_weight_magnitude_table)
    assert table.size == 1
    np.testing.assert_allclose(float(table[0]), 0.03)
    sign = np.asarray(model.params["emitter"].sign)
    w = np.asarray(
        resolve_edge_weight(el, el.weight.dtype, presynaptic_sign=jnp.asarray(sign))
    )
    np.testing.assert_allclose(np.sort(np.unique(w)), [-0.03, 0.03], rtol=1e-6, atol=1e-6)


def test_compact_weight_bit_exact_observables():
    cfg, model = _bounded_model(n=200, seed=3)
    before = np.asarray(jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=2).V_m)
    el = model.params["edge_list"]
    sign = jnp.asarray(model.params["emitter"].sign)
    expanded = resolve_edge_weight(el, el.weight.dtype, presynaptic_sign=sign)
    assert expanded.shape == (el.n_edges,)
    after = np.asarray(jtfne.simulate(model, duration_ms=10.0, dt_ms=0.5, seed=2).V_m)
    np.testing.assert_array_equal(before, after)


def test_weight_compaction_reduces_persistent_bytes():
    model = _bounded_model(n=1000, seed=1)[1]
    nbytes = {
        path.split(".")[-1]: int(np.asarray(arr).nbytes)
        for path, arr in walk_arrays(model)
        if "edge_list" in path and path.endswith("weight")
    }
    assert nbytes.get("weight", 0) == 0
    assert total_bytes(model) < 1_100_000


def test_inhomogeneous_weights_remain_per_edge():
    cfg = (
        jtfne.Configuration()
        .runtime(seed=0, dtype="float32", duration_ms=20.0, dt_ms=0.5)
        .column(name="c", layers=["L2/3", "L4"], n=40)
        .cell_types({"E": 0.75, "PV": 0.25})
        .connectivity(kind="laminar_signed_metadata", recurrent=True)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes"])
    )
    model = jtfne.construct(cfg)
    el = model.params["edge_list"]
    assert el.weight_storage == "per_edge"
    assert int(np.asarray(el.weight).shape[0]) == el.n_edges
