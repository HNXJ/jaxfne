"""Gate 10: Model.edge_table() connectivity observability test.

Verifies that:
1. Model.edge_table() returns realized edges from actual state, not configured probabilities.
2. In an asymmetric deterministic fixture with known expected edges, the correspondence is exact.
3. Edge table columns match exact authoritative values:
   (pre, post, weight, receptor_index, receptor_type, tau_ms, delay_steps).
4. Neuron attributes join accurately against Model.neuron_table().
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne.emitters import EdgeList


def test_gate10_edge_table_asymmetric_deterministic_fixture():
    """Verify edge_table on an asymmetric deterministic circuit with heterogeneous delay & receptor."""
    cfg = jtfne.suite2_net1_config(seed=10, n=3, duration_ms=10.0, dt_ms=0.5)
    model = jtfne.construct(cfg)

    # Construct an explicit asymmetric EdgeList with known properties:
    # Edge 0: 0 -> 1, weight=0.15, receptor_index=0 (AMPA), tau_ms=2.0, delay_steps=4
    # Edge 1: 1 -> 2, weight=-0.40, receptor_index=1 (GABA_A), tau_ms=5.0, delay_steps=0
    # Edge 2: 2 -> 0, weight=0.85, receptor_index=2 (NMDA), tau_ms=100.0, delay_steps=2
    pre = jnp.array([0, 1, 2], dtype=jnp.int32)
    post = jnp.array([1, 2, 0], dtype=jnp.int32)
    weight = jnp.array([0.15, -0.40, 0.85], dtype=jnp.float32)
    rec_idx = jnp.array([0, 1, 2], dtype=jnp.int32)
    tau_ms = jnp.array([2.0, 5.0, 100.0], dtype=jnp.float32)
    delay_steps = jnp.array([4, 0, 2], dtype=jnp.int32)

    explicit_el = EdgeList(
        pre=pre,
        post=post,
        weight=weight,
        receptor_index=rec_idx,
        tau_ms=tau_ms,
        delay_steps=delay_steps,
    )

    new_params = dict(model.params)
    new_params["edge_list"] = explicit_el
    model_with_el = jtfne.Model(
        cfg=model.cfg,
        params=new_params,
        static=model.static,
    )

    edges = model_with_el.edge_table()
    assert len(edges) == 3

    # Check Edge 0
    e0 = edges[0]
    assert e0["edge_id"] == 0
    assert e0["pre"] == 0
    assert e0["post"] == 1
    assert pytest.approx(e0["weight"], abs=1e-6) == 0.15
    assert e0["receptor_index"] == 0
    assert e0["receptor_type"] == "AMPA"
    assert pytest.approx(e0["tau_ms"], abs=1e-6) == 2.0
    assert e0["delay_steps"] == 4

    # Check Edge 1
    e1 = edges[1]
    assert e1["edge_id"] == 1
    assert e1["pre"] == 1
    assert e1["post"] == 2
    assert pytest.approx(e1["weight"], abs=1e-6) == -0.40
    assert e1["receptor_index"] == 1
    assert e1["receptor_type"] == "GABA_A"
    assert pytest.approx(e1["tau_ms"], abs=1e-6) == 5.0
    assert e1["delay_steps"] == 0

    # Check Edge 2
    e2 = edges[2]
    assert e2["edge_id"] == 2
    assert e2["pre"] == 2
    assert e2["post"] == 0
    assert pytest.approx(e2["weight"], abs=1e-6) == 0.85
    assert e2["receptor_index"] == 2
    assert e2["receptor_type"] == "NMDA"
    assert pytest.approx(e2["tau_ms"], abs=1e-6) == 100.0
    assert e2["delay_steps"] == 2

    # Verify join against neuron_table
    nt = model_with_el.neuron_table()
    for e in edges:
        pre_n = nt[e["pre"]]
        post_n = nt[e["post"]]
        assert e["pre_area"] == pre_n["area"]
        assert e["pre_layer"] == pre_n["layer"]
        assert e["pre_cell_type"] == pre_n["cell_type"]
        assert e["post_area"] == post_n["area"]
        assert e["post_layer"] == post_n["layer"]
        assert e["post_cell_type"] == post_n["cell_type"]


def test_gate10_edge_table_empty_model():
    """Verify empty edge_table behavior when no recurrence exists."""
    cfg = jtfne.suite2_single_neuron_config()
    model = jtfne.construct(cfg)
    edges = model.edge_table()
    assert isinstance(edges, list)
    assert len(edges) == 0
