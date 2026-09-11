"""23-H-01: H/RBS/RBD representation audit probes.

These tests inventory dynamic-state coordinates, run observable-specific
knockout probes (delta_jk style), and classify redundant vs required
representation. They are evidence for the audit receipt, not a specification
for immediate carrier-shape refactors.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxfne as jtfne
from jaxfne import _pipeline
from jaxfne.emitters import (
    EdgeList,
    IzhikevichParams,
    simulate_edge_recurrent_izhikevich_rbd,
)

AUDIT_JSON = (
    Path(__file__).resolve().parents[1]
    / "artifacts/audit/h_representation_probe_results.json"
)

COORDINATE_INVENTORY = [
    {
        "id": "dynamic.v",
        "owner": "neuron",
        "paths": ["baseline", "rbd", "hdp", "homeostasis"],
        "role": "fast activity state",
    },
    {
        "id": "dynamic.u",
        "owner": "neuron",
        "paths": ["baseline", "rbd", "hdp", "homeostasis"],
        "role": "Izhikevich recovery",
    },
    {
        "id": "dynamic.prev_spikes",
        "owner": "neuron",
        "paths": ["baseline", "rbd", "hdp"],
        "role": "continuation parity carry; unread on zero-delay baseline kernel",
    },
    {
        "id": "dynamic.syn_state",
        "owner": "edge",
        "paths": ["baseline", "rbd", "hdp"],
        "role": "per-edge exponential synaptic filter",
    },
    {
        "id": "dynamic.H",
        "owner": "neuron (or population vector)",
        "paths": ["rbd", "hdp", "registered_hdp"],
        "role": "RBS / HDP hidden state",
    },
    {
        "id": "dynamic.w",
        "owner": "edge",
        "paths": ["hdp", "registered_hdp"],
        "role": "plastic synaptic weight state",
    },
    {
        "id": "dynamic.theta_S",
        "owner": "population",
        "paths": ["hdp_population"],
        "role": "population controller coordinates",
    },
    {
        "id": "dynamic.aux",
        "owner": "rule",
        "paths": ["registered_hdp"],
        "role": "registered-rule auxiliary coordinates",
    },
    {
        "id": "continuation.delay_state",
        "owner": "network",
        "paths": ["delayed_baseline", "delayed_rbd", "delayed_hdp"],
        "role": "ring buffer B_t for finite edge delays",
    },
]


def _suite2_model():
    return jtfne.construct(jtfne.suite2_net1_config(seed=11, n=8))


def test_baseline_carrier_H_and_w_are_observably_inert():
    """Knockout: perturb H,w in continuation carrier; V_m and spikes unchanged."""
    model = _suite2_model()
    step_fn, init = _pipeline.compile_step_fn(
        model, dt_ms=0.5, kernel="baseline", noise_scale=0.0
    )
    sched = jnp.zeros((12, init.dynamic.v.shape[0]), dtype=init.dynamic.v.dtype)
    key = jax.random.PRNGKey(17)
    ref = init._replace(
        prng_key=key,
        dynamic=init.dynamic._replace(
            H=jnp.ones_like(init.dynamic.H),
            w=init.dynamic.w,
        ),
    )
    pert = init._replace(
        prng_key=key,
        dynamic=init.dynamic._replace(
            H=jnp.full_like(init.dynamic.H, 3.0),
            w=init.dynamic.w * 1.75,
        ),
    )
    _, out_ref = _pipeline.run_continuation(step_fn, ref, sched)
    _, out_pert = _pipeline.run_continuation(step_fn, pert, sched)
    assert jnp.array_equal(out_ref[0], out_pert[0])
    assert jnp.array_equal(out_ref[1], out_pert[1])


def test_rbd_H_couples_to_V_m_when_beta_h_active():
    """Knockout: beta_h=0 vs beta_h>0 with perturbed H0 changes V_m trajectory."""
    jdtype = jnp.float32
    n = 2
    params = IzhikevichParams(
        v0=jnp.full((n,), -65.0, dtype=jdtype),
        u0=jnp.zeros((n,), dtype=jdtype),
        a=jnp.full((n,), 0.02, dtype=jdtype),
        b=jnp.full((n,), 0.2, dtype=jdtype),
        c=jnp.full((n,), -65.0, dtype=jdtype),
        d=jnp.full((n,), 8.0, dtype=jdtype),
        drive=jnp.full((n,), 8.0, dtype=jdtype),
        sign=jnp.ones((n,), dtype=jdtype),
        W=jnp.zeros((n, n), dtype=jdtype),
        source_scale=jnp.ones((n,), dtype=jdtype),
        labels=tuple("E" for _ in range(n)),
        layer_labels=tuple("L4" for _ in range(n)),
        source_calibration_status="uncalibrated_izhikevich_native_current",
    )
    edges = EdgeList(
        pre=jnp.asarray([0, 1], dtype=jnp.int32),
        post=jnp.asarray([1, 0], dtype=jnp.int32),
        weight=jnp.asarray([8.0, 8.0], dtype=jdtype),
        receptor_index=jnp.zeros((2,), dtype=jnp.int32),
        tau_ms=jnp.asarray([2.0, 2.0], dtype=jdtype),
    )
    key = jax.random.PRNGKey(0)
    shared = dict(noise_scale=0.0, rbd_family="f1", kappa_h=0.05)
    v_null, _, _, _ = simulate_edge_recurrent_izhikevich_rbd(
        params, edges, 200, 0.5, key, **shared, beta_h=0.0
    )
    v_active, _, _, _ = simulate_edge_recurrent_izhikevich_rbd(
        params,
        edges,
        200,
        0.5,
        key,
        **shared,
        beta_h=0.5,
        init_state={"H_final": jnp.asarray([1.2, 0.8], dtype=jdtype)},
    )
    delta = float(np.max(np.abs(np.asarray(v_null) - np.asarray(v_active))))
    assert delta > 1e-3


def test_hdp_w_couples_to_V_m_when_plasticity_active():
    """Knockout: null HDP vs active HDP changes V_m (w coordinate required)."""
    cfg = jtfne.suite2_net1_config(seed=3, n=8)
    off = jtfne.construct(cfg.runtime(enable_hdp=False, recurrent_backend="edge_list"))
    on = jtfne.construct(
        cfg.runtime(
            enable_hdp=True,
            recurrent_backend="edge_list",
            hdp_params={"K_HDP": 0.05, "alpha": 0.02, "gamma": 0.1},
        )
    )
    v_off = np.asarray(
        jtfne.simulate(off, duration_ms=20.0, dt_ms=0.5, seed=5).V_m
    )
    v_on = np.asarray(
        jtfne.simulate(on, duration_ms=20.0, dt_ms=0.5, seed=5).V_m
    )
    assert not np.array_equal(v_off, v_on)
    diag = on.last_hdp_diagnostics()
    w0 = np.asarray(off.params["edge_list"].weight)
    wf = np.asarray(diag["w_final"])
    assert float(np.max(np.abs(wf - w0))) > 1e-6


def test_registered_aux_and_theta_slots_exist_with_zero_default():
    """Cold-start carrier exposes theta_S and aux with shape (0,) when unused."""
    model = _suite2_model()
    state = _pipeline.dynamic_state_from_model(model)
    assert state.theta_S.shape == (0,)
    assert state.aux.shape == (0,)


def test_audit_json_matches_probe_inventory():
    """Machine-readable inventory stays aligned with probe classifications."""
    payload = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    assert payload["audit_id"] == "23-H-01"
    assert payload["classification"] == "REDUCTION_CANDIDATES_IDENTIFIED"
    assert len(payload["coordinates"]) == len(COORDINATE_INVENTORY)
    by_id = {row["id"]: row for row in payload["coordinates"]}
    assert by_id["dynamic.H"]["baseline_carrier"]["observable_dependence"] == "inert"
    assert by_id["dynamic.w"]["baseline_carrier"]["observable_dependence"] == "inert"
    assert by_id["dynamic.H"]["rbd_kernel"]["observable_dependence"] == "required_when_beta_or_kappa_active"
    assert by_id["dynamic.w"]["hdp_kernel"]["observable_dependence"] == "required_when_plasticity_active"
