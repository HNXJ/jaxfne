"""P-051: the named edge mechanism ``instantaneous`` (tau_ms 0), the tau -> 0 limit of the edge
route's filtered current. The synaptic drive at step t is the weighted sum of presynaptic spikes
at t-1, with no carry-over: the dense route's ``weights @ prev_spikes``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne as jtfne
from jaxfne._edge_class_storage import build_declared_mechanism_tau_table
from jaxfne.emitters import (
    materialize_dense_W_from_edge_list,
    resolve_edge_tau_ms,
    resolve_edge_weight,
    simulate_edge_recurrent_izhikevich,
    simulate_eig_izhikevich,
)

N = 40  # 32 E and 8 I cells at the 0.8 / 0.2 split
N_STEPS = 400
DT_MS = 0.5


def _ei_cfg(n=N, name="instantaneous", mech_params=None, kind="receptor"):
    mech_params = {"tau_ms": 0.0} if mech_params is None else mech_params
    return (
        jtfne.configuration()
        .network(name="inst", kind="balanced_ei_population", n=n, cell_types={"E": 0.8, "I": 0.2})
        .mechanisms(name=name, kind=kind, params=mech_params)
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=1.0, weight_range=(0.2, 0.6), sign="excitatory", mechanism=name)
        .connections(name="e_i", source={"cell_type": "E"}, target={"cell_type": "I"},
                     probability=1.0, weight_range=(0.3, 0.9), sign="excitatory", mechanism=name)
        .connections(name="i_e", source={"cell_type": "I"}, target={"cell_type": "E"},
                     probability=1.0, weight_range=(0.1, 0.4), sign="inhibitory", mechanism=name)
        .update_metadata(connectivity_mode="explicit")
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )


def test_instantaneous_mechanism_compiles_to_zero_tau_edges_with_weight_range():
    model = jtfne.construct(_ei_cfg())
    edges = model.params["edge_list"]
    labels = np.asarray(model.params["emitter"].labels)
    n_e, n_i = int(np.sum(labels == "E")), int(np.sum(labels == "I"))
    assert n_e > 0 and n_i > 0, "fixture must hold both E and I cells"
    assert int(edges.n_edges) == n_e * (n_e - 1) + n_e * n_i + n_i * n_e
    tau = np.asarray(resolve_edge_tau_ms(edges, jnp.float32))
    assert tau.shape == (int(edges.n_edges),)
    assert np.all(tau == 0.0), "every instantaneous edge carries tau 0"
    pre, post = np.asarray(edges.pre), np.asarray(edges.post)
    mag = np.abs(np.asarray(resolve_edge_weight(edges, jnp.float32)))
    for (pl, ql), (lo, hi) in {("E", "E"): (0.2, 0.6), ("E", "I"): (0.3, 0.9), ("I", "E"): (0.1, 0.4)}.items():
        sel = (labels[pre] == pl) & (labels[post] == ql)
        assert sel.any(), f"fixture must hold {pl}->{ql} edges"
        assert np.all((mag[sel] >= lo) & (mag[sel] < hi)), f"{pl}->{ql} magnitudes outside weight_range"
        assert np.ptp(mag[sel]) > 0.0, f"{pl}->{ql} weights must be drawn, not constant"


@pytest.mark.parametrize("mech_params", [{"tau_ms": 2.0}, {}])
def test_instantaneous_must_declare_tau_zero_exactly(mech_params):
    with pytest.raises(ValueError, match="declare tau_ms=0.0"):
        jtfne.construct(_ei_cfg(mech_params=mech_params))


def test_other_mechanisms_still_refuse_zero_tau():
    with pytest.raises(ValueError, match="finite and positive"):
        jtfne.construct(_ei_cfg(name="fast_inh", mech_params={"tau_ms": 0.0}))


def test_declared_tau_table_carries_the_instantaneous_zero():
    meta = {"circuit": {"mechanisms": [
        {"name": "instantaneous", "kind": "receptor", "params": {"tau_ms": 0.0}},
        {"name": "ampa", "kind": "receptor", "params": {"tau_ms": 2.0}},
    ]}}
    np.testing.assert_array_equal(build_declared_mechanism_tau_table(meta), [0.0, 2.0])


def _f64_params(params):
    fields = ("a", "b", "c", "d", "drive", "sign", "v0", "u0", "source_scale")
    return dataclasses.replace(
        params, **{f: jnp.asarray(getattr(params, f), dtype=jnp.float64) for f in fields}
    )


@pytest.mark.x64
def test_instantaneous_edge_route_matches_dense_route():
    model = jtfne.construct(_ei_cfg())
    params = _f64_params(model.params["emitter"])
    edges = model.params["edge_list"]
    n = int(params.n_neurons)
    W = materialize_dense_W_from_edge_list(edges, n, dtype=jnp.float64, presynaptic_sign=params.sign)
    assert int(np.count_nonzero(np.asarray(W))) == int(edges.n_edges), "fixture: one edge per (pre, post)"
    key = jax.random.PRNGKey(7)

    v_dense, s_dense, _ = simulate_eig_izhikevich(
        dataclasses.replace(params, W=W), N_STEPS, DT_MS, key, dtype="float64"
    )
    v_edge, s_edge, _, _ = simulate_edge_recurrent_izhikevich(
        params, edges, N_STEPS, DT_MS, key, dtype="float64"
    )
    assert int(np.asarray(s_dense).sum()) > 0, "fixture must spike"
    np.testing.assert_array_equal(np.asarray(s_edge), np.asarray(s_dense))
    np.testing.assert_allclose(np.asarray(v_edge), np.asarray(v_dense), rtol=1e-10)

    # Control: the same edges with a 5 ms filter are a different network, so the match is not vacuous.
    n_edges = int(edges.n_edges)
    filtered = dataclasses.replace(
        edges, tau_storage="per_edge", mechanism_tau_table=None,
        tau_ms=jnp.full((n_edges,), 5.0, dtype=jnp.float64),
    )
    v_filt, _, _, _ = simulate_edge_recurrent_izhikevich(
        params, filtered, N_STEPS, DT_MS, key, dtype="float64"
    )
    assert not np.allclose(np.asarray(v_filt), np.asarray(v_dense), rtol=1e-10), \
        "a 5 ms filter must change the trace"


def test_a_presynaptic_spike_drives_exactly_one_step():
    cfg = (
        jtfne.configuration()
        .network(name="pair", kind="balanced_ei_population", n=2, cell_types={"E": 1.0})
        .mechanisms(name="instantaneous", kind="receptor", params={"tau_ms": 0.0})
        .connections(name="e_e", source={"cell_type": "E"}, target={"cell_type": "E"},
                     probability=1.0, weight_range=(0.2, 0.6), sign="excitatory", mechanism="instantaneous")
        .update_metadata(connectivity_mode="explicit")
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="declared_proxy", gauge="mean_zero")
        .probe(name="spike_probe", modes=["spikes"])
    )
    model = jtfne.construct(cfg)
    edges = model.params["edge_list"]
    params = dataclasses.replace(model.params["emitter"], drive=jnp.zeros((2,), dtype=jnp.float32))
    n_steps, s = 60, 10
    sched = np.zeros((n_steps, 2), dtype=np.float32)
    sched[s, 0] = 300.0  # neuron 0 is forced to spike once, at step s
    key = jax.random.PRNGKey(0)

    def run(e):
        _, spikes, _, state = simulate_edge_recurrent_izhikevich(
            params, e, n_steps, DT_MS, key, drive_schedule=jnp.asarray(sched),
            noise_scale=0.0, record_edge_current=True,
        )
        return np.asarray(spikes), np.asarray(state["edge_current_trace"])

    spk, cur = run(edges)
    assert spk[:, 0].sum() == 1 and spk[s, 0] == 1, "fixture: neuron 0 spikes exactly once, at step s"
    from_0 = np.asarray(edges.pre) == 0
    assert from_0.any(), "fixture must hold edges from neuron 0"
    w = np.asarray(resolve_edge_weight(edges, jnp.float32))
    np.testing.assert_array_equal(cur[s, from_0], 0.0)  # the spike has not arrived yet
    np.testing.assert_array_equal(cur[s + 1, from_0], w[from_0])  # one step later, the full weight
    np.testing.assert_array_equal(cur[s + 2:, from_0], 0.0)  # and no carry-over afterwards

    # Control: a 5 ms filter on the same edges does carry the spike forward.
    filtered = dataclasses.replace(
        edges, tau_storage="per_edge", mechanism_tau_table=None,
        tau_ms=jnp.full((int(edges.n_edges),), 5.0, dtype=jnp.float32),
    )
    _, cur5 = run(filtered)
    assert np.all(cur5[s + 2, from_0] > 0.0), "a 5 ms filter must carry the spike past one step"
