
import json

import jax
import jax.numpy as jnp

import jaxfne as jtfne


def _cfg(n=6):
    return (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n, cell_types={"E": 0.7, "PV": 0.2, "SST": 0.1})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )


def test_edge_list_export_and_json_safety():
    model = jtfne.construct(_cfg())
    edges = model.params["edge_list"]
    assert isinstance(edges, jtfne.EdgeList)
    assert edges.n_edges > 0
    payload = edges.to_dict()
    assert payload["backend"] == "edge_list_recurrent_v0.0.9"
    assert payload["physical_amplitude_claim_allowed"] is False
    json.dumps(payload, allow_nan=False)


def test_edge_list_is_jax_pytree():
    model = jtfne.construct(_cfg())
    leaves, treedef = jax.tree_util.tree_flatten(model.params["edge_list"])
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, jtfne.EdgeList)
    assert rebuilt.n_edges == model.params["edge_list"].n_edges


def test_make_edge_list_from_dense_preserves_dense_synapse_direction():
    W = jnp.array([[0.0, 2.0], [-3.0, 0.0]], dtype=jnp.float32)
    edges = jtfne.make_edge_list_from_dense(W)
    assert edges.n_edges == 2
    assert set(map(float, list(edges.weight))) == {2.0, -3.0}
    assert set(map(int, list(edges.receptor_index))) == {0, 1}


def test_edge_recurrent_simulation_shapes_and_truth_status():
    model = jtfne.construct(_cfg(n=5))
    rt = jtfne.runtime(jit=True, recurrent_backend="edge_list", seed=10)
    signals = model.simulate(jtfne.simulation(duration_ms=3.0, dt_ms=0.1, seed=10, runtime=rt))
    assert signals.V_m.shape == (30, 5)
    assert signals.spikes.shape == (30, 5)
    assert signals.sources.shape == (30, 5)
    assert signals.metadata["recurrent_backend"] == "edge_list"
    assert signals.metadata["field_claim_level"] == "proxy_readout_only"
    assert signals.summary()["field_claim_level"] == "proxy_readout_only"


def test_edge_recurrent_batch_vmap_metadata():
    model = jtfne.construct(_cfg(n=4))
    rt = jtfne.runtime(jit=True, vmap=True, recurrent_backend="edge_list", seed=11)
    batch = model.simulate_batch(jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=11, runtime=rt), n_seeds=3)
    assert batch["V_m"].shape == (3, 20, 4)
    assert batch["metadata"]["batch_status"] == "vmap_seed_batch_v0.0.9"
    assert batch["metadata"]["physical_amplitude_claim_allowed"] is False
    json.dumps(batch["metadata"], allow_nan=False)


def test_dense_and_edge_backend_do_not_change_truth_gates():
    model = jtfne.construct(_cfg(n=5))
    dense = model.simulate(jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=12, runtime=jtfne.runtime(recurrent_backend="dense")))
    edge = model.simulate(jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=12, runtime=jtfne.runtime(recurrent_backend="edge_list")))
    assert dense.metadata["field_claim_level"] == "proxy_readout_only"
    assert edge.metadata["field_claim_level"] == "proxy_readout_only"
    assert edge.metadata["runtime"]["recurrent_backend"] == "edge_list"
