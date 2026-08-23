
import json

import jax
import jax.numpy as jnp
import pytest

import jaxfne as jtfne

model = None  # set lazily by _get_model for rejection tests


def _get_model():
    global model
    if model is None:
        model = jtfne.construct(_cfg())
    return model


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
    assert payload["physical_amplitude_calibrated"] is False
    json.dumps(payload, allow_nan=False)


def test_edge_list_roundtrip_bit_exact():
    """Full to_dict/from_dict cycle restores every array bit-exactly."""
    import numpy as np
    from dataclasses import replace

    model = jtfne.construct(_cfg())
    edges = model.params["edge_list"]
    delayed = replace(
        edges,
        delay_steps=jax.random.randint(jax.random.PRNGKey(0), (edges.n_edges,), 0, 4, dtype=jnp.int32),
    )
    restored = type(edges).from_dict(delayed.to_dict())
    for name in ("pre", "post", "weight", "receptor_index", "tau_ms", "delay_steps"):
        a = np.asarray(getattr(delayed, name))
        b = np.asarray(getattr(restored, name))
        assert a.dtype == b.dtype, f"{name}: dtype drift {a.dtype} vs {b.dtype}"
        assert np.array_equal(a, b), f"{name}: values differ after roundtrip"
    assert restored.source_calibration_status == delayed.source_calibration_status


def test_edge_list_roundtrip_survives_json_text():
    """The documented persistence path is JSON text; roundtrip through it."""
    import numpy as np

    model = jtfne.construct(_cfg())
    edges = model.params["edge_list"]
    payload = json.loads(json.dumps(edges.to_dict(), allow_nan=False))
    restored = type(edges).from_dict(payload)
    assert np.array_equal(np.asarray(edges.weight), np.asarray(restored.weight))
    assert np.array_equal(np.asarray(edges.tau_ms), np.asarray(restored.tau_ms))


def test_edge_list_from_dict_rejects_summary_only_payload():
    with pytest.raises(ValueError, match="cannot be reconstructed"):
        type(_get_model().params["edge_list"]).from_dict(
            {"backend": "edge_list_recurrent_v0.0.9", "n_edges": 3}
        )


def test_edge_list_from_dict_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported EdgeList payload backend"):
        type(_get_model().params["edge_list"]).from_dict({"backend": "something_else"})


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
    assert signals.metadata["field_claim_level"] == "proxy_readout"
    assert signals.summary()["field_claim_level"] == "proxy_readout"


def test_edge_recurrent_batch_vmap_metadata():
    model = jtfne.construct(_cfg(n=4))
    rt = jtfne.runtime(jit=True, vmap=True, recurrent_backend="edge_list", seed=11)
    batch = model.simulate_batch(jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=11, runtime=rt), n_seeds=3)
    assert batch["V_m"].shape == (3, 20, 4)
    assert batch["metadata"]["batch_status"] == "vmap_seed_batch_v0.0.9"
    assert batch["metadata"]["physical_amplitude_calibrated"] is False
    json.dumps(batch["metadata"], allow_nan=False)


def test_dense_and_edge_backend_do_not_change_truth_gates():
    model = jtfne.construct(_cfg(n=5))
    dense = model.simulate(jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=12, runtime=jtfne.runtime(recurrent_backend="dense")))
    edge = model.simulate(jtfne.simulation(duration_ms=2.0, dt_ms=0.1, seed=12, runtime=jtfne.runtime(recurrent_backend="edge_list")))
    assert dense.metadata["field_claim_level"] == "proxy_readout"
    assert edge.metadata["field_claim_level"] == "proxy_readout"
    assert edge.metadata["runtime"]["recurrent_backend"] == "edge_list"
