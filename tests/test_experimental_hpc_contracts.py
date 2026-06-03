import pytest
import jax.numpy as jnp

from jaxfne.experimental_hpc import (
    Config,
    RuntimeStatic,
    FlatNet,
    NodeIdentity,
    Paradigm,
    compile_connection_rules,
)


def test_node_identity_quartet_six_digits():
    node = NodeIdentity(global_id=7, area="V1", area_id="V1", local_id=7, layer="L4", cell_type="PV")
    assert node.quartet == "V1:000007:L4:PV"


def test_runtime_static_validate_smoke():
    report = RuntimeStatic(n_steps=10, dt_ms=0.1).validate()
    assert report["valid"] is True
    assert report["dtype"] == "float32"


def test_flatnet_is_pytree_roundtrip():
    flat = FlatNet(
        neuron_params=jnp.zeros((2, 4), dtype=jnp.float32),
        positions=jnp.zeros((2, 3), dtype=jnp.float32),
        area_id=jnp.zeros((2,), dtype=jnp.int32),
        layer_id=jnp.zeros((2,), dtype=jnp.int32),
        cell_type_id=jnp.zeros((2,), dtype=jnp.int32),
        edge_pre=jnp.zeros((0,), dtype=jnp.int32),
        edge_post=jnp.zeros((0,), dtype=jnp.int32),
        edge_weight=jnp.zeros((0,), dtype=jnp.float32),
        edge_mechanism=jnp.zeros((0,), dtype=jnp.int32),
        probe_positions=jnp.zeros((4, 3), dtype=jnp.float32),
        static={"schema_version": "test"},
    )
    leaves, treedef = __import__("jax").tree_util.tree_flatten(flat)
    out = __import__("jax").tree_util.tree_unflatten(treedef, leaves)
    assert out.n_nodes == 2
    assert out.n_edges == 0
    assert out.static["schema_version"] == "test"


def test_tbi_methods_fail_loudly():
    with pytest.raises(NotImplementedError, match=">TBI-not-ready"):
        Config().validate()
    with pytest.raises(NotImplementedError, match=">TBI-not-ready"):
        Paradigm.constant_dc(target={"area": "V1"}, amplitude=1.0, duration_ms=10.0)
    with pytest.raises(NotImplementedError, match=">TBI-not-ready"):
        compile_connection_rules([], [], {})
