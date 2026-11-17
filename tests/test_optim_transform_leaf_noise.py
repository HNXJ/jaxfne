"""The SDR-family Optax transforms draw independent noise per parameter leaf (ISSUE_LOG P-059).

With zero gradients the inner optimizer contributes nothing, so each update is the stochastic term
alone. Two leaves of equal shape must then receive different noise; before the fix one key served
every leaf and equal-shape leaves moved in lockstep.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from jaxfne.optim.core import _leafwise_normal, agsdr_transform, gsdr_transform, sdr_transform

FACTORIES = {"sdr": sdr_transform, "gsdr": gsdr_transform, "agsdr": agsdr_transform}


def _params():
    return {
        "w1": jnp.zeros((4,)),
        "w2": jnp.ones((4,)),
        "nested": {"w3": jnp.full((4,), 2.0), "b": jnp.zeros((3,))},
        "m": jnp.zeros((2, 3)),
    }


def _run(name, steps=2):
    params = _params()
    grads = jax.tree_util.tree_map(jnp.zeros_like, params)
    tx = FACTORIES[name](inner_optimizer=optax.sgd(0.1))
    state = tx.init(params)
    key = jax.random.PRNGKey(7)
    out = []
    for _ in range(steps):
        upd, state = tx.update(grads, state, params, key=key)
        out.append(upd)
    return out


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_equal_shape_leaves_get_independent_noise(name):
    for upd in _run(name):
        w1, w2, w3 = upd["w1"], upd["w2"], upd["nested"]["w3"]
        assert bool(jnp.any(w1 != 0)), "fixture must produce a nonzero stochastic term"
        assert not bool(jnp.array_equal(w1, w2))
        assert not bool(jnp.array_equal(w1, w3))
        assert not bool(jnp.array_equal(w2, w3))


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_structure_shape_dtype_kept(name):
    params = _params()
    upd = _run(name, steps=1)[0]
    assert jax.tree_util.tree_structure(upd) == jax.tree_util.tree_structure(params)
    for u, p in zip(jax.tree_util.tree_leaves(upd), jax.tree_util.tree_leaves(params)):
        assert u.shape == p.shape and u.dtype == p.dtype


@pytest.mark.parametrize("name", sorted(FACTORIES))
def test_fixed_seed_replays_and_steps_differ(name):
    a, b = _run(name), _run(name)
    for ua, ub in zip(a, b):
        for x, y in zip(jax.tree_util.tree_leaves(ua), jax.tree_util.tree_leaves(ub)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
    assert not bool(jnp.array_equal(a[0]["w1"], a[1]["w1"])), "step counter must re-key the noise"


def test_leafwise_noise_jit_matches_eager():
    # The transforms' update() is not jittable (state dataclasses are not pytrees); the noise draw is.
    tree = _params()
    key = jax.random.PRNGKey(3)
    eager = _leafwise_normal(key, tree)
    jitted = jax.jit(_leafwise_normal)(key, tree)
    for x, y in zip(jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(jitted)):
        np.testing.assert_allclose(np.asarray(x), np.asarray(y), rtol=1e-6, atol=1e-7)


def test_leafwise_noise_passes_non_array_leaves():
    tree = {"a": jnp.zeros((2,)), "s": "tag"}
    out = _leafwise_normal(jax.random.PRNGKey(0), tree)
    assert out["s"] == "tag" and out["a"].shape == (2,)
