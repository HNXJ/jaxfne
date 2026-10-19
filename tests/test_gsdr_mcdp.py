"""GSDR MCDP core (P-039): factors match the Fig. 6 oracle; transform gates and reverts."""

import numpy as np
import pytest


def _oracle_factors(v, pre, post, shift):
    """Hand transcription of the lane oracle math (PORT_SPEC section 10)."""
    v = np.asarray(v, dtype=np.float64)
    pre = np.asarray(pre)
    post = np.asarray(post)
    pv = v[:, :-shift][:, :, pre] if v.ndim == 3 else v[:-shift, pre]
    qv = v[:, shift:][:, :, post] if v.ndim == 3 else v[shift:, post]
    pv = pv - pv.mean(axis=-2, keepdims=True)
    qv = qv - qv.mean(axis=-2, keepdims=True)
    num = (pv * qv).sum(axis=-2)
    den = np.sqrt((pv ** 2).sum(axis=-2) * (qv ** 2).sum(axis=-2)) + 1e-6
    corr = (num / den) ** 2
    if corr.ndim == 2:
        corr = corr.mean(axis=0)
    return (corr - corr.mean()) / (corr.std() + 1e-6)


def _oracle_smooth(x, k):
    taps = np.arange(-k, k + 1, dtype=np.float64)
    w = np.exp(-(taps ** 2) / (2.0 * (k / 2.0) ** 2))
    w = w / w.sum()
    return np.convolve(np.pad(x, k, mode="edge"), w, mode="valid")


def test_mcdp_factors_match_oracle_3d():
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaxfne.optim.gsdr import mcdp_factors

    rng = np.random.default_rng(7)
    v = rng.standard_normal((2, 60, 6))
    pre = np.asarray([0, 1, 2, 3])
    post = np.asarray([2, 3, 4, 5])
    got = np.asarray(mcdp_factors(v, pre, post, 10))
    want = _oracle_factors(v, pre, post, 10)
    assert got.shape == want.shape == (4,)
    np.testing.assert_allclose(got, want, rtol=1e-5)


def test_mcdp_factors_match_oracle_2d():
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaxfne.optim.gsdr import mcdp_factors

    rng = np.random.default_rng(11)
    v = rng.standard_normal((60, 6))
    pre = np.asarray([0, 2, 4])
    post = np.asarray([1, 3, 5])
    np.testing.assert_allclose(
        np.asarray(mcdp_factors(v, pre, post, 10)),
        _oracle_factors(v, pre, post, 10),
        rtol=1e-5,
    )


def test_mcdp_factors_zscored_per_group_like_oracle():
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaxfne.optim.gsdr import mcdp_factors

    rng = np.random.default_rng(23)
    v = rng.standard_normal((2, 60, 8))
    pre_a = np.asarray([0, 1, 2])
    post_a = np.asarray([3, 4, 5])
    pre_g = np.asarray([1, 2, 6])
    post_g = np.asarray([0, 5, 7])
    got_a = np.asarray(mcdp_factors(v, pre_a, post_a, 10))
    got_g = np.asarray(mcdp_factors(v, pre_g, post_g, 10))
    np.testing.assert_allclose(got_a, _oracle_factors(v, pre_a, post_a, 10), rtol=1e-5)
    np.testing.assert_allclose(got_g, _oracle_factors(v, pre_g, post_g, 10), rtol=1e-5)
    joint = np.asarray(
        mcdp_factors(
            v,
            np.concatenate([pre_a, pre_g]),
            np.concatenate([post_a, post_g]),
            10,
        )
    )
    assert joint.shape == (6,)
    assert float(np.abs(joint - np.concatenate([got_a, got_g])).max()) > 1e-3


def test_mcdp_factors_smoothing_matches_hand_gaussian():
    import jax

    jax.config.update("jax_enable_x64", True)
    from jaxfne.optim.gsdr import mcdp_factors

    rng = np.random.default_rng(3)
    v = rng.standard_normal((2, 60, 8))
    pre = np.arange(6)
    post = np.arange(2, 8)
    v64 = np.asarray(v, dtype=np.float64)
    pv = v64[:, :-10][:, :, pre]
    qv = v64[:, 10:][:, :, post]
    pv = pv - pv.mean(axis=-2, keepdims=True)
    qv = qv - qv.mean(axis=-2, keepdims=True)
    num = (pv * qv).sum(axis=-2)
    den = np.sqrt((pv ** 2).sum(axis=-2) * (qv ** 2).sum(axis=-2)) + 1e-6
    corr = ((num / den) ** 2).mean(axis=0)
    want = (_oracle_smooth(corr, 2) - _oracle_smooth(corr, 2).mean()) / (
        _oracle_smooth(corr, 2).std() + 1e-6
    )
    np.testing.assert_allclose(
        np.asarray(mcdp_factors(v, pre, post, 10, smooth_k=2)), want, rtol=1e-5
    )


def test_mcdp_factors_zero_variance_edges_read_one():
    from jaxfne.optim.gsdr import mcdp_factors

    v = np.full((40, 5), 3.0)
    got = np.asarray(mcdp_factors(v, np.asarray([0, 1]), np.asarray([2, 3]), 10))
    np.testing.assert_allclose(got, np.ones(2))


def test_mcdp_transform_requires_explicit_key():
    pytest.importorskip("optax")
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    transform = gsdr_mcdp_transform()
    params = jnp.asarray([1.0, 2.0])
    state = transform.init(params)
    try:
        transform.update(
            jnp.asarray([0.1, 0.1]), state, params=params, loss=1.0, key=None
        )
    except ValueError:
        return
    raise AssertionError("update without key must raise ValueError")


def test_mcdp_transform_refuses_invalid_hyperparams():
    pytest.importorskip("optax")
    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    for kwargs in (
        {"deselection_threshold": 0},
        {"checkpoint_n": 0},
        {"a_init": 1.5},
        {"a_init": -0.1},
        {"lambda_d": 0.0},
        {"stochastic_scale": -1.0},
        {"a_step": -0.1},
        {"a_step": 1.5},
        {"w_max": 0.0},
        {"w_max": -2.0},
    ):
        try:
            gsdr_mcdp_transform(**kwargs)
        except ValueError:
            continue
        raise AssertionError("gsdr_mcdp_transform(%r) must raise" % (kwargs,))


def test_mcdp_transform_stagnation_reset_reverts_to_best():
    optax = pytest.importorskip("optax")
    import jax
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    key = jax.random.PRNGKey(0)
    transform = gsdr_mcdp_transform(
        checkpoint_n=3, stochastic_scale=0.0, a_init=0.0, a_dynamic=False
    )
    params = jnp.asarray([1.0])
    state = transform.init(params)
    best = None
    for step, loss in enumerate((10.0, 11.0, 12.0, 13.0)):
        if step == 0:
            best = params
        updates, state = transform.update(
            jnp.asarray([1.0]),
            state,
            params=params,
            loss=loss,
            key=jax.random.fold_in(key, step),
            mcdp=jnp.asarray([1.0]),
        )
        params = optax.apply_updates(params, updates)
    gsdr_state, _ = state
    assert gsdr_state.reset_counter == 1
    assert float(params[0]) == float(best[0])


def test_mcdp_transform_nan_loss_reverts_to_best():
    optax = pytest.importorskip("optax")
    import jax
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    key = jax.random.PRNGKey(2)
    transform = gsdr_mcdp_transform(stochastic_scale=0.0)
    params = jnp.asarray([1.0])
    state = transform.init(params)
    updates, state = transform.update(
        jnp.asarray([0.5]), state, params=params, loss=5.0, key=key,
        mcdp=jnp.asarray([1.0]),
    )
    params = optax.apply_updates(params, updates)
    best = state[0].best_param
    updates, state = transform.update(
        jnp.asarray([0.5]),
        state,
        params=params,
        loss=float("nan"),
        key=jax.random.fold_in(key, 1),
        mcdp=jnp.asarray([1.0]),
    )
    params = optax.apply_updates(params, updates)
    assert float(params[0]) == float(jnp.asarray(best)[0])


def test_mcdp_none_voids_epoch_with_zero_updates():
    pytest.importorskip("optax")
    import jax
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    transform = gsdr_mcdp_transform()
    params = jnp.asarray([1.0, -2.0])
    state = transform.init(params)
    updates, state2 = transform.update(
        jnp.asarray([0.5, 0.5]),
        state,
        params=params,
        loss=1.0,
        key=jax.random.PRNGKey(0),
        mcdp=None,
    )
    np.testing.assert_allclose(np.asarray(updates), np.zeros(2))
    assert state2[0].step == state[0].step


def test_mcdp_factors_gate_the_stochastic_delta():
    pytest.importorskip("optax")
    import jax
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    transform = gsdr_mcdp_transform(
        stochastic_scale=1.0, a_init=1.0, a_dynamic=False
    )
    params = jnp.asarray([1.0, 2.0, 4.0])
    zero = jnp.zeros(3)
    key = jax.random.PRNGKey(9)
    _, state = transform.update(
        zero, transform.init(params), params=params, loss=3.0, key=key,
        mcdp=jnp.zeros(3),
    )
    updates_off, _ = transform.update(
        zero, state, params=params, loss=4.0, key=jax.random.fold_in(key, 1),
        mcdp=jnp.zeros(3),
    )
    np.testing.assert_allclose(np.asarray(updates_off), np.zeros(3))
    updates_on, _ = transform.update(
        zero, state, params=params, loss=4.0, key=jax.random.fold_in(key, 1),
        mcdp=jnp.ones(3),
    )
    assert bool((np.asarray(updates_on) != 0.0).any())


def test_a_step_drives_walk():
    pytest.importorskip("optax")
    import jax
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    key = jax.random.PRNGKey(4)
    params = jnp.asarray([1.0, 2.0])
    frozen = gsdr_mcdp_transform(a_init=0.5, a_dynamic=True, a_step=0.0)
    _, s_frozen = frozen.update(
        jnp.zeros(2), frozen.init(params), params=params, loss=2.0, key=key,
        mcdp=jnp.ones(2),
    )
    assert s_frozen[0].a == 0.5
    assert s_frozen[0].a_proposed == 0.5
    walked = gsdr_mcdp_transform(a_init=0.5, a_dynamic=True, a_step=0.4)
    _, s_walked = walked.update(
        jnp.zeros(2), walked.init(params), params=params, loss=2.0, key=key,
        mcdp=jnp.ones(2),
    )
    assert abs(s_walked[0].a - 0.5) <= 0.4
    assert s_walked[0].a_proposed != 0.5


def test_dynamic_a_kept_only_on_improvement():
    pytest.importorskip("optax")
    import jax
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    transform = gsdr_mcdp_transform(a_init=0.5, a_dynamic=True, a_step=0.1)
    params = jnp.asarray([1.0])
    key = jax.random.PRNGKey(5)
    _, s1 = transform.update(
        jnp.zeros(1), transform.init(params), params=params, loss=5.0, key=key,
        mcdp=jnp.ones(1),
    )
    accepted1 = s1[0].a
    _, s2 = transform.update(
        jnp.zeros(1), s1, params=params, loss=6.0,
        key=jax.random.fold_in(key, 1), mcdp=jnp.ones(1),
    )
    assert s2[0].a == accepted1
    _, s3 = transform.update(
        jnp.zeros(1), s2, params=params, loss=4.0,
        key=jax.random.fold_in(key, 2), mcdp=jnp.ones(1),
    )
    assert s3[0].a == s2[0].a_proposed


def test_w_max_clamps_params_plus_update():
    pytest.importorskip("optax")
    import jax
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    key = jax.random.PRNGKey(6)
    params = jnp.asarray([1.0, 9.0])
    grads = jnp.asarray([100.0, -100.0])
    free = gsdr_mcdp_transform(a_init=0.0, a_dynamic=False)
    updates_free, _ = free.update(
        grads, free.init(params), params=params, loss=1.0, key=key,
        mcdp=jnp.ones(2),
    )
    clamped = gsdr_mcdp_transform(a_init=0.0, a_dynamic=False, w_max=8.0)
    updates_cl, _ = clamped.update(
        grads, clamped.init(params), params=params, loss=1.0, key=key,
        mcdp=jnp.ones(2),
    )
    new = np.asarray(params + updates_cl)
    assert bool((new >= 0.0).all() and (new <= 8.0).all())
    assert bool((np.asarray(updates_cl) != np.asarray(updates_free)).any())


def test_stochastic_scale_changes_step():
    pytest.importorskip("optax")
    import jax
    import jax.numpy as jnp

    from jaxfne.optim.gsdr import gsdr_mcdp_transform

    key = jax.random.PRNGKey(8)
    params = jnp.asarray([1.0, 2.0])
    one = gsdr_mcdp_transform(a_init=1.0, a_dynamic=False, stochastic_scale=1.0)
    two = gsdr_mcdp_transform(a_init=1.0, a_dynamic=False, stochastic_scale=2.0)
    updates_one, _ = one.update(
        jnp.zeros(2), one.init(params), params=params, loss=1.0, key=key,
        mcdp=jnp.ones(2),
    )
    updates_two, _ = two.update(
        jnp.zeros(2), two.init(params), params=params, loss=1.0, key=key,
        mcdp=jnp.ones(2),
    )
    np.testing.assert_allclose(np.asarray(updates_two), 2.0 * np.asarray(updates_one))
