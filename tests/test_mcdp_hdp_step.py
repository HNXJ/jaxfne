"""MCDP as an outer-loop HDP step, simple form (Hamm 2026-10-20, P-039): W' = clip(W + eta * xi * W * m)."""

import jax
import numpy as np
import pytest
from numpy.testing import assert_allclose

from jaxfne.optim.gsdr import mcdp_hdp_step

W = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float32)
M = np.array([-1.5, 0.25, 1.0, 2.0], dtype=np.float32)


def test_step_is_eta_xi_w_m():
    key = jax.random.PRNGKey(3)
    xi = np.asarray(jax.random.normal(key, W.shape, dtype=W.dtype))
    assert np.all(xi != 0), "fixture must draw non-zero noise"
    out = np.asarray(mcdp_hdp_step(W, M, key, eta=0.3))
    assert_allclose(out, np.clip(W + 0.3 * xi * W * M, 0.0, None), rtol=1e-6)


def test_factors_change_the_update():
    key = jax.random.PRNGKey(3)
    a = np.asarray(mcdp_hdp_step(W, M, key, eta=0.3))
    b = np.asarray(mcdp_hdp_step(W, 2.0 * M, key, eta=0.3))
    assert not np.allclose(a, b)


def test_zero_eta_and_zero_factors_are_identity():
    key = jax.random.PRNGKey(0)
    assert_allclose(np.asarray(mcdp_hdp_step(W, M, key, eta=0.0)), W)
    assert_allclose(np.asarray(mcdp_hdp_step(W, np.zeros_like(M), key, eta=0.3)), W)


def test_seed_changes_the_update():
    a = np.asarray(mcdp_hdp_step(W, M, jax.random.PRNGKey(1), eta=0.3))
    b = np.asarray(mcdp_hdp_step(W, M, jax.random.PRNGKey(2), eta=0.3))
    assert not np.allclose(a, b)


def test_w_max_bounds_the_weights():
    out = np.asarray(mcdp_hdp_step(W, 50.0 * M, jax.random.PRNGKey(5), eta=1.0, w_max=3.0))
    assert out.min() >= 0.0 and out.max() <= 3.0
    unclipped = W + 1.0 * np.asarray(jax.random.normal(jax.random.PRNGKey(5), W.shape)) * W * 50.0 * M
    assert unclipped.max() > 3.0 or unclipped.min() < 0.0, "fixture must hit a bound"


@pytest.mark.parametrize("kwargs", [dict(eta=-0.1), dict(eta=0.1, w_max=0.0)])
def test_refuses_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        mcdp_hdp_step(W, M, jax.random.PRNGKey(0), **kwargs)


def test_refuses_shape_mismatch():
    with pytest.raises(ValueError):
        mcdp_hdp_step(W, M[:3], jax.random.PRNGKey(0), eta=0.1)
