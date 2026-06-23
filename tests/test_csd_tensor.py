"""CSD tensor: spatial second-derivative readout stage, factored out of
project_laminar_sources (jaxfne/fields/proxy.py) so it is independently
named, testable, and reusable.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import jaxfne as jtfne
from jaxfne.fields import csd_tensor


def test_csd_tensor_matches_project_laminar_sources_inline_computation():
    rng = np.random.default_rng(0)
    T, N = 200, 30
    sources = rng.normal(size=(T, N)).astype(np.float32)
    positions = np.stack([np.zeros(N), np.zeros(N), np.linspace(0, 1, N)], axis=1)

    fo = jtfne.project_laminar_sources(sources, positions, n_contacts=16)
    dz = fo.contact_depths[1] - fo.contact_depths[0]
    standalone = csd_tensor(fo.phi_e_proxy, dz)

    np.testing.assert_allclose(np.asarray(fo.csd_proxy), np.asarray(standalone))


def test_csd_tensor_shape_and_finite():
    rng = np.random.default_rng(1)
    T, C = 64, 10
    phi = jnp.asarray(rng.normal(size=(T, C)).astype(np.float32))
    csd = csd_tensor(phi, dz=0.1)

    assert csd.shape == phi.shape
    assert jnp.all(jnp.isfinite(csd))


def test_csd_tensor_fewer_than_3_contacts_returns_zeros():
    phi = jnp.ones((10, 2), dtype=jnp.float32)
    csd = csd_tensor(phi, dz=0.1)
    assert jnp.all(csd == 0.0)


def test_csd_tensor_flat_potential_gives_zero_csd():
    """A spatially flat (constant across contacts) potential has zero 2nd derivative."""
    phi = jnp.ones((5, 8), dtype=jnp.float32) * 3.7
    csd = csd_tensor(phi, dz=0.1)
    np.testing.assert_allclose(np.asarray(csd), np.zeros((5, 8)), atol=1e-5)


def test_csd_tensor_exported_at_top_level():
    assert jtfne.csd_tensor is csd_tensor
