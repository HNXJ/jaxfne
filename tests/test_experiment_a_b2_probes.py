"""0.4.17-B B2 — independent probe operators and lfp_proxy_probe fix."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxfne.experiment_a.canonical import freeze_canonical_dataset
from jaxfne.experiment_a.observe import (
    apply_independent_probe,
    materialize_field,
    verify_b2_invariants,
    write_b2_receipt,
)
from jaxfne.fields import lfp_proxy_probe, sample_phi_at_probe_depths


def test_lfp_proxy_probe_pass_through_without_contact_depths():
    phi = jnp.ones((10, 4), dtype=jnp.float32)
    out = lfp_proxy_probe(phi)
    np.testing.assert_array_equal(np.asarray(out.data), np.asarray(phi))


def test_lfp_proxy_probe_resamples_at_contact_depths():
    t, c = 20, 8
    field_z = jnp.linspace(0.0, 1.0, c, dtype=jnp.float32)
    # phi increases with depth at each time row
    phi = (field_z[None, :] + 0.1) * jnp.arange(1, t + 1, dtype=jnp.float32)[:, None]
    shallow = lfp_proxy_probe(phi, contact_depths=jnp.asarray([0.1]), field_contact_depths=field_z)
    deep = lfp_proxy_probe(phi, contact_depths=jnp.asarray([0.9]), field_contact_depths=field_z)
    assert shallow.data.shape == (t, 1)
    assert float(shallow.data[5, 0]) < float(deep.data[5, 0])


def test_sample_phi_at_probe_depths_superposition_not_required():
    phi = jnp.array([[0.0, 1.0, 2.0]], dtype=jnp.float32)
    field_z = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    y = sample_phi_at_probe_depths(phi, field_z, jnp.asarray([0.5]))
    np.testing.assert_allclose(np.asarray(y), [[1.0]], rtol=1e-5)


def test_b2_q_invariant_probe_distinct_on_canonical():
    ds = freeze_canonical_dataset(duration_ms=80.0, package_head="test_b2")
    results = verify_b2_invariants(ds)
    assert results["q_hash_invariant"]
    assert results["probe_distinct"]
    assert results["Q_zero_implies_Y_zero"]


def test_b2_factorized_path_does_not_touch_Q():
    ds = freeze_canonical_dataset(duration_ms=40.0, package_head="test_b2")
    q0 = ds.Q.copy()
    field = materialize_field(ds)
    apply_independent_probe(ds, field, "lfp_contact_shallow")
    apply_independent_probe(ds, field, "lfp_contact_deep")
    np.testing.assert_array_equal(ds.Q, q0)


def test_b2_receipt_writes(tmp_path):
    ds = freeze_canonical_dataset(duration_ms=40.0, package_head="test")
    receipt = write_b2_receipt(verify_b2_invariants(ds), tmp_path / "b2.json")
    assert receipt["checkpoint"] == "B2"
    loaded = json.loads((tmp_path / "b2.json").read_text(encoding="utf-8"))
    assert loaded["invariants"]["probe_distinct"]
