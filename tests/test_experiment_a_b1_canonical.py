"""0.4.17-B B1 — canonical X/H/Q dataset tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from jaxfne.experiment_a.canonical import (
    array_sha256,
    freeze_canonical_dataset,
    write_b1_receipt,
    write_canonical_npz,
)
from jaxfne.experiment_a.protocol import load_protocol_spec

RECEIPT = Path("artifacts/etudes/experiment_a/b1_canonical_receipt.json")


def test_b1_single_simulate_produces_finite_XHQ():
    ds = freeze_canonical_dataset(duration_ms=40.0, package_head="test_head")
    assert np.all(np.isfinite(ds.X_V_m))
    assert np.all(np.isfinite(ds.Q))
    assert np.all(np.isfinite(ds.H))
    assert ds.Q.shape == ds.X_V_m.shape
    assert ds.H.shape == ds.Q.shape


def test_b1_H_identity_when_hdp_off():
    spec = load_protocol_spec()
    ds = freeze_canonical_dataset(spec=spec, duration_ms=20.0, package_head="test")
    assert ds.H_semantic == spec["neural_system"]["H_semantic_when_hdp_off"]
    np.testing.assert_array_equal(ds.H, np.ones_like(ds.H))


def test_b1_cause_hashes_stable_on_reread():
    ds = freeze_canonical_dataset(duration_ms=20.0, package_head="test")
    assert ds.cause_hashes["Q"] == array_sha256(ds.Q)
    assert ds.cause_hashes["V_m"] == array_sha256(ds.X_V_m)


def test_b1_receipt_schema(tmp_path):
    ds = freeze_canonical_dataset(duration_ms=16.0, package_head="abc123")
    receipt = write_b1_receipt(ds, tmp_path / "b1.json")
    assert receipt["checkpoint"] == "B1"
    assert receipt["status"] == "FROZEN"
    assert "cause_hashes" in receipt
    assert receipt["shapes"]["Q"][1] == ds.Q.shape[1]
