"""v0.3.30 connectivity compiler: weight modes (scalar/uniform/normal/matrix/artifact_ref)."""

import numpy as np
import pytest

import jaxfne as jtfne

NEURONS = [
    {"neuron_id": 0, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 1, "area": "V1", "layer": "L4", "cell_type": "E"},
    {"neuron_id": 2, "area": "V1", "layer": "L4", "cell_type": "PV"},
    {"neuron_id": 3, "area": "V1", "layer": "L4", "cell_type": "PV"},
]
MECHS = [{"name": "ampa", "kind": "synapse", "params": {"tau_ms": 5.0}}]


def _compile(weight, **kw):
    conn = [{"name": "E_PV", "source": {"cell_type": "E"}, "target": {"cell_type": "PV"},
             "weight": weight, "mechanism": "ampa"}]
    return jtfne.compile_connection_rules(NEURONS, conn, MECHS, **kw)


def test_scalar_float_weight():
    r = _compile(0.25)
    assert r.n_edges == 4
    assert np.allclose(np.asarray(r.edge_weight), 0.25)


def test_scalar_mode_dict():
    r = _compile({"mode": "scalar", "value": 0.7})
    assert np.allclose(np.asarray(r.edge_weight), 0.7)


def test_random_uniform_mode_deterministic_and_in_range():
    spec = {"mode": "random_uniform", "low": 0.0, "high": 1.0, "seed": 11}
    a = _compile(spec, seed=0)
    b = _compile(spec, seed=0)
    w = np.asarray(a.edge_weight)
    assert np.array_equal(w, np.asarray(b.edge_weight))
    assert np.all((w >= 0.0) & (w <= 1.0))


def test_random_normal_mode_finite():
    r = _compile({"mode": "random_normal", "value": 0.0, "scale": 0.1, "seed": 5})
    assert np.all(np.isfinite(np.asarray(r.edge_weight)))


def test_matrix_mode_maps_positions():
    # n_pre=2 (E: 0,1), n_post=2 (PV: 2,3)
    mat = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    r = _compile({"mode": "matrix", "array": mat})
    # all-to-all (no probability) => 4 edges; weights equal the matrix entries
    weights = {}
    for a, b, w in zip(np.asarray(r.edge_pre), np.asarray(r.edge_post), np.asarray(r.edge_weight)):
        weights[(int(a), int(b))] = float(w)
    # float32 default dtype → compare with tolerance
    assert np.isclose(weights[(0, 2)], 0.1) and np.isclose(weights[(0, 3)], 0.2)
    assert np.isclose(weights[(1, 2)], 0.3) and np.isclose(weights[(1, 3)], 0.4)


def test_matrix_wrong_shape_fails():
    with pytest.raises(ValueError, match="matrix shape"):
        _compile({"mode": "matrix", "array": np.zeros((3, 3))})


def test_artifact_ref_requires_fields():
    with pytest.raises(ValueError, match="artifact_ref requires"):
        _compile({"mode": "artifact_ref", "path": "w.npy", "array_name": "w"})  # missing sha256


def _sha(arr):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def test_artifact_ref_requires_supplied_array():
    spec = {"mode": "artifact_ref", "path": "w.npy", "array_name": "w", "sha256": "abc"}
    with pytest.raises(ValueError, match="not supplied"):
        _compile(spec)


def test_artifact_ref_uses_supplied_array_with_matching_hash():
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    spec = {"mode": "artifact_ref", "path": "w.npy", "array_name": "w", "sha256": _sha(mat)}
    r = _compile(spec, artifacts={"w": mat})
    assert r.n_edges == 4
    assert np.all(np.isfinite(np.asarray(r.edge_weight)))


def test_artifact_ref_sha256_mismatch_fails():
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    spec = {"mode": "artifact_ref", "path": "w.npy", "array_name": "w", "sha256": "deadbeef"}
    with pytest.raises(ValueError, match="sha256 mismatch"):
        _compile(spec, artifacts={"w": mat})


def test_unknown_weight_mode_fails():
    with pytest.raises(ValueError, match="unknown weight mode"):
        _compile({"mode": "lognormal"})


def test_nonfinite_scalar_weight_fails():
    with pytest.raises(ValueError, match="finite"):
        _compile(float("inf"))
