"""0.4.17-E E4: observation-chain composition implementation tests."""

from __future__ import annotations


import jax.numpy as jnp
import numpy as np
import pytest

import jaxfne.protocol_e_integration.e4_execution as e4x
from jaxfne.fields.proxy import project_laminar_sources
from jaxfne.protocol_e_integration.e2_execution import build_e2_model
from jaxfne.protocol_e_integration.e3_execution import run_e3_kernel
from jaxfne.protocol_e_integration.e4_execution import (
    aggregate_q_by_hierarchy,
    apply_e4_probe,
    build_hierarchy_aware_source_table_summary,
    freeze_e3_trajectory,
    materialize_phi_ref,
    run_all_primary_observations,
    run_e4_neural_only,
    verify_field_linearity,
    verify_zero_source_operators,
    write_e4_execution_receipt,
)
from jaxfne.protocol_e_integration.e4_protocol import (
    load_e4_spec,
    validate_e4_spec,
)


def test_e4_neural_only_matches_e3_null():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    e3 = run_e3_kernel(
        model,
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
        seed=11,
        mode="E3-null",
    )
    e4 = run_e4_neural_only(
        model,
        seed=11,
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    assert np.array_equal(e3["V_m"], e4.V_m)
    assert np.array_equal(e3["sources"], e4.Q)
    assert np.array_equal(e3["H_K_trace"], e4.H_K)


def test_e4_q_conservation_under_hierarchy_aggregation():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-null",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    _, conservation = aggregate_q_by_hierarchy(traj)
    assert conservation["passed"]


def test_e4_probe_factorization_shared_phi_distinct_y():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-null",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    field = materialize_phi_ref(traj, spec)
    shallow = apply_e4_probe(traj, field, "lfp_contact_shallow", spec=spec)
    deep = apply_e4_probe(traj, field, "lfp_contact_deep", spec=spec)
    assert shallow.phi_ref_hash == deep.phi_ref_hash
    assert shallow.q_hash_before == shallow.q_hash_after == traj.cause_hashes["Q"]
    assert not np.array_equal(shallow.Y, deep.Y)


def test_e4_zero_source_and_linearity_operators():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-null",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    assert verify_zero_source_operators(traj, spec=spec)["passed"]
    assert verify_field_linearity(traj, spec=spec)["passed"]


def test_e4_all_primary_observations_preserve_q():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-null",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    obs = run_all_primary_observations(traj, spec=spec)
    assert obs["cause_hashes_unchanged"]
    table = build_hierarchy_aware_source_table_summary(traj)
    assert table["conservation"]["passed"]


def test_e4_dynamic_diagnostic_observation_runs():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    traj = freeze_e3_trajectory(
        model,
        seed=11,
        mode="E3-dynamic",
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
    )
    obs = run_all_primary_observations(traj, spec=spec)
    assert obs["cause_hashes_unchanged"]
    assert float(np.max(np.abs(traj.H_K - 1.0))) > 0.0


def test_e4_run_gates_and_write_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jaxfne.protocol_e_integration.e4_execution.E4_EXECUTION_RECEIPT_PATH",
        tmp_path / "e4_execution_receipt.json",
    )
    receipt = write_e4_execution_receipt()
    assert receipt["checkpoint"] == "E4"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
    assert receipt["source_of_truth_hashes"]["Q"]
    assert receipt["e3_dynamic_observation_diagnostic"]["observation_completed"] is True
    assert len(receipt["semantic_status_table"]) == 9


def test_e4_frozen_execution_receipt_on_disk():
    from jaxfne.protocol_e_integration.e4_execution import load_e4_execution_receipt

    receipt = load_e4_execution_receipt()
    validate_e4_spec()
    assert receipt["status"] == "FROZEN"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])


# D4 (2026-11-25): verify_field_linearity combines 0.7 Q and 0.3 Q, so it tests homogeneity only. The checks below
# test additivity on independent, non-proportional sources at the E4 field parameters; production code is unchanged.


@pytest.fixture(scope="module")
def e4_null():
    spec = load_e4_spec()
    sim = spec["simulation_policy"]
    traj = freeze_e3_trajectory(
        build_e2_model(zero_delays=False), seed=11, mode="E3-null",
        n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"]),
    )
    params = {**e4x._field_params(spec, "lfp_ref"), "column_height_mm": traj.column_height_mm}
    q1 = jnp.asarray(traj.Q, dtype=jnp.float32)
    assert q1.ndim == 2 and float(jnp.max(jnp.abs(q1))) > 0, "fixture: Q must be a nonzero [T, N] trace"
    rng = np.random.default_rng(20261125)
    q2 = jnp.asarray(rng.standard_normal(q1.shape) * float(jnp.std(q1)), dtype=jnp.float32)
    corr = np.corrcoef(np.ravel(q1), np.ravel(q2))[0, 1]
    assert abs(corr) < 0.1, f"fixture: q2 must not be proportional to Q (corr {corr:.3f})"
    return spec, traj, jnp.asarray(traj.positions, dtype=jnp.float32), params, q1, q2


def _additivity_excess(op, q1, q2):
    """max |op(q1 + q2) - op(q1) - op(q2)| / B for an operator op(q) = q @ K.T run in float32.

    B = gamma_N (|q12| + |q1| + |q2|) |K|^T + u |q1 + q2| |K|^T, with q12 the float32 sum: three length-N dot
    products in any order (Higham, Accuracy and Stability, Lemma 3.1, Sec. 3.1) and one rounded sum; the
    difference is formed in float64. Excess <= 1 is additivity up to float32 rounding.
    """
    q12 = q1 + q2
    f12, f1, f2 = op(q12), op(q1), op(q2)
    for f in (f12, f2):
        assert np.array_equal(np.asarray(f.kernel), np.asarray(f1.kernel)), "kernel must not depend on q"
    f64 = lambda x: np.asarray(x).astype(np.float64)  # float32 -> float64 is exact
    k = np.abs(f64(f1.kernel))
    u = 2.0 ** -24
    gamma = k.shape[1] * u / (1 - k.shape[1] * u)
    a1, a2, a12 = f64(q1), f64(q2), f64(q12)
    bound = gamma * (np.abs(a12) + np.abs(a1) + np.abs(a2)) @ k.T + u * np.abs(a1 + a2) @ k.T
    diff = np.abs(f64(f12.phi_e_proxy) - f64(f1.phi_e_proxy) - f64(f2.phi_e_proxy))
    assert np.all(diff[bound == 0] == 0)
    pos = bound > 0
    assert pos.any(), "bound must cover at least one element"
    return float(np.max(diff[pos] / bound[pos]))


def test_e4_field_operator_is_additive_on_independent_sources(e4_null):
    _, _, pos, params, q1, q2 = e4_null
    assert _additivity_excess(lambda q: project_laminar_sources(q, pos, **params), q1, q2) <= 1.0


def test_additivity_check_fails_on_a_homogeneous_nonlinearity(e4_null):
    """Known negative: q -> F(|q|) is positively homogeneous, not additive on mixed-sign sources."""
    _, _, pos, params, q1, q2 = e4_null
    assert _additivity_excess(lambda q: project_laminar_sources(jnp.abs(q), pos, **params), q1, q2) > 1e3


@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="D4 (2026-11-25): the E4 linearity gate uses proportional inputs and passes a positively homogeneous "
    "nonlinearity; the production change awaits a ruling",
)
def test_e4_linearity_gate_detects_a_homogeneous_nonlinearity(e4_null, monkeypatch):
    spec, traj, *_ = e4_null
    real = e4x.project_laminar_sources
    monkeypatch.setattr(e4x, "project_laminar_sources", lambda q, p, **kw: real(jnp.abs(q), p, **kw))
    assert not verify_field_linearity(traj, spec=spec)["passed"]
