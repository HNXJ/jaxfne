"""0.4.17-E E3 — sparse owned H_K RBS composition implementation tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

from jaxfne.protocol_e_integration.e2_execution import build_e2_model, delay_step_occupancy
from jaxfne.protocol_e_integration.e3_execution import (
    build_h_k0,
    build_owner_mask,
    h_k_owner_euler_reference,
    load_e3_execution_receipt,
    owner_flat_indices,
    run_e2_reference_kernel,
    run_e3_kernel,
    run_e3_rbs_composition,
    verify_e3_continuation,
    verify_non_owner_h_k_support,
    write_e3_execution_receipt,
)
from jaxfne.protocol_e_integration.e3_protocol import (
    E3_EXECUTION_RECEIPT_PATH,
    load_e3_spec,
    validate_e3_spec,
)


def test_e3_null_matches_e2_delayed_reference_kernel():
    spec = load_e3_spec()
    sim = spec["simulation_policy"]
    model = build_e2_model(zero_delays=False)
    e2 = run_e2_reference_kernel(
        model, n_steps=int(sim["n_steps"]), dt_ms=float(sim["dt_ms"]), seed=11
    )
    e3 = run_e3_kernel(
        model,
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
        seed=11,
        mode="E3-null",
    )
    assert np.array_equal(e2["V_m"], e3["V_m"])
    assert np.array_equal(e2["spikes"], e3["spikes"])
    assert np.array_equal(e2["sources"], e3["sources"])


def test_e3_non_owner_h_k_fixed_at_reference():
    spec = load_e3_spec()
    sim = spec["simulation_policy"]
    owners = owner_flat_indices(spec)
    model = build_e2_model(zero_delays=False)
    dyn = run_e3_kernel(
        model,
        n_steps=int(sim["n_steps"]),
        dt_ms=float(sim["dt_ms"]),
        seed=11,
        mode="E3-dynamic",
    )
    g3 = verify_non_owner_h_k_support(
        dyn["H_K_trace"], owners, int(model.params["emitter"].n_neurons)
    )
    assert g3["passed"]
    assert g3["max_abs_H_minus_1_non_owners"] == 0.0


def test_e3_owner_h_k_follows_d2a_euler_recurrence():
    spec = load_e3_spec()
    sim = spec["simulation_policy"]
    dt_ms = float(sim["dt_ms"])
    n_steps = int(sim["n_steps"])
    tau_k = float(spec["rbs_primitive"]["dynamics_F_H"]["dynamic_mode"]["tau_K_ms"])
    h0 = float(spec["rbs_primitive"]["initial_perturbation"]["H_K0_dynamic"])
    ref = h_k_owner_euler_reference(h0, n_steps, dt_ms=dt_ms, tau_k_ms=tau_k)
    model = build_e2_model(zero_delays=False)
    dyn = run_e3_kernel(model, n_steps=n_steps, dt_ms=dt_ms, seed=11, mode="E3-dynamic")
    owners = owner_flat_indices(spec)
    for oid in owners:
        assert np.max(np.abs(dyn["H_K_trace"][:, oid] - ref)) == 0.0


def test_e3_delay_table_unchanged_from_e2():
    model = build_e2_model(zero_delays=False)
    occ = delay_step_occupancy(model)
    assert occ["2"] == 651
    assert occ["4"] == 140
    assert occ["8"] == 140


def test_e3_continuation_inflight_120ms_dynamic():
    spec = load_e3_spec()
    sim = spec["simulation_policy"]
    dt_ms = float(sim["dt_ms"])
    split_steps = int(round(120.0 / dt_ms))
    model = build_e2_model(zero_delays=False)
    g7 = verify_e3_continuation(
        model,
        total_steps=int(sim["n_steps"]),
        split_steps=split_steps,
        dt_ms=dt_ms,
        seed=11,
        mode="E3-dynamic",
    )
    assert g7["passed"]
    assert g7["H_K_bit_exact"]
    assert g7["delay_state_bit_exact"]
    assert abs((g7["H_K_at_split_ms"] - 1.0) - 0.2 * np.exp(-1.2)) < 0.01


def test_e3_owner_mask_and_h_k0_builders():
    owners = owner_flat_indices()
    n = 80
    mask = np.asarray(build_owner_mask(n, owners))
    assert np.all(mask[owners] > 0.5)
    assert np.all(mask[np.setdiff1d(np.arange(n), owners)] < 0.5)
    h_dyn = np.asarray(build_h_k0(n, owners, mode="E3-dynamic", h_k0_dynamic=1.2))
    h_null = np.asarray(build_h_k0(n, owners, mode="E3-null", h_k0_dynamic=1.2))
    assert np.all(h_dyn[owners] == 1.2)
    assert np.all(h_dyn[np.setdiff1d(np.arange(n), owners)] == 1.0)
    assert np.all(h_null == 1.0)


def test_e3_run_gates_and_write_receipt(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "jaxfne.protocol_e_integration.e3_execution.E3_EXECUTION_RECEIPT_PATH",
        tmp_path / "e3_execution_receipt.json",
    )
    receipt = write_e3_execution_receipt()
    assert receipt["checkpoint"] == "E3"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
    assert receipt["rbs_owner"]["flat_indices"] == [70, 71, 72, 73, 74, 75, 76]
    assert receipt["rbs_primitive"]["semantic_type"] == "effective_K_associated_recovery"
    assert (
        receipt["rbs_primitive"]["non_owner_H_K_semantics"]
        == "fixed reference coordinate H_K=1; F1 recurrence masked off outside owner_mask"
    )
    on_disk = json.loads((tmp_path / "e3_execution_receipt.json").read_text())
    assert on_disk["status"] == "FROZEN"


@pytest.mark.skipif(
    not E3_EXECUTION_RECEIPT_PATH.exists(),
    reason="frozen E3 execution receipt not present",
)
def test_e3_frozen_execution_receipt_on_disk():
    receipt = load_e3_execution_receipt()
    validate_e3_spec()
    assert receipt["status"] == "FROZEN"
    assert all(receipt["gates"][g]["passed"] for g in receipt["gates"])
