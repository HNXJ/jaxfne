"""Protocol W3 — stability analysis receipts (analysis only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jaxfne.w3_stability_analysis import (
    W3NominalParameters,
    analytical_reduced_jacobian_continuous,
    export_w3_stability_receipt,
    gate_zero_doctrinal_point,
    izhikevich_silent_fixed_point,
    run_w3_stability_analysis,
)

RECEIPT = Path("artifacts/protocol_w/w3_stability/w3_stability_receipt.json")


def test_gate_zero_doctrinal_reference_is_not_fixed_point():
    gate = gate_zero_doctrinal_point(W3NominalParameters())
    assert not gate["is_fixed_point"]
    assert gate["residual_norm"] > 1.0


def test_verified_equilibrium_converges_at_silent_fixed_point():
    v_s, u_s = izhikevich_silent_fixed_point()
    out = run_w3_stability_analysis()
    z = out["equilibrium"]["z_star"]
    assert out["equilibrium"]["convergence"]["converged"]
    assert z[0] == pytest.approx(v_s, abs=1e-6)
    assert z[2] == pytest.approx(u_s, abs=1e-6)
    assert out["equilibrium"]["H_A"] == pytest.approx(1.0)
    assert out["equilibrium"]["H_B"] == pytest.approx(1.0)
    assert out["equilibrium"]["omega_AB"] == pytest.approx(0.0)


def test_b_hw_zero_at_syn_zero_equilibrium():
    out = run_w3_stability_analysis()
    assert out["b_HW_derivation"]["b_HW_from_autodiff"] == pytest.approx(0.0, abs=1e-15)
    assert not out["reduced_antisymmetric"]["closed_loop_linearly_active_at_equilibrium"]


def test_reduced_analytic_jacobian_matches_block_structure():
    j = analytical_reduced_jacobian_continuous(W3NominalParameters(), 0.0)
    assert j[0, 1] == pytest.approx(0.0)
    assert j[0, 0] == pytest.approx(-1.0 / 80.0)
    assert j[1, 0] == pytest.approx(2.0 * 1.0 / 100.0)


def test_full_step_jacobian_autodiff_matches_finite_difference():
    out = run_w3_stability_analysis()
    assert out["full_step_jacobian"]["autodiff_vs_finite_difference_max_abs"] < 1e-5


def test_nominal_discrete_stability_gate_passes_marginally():
    out = run_w3_stability_analysis()
    gate = out["stability_gate"]
    assert gate["discrete_pass"]
    assert gate["discrete_spectral_margin"] == pytest.approx(0.001, abs=1e-6)
    assert gate["continuous_from_step_pass"]
    assert not gate["w3_kernel_implementation_authorized"]


def test_frozen_receipt_on_disk_matches_exporter():
    assert RECEIPT.is_file()
    frozen = json.loads(RECEIPT.read_text())
    live = export_w3_stability_receipt()
    assert frozen["schema"] == live["schema"]
    assert frozen["stability_gate"]["discrete_pass"] == live["stability_gate"]["discrete_pass"]
    assert frozen["b_HW_derivation"]["b_HW_from_autodiff"] == live["b_HW_derivation"]["b_HW_from_autodiff"]
