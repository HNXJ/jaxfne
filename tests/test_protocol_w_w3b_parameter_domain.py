"""Protocol W3b — parameter-domain regime classification."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from jaxfne.w3b_parameter_domain import (
    Regime,
    W3bDomainScanConfig,
    W3bFrozenGates,
    W3bParameterPoint,
    classify_regime,
    export_w3b_domain_receipt,
    floquet_margin_nonneutral,
    gamma_hdp,
    run_w3b_domain_scan,
    validate_period_for_floquet,
)

AUDIT = Path("artifacts/protocol_w/w3a_stability/w3a_margin_audit.json")
SPEC = Path("artifacts/protocol_w/w3b_parameter_domain/w3b_parameter_domain_spec.json")


def test_margin_audit_documents_period_one_issue():
    audit = json.loads(AUDIT.read_text())
    assert audit["root_causes"][0]["id"] == "period_one_false_positive"
    assert audit["W3b_blocking_rule"]


def test_w3b_spec_exists_and_allows_empty_domain():
    spec = json.loads(SPEC.read_text())
    assert spec["D_useful"]["may_be_empty"] is True
    assert "m_F > 0.02" in spec["frozen_gates_before_scan"]["robust_stability"]


def test_floquet_margin_nonneutral_excludes_unit_multipliers():
    m = np.diag([1.0, 1.2, 0.9])
    out = floquet_margin_nonneutral(m, epsilon_neutral=0.05)
    assert out["rho_nonneutral"] == pytest.approx(1.2)
    assert out["m_F"] == pytest.approx(-0.2)


def test_period_one_rejected_for_floquet():
    gates = W3bFrozenGates()
    trace = np.zeros((10, 10))
    val = validate_period_for_floquet(trace, {"found": True, "period": 1}, gates=gates)
    assert not val["valid"]
    assert val["reason"] == "period_one_rejected"


def test_classify_regime_dormant_at_zero_syn():
    gates = W3bFrozenGates()
    assert classify_regime(mean_syn=0.0, l_hdp=0.0, r_tau=10.0, m_f=0.5, floquet_available=True, gates=gates) == Regime.DORMANT


def test_classify_regime_stable_requires_margin_and_gain():
    gates = W3bFrozenGates()
    assert classify_regime(mean_syn=0.1, l_hdp=1e-5, r_tau=10.0, m_f=0.05, floquet_available=True, gates=gates) == Regime.STABLE
    assert classify_regime(mean_syn=0.1, l_hdp=1e-5, r_tau=10.0, m_f=0.01, floquet_available=True, gates=gates) == Regime.CRITICAL
    assert classify_regime(mean_syn=0.1, l_hdp=1e-5, r_tau=10.0, m_f=-0.1, floquet_available=True, gates=gates) == Regime.UNSTABLE


def test_frozen_lattice_size():
    cfg = W3bDomainScanConfig.frozen_full_lattice()
    assert len(cfg.parameter_points) == 243
    assert len(cfg.i_tonic_grid) == 9


def test_frozen_domain_receipt_three_branch_interpretation():
    receipt = json.loads(
        Path("artifacts/protocol_w/w3b_parameter_domain/w3b_domain_receipt.json").read_text()
    )
    agg = receipt["aggregate_quantities"]
    assert agg["N_S"] == 0
    assert agg["N_X"] > 0
    assert receipt["interpretation"]["branch"] == "N_S_eq_0_and_N_X_gt_0"
    assert len(receipt["scan_results"]) == 2187


def test_gamma_hdp_matches_reduced_convention():
    p = W3bParameterPoint(0.05, 1.0, 0.1, 80.0, 100.0)
    assert gamma_hdp(p, 0.01) == pytest.approx(abs(2.0 * 1.0 * 0.01 / 0.1))
