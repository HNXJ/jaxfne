"""Tests for v0.0.16 RunReceipt — complete run receipt with truth gate enforcement.

A. Model.run_receipt(signals) returns RunReceipt
B. RunReceipt.to_dict() is JSON-safe (no NaN/Inf, round-trips json.dumps)
C. receipt_id is stable: same config + seed → same ID
D. receipt_id differs for different seeds
E. Truth gates are frozen at conservative defaults (all required keys present)
F. save_receipt() creates JSON file on disk
G. save_receipt() raises ValueError with token 'receipt_file_exists' on overwrite
H. save_receipt(overwrite=True) succeeds
I. Module-level run_receipt() factory delegates correctly
J. RunReceipt works with edge_list backend (backend dict populated)
"""

import json
import tempfile
from pathlib import Path

import pytest

import jaxfne as jtfne
from jaxfne.core import _JAXFNE_VERSION, _RECEIPT_SCHEMA_VERSION


_TRUTH_REQUIRED_KEYS = {
    "truth_mode",
    "claim_level",
    "source_calibration_status",
    "field_solver_status",
    "field_claim_level",
    "physical_amplitude_claim_allowed",
    "empirical_validation_status",
    "mechanism_claim_status",
}

_TRUTH_CONSERVATIVE = {
    "truth_mode": "truth_safe_unverified",
    "claim_level": "computational_scaffold",
    "source_calibration_status": "uncalibrated_izhikevich_native_current",
    "field_solver_status": "laminar_proxy_no_pde",
    "field_claim_level": "proxy_readout_only",
    "physical_amplitude_claim_allowed": False,
    "empirical_validation_status": "not_empirically_validated",
    "mechanism_claim_status": "not_claimed",
}


def _make_model_signals(n: int = 8, duration_ms: float = 5.0, seed: int = 0):
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=duration_ms, dt_ms=0.5, seed=seed)
    signals = model.simulate(sim)
    return model, signals


def test_a_run_receipt_returns_runreceipt():
    model, signals = _make_model_signals()
    receipt = model.run_receipt(signals)
    assert isinstance(receipt, jtfne.RunReceipt)
    assert isinstance(receipt.receipt_id, str)
    assert len(receipt.receipt_id) == 16
    assert receipt.jaxfne_version == _JAXFNE_VERSION
    assert isinstance(receipt.config_hash, str)
    assert len(receipt.config_hash) == 16
    assert isinstance(receipt.simulation, dict)
    assert isinstance(receipt.signals_summary, dict)
    assert isinstance(receipt.truth, dict)
    assert isinstance(receipt.claim_labels, dict)
    assert isinstance(receipt.backend, dict)
    assert isinstance(receipt.tags, dict)


def test_b_to_dict_json_safe():
    model, signals = _make_model_signals()
    receipt = model.run_receipt(signals, tags={"paper": "paper_1", "condition": "A"})
    d = receipt.to_dict()
    assert isinstance(d, dict)
    assert "receipt_id" in d
    assert "truth" in d
    assert "claim_labels" in d
    # Must round-trip through json.dumps with allow_nan=False
    json_str = json.dumps(d, allow_nan=False)
    assert isinstance(json_str, str)
    # Tags must be preserved
    assert d["tags"]["paper"] == "paper_1"


def test_c_receipt_id_stable_same_inputs():
    model, signals = _make_model_signals(seed=42)
    r1 = model.run_receipt(signals)
    r2 = model.run_receipt(signals)
    assert r1.receipt_id == r2.receipt_id
    assert r1.config_hash == r2.config_hash


def test_d_receipt_id_differs_different_seed():
    model, s1 = _make_model_signals(seed=0)
    _, s2 = _make_model_signals(seed=99)
    r1 = model.run_receipt(s1)
    r2 = model.run_receipt(s2)
    assert r1.receipt_id != r2.receipt_id


def test_e_truth_gates_frozen_conservative():
    model, signals = _make_model_signals()
    receipt = model.run_receipt(signals)
    truth = receipt.truth
    # All required keys present
    assert _TRUTH_REQUIRED_KEYS.issubset(set(truth.keys()))
    # All values match conservative defaults
    for k, v in _TRUTH_CONSERVATIVE.items():
        assert truth[k] == v, f"truth[{k!r}] = {truth[k]!r}, expected {v!r}"
    # physical_amplitude_claim_allowed is False
    assert truth["physical_amplitude_claim_allowed"] is False
    # Claim labels must reference the schema version
    assert receipt.claim_labels["receipt_status"] == _RECEIPT_SCHEMA_VERSION
    assert receipt.claim_labels["empirical_validation_status"] == "not_empirically_validated"
    assert receipt.claim_labels["mechanism_claim_status"] == "not_claimed"
    assert receipt.claim_labels["physical_amplitude_claim_allowed"] is False


def test_f_save_receipt_creates_file():
    model, signals = _make_model_signals()
    receipt = model.run_receipt(signals)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "receipt.json"
        jtfne.save_receipt(receipt, path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["receipt_id"] == receipt.receipt_id
        assert loaded["truth"]["physical_amplitude_claim_allowed"] is False


def test_g_save_receipt_raises_on_overwrite():
    model, signals = _make_model_signals()
    receipt = model.run_receipt(signals)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "receipt.json"
        jtfne.save_receipt(receipt, path)
        # Second save without overwrite=True must raise
        with pytest.raises(ValueError, match="receipt_file_exists"):
            jtfne.save_receipt(receipt, path)


def test_h_save_receipt_overwrite_true_succeeds():
    model, signals = _make_model_signals()
    receipt = model.run_receipt(signals)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "receipt.json"
        jtfne.save_receipt(receipt, path)
        # Should not raise
        jtfne.save_receipt(receipt, path, overwrite=True)
        loaded = json.loads(path.read_text())
        assert loaded["receipt_id"] == receipt.receipt_id


def test_i_module_level_run_receipt_factory():
    model, signals = _make_model_signals()
    receipt_method = model.run_receipt(signals)
    receipt_factory = jtfne.run_receipt(model, signals)
    # Both should produce identical receipts for same inputs
    assert receipt_method.receipt_id == receipt_factory.receipt_id
    assert receipt_method.config_hash == receipt_factory.config_hash
    assert receipt_method.truth == receipt_factory.truth
    assert isinstance(receipt_factory, jtfne.RunReceipt)


def test_j_run_receipt_edge_list_backend():
    cfg = (
        jtfne.configuration()
        .network(name="V1", kind="cortical_column", n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="laminar_probe", modes=["spikes", "V_m"])
    )
    model = jtfne.construct(cfg)
    sim = jtfne.simulation(duration_ms=5.0, dt_ms=0.5, seed=0,
                           runtime=jtfne.runtime(recurrent_backend="edge_list"))
    signals = model.simulate(sim)
    receipt = model.run_receipt(signals)
    assert isinstance(receipt, jtfne.RunReceipt)
    # Edge list backend details should be populated
    assert receipt.backend["edge_list_n_edges"] > 0
    assert receipt.backend["edge_list_backend"] == "edge_list_recurrent_v0.0.9"
    # Truth gates still frozen
    assert receipt.truth["physical_amplitude_claim_allowed"] is False
    assert receipt.truth["empirical_validation_status"] == "not_empirically_validated"
    # JSON-safe
    json.dumps(receipt.to_dict(), allow_nan=False)
