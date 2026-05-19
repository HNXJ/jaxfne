"""Tests for manifest() readout argument compatibility — v0.0.23.

Covers the bug where model.manifest(signals, readouts) raised
``AttributeError: 'list' object has no attribute 'get'`` when readouts
was the list returned by model.compute_readout().

Tests A–E enumerate every supported readout argument shape.
Test F validates the exact canonical v0.1 workflow documented in COLAB.md.
Test G validates JSON strictness (no NaN).
Test H validates truth gates are never escalated by the readout normaliser.
"""

import json
import pytest
import jaxfne


# ─── shared fixture ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def _base():
    cfg = (
        jaxfne.Configuration()
        .network(n=8)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy",
               boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="lp", n_contacts=8)
    )
    model = jaxfne.construct(cfg)
    signals = model.simulate(jaxfne.Simulation(duration_ms=20.0, dt_ms=0.5))
    readouts = model.compute_readout(signals, [
        jaxfne.ReadoutSpec(name="rate", metric="spike_rate_hz"),
        jaxfne.ReadoutSpec(name="csd", metric="csd_abs_mean"),
    ])
    return model, signals, readouts


# ─── A. readout=None ─────────────────────────────────────────────────────────


def test_a_manifest_readout_none(_base):
    model, signals, _ = _base
    mf = model.manifest(signals)
    assert mf["truth_mode"] == "truth_safe_unverified"
    assert mf["physical_amplitude_claim_allowed"] is False


# ─── B. readout=list of ReadoutResult (the bug case) ────────────────────────


def test_b_manifest_readout_list_of_results(_base):
    model, signals, readouts = _base
    mf = model.manifest(signals, readouts)
    assert "readout_results" in mf
    assert mf["readout_results"]["n_results"] == 2
    assert mf["readout_results"]["physical_amplitude_claim_allowed"] is False
    metrics = mf["readout_results"]["requested_metrics"]
    assert "spike_rate_hz" in metrics
    assert "csd_abs_mean" in metrics


# ─── C. readout=dict (legacy shape) ─────────────────────────────────────────


def test_c_manifest_readout_legacy_dict(_base):
    model, signals, _ = _base
    mf = model.manifest(signals, {"requested_modes": ["CSD"]})
    assert mf["truth_mode"] == "truth_safe_unverified"


# ─── D. readout=single ReadoutResult ────────────────────────────────────────


def test_d_manifest_readout_single_result(_base):
    model, signals, readouts = _base
    mf = model.manifest(signals, readouts[0])
    assert "readout_results" in mf
    assert mf["readout_results"]["n_results"] == 1


# ─── E. readout=list of dicts ────────────────────────────────────────────────


def test_e_manifest_readout_list_of_dicts(_base):
    model, signals, readouts = _base
    mf = model.manifest(signals, [r.to_dict() for r in readouts])
    assert "readout_results" in mf
    assert mf["readout_results"]["n_results"] == 2


# ─── F. canonical v0.1 workflow (exact COLAB.md pattern) ────────────────────


def test_f_canonical_v1_workflow(_base):
    """Exact copy-paste pattern from docs/COLAB.md must execute without error."""
    model, signals, readouts = _base
    manifest = model.manifest(signals, readouts)
    json.dumps(manifest, allow_nan=False)  # must not raise
    assert manifest["truth_mode"] == "truth_safe_unverified"
    assert manifest["claim_level"] == "computational_scaffold"
    assert manifest["field_solver_status"] == "laminar_proxy_no_pde"
    assert manifest["physical_amplitude_claim_allowed"] is False


# ─── G. JSON strictness ──────────────────────────────────────────────────────


def test_g_manifest_json_no_nan(_base):
    model, signals, readouts = _base
    for readout_arg in [None, readouts, readouts[0], [r.to_dict() for r in readouts]]:
        mf = model.manifest(signals, readout_arg)
        json.dumps(mf, allow_nan=False)  # must not raise


# ─── H. truth gates never escalated ─────────────────────────────────────────


def test_h_readout_normaliser_no_truth_escalation(_base):
    model, signals, readouts = _base
    for readout_arg in [None, readouts, readouts[0]]:
        mf = model.manifest(signals, readout_arg)
        assert mf["truth_mode"] == "truth_safe_unverified"
        assert mf["physical_amplitude_claim_allowed"] is False
        rr = mf.get("readout_results")
        if rr is not None:
            assert rr["physical_amplitude_claim_allowed"] is False
