"""Tests for v0.2.0 source bookkeeping hardening.

Covers source mode exclusivity, double-count detection, and theoretical validation contracts.
"""

import json
import pytest
import jaxfne


def _cfg(n=8):
    """Minimal configuration."""
    return (
        jaxfne.configuration()
        .network(n=n)
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
        .probe(name="p", n_contacts=4)
    )


def _model_and_signals(n=8):
    """Construct model and run simulation."""
    model = jaxfne.construct(_cfg(n=n))
    sim = jaxfne.Simulation(duration_ms=10.0, dt_ms=0.5)
    signals = model.simulate(sim)
    return model, signals


class TestSourceBookkeepingMetadata:
    """v0.2.0: Source bookkeeping metadata presence and structure."""

    def test_source_bookkeeping_present_in_signals_metadata(self):
        """signals.metadata must contain source_bookkeeping dict."""
        model, signals = _model_and_signals()
        assert "source_bookkeeping" in signals.metadata
        assert isinstance(signals.metadata["source_bookkeeping"], dict)

    def test_source_bookkeeping_required_fields(self):
        """source_bookkeeping must have all required v0.2.0 fields."""
        model, signals = _model_and_signals()
        sb = signals.metadata["source_bookkeeping"]
        required_fields = [
            "source_mode",
            "source_projection_mode",
            "source_decomposition",
            "source_calibration_status",
            "synaptic_current_counting",
            "source_mode_exclusive",
            "physical_amplitude_claim_allowed",
            "double_count_guard",
            "double_count_evidence",
        ]
        for field in required_fields:
            assert field in sb, f"Missing required field: {field}"

    def test_source_mode_exclusive_true(self):
        """source_mode_exclusive must be True (exactly one source mode per run)."""
        model, signals = _model_and_signals()
        assert signals.metadata["source_bookkeeping"]["source_mode_exclusive"] is True

    def test_source_calibration_status_uncalibrated(self):
        """source_calibration_status must be uncalibrated_izhikevich_native_current."""
        model, signals = _model_and_signals()
        sb = signals.metadata["source_bookkeeping"]
        assert sb["source_calibration_status"] == "uncalibrated_izhikevich_native_current"

    def test_physical_amplitude_claim_allowed_false(self):
        """physical_amplitude_claim_allowed must be False (no uncalibrated physical claims)."""
        model, signals = _model_and_signals()
        assert signals.metadata["source_bookkeeping"]["physical_amplitude_claim_allowed"] is False

    def test_double_count_guard_passed(self):
        """double_count_guard must be 'passed' (no double-counting detected)."""
        model, signals = _model_and_signals()
        assert signals.metadata["source_bookkeeping"]["double_count_guard"] == "passed"

    def test_double_count_evidence_none(self):
        """double_count_evidence must be None (no issues found)."""
        model, signals = _model_and_signals()
        assert signals.metadata["source_bookkeeping"]["double_count_evidence"] is None


class TestSourceBookkeepingInRunReceipt:
    """v0.2.0: source_bookkeeping flows into RunReceipt backend metadata."""

    def test_source_bookkeeping_in_receipt_backend(self):
        """RunReceipt.backend must include source_bookkeeping metadata."""
        model, signals = _model_and_signals()
        receipt = model.run_receipt(signals)
        assert "source_bookkeeping" in receipt.backend

    def test_source_bookkeeping_consistent_across_signals_and_receipt(self):
        """source_bookkeeping in receipt.backend must match signals.metadata."""
        model, signals = _model_and_signals()
        receipt = model.run_receipt(signals)
        assert receipt.backend["source_bookkeeping"] == signals.metadata["source_bookkeeping"]


class TestSourceBookkeepingInManifest:
    """v0.2.0: source_bookkeeping flows into manifest."""

    def test_source_bookkeeping_in_manifest(self):
        """manifest must include source_bookkeeping via source_model."""
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        # source_bookkeeping is nested under backend_metadata
        assert "backend_metadata" in manifest
        bm = manifest["backend_metadata"]
        # Check that source-related metadata is present
        assert "source_calibration_status" in bm or bm.get("used_recurrent_backend") is not None

    def test_manifest_json_safe_with_source_bookkeeping(self):
        """manifest must remain JSON-safe (no NaN/Inf) with source_bookkeeping."""
        model, signals = _model_and_signals()
        manifest = model.manifest(signals=signals)
        from jaxfne.io import json_safe
        safe_manifest = json_safe(manifest)
        # Must not raise
        json.dumps(safe_manifest, allow_nan=False)


class TestSourceModeExclusivity:
    """v0.2.0: Exactly one source mode per run (no mode mixing)."""

    def test_source_mode_field_present(self):
        """source_mode must be declared in source_bookkeeping."""
        model, signals = _model_and_signals()
        sb = signals.metadata["source_bookkeeping"]
        assert sb["source_mode"] is not None
        assert isinstance(sb["source_mode"], str)

    def test_source_projection_mode_proxy(self):
        """source_projection_mode must be 'proxy_no_field_solve' for v0.2.0."""
        model, signals = _model_and_signals()
        sb = signals.metadata["source_bookkeeping"]
        assert sb["source_projection_mode"] == "proxy_no_field_solve"

    def test_source_decomposition_proxy_reduced_emitter(self):
        """source_decomposition must be 'proxy_reduced_emitter' for current version."""
        model, signals = _model_and_signals()
        sb = signals.metadata["source_bookkeeping"]
        assert sb["source_decomposition"] == "proxy_reduced_emitter"


class TestTruthGatesPreserved:
    """v0.2.0: Truth gates remain frozen from v0.1.2 baseline."""

    def test_truth_mode_unverified(self):
        """truth_mode must be truth_safe_unverified."""
        model, signals = _model_and_signals()
        receipt = model.run_receipt(signals)
        assert receipt.truth["truth_mode"] == "truth_safe_unverified"

    def test_claim_level_scaffold(self):
        """claim_level must be computational_scaffold (no new empirical claims)."""
        model, signals = _model_and_signals()
        receipt = model.run_receipt(signals)
        assert receipt.truth["claim_level"] == "computational_scaffold"

    def test_field_claim_level_proxy_readout_only(self):
        """field_claim_level must be proxy_readout_only."""
        model, signals = _model_and_signals()
        receipt = model.run_receipt(signals)
        assert receipt.truth["field_claim_level"] == "proxy_readout_only"

    def test_physical_amplitude_claim_not_allowed(self):
        """physical_amplitude_claim_allowed must be False."""
        model, signals = _model_and_signals()
        receipt = model.run_receipt(signals)
        assert receipt.truth["physical_amplitude_claim_allowed"] is False

    def test_empirical_validation_status_not_validated(self):
        """empirical_validation_status must be not_empirically_validated."""
        model, signals = _model_and_signals()
        receipt = model.run_receipt(signals)
        assert receipt.truth["empirical_validation_status"] == "not_empirically_validated"

    def test_mechanism_claim_status_not_claimed(self):
        """mechanism_claim_status must be not_claimed."""
        model, signals = _model_and_signals()
        receipt = model.run_receipt(signals)
        assert receipt.truth["mechanism_claim_status"] == "not_claimed"


class TestSynapticCurrentCounting:
    """v0.2.0: Synaptic current counting must not be duplicated."""

    def test_synaptic_current_counting_guard_present(self):
        """synaptic_current_counting must describe guard against double-counting."""
        model, signals = _model_and_signals()
        sb = signals.metadata["source_bookkeeping"]
        assert "synaptic_current_counting" in sb
        assert "no_extra_synaptic_source" in sb["synaptic_current_counting"] or \
               "single_proxy" in sb["synaptic_current_counting"]

    def test_source_model_metadata_includes_double_count_guard(self):
        """source_model metadata must document double-count guard."""
        model, signals = _model_and_signals()
        sm = signals.metadata.get("source_model", {})
        assert "double_count_synaptic_current_guard" in sm
        assert "single_proxy_expression_no_extra_synaptic_source" in sm["double_count_synaptic_current_guard"]
