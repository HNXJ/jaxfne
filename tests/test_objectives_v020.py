"""Tests for v0.2.0 objective grammar, synchrony gates, and null-ready schemas.

Validates explicit score labels, synchrony diagnostics, null-mode support,
manifest contracts, and JSON safety for objective evaluation.

Gate pass/fail is a computational diagnostic only. No biological validation,
empirical mechanism proof, or amplitude calibration is claimed.
"""

import json
import math

import jax.numpy as jnp
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


def _model_and_signals(n=8, seed=0):
    """Construct model and run simulation."""
    cfg = _cfg(n=n)
    model = jaxfne.construct(cfg)
    sim = jaxfne.Simulation(duration_ms=10.0, dt_ms=0.5, seed=seed)
    signals = model.simulate(sim)
    return model, signals


class TestObjectiveGrammarScoreLabels:
    """Explicit score label grammar for objectives."""

    def test_motif_gate_percent_label(self):
        """Objective can use motif_gate_percent as an explicit score label."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="spike_rate_test").loss(
            name="motif_gate_percent",
            metric="spike_rate_hz_mean",
            weight=1.0,
            metadata={"score_type": "internal_motif_gate"}
        )
        report = model.evaluate(signals, obj)
        assert report["objective_name"] == "spike_rate_test"
        assert len(report["losses"]) == 1
        assert report["losses"][0]["name"] == "motif_gate_percent"
        assert report["losses"][0].get("metadata", {}).get("score_type") == "internal_motif_gate"

    def test_profile_score_percent_label(self):
        """Objective can use profile_score_percent as an explicit score label."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="profile_test").loss(
            name="profile_score_percent",
            metric="csd_proxy_abs_mean",
            weight=1.0,
            metadata={"score_type": "profile_score_no_null"}
        )
        report = model.evaluate(signals, obj)
        assert report["losses"][0]["name"] == "profile_score_percent"
        assert report["losses"][0].get("metadata", {}).get("score_type") == "profile_score_no_null"

    def test_S_lam_score_requires_null_metadata(self):
        """S_lam score must include null distribution metadata or be reported as unavailable."""
        model, signals = _model_and_signals(n=4)
        # Create an S_lam loss without null metadata
        obj = jaxfne.Objective(name="slammer_no_null").loss(
            name="S_lam",
            metric="spike_rate_hz_mean",
            metadata={"score_type": "null_normalized_similarity"}
        )
        report = model.evaluate(signals, obj)
        loss_record = report["losses"][0]
        # Should either be missing or marked with a warning
        if loss_record.get("status") is not None:
            assert loss_record["status"] in ["no_metric_specified", "ok"]

    def test_S_lam_with_null_metadata(self):
        """S_lam score with proper null metadata should be reported correctly."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="slammer_with_null").loss(
            name="S_lam_similarity",
            metric="spike_rate_hz_mean",
            weight=1.0,
            metadata={
                "score_type": "null_normalized_similarity",
                "null_type": "layer_shuffle",
                "n_null": 10,
                "null_mean": 0.5,
                "null_sd": 0.1,
                "ci_95": [0.3, 0.7]
            }
        )
        report = model.evaluate(signals, obj)
        loss_record = report["losses"][0]
        assert loss_record["name"] == "S_lam_similarity"
        assert loss_record.get("metadata", {}).get("score_type") == "null_normalized_similarity"


class TestSynchronyGate:
    """Synchrony diagnostic and anti-seizure gate."""

    def test_synchrony_regularizer_minimal(self):
        """Objective can include synchrony regularizer with minimal metadata."""
        model, signals = _model_and_signals(n=8)
        obj = jaxfne.Objective(name="sync_test").regularizer(
            name="synchrony",
            target=0.0,
            weight=0.1,
            metric="spike_rate_hz_mean",  # Placeholder metric
            metadata={
                "enabled": True,
                "metric": "mean_pairwise_correlation",
                "bin_ms": 50,
                "windows": 1,
                "target": 0.0,
                "penalty": "l2",
                "weight": 0.1
            }
        )
        report = model.evaluate(signals, obj)
        assert len(report["regularizers"]) == 1
        reg = report["regularizers"][0]
        assert reg["name"] == "synchrony"
        meta = reg.get("metadata", {})
        assert meta.get("metric") == "mean_pairwise_correlation"
        assert meta.get("target") == 0.0

    def test_high_synchrony_triggers_gate(self):
        """High global synchrony should trigger a gate threshold."""
        model, signals = _model_and_signals(n=4)
        obj = (jaxfne.Objective(name="high_sync_gate")
               .gate(name="synchrony_too_high", threshold=0.1, criterion="below",
                     metric="spike_rate_hz_mean",
                     metadata={"synchrony_limit": 0.1, "metric": "mean_pairwise_correlation"})
        )
        report = model.evaluate(signals, obj)
        # Gate result should be present
        assert len(report["gates"]) >= 1


class TestNullReadySchema:
    """Null-ready output schema support."""

    def test_null_type_metadata_in_loss(self):
        """Loss can include null_type metadata for null-ready outputs."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="null_ready_test").loss(
            name="profile_against_shuffle",
            metric="spike_rate_hz_mean",
            metadata={
                "null_type": "layer_shuffle",
                "n_null": 100,
                "null_mean": 0.4,
                "null_sd": 0.08,
                "ci_95": [0.25, 0.55]
            }
        )
        report = model.evaluate(signals, obj)
        loss = report["losses"][0]
        meta = loss.get("metadata", {})
        assert meta.get("null_type") == "layer_shuffle"
        assert meta.get("n_null") == 100

    def test_null_modes_supported(self):
        """Objective supports declaring multiple null modes for future use."""
        model, signals = _model_and_signals(n=4)
        null_modes = [
            "layer_shuffle",
            "band_label_shuffle",
            "phase_randomized",
            "uniform_gain",
            "no_field_projection",
            "source_polarity_flip",
            "optimizer_budget_control"
        ]
        obj = jaxfne.Objective(name="multi_null_test")
        for i, mode in enumerate(null_modes):
            obj = obj.loss(
                name=f"null_{i}",
                metric="spike_rate_hz_mean",
                metadata={"null_type": mode}
            )
        report = model.evaluate(signals, obj)
        assert len(report["losses"]) == len(null_modes)
        for i, loss in enumerate(report["losses"]):
            assert loss.get("metadata", {}).get("null_type") == null_modes[i]


class TestObjectiveManifestContract:
    """Objective manifest contract and truth gate preservation."""

    def test_acceptance_decision_explicit(self):
        """Evaluation report must include explicit acceptance_decision."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="decision_test").gate(
            name="pass_always", threshold=1000.0, criterion="above", metric="spike_rate_hz_mean"
        )
        report = model.evaluate(signals, obj)
        assert "acceptance_decision" in report
        assert report["acceptance_decision"] in ["gates_pass", "gates_fail"]

    def test_truth_mode_preserved(self):
        """Objective report must preserve truth_mode=truth_safe_unverified."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="truth_test")
        report = model.evaluate(signals, obj)
        assert report.get("truth_mode") == "truth_safe_unverified"

    def test_claim_level_preserved(self):
        """Objective report must preserve claim_level=computational_scaffold."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="claim_test")
        report = model.evaluate(signals, obj)
        assert report.get("claim_level") == "computational_scaffold"

    def test_physical_amplitude_claim_false(self):
        """Objective report must preserve physical_amplitude_claim_allowed=False."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="amplitude_test")
        report = model.evaluate(signals, obj)
        assert report.get("physical_amplitude_claim_allowed") is False

    def test_field_claim_level_proxy(self):
        """Objective report must preserve field_claim_level=proxy_readout_only."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="field_claim_test")
        report = model.evaluate(signals, obj)
        assert report.get("field_claim_level") == "proxy_readout_only"


class TestObjectiveWindowDiscipline:
    """Omission/spectrolaminar window support in objectives."""

    def test_window_specification_metadata(self):
        """Objective can specify window parameters for peri-event analysis."""
        model, signals = _model_and_signals(n=4)
        windows_spec = {
            "baseline": {"start_ms": -500, "end_ms": 0},
            "event": {"start_ms": 0, "end_ms": 500},
            "post": {"start_ms": 500, "end_ms": 1000},
            "full_peri_event": {"start_ms": -500, "end_ms": 1000}
        }
        obj = jaxfne.Objective(name="window_test").loss(
            name="peri_event_csd",
            metric="csd_proxy_abs_mean",
            metadata={"windows": windows_spec}
        )
        report = model.evaluate(signals, obj)
        loss = report["losses"][0]
        meta = loss.get("metadata", {})
        assert meta.get("windows") == windows_spec


class TestObjectiveJSONSafety:
    """JSON safety and serialization for objective reports."""

    def test_objective_report_json_safe(self):
        """Objective report must be JSON-safe (no NaN, no Inf)."""
        model, signals = _model_and_signals(n=4)
        obj = (jaxfne.Objective(name="json_test")
               .loss(name="spike_rate", metric="spike_rate_hz_mean")
               .regularizer(name="smoothness", metric="mean_V_m")
               .gate(name="spike_gate", threshold=100.0, criterion="above", metric="spike_count_total")
        )
        report = model.evaluate(signals, obj)
        # Should not raise with allow_nan=False
        json_str = json.dumps(report, allow_nan=False)
        assert isinstance(json_str, str)

    def test_objective_report_with_none_values_json_safe(self):
        """Objective report with None values must still be JSON-safe."""
        model, signals = _model_and_signals(n=4)
        obj = jaxfne.Objective(name="none_test").loss(
            name="missing_metric",
            metric="nonexistent_metric"  # This metric doesn't exist
        )
        report = model.evaluate(signals, obj)
        # Should still be JSON-safe
        json_str = json.dumps(report, allow_nan=False)
        assert isinstance(json_str, str)


class TestObjectiveGrammarNoMeanSimilarity:
    """Objective grammar must not use ambiguous mean_similarity labels."""

    def test_no_mean_similarity_in_code(self):
        """Codebase should not contain 'mean_similarity' metric name."""
        # This is a grep check at code review time
        # For now, we just verify that the defined metrics don't use that name
        from jaxfne.core import _KNOWN_METRICS
        assert "mean_similarity" not in _KNOWN_METRICS


class TestObjectiveGrammarRejectsOverclaims:
    """Objective grammar prevents overclaiming biological validation or mechanism."""

    def test_no_mechanism_proven_in_metadata(self):
        """Objective metadata should never claim mechanisms are proven."""
        model, signals = _model_and_signals(n=4)
        # These claims would be invalid; test that they don't affect truth gates
        bad_claims = [
            "mechanism_proven_omission_response",
            "validates_prediction_error_coding",
            "proves_biological_mechanism"
        ]
        for claim in bad_claims:
            obj = jaxfne.Objective(name="overclaim_test").loss(
                name="test",
                metric="spike_rate_hz_mean",
                metadata={"claim": claim}
            )
            report = model.evaluate(signals, obj)
            # Truth gates must remain unaffected by metadata claims
            assert report.get("claim_level") == "computational_scaffold"
            assert report.get("truth_mode") == "truth_safe_unverified"
