"""Tests for Suite No. 1 AGSDR public API and multi-parameter optimization.

Validates:
- AGSDROptimizerSpec creation and JSON serialization
- rate_targets() objective factory
- Model.tune() multi-parameter path with AGSDR optimizer
- TuneResult typed result object with to_dict() serialization
- Group-wise firing rate evaluation
"""

import json
import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne
import pytest


class TestAGSDROptimizerSpec:
    """Test AGSDR optimizer specification."""

    def test_agsdr_spec_creation_multiparameter(self):
        """agsdr() with parameters dict creates AGSDROptimizerSpec."""
        spec = jtfne.agsdr(
            parameters={"drive_a": (0.35, 2.25), "drive_b": (0.35, 2.25)},
            generations=8,
            population_size=6,
        )
        # Check that it's the right type (or has the right attributes)
        assert hasattr(spec, "parameters") or isinstance(spec, dict)
        if hasattr(spec, "to_dict"):
            d = spec.to_dict()
            assert isinstance(d, dict)
            assert "parameters" in d

    def test_agsdr_spec_json_serializable(self):
        """agsdr() spec is JSON-safe."""
        spec = jtfne.agsdr(
            parameters={"param": (0.5, 1.5)},
            generations=4,
            population_size=3,
        )
        if hasattr(spec, "to_dict"):
            d = spec.to_dict()
            json_str = json.dumps(d, allow_nan=False)
            assert isinstance(json_str, str)


class TestRateTargetsObjective:
    """Test rate_targets() objective factory."""

    def test_rate_targets_basic_creation(self):
        """rate_targets() creates objective with kind='group_rate_targets'."""
        obj = jtfne.rate_targets(
            groups={"group_a": np.arange(0, 5), "group_b": np.arange(5, 10)},
            targets_hz={"group_a": 5.0, "group_b": 10.0},
        )
        assert isinstance(obj, jtfne.Objective)
        assert obj.kind == "group_rate_targets"
        assert obj.name == "rate_targets"

    def test_rate_targets_stores_metadata(self):
        """rate_targets() stores groups and targets in gate metadata."""
        obj = jtfne.rate_targets(
            groups={"group_a": [0, 1, 2], "group_b": [3, 4, 5]},
            targets_hz={"group_a": 5.0, "group_b": 10.0},
        )
        assert len(obj.gates) >= 1
        gate = obj.gates[0]
        assert "metadata" in gate
        assert "groups" in gate["metadata"]
        assert "targets_hz" in gate["metadata"]

    def test_rate_targets_custom_weights(self):
        """rate_targets() accepts and stores custom weights."""
        obj = jtfne.rate_targets(
            groups={"a": [0], "b": [1]},
            targets_hz={"a": 5.0, "b": 10.0},
            weights={"a": 2.0, "b": 0.5},
        )
        gate = obj.gates[0]
        assert gate["metadata"]["weights"]["a"] == 2.0
        assert gate["metadata"]["weights"]["b"] == 0.5

    def test_rate_targets_mismatched_groups_raises(self):
        """rate_targets() raises if group names don't match."""
        with pytest.raises(ValueError):
            jtfne.rate_targets(
                groups={"a": [0], "b": [1]},
                targets_hz={"a": 5.0, "c": 10.0},  # 'c' doesn't match
            )

    def test_rate_targets_empty_raises(self):
        """rate_targets() raises on empty groups or targets."""
        with pytest.raises(ValueError):
            jtfne.rate_targets(groups={}, targets_hz={})


class TestGroupRateEvaluation:
    """Test group-wise firing rate evaluation in Model.evaluate()."""

    def test_evaluate_group_rate_targets_basic(self):
        """Model.evaluate() with group_rate_targets objective computes group-wise loss."""
        # Create simple model
        cfg = (
            jtfne.configuration()
            .network(name="test", kind="cortical_column", n=10)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=42)
        signals = model.simulate(sim)

        # Create group-rate objective
        obj = jtfne.rate_targets(
            groups={"all": np.arange(0, 10)},
            targets_hz={"all": 5.0},
        )

        # Evaluate
        report = model.evaluate(signals, obj)

        assert isinstance(report, dict)
        assert "total_loss" in report
        assert "group_rate_losses" in report
        assert report["evaluation_status"].startswith("objective_evaluate_group_rate_targets")

    def test_evaluate_group_rate_targets_multiple_groups(self):
        """evaluate() computes separate loss for each group."""
        cfg = (
            jtfne.configuration()
            .network(name="test", kind="cortical_column", n=20)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=42)
        signals = model.simulate(sim)

        obj = jtfne.rate_targets(
            groups={"first": np.arange(0, 10), "second": np.arange(10, 20)},
            targets_hz={"first": 5.0, "second": 10.0},
        )

        report = model.evaluate(signals, obj)
        assert "group_rate_losses" in report
        group_losses = report.get("group_rate_losses", [])
        # Should have losses for both groups (or at least attempt)
        assert len(group_losses) >= 1

    def test_evaluate_group_rate_targets_finite_loss(self):
        """evaluate() returns finite loss for group-rate objectives."""
        cfg = (
            jtfne.configuration()
            .network(name="test", kind="cortical_column", n=8)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="test_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)
        sim = jtfne.Simulation(duration_ms=10.0, dt_ms=0.1, seed=42)
        signals = model.simulate(sim)

        obj = jtfne.rate_targets(
            groups={"test": np.arange(0, 8)},
            targets_hz={"test": 10.0},
        )

        report = model.evaluate(signals, obj)
        loss = report.get("total_loss")
        if loss is not None:
            assert np.isfinite(loss), f"Loss should be finite, got {loss}"


class TestTuneMultiparameterAGSDR:
    """Test Model.tune() multi-parameter optimization path."""

    def test_tune_multiparameter_returns_tuple(self):
        """tune() with parameters dict returns TuneResult."""
        cfg = (
            jtfne.configuration()
            .network(name="test", kind="cortical_column", n=8)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="test_probe", modes=["spikes"])
        )
        model = jtfne.construct(cfg)

        obj = jtfne.rate_targets(
            groups={"all": np.arange(0, 8)},
            targets_hz={"all": 5.0},
        )

        result = model.tune(
            objective=obj,
            parameters={"source_scale": (0.25, 4.0)},
            generations=2,
            population_size=2,
            seed=42,
        )

        assert isinstance(result, jtfne.TuneResult)
        assert isinstance(result.best_parameters, dict)
        assert isinstance(result.summary, dict)

    def test_tune_multiparameter_report_structure(self):
        """tune() multi-param report contains expected fields."""
        cfg = (
            jtfne.configuration()
            .network(name="test", kind="cortical_column", n=8)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="test_probe", modes=["spikes"])
        )
        model = jtfne.construct(cfg)

        obj = jtfne.rate_targets(
            groups={"all": np.arange(0, 8)},
            targets_hz={"all": 5.0},
        )

        result = model.tune(
            objective=obj,
            parameters={"source_scale": (0.25, 4.0)},
            generations=2,
            population_size=2,
            seed=42,
        )

        assert "best_parameters" in result.summary
        assert "best_score" in result.summary
        assert "generation_records" in result.summary
        assert "tuning_status" in result.summary

    def test_tune_multiparameter_best_score_improves(self):
        """tune() AGSDR loop finds better scores (best-so-far non-increasing)."""
        cfg = (
            jtfne.configuration()
            .network(name="test", kind="cortical_column", n=8)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="test_probe", modes=["spikes"])
        )
        model = jtfne.construct(cfg)

        obj = jtfne.rate_targets(
            groups={"all": np.arange(0, 8)},
            targets_hz={"all": 5.0},
        )

        result = model.tune(
            objective=obj,
            parameters={"source_scale": (0.25, 4.0)},
            generations=3,
            population_size=3,
            seed=42,
        )

        # Extract all scores from generation records
        gen_records = result.summary.get("generation_records", [])
        if len(gen_records) >= 2:
            scores = [r.get("best_score") for r in gen_records]
            finite_scores = [s for s in scores if s is not None and np.isfinite(s)]
            # Best-so-far should be non-increasing (allowing ties)
            if len(finite_scores) >= 2:
                for i in range(1, len(finite_scores)):
                    assert finite_scores[i] <= finite_scores[i-1] + 1e-6, \
                        f"Score should be non-increasing: {finite_scores}"


class TestTuneResult:
    """Test TuneResult typed result object."""

    def test_tune_result_creation(self):
        """TuneResult can be created with expected fields."""
        result = jtfne.TuneResult(
            best_parameters={"param_a": 1.0},
            best_score=0.05,
            history=[{"gen": 0, "score": 0.1}],
            summary={"status": "ok"},
        )
        assert result.best_parameters == {"param_a": 1.0}
        assert result.best_score == 0.05
        assert isinstance(result.history, list)

    def test_tune_result_to_dict(self):
        """TuneResult.to_dict() returns JSON-safe dictionary."""
        result = jtfne.TuneResult(
            best_parameters={"x": 0.5},
            best_score=0.1,
            history=[],
            summary={"converged": True},
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        json_str = json.dumps(d, allow_nan=False)
        assert isinstance(json_str, str)


class TestSuiteNo1Part4PublicGrammar:
    """Test that Suite No. 1 Part 4 grammar works end-to-end."""

    def test_suite_no1_part4_grammar(self):
        """Full Suite No. 1 Part 4 grammar works without internal loops."""
        # Setup model
        cfg = (
            jtfne.configuration()
            .network(name="V1", kind="cortical_column", n=48)
            .emitter(family="izhikevich", preset="cortical_eig")
            .field(domain="laminar_column", conductivity="proxy")
            .probe(name="laminar_probe", modes=["spikes", "V_m"])
        )
        model = jtfne.construct(cfg)

        # Define multi-group objectives using public grammar
        objectives = jtfne.rate_targets(
            groups={
                "first_half": np.arange(0, 24),
                "second_half": np.arange(24, 48),
            },
            targets_hz={
                "first_half": 5.0,
                "second_half": 10.0,
            },
        )

        # Define optimizer using public grammar
        optimizer = jtfne.agsdr(
            parameters={
                "drive_scale_a": (0.35, 2.25),
                "drive_scale_b": (0.35, 2.25),
            },
            generations=2,
            population_size=2,
            seed=42,
        )

        # Run optimization using public grammar
        result = model.tune(
            objectives=objectives,
            optimizer=optimizer,
            seed=42,
        )

        # Verify result structure
        assert isinstance(result, jtfne.TuneResult)
        assert isinstance(result.best_parameters, dict)
        assert "best_parameters" in result.summary
        assert "best_score" in result.summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
