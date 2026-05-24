"""Tests for jtfne.with_emitter_parameters and Model.with_emitter_parameters.

truth_mode: truth_safe_unverified
claim_level: computational_scaffold
"""

import json
import pytest
import jax.numpy as jnp
import jaxfne as jtfne


def _base_model():
    cfg = (
        jtfne.configuration()
        .network(name="test", kind="isolated_neuron", n=1, cell_types={"E": 1.0})
        .emitter(family="izhikevich", preset="cortical_eig")
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .probe(name="p", modes=["spikes", "V_m", "source"], n_contacts=16)
    )
    return jtfne.construct(cfg)


class TestWithEmitterParametersExport:
    """with_emitter_parameters is exported from jaxfne public API."""

    def test_callable_from_jtfne(self):
        assert callable(jtfne.with_emitter_parameters)

    def test_in_all(self):
        assert "with_emitter_parameters" in jtfne.__all__

    def test_model_method_callable(self):
        model = _base_model()
        assert callable(model.with_emitter_parameters)


class TestWithEmitterParametersReturnType:
    """with_emitter_parameters returns a valid Model."""

    def test_returns_model_instance(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, a=0.05)
        assert isinstance(result, jtfne.Model)

    def test_original_not_mutated(self):
        base = _base_model()
        original_a = float(base.params["emitter"].a[0])
        _ = jtfne.with_emitter_parameters(base, a=0.10)
        assert float(base.params["emitter"].a[0]) == pytest.approx(original_a)

    def test_no_args_returns_equivalent_model(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base)
        assert float(result.params["emitter"].a[0]) == pytest.approx(
            float(base.params["emitter"].a[0])
        )


class TestWithEmitterParametersSingleOverrides:
    """Each parameter override is applied correctly."""

    def test_a_override(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, a=0.10)
        assert float(result.params["emitter"].a[0]) == pytest.approx(0.10)

    def test_b_override(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, b=0.25)
        assert float(result.params["emitter"].b[0]) == pytest.approx(0.25)

    def test_c_override(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, c=-70.0)
        assert float(result.params["emitter"].c[0]) == pytest.approx(-70.0)

    def test_d_override(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, d=12.0)
        assert float(result.params["emitter"].d[0]) == pytest.approx(12.0)

    def test_drive_scale_override(self):
        base = _base_model()
        original_drive = float(base.params["emitter"].drive[0])
        result = jtfne.with_emitter_parameters(base, drive_scale=1.5)
        assert float(result.params["emitter"].drive[0]) == pytest.approx(original_drive * 1.5)

    def test_drive_scale_1_unchanged(self):
        base = _base_model()
        original_drive = float(base.params["emitter"].drive[0])
        result = jtfne.with_emitter_parameters(base, drive_scale=1.0)
        assert float(result.params["emitter"].drive[0]) == pytest.approx(original_drive)

    def test_unspecified_params_unchanged(self):
        base = _base_model()
        original_b = float(base.params["emitter"].b[0])
        result = jtfne.with_emitter_parameters(base, a=0.10)  # only a changed
        assert float(result.params["emitter"].b[0]) == pytest.approx(original_b)


class TestWithEmitterParametersClaimGatesPreserved:
    """Claim gates are preserved through with_emitter_parameters."""

    def test_physical_amplitude_claim_remains_false(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, a=0.05, drive_scale=1.2)
        assert result.summary()["physical_amplitude_claim_allowed"] is False

    def test_claim_level_unchanged(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, c=-70.0)
        assert result.summary()["claim_level"] == "computational_scaffold"

    def test_truth_mode_unchanged(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, d=12.0)
        assert result.summary()["truth_mode"] == "truth_safe_unverified"


class TestWithEmitterParametersSimulates:
    """Model modified via with_emitter_parameters produces valid simulations."""

    def test_modified_model_simulates(self):
        base = _base_model()
        model = jtfne.with_emitter_parameters(base, a=0.05)
        run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=0)
        signals = model.simulate(jtfne.simulation(duration_ms=200.0, dt_ms=0.1, seed=0, runtime=run))
        assert jnp.all(jnp.isfinite(signals.V_m))
        assert jnp.all(jnp.isfinite(signals.spikes))

    def test_drive_scale_affects_firing_rate(self):
        """Stronger drive raises firing rate (smoke-level check)."""
        base = _base_model()
        run = jtfne.runtime(device_type="auto", dtype="float32", x64_enabled=False, seed=0)
        sim_spec = jtfne.simulation(duration_ms=500.0, dt_ms=0.1, seed=0, runtime=run)

        baseline_signals = base.simulate(sim_spec)
        strong_model = jtfne.with_emitter_parameters(base, drive_scale=1.5)
        strong_signals = strong_model.simulate(sim_spec)

        rate_baseline = float(jnp.mean(baseline_signals.spikes) * (1000.0 / 0.1))
        rate_strong = float(jnp.mean(strong_signals.spikes) * (1000.0 / 0.1))
        # Stronger drive should not decrease the firing rate
        assert rate_strong >= rate_baseline - 5.0  # allow ±5 Hz tolerance


class TestWithEmitterParametersJSONSafe:
    """Outputs from modified model are JSON-safe."""

    def test_manifest_json_safe(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, a=0.05, drive_scale=1.1)
        manifest = result.manifest()
        json_str = json.dumps(manifest, allow_nan=False)
        assert "NaN" not in json_str
        assert "Infinity" not in json_str

    def test_summary_json_safe(self):
        base = _base_model()
        result = jtfne.with_emitter_parameters(base, b=0.25)
        summary = result.summary()
        json_str = json.dumps(summary, allow_nan=False)
        assert len(json_str) > 0


class TestWithEmitterParametersMethodEquivalence:
    """Standalone function and Model method produce identical results."""

    def test_standalone_equals_method(self):
        base = _base_model()
        result_fn = jtfne.with_emitter_parameters(base, a=0.05, d=10.0)
        result_method = base.with_emitter_parameters(a=0.05, d=10.0)
        assert float(result_fn.params["emitter"].a[0]) == pytest.approx(
            float(result_method.params["emitter"].a[0])
        )
        assert float(result_fn.params["emitter"].d[0]) == pytest.approx(
            float(result_method.params["emitter"].d[0])
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
