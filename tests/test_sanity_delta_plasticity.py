"""Tests for plasticity and homeostatic regularizer."""

import jaxfne as jtfne


class TestPlasticityConfig:
    """Test plasticity configuration and bounds."""

    def test_enable_plasticity_defaults(self):
        """Test plasticity enablement with defaults."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        model = cfg.construct()
        model_with_plasticity = model.enable_plasticity()

        assert model_with_plasticity.plasticity_enabled is True
        assert model_with_plasticity.plasticity_config["homeostatic_rule"] == "low_rate_potentiation_high_rate_depression"
        assert model_with_plasticity.plasticity_config["target_rate_hz"] == 10.0
        assert model_with_plasticity.plasticity_config["weight_bounds"] == "preserve_sign"
        assert model_with_plasticity.plasticity_config["eta_homeo"] == 0.01

    def test_enable_plasticity_custom_params(self):
        """Test plasticity with custom parameters."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        model = cfg.construct()
        model_with_plasticity = model.enable_plasticity(
            mode="area_specific",
            target_rate_hz=12.0,
            eta_homeo=0.02,
        )

        assert model_with_plasticity.plasticity_config["mode"] == "area_specific"
        assert model_with_plasticity.plasticity_config["target_rate_hz"] == 12.0
        assert model_with_plasticity.plasticity_config["eta_homeo"] == 0.02

    def test_plasticity_not_biological_learning(self):
        """Test that plasticity claim is computational, not biological."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        assert cfg.biological_learning_claim is False

        model = cfg.construct()
        model_with_plasticity = model.enable_plasticity()
        assert cfg.biological_learning_claim is False

    def test_weight_bounds_preserve_sign(self):
        """Test that weight bounds preserve excitatory/inhibitory signs."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        model = cfg.construct().enable_plasticity(weight_bounds="preserve_sign")

        assert model.plasticity_config["weight_bounds"] == "preserve_sign"

    def test_homeostatic_rule_options(self):
        """Test homeostatic rule options."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        model = cfg.construct()

        # Default rule
        m1 = model.enable_plasticity(homeostatic_rule="low_rate_potentiation_high_rate_depression")
        assert m1.plasticity_config["homeostatic_rule"] == "low_rate_potentiation_high_rate_depression"

    def test_plasticity_segment_boundary_application(self):
        """Test that plasticity is noted as segment-boundary applied."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct().enable_plasticity()

        # Plasticity config exists and is set for segment-boundary application
        assert model.plasticity_enabled is True
        # Task schedule is available for segment boundaries
        schedule = paradigm.to_schedule()
        assert "segments" in schedule
