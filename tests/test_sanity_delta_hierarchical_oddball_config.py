"""Tests for SanityDeltaConfig and model construction."""

import pytest
import jaxfne as jtfne


class TestSanityDeltaConfigFactory:
    """Test configuration factory and validation."""

    def test_hierarchical_global_local_oddball_factory(self):
        """Test factory creates valid config."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()

        assert cfg.seed == 0
        assert cfg.duration_ms == 12000.0
        assert cfg.dt_ms == 0.1
        assert cfg.neurons_per_area == 100
        assert cfg.areas == ("V1a", "V1b", "V4", "MT", "PFC")
        assert len(cfg.hierarchy) == 4
        assert cfg.cell_counts["E"] == 70
        assert cfg.stimulus_frequency_hz == 100.0

    def test_truth_gates_preserved(self):
        """Test that truth gates are set correctly."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()

        assert cfg.truth_mode == "truth_safe_unverified"
        assert cfg.claim_level == "computational_scaffold"
        assert cfg.field_solver_status == "laminar_proxy_no_pde"
        assert cfg.physical_amplitude_claim_allowed is False
        assert cfg.biological_learning_claim is False

    def test_stimulus_map_structure(self):
        """Test stimulus map contains correct drives."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()

        assert "A" in cfg.stimulus_map
        assert "B" in cfg.stimulus_map
        assert cfg.stimulus_map["A"]["orientation_deg"] == 45.0
        assert cfg.stimulus_map["A"]["V1a"] == 0.75
        assert cfg.stimulus_map["A"]["V1b"] == 0.25
        assert cfg.stimulus_map["B"]["orientation_deg"] == 135.0

    def test_construct_returns_model(self):
        """Test that construct() returns a SanityDeltaModel."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        model = cfg.construct()

        assert isinstance(model, jtfne.SanityDeltaModel)
        assert model.n_neurons == 500
        assert model.config is cfg

    def test_make_paradigm_creates_oddball(self):
        """Test paradigm creation."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()

        assert isinstance(paradigm, jtfne.HierarchicalOddballParadigm)
        assert paradigm.name == "global_local_oddball_AAAB"
        assert paradigm.sequence == ("A", "A", "A", "B")

    def test_five_areas_validation(self):
        """Test that 5 areas are required."""
        with pytest.raises(AssertionError):
            jtfne.SanityDeltaConfig.hierarchical_global_local_oddball(
                areas=("V1", "V4", "MT")
            )

    def test_500_neurons_total(self):
        """Test that total neurons equals 500."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        assert len(cfg.areas) * cfg.neurons_per_area == 500

    def test_cell_counts_100_per_area(self):
        """Test that cell counts sum to 100."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        assert sum(cfg.cell_counts.values()) == 100
        assert cfg.cell_counts["E"] == 70
        assert sum(c for k, c in cfg.cell_counts.items() if k != "E") == 30


class TestSanityDeltaModel:
    """Test model wrapping and methods."""

    def test_enable_plasticity(self):
        """Test plasticity enablement."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        model = cfg.construct()
        model_with_plasticity = model.enable_plasticity()

        assert model_with_plasticity.plasticity_enabled is True
        assert model_with_plasticity.plasticity_config["homeostatic_rule"] == "low_rate_potentiation_high_rate_depression"
        assert model_with_plasticity.plasticity_config["target_rate_hz"] == 10.0

    def test_initialize_backup(self):
        """Test backup state initialization."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        model = cfg.construct()
        paradigm = cfg.make_paradigm()
        backup = model.initialize_backup(paradigm)

        assert isinstance(backup, jtfne.BackupState)
        assert backup.time_ms == 0.0
        assert backup.fixation_counter == 0
        assert backup.runtime_metadata["n_neurons"] == 500
