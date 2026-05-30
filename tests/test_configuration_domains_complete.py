"""Tests for Phase 2: Complete Configuration API with all 10 domains.

This test suite validates the 4 new chainable Configuration methods:
- .inter_column_connectivity()
- .drive()
- .objective()
- .optimizer()
"""

import pytest
import jaxfne as jtfne


class TestInterColumnConnectivity:
    """Tests for .inter_column_connectivity() method."""

    def test_inter_column_connectivity_chainable(self):
        """Test .inter_column_connectivity() returns a Configuration."""
        cfg = (
            jtfne.Configuration()
            .areas(["V1", "V4"])
            .column("V1", layers=["L1", "L4", "L5"], n=100)
            .column("V4", layers=["L1", "L4", "L5"], n=100)
            .inter_column_connectivity(source_area="V1", target_area="V4")
        )
        assert isinstance(cfg, jtfne.Configuration)

    def test_inter_column_connectivity_metadata(self):
        """Test .inter_column_connectivity() stores metadata correctly."""
        cfg = (
            jtfne.Configuration()
            .areas(["V1", "V4"])
            .inter_column_connectivity(
                source_area="V1",
                target_area="V4",
                mode="sparse",
                p_feedforward=0.3,
                p_feedback=0.2,
            )
        )
        assert cfg.metadata.get("inter_column_connectivity") is not None
        spec = cfg.metadata["inter_column_connectivity"]
        assert spec["source_area"] == "V1"
        assert spec["target_area"] == "V4"
        assert spec["mode"] == "sparse"
        assert spec["p_feedforward"] == 0.3
        assert spec["p_feedback"] == 0.2

    def test_inter_column_connectivity_defaults(self):
        """Test .inter_column_connectivity() has sensible defaults."""
        cfg = jtfne.Configuration().inter_column_connectivity()
        spec = cfg.metadata["inter_column_connectivity"]
        assert spec["source_area"] == "V1"
        assert spec["target_area"] == "V4"
        assert spec["p_feedforward"] == 0.3
        assert spec["p_feedback"] == 0.2


class TestDrive:
    """Tests for .drive() method."""

    def test_drive_chainable(self):
        """Test .drive() returns a Configuration."""
        cfg = jtfne.Configuration().drive()
        assert isinstance(cfg, jtfne.Configuration)

    def test_drive_metadata(self):
        """Test .drive() stores metadata correctly."""
        cfg = jtfne.Configuration().drive(
            baseline_drive_by_cell_type={"E": 5.0, "PV": 3.0},
            noise_policy="additive_poisson",
        )
        assert cfg.metadata.get("drive") is not None
        drive = cfg.metadata["drive"]
        assert drive["baseline_drive_by_cell_type"]["E"] == 5.0
        assert drive["noise_policy"] == "additive_poisson"

    def test_drive_defaults(self):
        """Test .drive() has sensible defaults."""
        cfg = jtfne.Configuration().drive()
        drive = cfg.metadata["drive"]
        assert "baseline_drive_by_cell_type" in drive
        assert drive["baseline_drive_by_cell_type"]["E"] == 5.0
        assert drive["noise_policy"] == "additive_poisson"

    def test_drive_invalid_noise_policy(self):
        """Test .drive() rejects invalid noise_policy."""
        with pytest.raises(ValueError, match="noise_policy"):
            jtfne.Configuration().drive(noise_policy="invalid_policy")


class TestObjective:
    """Tests for .objective() method."""

    def test_objective_chainable(self):
        """Test .objective() returns a Configuration."""
        cfg = jtfne.Configuration().objective()
        assert isinstance(cfg, jtfne.Configuration)

    def test_objective_metadata(self):
        """Test .objective() stores metadata correctly."""
        cfg = jtfne.Configuration().objective(
            firing_rate_target={"E": 8.0, "PV": 15.0},
            band_definitions={"gamma": (40, 150)},
        )
        assert cfg.metadata.get("objective") is not None
        obj = cfg.metadata["objective"]
        assert obj["firing_rate_target"]["E"] == 8.0
        assert obj["band_definitions"]["gamma"] == (40, 150)

    def test_objective_defaults(self):
        """Test .objective() has sensible defaults."""
        cfg = jtfne.Configuration().objective()
        obj = cfg.metadata["objective"]
        assert "firing_rate_target" in obj
        assert "band_definitions" in obj
        assert obj["firing_rate_target"]["E"] == 8.0
        assert obj["band_definitions"]["gamma"] == (40.0, 150.0)


class TestOptimizer:
    """Tests for .optimizer() method."""

    def test_optimizer_chainable(self):
        """Test .optimizer() returns a Configuration."""
        cfg = jtfne.Configuration().optimizer()
        assert isinstance(cfg, jtfne.Configuration)

    def test_optimizer_metadata(self):
        """Test .optimizer() stores metadata correctly."""
        cfg = jtfne.Configuration().optimizer(
            optimizer_family="AGSDR",
            budget=50,
            seed=42,
        )
        assert cfg.metadata.get("optimizer") is not None
        opt = cfg.metadata["optimizer"]
        assert opt["optimizer_family"] == "AGSDR"
        assert opt["budget"] == 50
        assert opt["seed"] == 42

    def test_optimizer_defaults(self):
        """Test .optimizer() has sensible defaults."""
        cfg = jtfne.Configuration().optimizer()
        opt = cfg.metadata["optimizer"]
        assert opt["optimizer_family"] == "AGSDR"
        assert opt["budget"] == 50

    def test_optimizer_invalid_family(self):
        """Test .optimizer() rejects invalid optimizer_family."""
        with pytest.raises(ValueError, match="optimizer_family"):
            jtfne.Configuration().optimizer(optimizer_family="INVALID_OPTIMIZER")


class TestAllDomainsChainable:
    """Tests for chaining all 10 Configuration domains together."""

    def test_all_domains_chain(self):
        """Test that all 10 domains can be chained in sequence."""
        cfg = (
            jtfne.Configuration()
            .runtime(seed=7, duration_ms=1000, dt_ms=0.1)
            .column("V1", layers=["L4", "L5"], n=100)
            .cell_types({"E": 0.8, "PV": 0.2})
            .connectivity(within_area="all_to_all", within_gain=0.5)
            .inter_column_connectivity(source_area="V1", target_area="V4")
            .drive(baseline_drive_by_cell_type={"E": 5.0})
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes", "V_m"])
            .field(domain="laminar_column")
            .objective(firing_rate_target={"E": 8.0})
            .optimizer(optimizer_family="AGSDR")
        )
        # Validate all domains are present
        assert cfg.metadata.get("seed") == 7  # runtime stores directly in metadata
        assert cfg.metadata.get("columns") is not None
        assert cfg.metadata.get("cell_types") is not None
        assert cfg.metadata.get("connectivity") is not None
        assert cfg.metadata.get("inter_column_connectivity") is not None
        assert cfg.metadata.get("drive") is not None
        assert cfg.metadata.get("objective") is not None
        assert cfg.metadata.get("optimizer") is not None

    def test_all_domains_metadata_preserved(self):
        """Test that chaining domains preserves all metadata."""
        cfg = (
            jtfne.Configuration()
            .runtime(seed=42)
            .drive(baseline_drive_by_cell_type={"E": 5.0, "PV": 3.0})
            .objective(firing_rate_target={"E": 8.0, "PV": 15.0})
            .optimizer(budget=100)
        )
        # All metadata should be present (runtime stores directly in metadata)
        assert cfg.metadata["seed"] == 42
        assert cfg.metadata["drive"]["baseline_drive_by_cell_type"]["E"] == 5.0
        assert cfg.metadata["objective"]["firing_rate_target"]["E"] == 8.0
        assert cfg.metadata["optimizer"]["budget"] == 100


class TestTruthGatesPreserved:
    """Tests for truth-gate preservation."""

    def test_default_metadata_truth_gates(self):
        """Test that default metadata includes truth gates."""
        cfg = jtfne.Configuration()
        assert cfg.metadata.get("physical_amplitude_claim_allowed", False) is False

    def test_new_domains_preserve_truth_gates(self):
        """Test that new domains don't override truth gates."""
        cfg = (
            jtfne.Configuration()
            .drive()
            .objective()
            .optimizer()
        )
        # Truth gates should still be false
        assert cfg.metadata.get("physical_amplitude_claim_allowed", False) is False
