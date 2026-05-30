"""Tests for Phase 2: Public builder and helper functions.

This test suite validates all 11 public builders in jaxfne.builders:
- default_cortical_column_config
- default_spectrolaminar_config
- build_laminar_column
- build_multi_area_columns
- connect_columns
- sparse_intercolumn_connectivity
- all_to_all_intercolumn_connectivity
- layer_celltype_count_table (raises NotImplementedError)
- column_density_table (raises NotImplementedError)
- configuration_table
- validate_configuration
"""

import pytest
import jaxfne as jtfne


class TestDefaultConfigs:
    """Tests for default configuration generators."""

    def test_default_cortical_column_config(self):
        """Test default_cortical_column_config creates valid Configuration."""
        cfg = jtfne.default_cortical_column_config(column_name="V1", n=500, seed=42)
        assert isinstance(cfg, jtfne.Configuration)
        assert cfg.metadata["columns"][0]["n"] == 500
        assert cfg.metadata["seed"] == 42
        assert len(cfg.networks) > 0
        assert len(cfg.emitters) > 0

    def test_default_cortical_column_config_defaults(self):
        """Test default_cortical_column_config works with minimal arguments."""
        cfg = jtfne.default_cortical_column_config()
        assert isinstance(cfg, jtfne.Configuration)
        assert cfg.networks  # Should have networks
        assert cfg.emitters  # Should have emitters
        assert cfg.fields    # Should have fields

    def test_default_spectrolaminar_config(self):
        """Test default_spectrolaminar_config creates valid multi-area Configuration."""
        cfg = jtfne.default_spectrolaminar_config(areas=["V1", "V4"], n_per_area=200, seed=7)
        assert isinstance(cfg, jtfne.Configuration)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert "V1" in columns
        assert "V4" in columns
        assert columns["V1"]["n"] == 200
        assert cfg.metadata["seed"] == 7

    def test_default_spectrolaminar_config_inter_area_connectivity(self):
        """Test default_spectrolaminar_config includes inter-area connectivity."""
        cfg = jtfne.default_spectrolaminar_config(areas=["V1", "V4"], n_per_area=100)
        assert cfg.metadata.get("inter_column_connectivity") is not None
        assert cfg.metadata["inter_column_connectivity"]["source_area"] == "V1"
        assert cfg.metadata["inter_column_connectivity"]["target_area"] == "V4"

    def test_default_spectrolaminar_config_objectives(self):
        """Test default_spectrolaminar_config includes spectral objectives."""
        cfg = jtfne.default_spectrolaminar_config()
        assert cfg.metadata.get("objective") is not None
        obj = cfg.metadata["objective"]
        assert "band_definitions" in obj
        assert obj["band_definitions"]["gamma"] == (40.0, 150.0)


class TestBuilderFunctions:
    """Tests for column builder functions."""

    def test_build_laminar_column(self):
        """Test build_laminar_column creates single-column Configuration."""
        cfg = jtfne.build_laminar_column("M1", n=300, layers=["L1", "L5"])
        assert isinstance(cfg, jtfne.Configuration)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert columns["M1"]["n"] == 300
        assert columns["M1"]["layers"] == ["L1", "L5"]

    def test_build_laminar_column_defaults(self):
        """Test build_laminar_column uses sensible defaults."""
        cfg = jtfne.build_laminar_column("V1", n=100)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert columns["V1"]["layers"] == ["L1", "L2/3", "L4", "L5", "L6"]

    def test_build_multi_area_columns(self):
        """Test build_multi_area_columns creates multi-area Configuration."""
        cfg = jtfne.build_multi_area_columns(["V1", "V4", "PFC"], n_per_area=250)
        assert isinstance(cfg, jtfne.Configuration)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert "V1" in columns
        assert "V4" in columns
        assert "PFC" in columns
        assert columns["V1"]["n"] == 250

    def test_build_multi_area_columns_inter_area_connectivity(self):
        """Test build_multi_area_columns includes inter-area connectivity."""
        cfg = jtfne.build_multi_area_columns(["V1", "V4"], n_per_area=100)
        # Should have inter-area connectivity between V1 and V4
        assert cfg.metadata.get("inter_column_connectivity") is not None

    def test_connect_columns(self):
        """Test connect_columns adds inter-area connectivity."""
        cfg = jtfne.build_multi_area_columns(["V1", "V4"], n_per_area=100)
        cfg = jtfne.connect_columns(cfg, "V1", "V4", mode="all_to_all")
        assert cfg.metadata["inter_column_connectivity"]["mode"] == "all_to_all"


class TestConnectivitySpecs:
    """Tests for connectivity specification factories."""

    def test_sparse_intercolumn_connectivity_spec(self):
        """Test sparse_intercolumn_connectivity returns valid spec dict."""
        spec = jtfne.sparse_intercolumn_connectivity(p_feedforward=0.4)
        assert isinstance(spec, dict)
        assert spec["mode"] == "sparse"
        assert spec["p_feedforward"] == 0.4
        assert "feedforward_weight_range" in spec
        assert "feedback_weight_range" in spec

    def test_sparse_intercolumn_connectivity_defaults(self):
        """Test sparse_intercolumn_connectivity has sensible defaults."""
        spec = jtfne.sparse_intercolumn_connectivity()
        assert spec["p_feedforward"] == 0.3
        assert spec["p_feedback"] == 0.2

    def test_all_to_all_intercolumn_connectivity_spec(self):
        """Test all_to_all_intercolumn_connectivity returns valid spec dict."""
        spec = jtfne.all_to_all_intercolumn_connectivity(feedforward_gain=0.7)
        assert isinstance(spec, dict)
        assert spec["mode"] == "all_to_all"
        assert "feedforward_weight_range" in spec
        assert "feedback_weight_range" in spec


class TestAnalysisFunctions:
    """Tests for analysis/introspection functions."""

    def test_layer_celltype_count_table_raises_not_implemented(self):
        """Test layer_celltype_count_table raises NotImplementedError."""
        cfg = jtfne.default_cortical_column_config(n=100)
        with pytest.raises(NotImplementedError, match="layer_celltype_count_table"):
            jtfne.layer_celltype_count_table(cfg)

    def test_column_density_table_raises_not_implemented(self):
        """Test column_density_table raises NotImplementedError."""
        cfg = jtfne.default_cortical_column_config(n=100)
        with pytest.raises(NotImplementedError, match="column_density_table"):
            jtfne.column_density_table(cfg)

    def test_configuration_table(self):
        """Test configuration_table returns summary dict."""
        cfg = jtfne.default_spectrolaminar_config()
        table = jtfne.configuration_table(cfg)
        assert isinstance(table, dict)
        assert "runtime" in table
        assert "columns" in table
        assert "connectivity" in table
        assert "inter_column_connectivity" in table

    def test_validate_configuration_pass(self):
        """Test validate_configuration on valid Configuration."""
        cfg = (
            jtfne.Configuration()
            .column("V1", layers=["L1"], n=100)
            .cell_types({"E": 1.0})
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes"])
            .field(domain="laminar_column")
        )
        result = jtfne.validate_configuration(cfg, strict=False)
        assert "status" in result
        assert "truth_gates" in result

    def test_validate_configuration_truth_gates(self):
        """Test validate_configuration preserves truth gates."""
        cfg = jtfne.Configuration()
        result = jtfne.validate_configuration(cfg, strict=False)
        gates = result["truth_gates"]
        assert gates["physical_amplitude_claim_allowed"] is False


class TestPublicExports:
    """Tests for public API exports."""

    def test_builders_importable_from_jaxfne_root(self):
        """Test all builders are importable from jaxfne."""
        # These should not raise ImportError
        assert callable(jtfne.default_cortical_column_config)
        assert callable(jtfne.default_spectrolaminar_config)
        assert callable(jtfne.build_laminar_column)
        assert callable(jtfne.build_multi_area_columns)
        assert callable(jtfne.connect_columns)
        assert callable(jtfne.sparse_intercolumn_connectivity)
        assert callable(jtfne.all_to_all_intercolumn_connectivity)
        assert callable(jtfne.layer_celltype_count_table)
        assert callable(jtfne.column_density_table)
        assert callable(jtfne.configuration_table)
        assert callable(jtfne.validate_configuration)

    def test_builders_in_all(self):
        """Test that builder functions are documented/discoverable."""
        import jaxfne
        # All builder functions should be at module level
        assert hasattr(jaxfne, "default_cortical_column_config")
        assert hasattr(jaxfne, "default_spectrolaminar_config")
        assert hasattr(jaxfne, "build_laminar_column")
        assert hasattr(jaxfne, "configuration_table")


class TestJSONSafety:
    """Tests for JSON-safe output."""

    def test_configuration_table_json_safe(self):
        """Test configuration_table output is JSON-safe."""
        import json
        cfg = jtfne.default_spectrolaminar_config()
        table = jtfne.configuration_table(cfg)
        # Should not raise on JSON serialization with allow_nan=False
        try:
            json.dumps(table, allow_nan=False)
        except (ValueError, TypeError) as e:
            pytest.fail(f"configuration_table output not JSON-safe: {e}")

    def test_validate_configuration_json_safe(self):
        """Test validate_configuration output is JSON-safe."""
        import json
        cfg = jtfne.Configuration()
        result = jtfne.validate_configuration(cfg, strict=False)
        # Should not raise on JSON serialization with allow_nan=False
        try:
            json.dumps(result, allow_nan=False)
        except (ValueError, TypeError) as e:
            pytest.fail(f"validate_configuration output not JSON-safe: {e}")
