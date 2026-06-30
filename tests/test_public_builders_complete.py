"""Tests for Phase 2: Public builder and helper functions.

This test suite validates all 11 public builders in jaxfne.builders:
- default_cortical_column_config
- default_spectrolaminar_config
- build_laminar_column
- build_multi_area_columns
- connect_columns
- sparse_intercolumn_connectivity
- all_to_all_intercolumn_connectivity
- layer_celltype_count_table (Configuration or Model → counts by layer/cell_type)
- column_density_table (Configuration or Model → neurons/mm³ per layer)
- configuration_table
- validate_configuration
"""

import pytest
import jaxfne as jtfne
from jaxfne.builders import Configuration


def _default_spectrolaminar_config(
    areas=None,
    n_per_area: int = 100,
    seed=None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
) -> Configuration:
    """Local replacement for the removed jtfne.default_spectrolaminar_config.

    default_spectrolaminar_config and default_nuclei_config were removed
    from jaxfne/builders.py (archived as legacy-schema JSON at
    jaxfne/configs/legacy/spectrolaminar_default.json) but this test suite
    still needs the exact Configuration the old function built. Reproduced
    here verbatim from the pre-removal source (commit 1715ec2^) using the
    still-public Configuration fluent API -- not re-exposed on jaxfne's
    public surface.
    """
    if areas is None:
        areas = ["V1", "V4"]

    cfg = (
        Configuration()
        .runtime(seed=seed or 42, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
        .areas(areas)
    )

    for area in areas:
        cfg = cfg.column(area, layers=["L1", "L2/3", "L4", "L5", "L6"], n=n_per_area)

    cfg = (
        cfg.cell_types({"E": 0.75, "PV": 0.10, "SST": 0.08, "VIP": 0.07})
        .area_layer_cell_types(
            "V1",
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in ["L1", "L2/3", "L4", "L5", "L6"]},
        )
    )

    if len(areas) > 1:
        cfg = cfg.area_layer_cell_types(
            areas[1],
            {L: {"E": 0.75, "PV": 0.1, "SST": 0.08, "VIP": 0.07} for L in ["L1", "L2/3", "L4", "L5", "L6"]},
        )

    cfg = (
        cfg.uniform3d(radius_mm=0.25, height_mm=1.6)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.35, edge_seed=seed or 42)
    )

    if len(areas) >= 2:
        cfg = cfg.inter_column_connectivity(
            source_area=areas[0],
            target_area=areas[1],
            mode="sparse",
            p_feedforward=0.3,
            p_feedback=0.2,
            feedforward_weight_range=(0.5, 2.0),
            feedback_weight_range=(0.3, 1.5),
        )

    cfg = (
        cfg.set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source", "LFP", "CSD", "EEG", "MEG", "EMM"], n_contacts=16)
        .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        .objective(
            firing_rate_target={"E": 8.0, "PV": 15.0, "SST": 4.0, "VIP": 2.0},
            band_definitions={"alpha_beta": (8.0, 25.0), "gamma": (40.0, 150.0)},
        )
    )
    return cfg


def _default_nuclei_config(
    nucleus_name: str = "thalamus",
    n: int = 80,
    cell_type_fractions=None,
    seed=None,
    duration_ms: float = 1000.0,
    dt_ms: float = 0.1,
) -> Configuration:
    """Local replacement for the removed jtfne.default_nuclei_config.

    See _default_spectrolaminar_config docstring above -- reproduced
    verbatim from commit 1715ec2^.
    """
    if cell_type_fractions is None:
        cell_type_fractions = {"E": 0.70, "PV": 0.30}

    cfg = (
        Configuration()
        .runtime(seed=seed or 42, duration_ms=duration_ms, dt_ms=dt_ms, dtype="float32")
        .column(nucleus_name, layers=["core"], n=n)
        .cell_types(dict(cell_type_fractions))
        .uniform3d(radius_mm=0.25, height_mm=0.25)
        .connectivity(within_area="all_to_all_uniform_random", within_gain=0.45, edge_seed=seed or 42)
        .set_emitter("izhikevich", "cortical_eig")
        .probes(["spikes", "V_m", "source"], n_contacts=16)
    )
    return cfg


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

    def test_default_cortical_column_config_synaptic_kernel_choice(self):
        """synaptic_kernel is the one real behavioral choice in the native
        pipeline (preset= is metadata-only, never read back); defaults to
        'exponential'/dense, 'receptor_exponential' requires+sets edge_list."""
        cfg_default = jtfne.default_cortical_column_config(n=20, seed=0)
        assert cfg_default.metadata["synaptic_kernel"] == "exponential"
        assert cfg_default.metadata["recurrent_backend"] == "dense"

        cfg_re = jtfne.default_cortical_column_config(n=20, seed=0, synaptic_kernel="receptor_exponential")
        assert cfg_re.metadata["synaptic_kernel"] == "receptor_exponential"
        assert cfg_re.metadata["recurrent_backend"] == "edge_list"

        with pytest.raises(ValueError):
            jtfne.default_cortical_column_config(synaptic_kernel="bogus")

    def test_default_cortical_column_config_synaptic_kernel_constructs_and_differs(self):
        """Both kernel choices construct+simulate to finite output, and differ."""
        import numpy as np
        cfg_exp = jtfne.default_cortical_column_config(n=30, duration_ms=20.0, dt_ms=0.5, seed=0)
        cfg_re = jtfne.default_cortical_column_config(n=30, duration_ms=20.0, dt_ms=0.5, seed=0,
                                                       synaptic_kernel="receptor_exponential")
        sig_exp = jtfne.simulate(jtfne.construct(cfg_exp), duration_ms=20.0, dt_ms=0.5, seed=0)
        sig_re = jtfne.simulate(jtfne.construct(cfg_re), duration_ms=20.0, dt_ms=0.5, seed=0)
        assert bool(np.isfinite(np.asarray(sig_exp.get("V_m"))).all())
        assert bool(np.isfinite(np.asarray(sig_re.get("V_m"))).all())
        assert not np.array_equal(np.asarray(sig_exp.get("V_m")), np.asarray(sig_re.get("V_m")))

    def test_default_cortical_column_config_defaults(self):
        """Test default_cortical_column_config works with minimal arguments."""
        cfg = jtfne.default_cortical_column_config()
        assert isinstance(cfg, jtfne.Configuration)
        assert cfg.networks  # Should have networks
        assert cfg.emitters  # Should have emitters
        assert cfg.fields    # Should have fields

    def test_default_spectrolaminar_config(self):
        """Test default_spectrolaminar_config creates valid multi-area Configuration."""
        cfg = _default_spectrolaminar_config(areas=["V1", "V4"], n_per_area=200, seed=7)
        assert isinstance(cfg, jtfne.Configuration)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert "V1" in columns
        assert "V4" in columns
        assert columns["V1"]["n"] == 200
        assert cfg.metadata["seed"] == 7

    def test_default_spectrolaminar_config_inter_area_connectivity(self):
        """Test default_spectrolaminar_config includes inter-area connectivity."""
        cfg = _default_spectrolaminar_config(areas=["V1", "V4"], n_per_area=100)
        assert cfg.metadata.get("inter_column_connectivity") is not None
        assert cfg.metadata["inter_column_connectivity"][-1]["source_area"] == "V1"
        assert cfg.metadata["inter_column_connectivity"][-1]["target_area"] == "V4"

    def test_default_spectrolaminar_config_objectives(self):
        """Test default_spectrolaminar_config includes spectral objectives."""
        cfg = _default_spectrolaminar_config()
        assert cfg.metadata.get("objective") is not None
        obj = cfg.metadata["objective"]
        assert "band_definitions" in obj
        assert obj["band_definitions"]["gamma"] == (40.0, 150.0)


class TestThreeDefaultArchitecture:
    """default_nuclei_config / default_complete_configuration: the non-laminar
    nucleus default and the cortex+subcortex "complete" default, alongside the
    existing default_cortical_column_config."""

    def test_default_nuclei_config_creates_valid_configuration(self):
        cfg = _default_nuclei_config("thalamus", n=80, seed=42)
        assert isinstance(cfg, jtfne.Configuration)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert columns["thalamus"]["n"] == 80
        assert columns["thalamus"]["layers"] == ["core"]
        assert cfg.metadata["seed"] == 42

    def test_default_nuclei_config_is_flat_not_laminar(self):
        """A nucleus has exactly one structural layer -- no depth banding."""
        cfg = _default_nuclei_config("LGN", n=50)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert columns["LGN"]["layers"] == ["core"]

    def test_default_nuclei_config_constructs_and_simulates(self):
        import numpy as np
        cfg = _default_nuclei_config("thalamus", n=30, duration_ms=20.0, dt_ms=0.5, seed=0)
        model = jtfne.construct(cfg)
        assert model.params["emitter"].n_neurons == 30
        sig = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)
        assert bool(np.isfinite(np.asarray(sig.get("V_m"))).all())

    def test_default_nuclei_config_truth_gates_not_escalated(self):
        cfg = _default_nuclei_config()
        assert cfg.metadata["claim_level"] == "computational_scaffold"
        assert cfg.metadata["field_solver_status"] == "linear_solver"
        assert cfg.metadata["physical_amplitude_calibrated"] is False

    def test_default_complete_configuration_creates_valid_configuration(self):
        cfg = jtfne.default_complete_configuration("V1", "thalamus", n_column=80, n_nucleus=40, seed=7)
        assert isinstance(cfg, jtfne.Configuration)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert columns["V1"]["n"] == 80
        assert columns["thalamus"]["n"] == 40
        assert columns["thalamus"]["layers"] == ["core"]
        assert columns["V1"]["layers"] == ["L1", "L2/3", "L4", "L5", "L6"]

    def test_default_complete_configuration_wires_inter_area_connectivity(self):
        cfg = jtfne.default_complete_configuration("V1", "thalamus")
        assert cfg.metadata.get("inter_column_connectivity") is not None
        spec = cfg.metadata["inter_column_connectivity"][-1]
        assert spec["source_area"] == "V1"
        assert spec["target_area"] == "thalamus"

    def test_default_complete_configuration_constructs_and_simulates(self):
        import numpy as np
        cfg = jtfne.default_complete_configuration(
            "V1", "thalamus", n_column=60, n_nucleus=30, duration_ms=20.0, dt_ms=0.5, seed=0)
        model = jtfne.construct(cfg)
        assert model.params["emitter"].n_neurons == 90
        sig = jtfne.simulate(model, duration_ms=20.0, dt_ms=0.5, seed=0)
        assert bool(np.isfinite(np.asarray(sig.get("V_m"))).all())

    def test_default_complete_configuration_truth_gates_not_escalated(self):
        cfg = jtfne.default_complete_configuration()
        assert cfg.metadata["claim_level"] == "computational_scaffold"
        assert cfg.metadata["field_solver_status"] == "linear_solver"
        assert cfg.metadata["physical_amplitude_calibrated"] is False

    def test_three_defaults_importable_from_jaxfne_root(self):
        """default_nuclei_config was intentionally removed from the public
        API (archived as legacy JSON); only the two still-public defaults
        are asserted here."""
        assert callable(jtfne.default_complete_configuration)
        assert callable(jtfne.default_cortical_column_config)
        assert not hasattr(jtfne, "default_nuclei_config")


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

    def test_build_laminar_column_no_args(self):
        """build_laminar_column() works with zero arguments (canonical-size V1)."""
        cfg = jtfne.build_laminar_column()
        assert isinstance(cfg, jtfne.Configuration)
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert "V1" in columns
        assert columns["V1"]["n"] == 1000

    def test_build_laminar_column_canonical_profile(self):
        """ei_profile='canonical' yields the ground-truth E:I gradient via laminar geometry."""
        import collections
        cfg = (
            jtfne.build_laminar_column(n=1000, ei_profile="canonical")
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes"], n_contacts=8)
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        )
        tbl = jtfne.construct(cfg).neuron_table()
        by_layer = collections.defaultdict(collections.Counter)
        for r in tbl:
            by_layer[r["layer"]][r["cell_type"]] += 1
        # Real laminar layers preserved (not collapsed to "uniform_3d").
        assert set(by_layer) == set(jtfne.DEFAULT_LAYERS)
        # E-fraction rises with depth: L1 superficial < L6 deep.
        l1 = by_layer["L1"]; l6 = by_layer["L6"]
        e_l1 = l1["E"] / sum(l1.values())
        e_l6 = l6["E"] / sum(l6.values())
        assert e_l1 < 0.6 < e_l6
        # 2026-06-19 reference: L1 is VIP-rich with only a small PV fraction (0.05),
        # while the deepest layer L6 has no PV. L1 is strongly inhibitory overall.
        assert l1["VIP"] > l1["PV"]            # L1 VIP-dominated
        assert l6["PV"] == 0                   # no PV in the deepest layer
        assert (sum(l1.values()) - l1["E"]) / sum(l1.values()) >= 0.4  # L1 inhibition-heavy

    def test_build_laminar_column_canonical_requires_known_layers(self):
        """ei_profile='canonical' rejects unknown layer sets instead of silent flat fallback."""
        with pytest.raises(ValueError):
            jtfne.build_laminar_column(layers=["A", "B"], ei_profile="canonical")

    def test_build_laminar_column_flat_backward_compatible(self):
        """Default flat profile keeps legacy uniform3d placement."""
        import collections
        cfg = (
            jtfne.build_laminar_column("M1", 300)
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes"], n_contacts=8)
            .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann")
        )
        tbl = jtfne.construct(cfg).neuron_table()
        layers = collections.Counter(r["layer"] for r in tbl)
        assert set(layers) == {"uniform_3d"}

    def test_build_multi_area_columns_no_args(self):
        """build_multi_area_columns() defaults to the V1→V4→PFC hierarchy."""
        cfg = jtfne.build_multi_area_columns()
        columns = {col["name"]: col for col in cfg.metadata.get("columns", [])}
        assert {"V1", "V4", "PFC"} <= set(columns)
        assert columns["V1"]["n"] == 200

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
        links = cfg.metadata.get("inter_column_connectivity")
        assert isinstance(links, list) and len(links) > 0
        area_pairs = {(link["source_area"], link["target_area"]) for link in links}
        assert ("V1", "V4") in area_pairs
        assert ("V4", "V1") in area_pairs

    def test_connect_columns(self):
        """Test connect_columns adds inter-area connectivity."""
        cfg = jtfne.build_multi_area_columns(["V1", "V4"], n_per_area=100)
        cfg = jtfne.connect_columns(cfg, "V1", "V4", mode="all_to_all")
        assert cfg.metadata["inter_column_connectivity"][-1]["mode"] == "all_to_all"


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

    def test_layer_celltype_count_table(self):
        """Test layer_celltype_count_table returns nested count dict."""
        cfg = jtfne.default_cortical_column_config(n=100)
        table = jtfne.layer_celltype_count_table(cfg)
        assert isinstance(table, dict)
        total = sum(sum(ct.values()) for ct in table.values())
        assert total == 100
        for layer_counts in table.values():
            assert isinstance(layer_counts, dict)
            assert all(isinstance(v, int) and v >= 0 for v in layer_counts.values())

    def test_column_density_table(self):
        """Test column_density_table returns positive density per layer."""
        cfg = jtfne.default_cortical_column_config(n=100)
        table = jtfne.column_density_table(cfg)
        assert isinstance(table, dict)
        assert table
        assert all(isinstance(v, float) and v > 0.0 for v in table.values())

    def test_configuration_table(self):
        """Test configuration_table returns summary dict."""
        cfg = _default_spectrolaminar_config()
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
        assert gates["physical_amplitude_calibrated"] is False


class TestPublicExports:
    """Tests for public API exports."""

    def test_builders_importable_from_jaxfne_root(self):
        """Test all still-public builders are importable from jaxfne.

        default_spectrolaminar_config and default_nuclei_config were
        intentionally removed from the public API (archived as legacy-schema
        JSON at jaxfne/configs/legacy/) -- not part of this contract anymore.
        """
        # These should not raise ImportError
        assert callable(jtfne.default_cortical_column_config)
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
        assert not hasattr(jaxfne, "default_spectrolaminar_config"), (
            "default_spectrolaminar_config was intentionally removed from "
            "the public API (archived as legacy JSON); this asserts it "
            "stays removed"
        )
        assert hasattr(jaxfne, "build_laminar_column")
        assert hasattr(jaxfne, "configuration_table")

    def test_default_cortical_column_config_in_dunder_all(self):
        """default_cortical_column_config was importable from jaxfne root but
        missing from __all__ itself (so `from jaxfne import *` silently
        dropped it) -- regression guard. default_spectrolaminar_config was
        intentionally removed from the public API entirely (see
        test_builders_in_all) so it is no longer asserted here."""
        assert "default_cortical_column_config" in jtfne.__all__
        assert "default_spectrolaminar_config" not in jtfne.__all__


class TestJSONSafety:
    """Tests for JSON-safe output."""

    def test_configuration_table_json_safe(self):
        """Test configuration_table output is JSON-safe."""
        import json
        cfg = _default_spectrolaminar_config()
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
