"""Tests for jaxfne.tutorial_utils (laminar column helpers)."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

import jaxfne as jtfne
from jaxfne.tutorial_utils import (
    make_cell_dist,
    make_cell_type_catalog,
    cell_catalog_frame,
    make_laminar_column_config,
    config_summary_frame,
    build_laminar_column,
    build_laminar_connections,
    select_cells,
    make_stimulus,
    simulate_laminar_trials,
    spectrolaminar_from_trials,
    summarize_spectrolaminar_similarity,
    export_tutorial_artifacts,
    LaminarColumnConfig,
)


class TestMakeCellDist:
    """Test make_cell_dist helper."""

    def test_shape(self):
        """Test cell_dist shape (n_layers, n_cell_types)."""
        layers = ("L1", "L2", "L3", "L4", "L5", "L6")
        cell_types = ("E", "PV")
        cell_dist = make_cell_dist(layers, cell_types)
        assert cell_dist.shape == (6, 2)

    def test_row_normalization(self):
        """Test that each row sums to 1.0."""
        layers = ("L1", "L2", "L3", "L4", "L5", "L6")
        cell_types = ("E", "PV", "SST", "VIP")
        cell_dist = make_cell_dist(layers, cell_types)
        row_sums = cell_dist.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    def test_zero_mass_raises(self):
        """Test that zero-mass rows raise ValueError."""
        layers = ("L1",)
        cell_types = ("E", "PV")
        layer_cell_type_frac = {"L1": {"E": 0.0, "PV": 0.0}}
        with pytest.raises(ValueError):
            make_cell_dist(layers, cell_types, layer_cell_type_frac)

    def test_empty_layers_raises(self):
        """Test that empty layers raises ValueError."""
        with pytest.raises(ValueError):
            make_cell_dist([], ("E", "PV"))

    def test_empty_cell_types_raises(self):
        """Test that empty cell_types raises ValueError."""
        with pytest.raises(ValueError):
            make_cell_dist(("L1", "L2"), [])


class TestCellTypePresets:
    """Test cell type catalog and frame."""

    def test_catalog(self):
        """Test make_cell_type_catalog returns dict."""
        catalog = make_cell_type_catalog(selected=["E", "PV"])
        assert isinstance(catalog, dict)
        assert "E" in catalog
        assert "PV" in catalog
        assert "SST" not in catalog

    def test_catalog_frame(self):
        """Test cell_catalog_frame returns DataFrame."""
        catalog = make_cell_type_catalog()
        frame = cell_catalog_frame(catalog)
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) == len(catalog)
        assert "Cell Type" in frame.columns


class TestLaminarColumnConfig:
    """Test LaminarColumnConfig dataclass."""

    def test_creation(self):
        """Test config creation with defaults."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4", "PFC"),
            cell_types=("E", "PV"),
            layers=("L1", "L2", "L3", "L4", "L5", "L6"),
            n_neuron_per_column=100,
        )
        assert isinstance(cfg, LaminarColumnConfig)
        assert cfg.areas == ("V1", "V4", "PFC")
        assert cfg.cell_types == ("E", "PV")

    def test_cell_dist_frame(self):
        """Test cell_dist_frame property."""
        cfg = make_laminar_column_config(
            cell_types=("E", "PV"),
            layers=("L1", "L2", "L3", "L4", "L5", "L6"),
        )
        frame = cfg.cell_dist_frame
        assert isinstance(frame, pd.DataFrame)
        assert frame.shape == (6, 2)

    def test_truth_gates(self):
        """Test truth_gates property returns immutable dict."""
        cfg = make_laminar_column_config()
        gates = cfg.truth_gates
        assert gates['truth_mode'] == "truth_safe_unverified"
        assert gates['claim_level'] == "computational_scaffold"
        assert gates['field_solver_status'] == "laminar_proxy_no_pde"
        assert gates['physical_amplitude_claim_allowed'] is False

    def test_frozen_immutability(self):
        """Test that config is frozen (immutable)."""
        cfg = make_laminar_column_config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.seed = 999

    def test_manifest_dict(self):
        """Test to_manifest_dict JSON safety."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4"),
            n_neuron_per_column=200,
        )
        manifest = cfg.to_manifest_dict()
        assert isinstance(manifest, dict)
        assert 'areas' in manifest
        assert 'total_neurons' in manifest
        assert manifest['total_neurons'] == 400

    def test_config_summary(self):
        """Test config_summary_frame."""
        cfg = make_laminar_column_config()
        frame = config_summary_frame(cfg)
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) > 0
        assert 'Parameter' in frame.columns


class TestBuildLaminarColumn:
    """Test model building."""

    def test_build_returns_dict(self):
        """Test build_laminar_column returns dict with required keys."""
        cfg = make_laminar_column_config(n_neuron_per_column=50)
        model = build_laminar_column(cfg)
        assert isinstance(model, dict)
        assert 'neurons' in model
        assert 'positions_m' in model
        assert 'W_parts' in model
        assert 'truth_gates' in model

    def test_neuron_table_columns(self):
        """Test neuron table has required columns."""
        cfg = make_laminar_column_config(n_neuron_per_column=50)
        model = build_laminar_column(cfg)
        neurons = model['neurons']
        required_cols = {'neuron_id', 'area', 'layer', 'cell_type', 'x_m', 'y_m', 'z_m', 'pos_from_l4'}
        assert required_cols.issubset(set(neurons.columns))

    def test_neuron_count(self):
        """Test correct total neuron count."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4"),
            n_neuron_per_column=100,
        )
        model = build_laminar_column(cfg)
        assert len(model['neurons']) == 200

    def test_connection_matrices_finite(self):
        """Test connection matrices are finite."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        W_parts = model['W_parts']
        for name, W in W_parts.items():
            assert np.all(np.isfinite(W))

    def test_connection_matrix_shapes(self):
        """Test connection matrices have correct shape."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4", "PFC"),
            n_neuron_per_column=50,
        )
        model = build_laminar_column(cfg)
        n_neurons = 150  # 3 areas * 50
        W_parts = model['W_parts']
        for name, W in W_parts.items():
            assert W.shape == (n_neurons, n_neurons)

    def test_build_laminar_connections(self):
        """Test build_laminar_connections extraction."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        W_loc_e, W_loc_i, W_ff, W_fb = build_laminar_connections(model, cfg)
        assert W_loc_e.shape == W_loc_i.shape == W_ff.shape == W_fb.shape


class TestSelectCells:
    """Test cell selection."""

    def test_select_by_area(self):
        """Test selecting cells by area."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4"),
            n_neuron_per_column=50,
        )
        model = build_laminar_column(cfg)
        v1_cells = select_cells(model, area="V1")
        assert len(v1_cells) > 0
        assert len(v1_cells) <= 50

    def test_select_by_layer(self):
        """Test selecting cells by layer."""
        cfg = make_laminar_column_config(n_neuron_per_column=50)
        model = build_laminar_column(cfg)
        l4_cells = select_cells(model, layers=("L4",))
        assert len(l4_cells) > 0

    def test_select_fraction(self):
        """Test selecting a fraction of cells."""
        cfg = make_laminar_column_config(n_neuron_per_column=100)
        model = build_laminar_column(cfg)
        cells_25pct = select_cells(model, fraction=0.25, seed=42)
        assert 0 < len(cells_25pct) <= len(model['neurons']) * 0.25


class TestMakeStimulus:
    """Test stimulus creation."""

    def test_stimulus_shape(self):
        """Test stimulus array shape."""
        stim = make_stimulus(duration_ms=1000.0, dt_ms=0.1)
        assert stim.ndim == 1
        assert len(stim) == int(1000.0 / 0.1)

    def test_stimulus_finite(self):
        """Test stimulus is finite."""
        for kind in ["constant", "sine", "step", "pulses", "noise"]:
            stim = make_stimulus(kind=kind, duration_ms=100.0, seed=42)
            assert np.all(np.isfinite(stim))

    def test_stimulus_dtype(self):
        """Test stimulus dtype."""
        stim = make_stimulus(dtype="float32")
        assert stim.dtype == np.float32

    def test_constant_stimulus(self):
        """Test constant stimulus."""
        stim = make_stimulus(kind="constant", amplitude=2.0, baseline=1.0)
        assert np.all(stim == 3.0)

    def test_sine_stimulus(self):
        """Test sine stimulus oscillates."""
        stim = make_stimulus(kind="sine", duration_ms=100.0, dt_ms=0.1, frequency_hz=10.0, amplitude=1.0)
        assert stim.min() < 0.5
        assert stim.max() > 0.5


class TestSimulateTrials:
    """Test trial simulation."""

    def test_simulate_returns_dict(self):
        """Test simulate_laminar_trials returns dict."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        assert isinstance(trials, dict)

    def test_trials_keys(self):
        """Test trials dict has required keys."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        required_keys = {
            'time_ms', 'spikes', 'voltage_mV', 'source_native',
            'lfp_contacts', 'csd_contacts', 'contact_depths_m', 'area_names', 'control'
        }
        assert required_keys.issubset(trials.keys())

    def test_tensor_contract_shapes(self):
        """Test trial tensors follow the (trials, [areas,] T, N/contacts) contract."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4", "PFC"),
            cell_types=("E", "PV"),
            n_neuron_per_column=12,
            n_contacts=4,
        )
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        # Neuron-axis tensors: (trials, T, N)
        for key in ("spikes", "voltage_mV", "source_native"):
            assert trials[key].ndim == 3
        # Field-proxy tensors: (trials, areas, T, contacts)
        for key in ("lfp_contacts", "csd_contacts"):
            assert trials[key].ndim == 4, trials[key].shape
            assert trials[key].shape[0] == 2
            assert trials[key].shape[1] == len(cfg.areas)
            assert trials[key].shape[3] == cfg.n_contacts
            assert np.all(np.isfinite(trials[key]))

    def test_per_area_specs_distinct(self):
        """Test spectrolaminar specs differ across areas (no copy-same-spec)."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4", "PFC"), cell_types=("E", "PV"), n_neuron_per_column=12
        )
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        scores, specs = summarize_spectrolaminar_similarity(trials, cfg)
        assert set(specs.keys()) == set(cfg.areas)
        v1 = np.asarray(specs["V1"]["relative_power"])
        v4 = np.asarray(specs["V4"]["relative_power"])
        assert not np.allclose(v1, v4)
        assert np.all(np.isfinite(scores["similarity_percent"].to_numpy()))

    def test_per_area_specs_are_area_tagged(self):
        """Test spectrolaminar specs carry their area name."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4", "PFC"), cell_types=("E", "PV"), n_neuron_per_column=12
        )
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        scores, specs = summarize_spectrolaminar_similarity(trials, cfg, areas=cfg.areas)
        # Each spec dict must carry its own area name
        for area in cfg.areas:
            assert specs[area]["area"] == area
            # Verify all profile keys present
            for key in ["freq_hz", "pos_from_l4", "relative_power", "alpha_beta", "gamma"]:
                assert key in specs[area]

    def test_spikes_binary(self):
        """Test spikes are binary."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        spikes = trials['spikes']
        assert np.all((spikes == 0) | (spikes == 1))

    def test_voltage_range(self):
        """Test voltage is in reasonable range."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        voltage = trials['voltage_mV']
        assert voltage.min() >= -100
        assert voltage.max() <= 50

    def test_arrays_finite(self):
        """Test all arrays are finite."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        for key in ['voltage_mV', 'source_native', 'lfp_contacts', 'csd_contacts']:
            assert np.all(np.isfinite(trials[key]))


class TestSpectrolaminar:
    """Test spectrolaminar analysis."""

    def test_spectrolaminar_from_trials(self):
        """Test spectrolaminar_from_trials returns power and specs."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        power, specs = spectrolaminar_from_trials(trials, cfg)
        assert power.shape[0] == cfg.freq_count
        assert power.shape[1] == cfg.n_contacts

    def test_spectrolaminar_specs_keys(self):
        """Test spectrolaminar specs has required keys."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        power, specs = spectrolaminar_from_trials(trials, cfg)
        required_keys = {'freq_hz', 'pos_from_l4', 'relative_power', 'alpha_beta', 'gamma'}
        assert required_keys.issubset(specs.keys())

    def test_summarize_spectrolaminar_similarity(self):
        """Test summarize_spectrolaminar_similarity returns scores and specs."""
        cfg = make_laminar_column_config(n_neuron_per_column=30)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        scores, specs = summarize_spectrolaminar_similarity(trials, cfg)
        assert isinstance(scores, pd.DataFrame)
        assert 'area' in scores.columns
        assert 'similarity_percent' in scores.columns
        assert len(scores) == len(cfg.areas)


class TestExportArtifacts:
    """Test artifact export."""

    def test_export_writes_json(self, tmp_path):
        """Test export_tutorial_artifacts writes JSON files."""
        cfg = make_laminar_column_config(
            output_dir=tmp_path,
            n_neuron_per_column=30,
        )
        manifest = {'test': 'data'}
        paths = export_tutorial_artifacts(cfg, manifest_dict=manifest, output_dir=tmp_path)
        assert 'manifest_path' in paths
        assert Path(paths['manifest_path']).exists()

    def test_export_json_strict(self, tmp_path):
        """Test exported JSON is strict (no NaN/Inf)."""
        cfg = make_laminar_column_config(output_dir=tmp_path)
        manifest = cfg.to_manifest_dict()
        paths = export_tutorial_artifacts(cfg, manifest_dict=manifest, output_dir=tmp_path)

        # Read and re-serialize with allow_nan=False (should not fail)
        manifest_path = Path(paths['manifest_path'])
        text = manifest_path.read_text()
        data = json.loads(text)
        json_text = json.dumps(data, allow_nan=False)  # Should not raise
        assert json_text is not None
