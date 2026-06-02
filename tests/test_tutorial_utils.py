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


def make_test_config(**kwargs):
    """Small deterministic config for fast unit tests."""
    defaults = dict(
        n_neuron_per_column=12,
        duration_ms=20.0,
        dt_ms=0.1,
        n_trials=1,
        n_contacts=4,
        freq_count=8,
    )
    defaults.update(kwargs)
    return make_laminar_column_config(**defaults)


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
        cfg = make_test_config(
            areas=("V1", "V4", "PFC"),
            cell_types=("E", "PV"),
            layers=("L1", "L2", "L3", "L4", "L5", "L6"),
            n_neuron_per_column=12,
        )
        assert isinstance(cfg, LaminarColumnConfig)
        assert cfg.areas == ("V1", "V4", "PFC")
        assert cfg.cell_types == ("E", "PV")

    def test_cell_dist_frame(self):
        """Test cell_dist_frame property."""
        cfg = make_test_config(
            cell_types=("E", "PV"),
            layers=("L1", "L2", "L3", "L4", "L5", "L6"),
        )
        frame = cfg.cell_dist_frame
        assert isinstance(frame, pd.DataFrame)
        assert frame.shape == (6, 2)

    def test_truth_gates(self):
        """Test truth_gates property returns immutable dict."""
        cfg = make_test_config()
        gates = cfg.truth_gates
        assert gates['truth_mode'] == "truth_safe_unverified"
        assert gates['claim_level'] == "computational_scaffold"
        assert gates['field_solver_status'] == "laminar_proxy_no_pde"
        assert gates['physical_amplitude_claim_allowed'] is False

    def test_frozen_immutability(self):
        """Test that config is frozen (immutable)."""
        cfg = make_test_config()
        with pytest.raises((AttributeError, TypeError)):
            cfg.seed = 999

    def test_manifest_dict(self):
        """Test to_manifest_dict JSON safety."""
        cfg = make_test_config(
            areas=("V1", "V4"),
            n_neuron_per_column=12,
        )
        manifest = cfg.to_manifest_dict()
        assert isinstance(manifest, dict)
        assert 'areas' in manifest
        assert 'total_neurons' in manifest
        assert manifest['total_neurons'] == len(cfg.areas) * cfg.n_neuron_per_column

    def test_config_summary(self):
        """Test config_summary_frame."""
        cfg = make_test_config()
        frame = config_summary_frame(cfg)
        assert isinstance(frame, pd.DataFrame)
        assert len(frame) > 0
        assert 'Parameter' in frame.columns

    def test_rejects_negative_noise(self):
        """Config validation must reject negative per-type noise."""
        with pytest.raises(ValueError, match="noise must be nonnegative"):
            make_laminar_column_config(
                cell_type_izh_params={"E": {"drive": 4.0, "noise": -0.1}}
            )

    def test_rejects_nonfinite_drive(self):
        """Config validation must reject non-finite per-type drive."""
        with pytest.raises(ValueError, match="drive must be finite"):
            make_laminar_column_config(
                cell_type_izh_params={"E": {"drive": float("nan")}}
            )

    def test_cell_type_noise_override_changes_spikes(self):
        """Per-cell-type `noise` must reach the engine (zero vs high noise differ)."""
        def total_spikes(noise):
            cfg = make_laminar_column_config(
                areas=("V1",), cell_types=("E", "PV"), n_neuron_per_column=24,
                n_contacts=8, duration_ms=120.0, dt_ms=0.5, n_trials=1,
                cell_type_izh_params={
                    "E": {"a": 0.015, "b": 0.20, "c": -60.0, "d": 10.0, "drive": 2.0, "noise": noise},
                    "PV": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0, "drive": 2.0, "noise": noise},
                },
            )
            model = build_laminar_column(cfg)
            return int(simulate_laminar_trials(model, cfg, n_trials=1)["spikes"].sum())

        quiet = total_spikes(0.0)
        loud = total_spikes(6.0)
        assert loud != quiet, f"noise override had no effect: quiet={quiet}, loud={loud}"

    def test_default_config_firing_rates_are_stable(self):
        """Default per-cell-type drive/noise keep rates in a stable band (<~40 Hz)."""
        cfg = make_laminar_column_config(
            areas=("V1", "V4"), cell_types=("E", "PV", "SST", "VIP"),
            n_neuron_per_column=60, n_contacts=16, duration_ms=300.0, dt_ms=0.5,
            n_trials=2,
        )
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        per_neuron_hz = trials["spikes"].mean(axis=(0, 1)) * 1000.0 / cfg.dt_ms
        mean_hz = float(per_neuron_hz.mean())
        max_hz = float(per_neuron_hz.max())
        assert max_hz <= 45.0, f"peak rate {max_hz:.1f} Hz exceeds stable ceiling"
        assert 2.0 <= mean_hz <= 15.0, f"mean rate {mean_hz:.1f} Hz outside stable band"


class TestBuildLaminarColumn:
    """Test model building."""

    def test_build_returns_dict(self):
        """Test build_laminar_column returns dict with required keys."""
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        assert isinstance(model, dict)
        assert 'neurons' in model
        assert 'positions_m' in model
        assert 'W_parts' in model
        assert 'truth_gates' in model

    def test_neuron_table_columns(self):
        """Test neuron table has required columns."""
        cfg = make_test_config(n_neuron_per_column=12)
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
        cfg = make_test_config(n_neuron_per_column=12)
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
        cfg = make_test_config(n_neuron_per_column=12)
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
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        l4_cells = select_cells(model, layers=("L4",))
        assert len(l4_cells) > 0

    def test_select_fraction(self):
        """Test selecting a fraction of cells."""
        cfg = make_test_config(n_neuron_per_column=12)
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
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        assert isinstance(trials, dict)

    def test_trials_keys(self):
        """Test trials dict has required keys."""
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        required_keys = {
            'time_ms', 'spikes', 'voltage_mV', 'source_native',
            'lfp_contacts', 'csd_contacts', 'contact_depths_m', 'area_names', 'control'
        }
        assert required_keys.issubset(trials.keys())

    def test_tensor_contract_shapes(self):
        """Test trial tensors follow the (trials, [areas,] T, N/contacts) contract."""
        cfg = make_test_config(
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

    def test_real_engine_produces_depth_structure_and_activity(self):
        """Regression: trials come from the real Izhikevich engine + a
        depth-dependent leadfield, NOT white-noise mock.

        Guards against reverting to the old mock path (which broadcast one
        area-mean to every contact + noise, giving no depth structure and a
        white-noise power spectrum). Asserts: plausible spiking activity, and
        that LFP band power genuinely varies across depth/contacts.
        """
        cfg = make_test_config(
            areas=("V1",), cell_types=("E", "PV", "SST", "VIP"),
            n_neuron_per_column=40, n_contacts=16,
            duration_ms=150.0, dt_ms=0.5,
        )
        model = build_laminar_column(cfg)
        stim = make_stimulus(
            kind="sine", duration_ms=cfg.duration_ms, dt_ms=cfg.dt_ms,
            frequency_hz=10.0, amplitude=6.0,
        )
        target = select_cells(model, layers=("L4",), cell_types=("E",), seed=cfg.seed)
        trials = simulate_laminar_trials(model, cfg, cfg.base_control, stim, target, n_trials=3)

        # Plausible, finite activity (the mock used a flat 5 Hz Bernoulli rate;
        # the real engine produces spikes driven by dynamics).
        rate_hz = float(trials["spikes"].mean()) * 1000.0 / cfg.dt_ms
        assert 0.1 < rate_hz < 500.0, rate_hz
        assert np.all(np.isfinite(trials["lfp_contacts"]))
        assert np.all(np.isfinite(trials["source_native"]))

        # Depth structure: top and bottom contacts must see genuinely different
        # signals. The old mock broadcast ONE area-mean to every contact, so all
        # contact traces correlated ~1.0; the depth-dependent leadfield makes the
        # shallowest and deepest contacts sample different neuron populations, so
        # their correlation drops well below 1.0.
        lfp = trials["lfp_contacts"][:, 0]          # (trials, T, contacts)
        pooled = np.concatenate(list(lfp), axis=0)  # (trials*T, contacts)
        corr = np.corrcoef(pooled.T)
        first_last_corr = float(corr[0, -1])
        assert first_last_corr < 0.9, (
            f"top/bottom contacts nearly identical ({first_last_corr:.3f}) "
            "— leadfield depth structure missing (mock regression?)"
        )

    def test_cell_type_izh_params_override_reaches_engine(self):
        """Per-cell-type Izhikevich overrides must change the real dynamics.

        Drives the E population hard vs not at all via cell_type_izh_params and
        asserts the resulting E spike counts differ — confirming the override is
        applied to simulate_eig_izhikevich, not silently dropped.
        """
        from jaxfne.tutorial_utils import make_laminar_column_config

        def e_rate(drive):
            cfg = make_laminar_column_config(
                areas=("V1",), cell_types=("E", "PV"), n_neuron_per_column=20,
                n_contacts=8, duration_ms=120.0, dt_ms=0.5, n_trials=1,
                cell_type_izh_params={"E": {"drive": drive}},
            )
            model = build_laminar_column(cfg)
            trials = simulate_laminar_trials(model, cfg, n_trials=1)
            e_idx = np.flatnonzero(model["neurons"]["cell_type"].to_numpy() == "E")
            return float(trials["spikes"][0][:, e_idx].mean())

        quiet = e_rate(0.0)
        loud = e_rate(12.0)
        assert loud > quiet, f"E drive override had no effect: quiet={quiet}, loud={loud}"

    def test_cell_type_noise_override_changes_spikes(self):
        """Per-cell-type `noise` must reach the engine (zero vs high noise differ)."""
        from jaxfne.tutorial_utils import make_laminar_column_config

        def total_spikes(noise):
            cfg = make_laminar_column_config(
                areas=("V1",), cell_types=("E", "PV"), n_neuron_per_column=24,
                n_contacts=8, duration_ms=120.0, dt_ms=0.5, n_trials=1,
                cell_type_izh_params={
                    "E": {"a": 0.015, "b": 0.20, "c": -60.0, "d": 10.0, "drive": 2.0, "noise": noise},
                    "PV": {"a": 0.10, "b": 0.20, "c": -65.0, "d": 2.0, "drive": 2.0, "noise": noise},
                },
            )
            model = build_laminar_column(cfg)
            return int(simulate_laminar_trials(model, cfg, n_trials=1)["spikes"].sum())

        quiet = total_spikes(0.0)
        loud = total_spikes(6.0)
        assert loud != quiet, f"noise override had no effect: quiet={quiet}, loud={loud}"

    def test_default_config_firing_rates_are_stable(self):
        """Default per-cell-type drive/noise keep rates in a stable band (<~40 Hz)."""
        from jaxfne.tutorial_utils import make_laminar_column_config
        cfg = make_laminar_column_config(
            areas=("V1", "V4"), cell_types=("E", "PV", "SST", "VIP"),
            n_neuron_per_column=60, n_contacts=16, duration_ms=300.0, dt_ms=0.5,
            n_trials=2,
        )
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=2)
        per_neuron_hz = trials["spikes"].mean(axis=(0, 1)) * 1000.0 / cfg.dt_ms
        mean_hz = float(per_neuron_hz.mean())
        max_hz = float(per_neuron_hz.max())
        assert max_hz <= 45.0, f"peak rate {max_hz:.1f} Hz exceeds stable ceiling"
        assert 2.0 <= mean_hz <= 15.0, f"mean rate {mean_hz:.1f} Hz outside stable band"

    def test_default_config_applies_wider_E_waveform(self):
        """Default tutorial config ships the wider 'E-Wide' excitatory profile."""
        from jaxfne.tutorial_utils import make_laminar_column_config
        cfg = make_laminar_column_config(areas=("V1",), cell_types=("E", "PV"))
        assert "E" in cfg.cell_type_izh_params
        assert cfg.cell_type_izh_params["E"]["a"] < 0.02  # slower recovery => wider AP

    def test_inter_area_connectivity_applied(self):
        """Default V1/V4 config wires feedforward + feedback projections."""
        from jaxfne.tutorial_utils import make_laminar_column_config
        cfg = make_laminar_column_config(
            areas=("V1", "V4"), cell_types=("E", "PV"), n_neuron_per_column=24,
            n_contacts=8, duration_ms=60.0, dt_ms=0.5, n_trials=1,
        )
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg, n_trials=1)
        assert set(trials["connectivity_applied"]) == {"feedforward", "feedback"}

    def test_lesion_silences_target_neurons(self):
        """lesion_spec must zero spikes for matching neurons (knock-out)."""
        from jaxfne.tutorial_utils import make_laminar_column_config, _laminar_match_mask
        cfg = make_laminar_column_config(
            areas=("V1",), cell_types=("E", "PV"), n_neuron_per_column=24,
            n_contacts=8, duration_ms=80.0, dt_ms=0.5, n_trials=1,
            lesion_spec=({"area": "V1", "layers": ("L4",), "cell_types": ("E",)},),
        )
        model = build_laminar_column(cfg)
        les = np.flatnonzero(_laminar_match_mask(model["neurons"], "V1", ("L4",), ("E",)))
        trials = simulate_laminar_trials(model, cfg, n_trials=1)
        assert trials["n_lesioned"] == int(les.size) > 0
        assert trials["spikes"][0][:, les].sum() == 0

    def test_single_cell_waveforms_shapes_and_E_slower_than_PV(self):
        """single_cell_waveforms returns each type; wider/slower E fires less than PV."""
        from jaxfne.tutorial_utils import single_cell_waveforms
        wf = single_cell_waveforms(
            cell_types=("E", "PV"), duration_ms=150.0, dt_ms=0.25,
            cell_type_izh_params={"E": {"a": 0.015, "c": -60.0, "d": 10.0}},
        )
        assert set(wf) == {"E", "PV"}
        for w in wf.values():
            assert np.all(np.isfinite(w["voltage_mV"]))
        assert int(wf["E"]["spikes"].sum()) < int(wf["PV"]["spikes"].sum())

    def test_tune_laminar_agsdr_improves_loss(self):
        """AGSDR-style tuner returns a control dict and does not worsen loss."""
        from jaxfne.tutorial_utils import make_laminar_column_config, tune_laminar_agsdr
        cfg = make_laminar_column_config(
            areas=("V1",), cell_types=("E", "PV"), n_neuron_per_column=24,
            n_contacts=8, duration_ms=80.0, dt_ms=0.5, n_trials=1,
        )
        model = build_laminar_column(cfg)
        best, hist = tune_laminar_agsdr(
            model, cfg, target_rate_hz=8.0, generations=3, population_size=3,
            tune_duration_ms=80.0, seed=1,
        )
        assert "local_exc_gain" in best and "feedforward_gain" in best
        losses = list(hist["loss"]) if hasattr(hist, "loss") else [h["loss"] for h in hist]
        assert losses[-1] <= losses[0] + 1e-9

    def test_simulate_is_reproducible_for_fixed_seed(self):
        """Real-engine trials must be deterministic given the config seed."""
        cfg = make_test_config(
            areas=("V1",), cell_types=("E", "PV"), n_neuron_per_column=20,
            n_contacts=8, duration_ms=80.0, dt_ms=0.5,
        )
        model = build_laminar_column(cfg)
        stim = make_stimulus(kind="sine", duration_ms=cfg.duration_ms, dt_ms=cfg.dt_ms)
        target = select_cells(model, layers=("L4",), cell_types=("E",), seed=cfg.seed)
        a = simulate_laminar_trials(model, cfg, cfg.base_control, stim, target, n_trials=2)
        b = simulate_laminar_trials(model, cfg, cfg.base_control, stim, target, n_trials=2)
        assert np.array_equal(a["spikes"], b["spikes"])
        assert np.allclose(a["lfp_contacts"], b["lfp_contacts"])

    def test_per_area_specs_distinct(self):
        """Test spectrolaminar specs differ across areas (no copy-same-spec)."""
        cfg = make_test_config(
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
        cfg = make_test_config(
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
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        spikes = trials['spikes']
        assert np.all((spikes == 0) | (spikes == 1))

    def test_voltage_range(self):
        """Test voltage is in reasonable range."""
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        voltage = trials['voltage_mV']
        assert voltage.min() >= -100
        assert voltage.max() <= 50

    def test_arrays_finite(self):
        """Test all arrays are finite."""
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        for key in ['voltage_mV', 'source_native', 'lfp_contacts', 'csd_contacts']:
            assert np.all(np.isfinite(trials[key]))


class TestSpectrolaminar:
    """Test spectrolaminar analysis."""

    def test_spectrolaminar_from_trials(self):
        """Test spectrolaminar_from_trials returns power and specs."""
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        power, specs = spectrolaminar_from_trials(trials, cfg)
        assert power.shape[0] == cfg.freq_count
        assert power.shape[1] == cfg.n_contacts

    def test_spectrolaminar_specs_keys(self):
        """Test spectrolaminar specs has required keys."""
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        power, specs = spectrolaminar_from_trials(trials, cfg)
        required_keys = {'freq_hz', 'pos_from_l4', 'relative_power', 'alpha_beta', 'gamma'}
        assert required_keys.issubset(specs.keys())

    def test_summarize_spectrolaminar_similarity(self):
        """Test summarize_spectrolaminar_similarity returns scores and specs."""
        cfg = make_test_config(n_neuron_per_column=12)
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
        cfg = make_test_config(
            output_dir=tmp_path,
            n_neuron_per_column=12,
        )
        manifest = {'test': 'data'}
        paths = export_tutorial_artifacts(cfg, manifest_dict=manifest, output_dir=tmp_path)
        assert 'manifest_path' in paths
        assert Path(paths['manifest_path']).exists()

    def test_export_json_strict(self, tmp_path):
        """Test exported JSON is strict (no NaN/Inf)."""
        cfg = make_test_config(output_dir=tmp_path)
        manifest = cfg.to_manifest_dict()
        paths = export_tutorial_artifacts(cfg, manifest_dict=manifest, output_dir=tmp_path)

        # Read and re-serialize with allow_nan=False (should not fail)
        manifest_path = Path(paths['manifest_path'])
        text = manifest_path.read_text()
        data = json.loads(text)
        json_text = json.dumps(data, allow_nan=False)  # Should not raise
        assert json_text is not None


class TestEmitterNoiseScale:
    """Test emitter noise_scale parameter."""

    def test_noise_scale_vector_contract(self):
        """Verify noise_scale accepts (n_neurons,) vector and applies per-neuron."""
        import jax
        import jax.numpy as jnp
        from jaxfne.emitters import izhikevich_eig_params, simulate_eig_izhikevich

        params = izhikevich_eig_params(4, {"E": 1.0})
        noise = jnp.asarray([0.0, 0.1, 0.2, 0.3], dtype=jnp.float32)
        v, spk, src = simulate_eig_izhikevich(
            params,
            n_steps=20,
            dt_ms=0.1,
            key=jax.random.PRNGKey(0),
            dtype="float32",
            noise_scale=noise,
        )
        assert v.shape == (20, 4)
        assert spk.shape == (20, 4)
        assert src.shape == (20, 4)
        assert np.isfinite(np.asarray(v)).all()
        assert np.isfinite(np.asarray(src)).all()

    def test_noise_scale_backward_compatible(self):
        """noise_scale=None preserves historical 0.5 behavior."""
        import jax
        from jaxfne.emitters import izhikevich_eig_params, simulate_eig_izhikevich

        params = izhikevich_eig_params(2, {"E": 1.0})
        v1, _, _ = simulate_eig_izhikevich(
            params, n_steps=10, dt_ms=0.1, key=jax.random.PRNGKey(0), noise_scale=None
        )
        v2, _, _ = simulate_eig_izhikevich(
            params, n_steps=10, dt_ms=0.1, key=jax.random.PRNGKey(0), noise_scale=0.5
        )
        # Same key + same noise_scale should give same result
        assert np.allclose(np.asarray(v1), np.asarray(v2))
