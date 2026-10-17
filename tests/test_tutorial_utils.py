"""Tests for jaxfne.tutorial_utils (laminar column helpers)."""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import json

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
    spectrolaminar_motif_score,
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
        assert gates['claim_level'] == "computational_scaffold"
        assert gates['field_solver_status'] == "linear_solver"
        assert gates['physical_amplitude_calibrated'] is False

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


class TestBuildLaminarColumn:
    """Test model building."""

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

    def test_select_fraction(self):
        """Test selecting a fraction of cells."""
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        cells_25pct = select_cells(model, fraction=0.25, seed=42)
        assert 0 < len(cells_25pct) <= len(model['neurons']) * 0.25


class TestMakeStimulus:
    """Test stimulus creation."""

    def test_stimulus_dtype(self):
        """Test stimulus dtype."""
        stim = make_stimulus(dtype="float32")
        assert stim.dtype == np.float32

    def test_constant_stimulus(self):
        """Test constant stimulus."""
        stim = make_stimulus(kind="constant", amplitude=2.0, baseline=1.0)
        assert np.all(stim == 3.0)


class TestSimulateTrials:
    """Test trial simulation."""

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
        assert {"feedforward", "feedback"}.issubset(set(trials["connectivity_applied"]))

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


class TestSpectrolaminar:
    """Test spectrolaminar analysis."""

    def test_spectrolaminar_motif_score_matches_inlined_formula(self):
        """summarize_spectrolaminar_similarity must delegate to this scorer, not duplicate it."""
        cfg = make_test_config(n_neuron_per_column=12)
        model = build_laminar_column(cfg)
        trials = simulate_laminar_trials(model, cfg)
        scores, specs = summarize_spectrolaminar_similarity(trials, cfg)
        for area in cfg.areas:
            spec = specs[area]
            expected = spectrolaminar_motif_score(spec['alpha_beta'], spec['gamma'])
            actual = scores.loc[scores['area'] == area, 'similarity_percent'].iloc[0]
            assert actual == pytest.approx(expected)

    def test_spectrolaminar_motif_score_bounds_and_degeneracy(self):
        """Anti-correlated -> 100, correlated -> 0, degenerate (size<2 or zero-std) -> 50."""
        ab_anti = np.array([1.0, 2.0, 3.0])
        gm_anti = np.array([3.0, 2.0, 1.0])
        assert spectrolaminar_motif_score(ab_anti, gm_anti) == pytest.approx(100.0)

        ab_corr = np.array([1.0, 2.0, 3.0])
        gm_corr = np.array([1.0, 2.0, 3.0])
        assert spectrolaminar_motif_score(ab_corr, gm_corr) == pytest.approx(0.0)

        assert spectrolaminar_motif_score([1.0], [1.0]) == pytest.approx(50.0)
        assert spectrolaminar_motif_score([1.0, 1.0], [1.0, 2.0]) == pytest.approx(50.0)


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
        text = manifest_path.read_text(encoding="utf-8")
        data = json.loads(text)
        json_text = json.dumps(data, allow_nan=False)  # Should not raise
        assert json.loads(json_text) == data


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

    def test_noise_dt_ref_bit_identical(self):
        """noise_scale is defined at DT_REF_MS=0.5: dt=0.5 matches the old formula bit-exactly."""
        import jax
        import jax.numpy as jnp
        from jaxfne import emitters
        from jaxfne.emitters import (
            _dtype_from_policy,
            _izhikevich_dv_du,
            _source_proxy_from_components,
            izhikevich_eig_params,
            simulate_eig_izhikevich,
        )

        assert emitters.DT_REF_MS == 0.5
        jdtype = _dtype_from_policy("float32")
        for ns in (None, 0.5, 0.0, 1.0):
            old = (
                jnp.asarray(0.5, dtype=jdtype)
                if ns is None
                else jnp.asarray(ns, dtype=jdtype)
            )
            new = emitters._noise_coef(ns, 0.5, jdtype)
            assert bool(jnp.array_equal(old, new)), f"helper must equal old formula at dt=0.5 for {ns!r}"
        nsv = jnp.asarray([0.0, 0.25, 0.5, 1.0], dtype=jdtype)
        assert bool(jnp.array_equal(nsv, emitters._noise_coef(nsv, 0.5, jdtype)))

        # Kernel level: dt=0.5 output equals a local scan using the old formula.
        params = izhikevich_eig_params(2, {"E": 1.0})
        n_steps, dt_ms, key = 10, 0.5, jax.random.PRNGKey(0)
        v_new, _, _ = simulate_eig_izhikevich(
            params, n_steps=n_steps, dt_ms=dt_ms, key=key, dtype="float32", noise_scale=0.5
        )
        a = params.a.astype(jdtype)
        b = params.b.astype(jdtype)
        c = params.c.astype(jdtype)
        d = params.d.astype(jdtype)
        drive = params.drive.astype(jdtype)
        weights = params.W.astype(jdtype)
        source_scale = params.source_scale.astype(jdtype)
        dt = jnp.asarray(dt_ms, dtype=jdtype)
        noise_coef = jnp.asarray(0.5, dtype=jdtype)  # baseline formula
        s_mask = jnp.ones(2, dtype=jdtype)
        _, noise_key = jax.random.split(key)
        bulk_noise = jax.random.normal(noise_key, shape=(n_steps, 2), dtype=jdtype)
        sched = jnp.zeros((n_steps, 2), dtype=jdtype)

        def step(carry, xs_t):
            sched_t, noise_t = xs_t
            v, u, prev_spikes = carry
            syn = weights @ prev_spikes
            current_native = drive + sched_t + syn + noise_coef * noise_t
            dv, du = _izhikevich_dv_du(v, u, current_native, a, b)
            v_next = v + dt * dv
            u_next = u + dt * du
            v_next = jnp.where(s_mask > 0.5, v_next, c)
            spikes_bool = (v_next >= 30.0) & (s_mask > 0.5)
            spikes = spikes_bool.astype(jdtype)
            v_reset = jnp.where(spikes_bool, c, v_next)
            u_reset = jnp.where(spikes_bool, u_next + d, u_next)
            source_proxy = _source_proxy_from_components(
                current_native, spikes, source_scale, dtype=jdtype
            )
            return (v_reset, u_reset, spikes), (v_reset, spikes, source_proxy)

        init = (
            params.v0.astype(jdtype),
            params.u0.astype(jdtype),
            jnp.zeros_like(params.v0, dtype=jdtype),
        )
        _, (v_old, _, _) = jax.lax.scan(step, init, xs=(sched, bulk_noise))
        assert np.array_equal(np.asarray(v_new), np.asarray(v_old))

    def test_noise_per_ms_variance_dt_invariant(self):
        """Per-ms noise variance of the voltage increment is equal at dt 0.1 and 0.5."""
        import dataclasses
        import jax
        import jax.numpy as jnp
        import pytest
        from jaxfne import emitters
        from jaxfne.emitters import (
            _dtype_from_policy,
            izhikevich_eig_params,
            simulate_eig_izhikevich,
        )

        # Helper algebra: dt * coef^2 is dt-invariant.
        jdtype = _dtype_from_policy("float32")
        c05 = float(emitters._noise_coef(0.5, 0.5, jdtype))
        c01 = float(emitters._noise_coef(0.5, 0.1, jdtype))
        assert (0.5 * c05**2) == pytest.approx(0.1 * c01**2, rel=1e-6)

        # Kernel level: isolated noise current read out via sources
        # (zero drive/recurrence, spike-free so sources == noise current).
        base = izhikevich_eig_params(1, {"E": 1.0})
        v0 = jnp.full((1,), -70.0, dtype=jnp.float32)
        params = dataclasses.replace(
            base,
            drive=jnp.zeros((1,), dtype=jnp.float32),
            W=jnp.zeros((1, 1), dtype=jnp.float32),
            v0=v0,
            u0=base.b.astype(jnp.float32) * v0,
        )

        def per_ms(dt_ms, n_steps, seed):
            _, spk, src = simulate_eig_izhikevich(
                params,
                n_steps=n_steps,
                dt_ms=dt_ms,
                key=jax.random.PRNGKey(seed),
                dtype="float32",
                noise_scale=0.5,
            )
            assert int(np.asarray(spk).sum()) == 0, "fixture must stay spike-free"
            return float(dt_ms) * float(np.var(np.asarray(src)))

        m05 = per_ms(0.5, 10000, 0)
        m01 = per_ms(0.1, 10000, 1)
        assert abs(m01 - m05) / ((m01 + m05) / 2) < 0.05

    def test_noise_scale_zero_exact_at_every_dt(self):
        """noise_scale=0.0 gives exactly zero noise at every dt."""
        import jax
        import jax.numpy as jnp
        from jaxfne import emitters
        from jaxfne.emitters import (
            _dtype_from_policy,
            izhikevich_eig_params,
            simulate_eig_izhikevich,
        )

        jdtype = _dtype_from_policy("float32")
        for dt_ms in (0.1, 0.5, 1.0):
            z = emitters._noise_coef(0.0, dt_ms, jdtype)
            assert bool(jnp.array_equal(z, jnp.zeros((), dtype=jdtype))), dt_ms
            zv = emitters._noise_coef(jnp.zeros((2,), dtype=jdtype), dt_ms, jdtype)
            assert bool(jnp.array_equal(zv, jnp.zeros((2,), dtype=jdtype))), dt_ms
        params = izhikevich_eig_params(2, {"E": 1.0})
        for dt_ms in (0.1, 0.5, 1.0):
            v1, _, _ = simulate_eig_izhikevich(
                params, n_steps=10, dt_ms=dt_ms, key=jax.random.PRNGKey(0),
                dtype="float32", noise_scale=0.0,
            )
            v2, _, _ = simulate_eig_izhikevich(
                params, n_steps=10, dt_ms=dt_ms, key=jax.random.PRNGKey(1),
                dtype="float32", noise_scale=0.0,
            )
            assert np.array_equal(np.asarray(v1), np.asarray(v2)), dt_ms

    def test_receptor_exponential_noise_scales(self):
        """The receptor-exponential kernel scales its noise term the same way."""
        import dataclasses
        import jax
        import jax.numpy as jnp
        from jaxfne.emitters import (
            EdgeList,
            izhikevich_eig_params,
            simulate_receptor_exponential_izhikevich,
        )

        base = izhikevich_eig_params(1, {"E": 1.0})
        v0 = jnp.full((1,), -70.0, dtype=jnp.float32)
        params = dataclasses.replace(
            base,
            drive=jnp.zeros((1,), dtype=jnp.float32),
            v0=v0,
            u0=base.b.astype(jnp.float32) * v0,
        )
        edges = EdgeList(
            pre=jnp.asarray([0]),
            post=jnp.asarray([0]),
            weight=jnp.asarray([0.0], dtype=jnp.float32),
            receptor_index=jnp.asarray([0], dtype=jnp.int32),
            tau_ms=jnp.asarray([10.0], dtype=jnp.float32),
        )

        def per_ms(dt_ms, n_steps, seed):
            _, spk, src, _ = simulate_receptor_exponential_izhikevich(
                params, edges, n_steps=n_steps, dt_ms=dt_ms,
                key=jax.random.PRNGKey(seed), dtype="float32",
            )
            assert int(np.asarray(spk).sum()) == 0, "fixture must stay spike-free"
            return float(dt_ms) * float(np.var(np.asarray(src)))

        m05 = per_ms(0.5, 10000, 0)
        m01 = per_ms(0.1, 10000, 1)
        assert abs(m01 - m05) / ((m01 + m05) / 2) < 0.05


def test_make_laminar_column_config_clamps_claim_level_escalation():
    """A caller passing claim_level="validated"/physical_amplitude_calibrated=True
    must NOT actually escalate the config's truth-gate fields -- found 2026-07-14
    via a direct write-site sweep: this factory never routed its claim kwargs
    through Configuration's clamp_truth_gate_metadata(), unlike
    Configuration.update_metadata()."""
    from jaxfne.tutorial_utils import make_laminar_column_config

    cfg = make_laminar_column_config(
        claim_level="validated",
        physical_amplitude_calibrated=True,
        field_solver_status="pde_solved",
        field_claim_level="real_field",
    )
    assert cfg.claim_level == "computational_scaffold"
    assert cfg.physical_amplitude_calibrated is False
    assert cfg.field_solver_status == "linear_solver"
    assert cfg.field_claim_level == "proxy_readout"


def test_kappa_synchrony_undefined_is_nan():
    """Undefined kappa (empty, < 2 neurons, < 2 valid) is NaN, not 0.0."""
    from jaxfne.tutorial_utils import kappa_synchrony

    assert np.isnan(kappa_synchrony(np.array([])))
    assert np.isnan(kappa_synchrony(np.ones((10, 1))))
    assert np.isnan(kappa_synchrony(np.zeros((20, 5))))
    assert np.isnan(kappa_synchrony(np.ones((10,))))


def test_kappa_synchrony_defined_zero_stays():
    """A defined 0.0 correlation stays 0.0 (orthogonal spike trains)."""
    from jaxfne.tutorial_utils import kappa_synchrony

    spikes = np.array(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    )
    value = kappa_synchrony(spikes)
    assert value == 0.0 and not np.isnan(value)


@pytest.mark.parametrize("inh_type", ["I", "Inl", "Ing"])
def test_simulate_laminar_trials_refuses_inhibitory_type_outside_same_layer_rule(inh_type):
    """The same-layer rule zeroes PV/SST/VIP input only. A config holding another
    inhibitory type is refused before simulation, not run with that type unzeroed."""
    from jaxfne.tutorial_utils import build_laminar_column, simulate_laminar_trials

    cfg = make_test_config(areas=("V1",), cell_types=("E", "PV", inh_type), n_neuron_per_column=24)
    model = build_laminar_column(cfg)
    assert inh_type in set(model["neurons"]["cell_type"]), "fixture must hold the type"
    with pytest.raises(ValueError, match=rf"\['{inh_type}'\]"):
        simulate_laminar_trials(model, cfg, n_trials=1)


def test_default_cell_signs_follow_emitter_table(monkeypatch):
    """Default cell_signs read the emitter table's sign, not a second hard-coded copy."""
    from jaxfne.emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS
    from jaxfne.tutorial_utils import make_laminar_column_config

    assert make_laminar_column_config().cell_signs == {"E": 1.0, "PV": -1.0, "SST": -1.0, "VIP": -1.0}
    monkeypatch.setitem(
        IZHIKEVICH_CELL_TYPE_DEFAULTS, "PV", {**IZHIKEVICH_CELL_TYPE_DEFAULTS["PV"], "sign": 1.0}
    )
    assert make_laminar_column_config().cell_signs["PV"] == 1.0


def test_canonical_summary_collapses_types_by_emitter_table_sign(monkeypatch):
    """Effective E/I grouping follows the emitter table's sign: a type the table marks
    inhibitory collapses to I under any name."""
    import types

    import jax.numpy as jnp
    from jaxfne.emitters import IZHIKEVICH_CELL_TYPE_DEFAULTS
    from jaxfne.util import canonical_compact_summary

    def inventory_for(labels):
        rows = [
            {"layer": "L2", "cell_type": ct, "area": "V1", "neuron_id": i}
            for i, ct in enumerate(labels)
        ]
        z = jnp.zeros(len(labels), dtype=jnp.float32)
        emitter = types.SimpleNamespace(
            v0=z, u0=z, a=z, b=z, c=z, d=z, drive=z, sign=z, source_scale=1.0
        )
        model = types.SimpleNamespace(
            params={"emitter": emitter},
            cfg=types.SimpleNamespace(metadata={}),
            static={},
            neuron_table=lambda: rows,
        )
        return canonical_compact_summary(model)["counts"]["realized"]["populations_inventory"]

    inv = inventory_for(["E", "PV", "Inl", "Ing", "I", "SST", "VIP"])
    assert inv["realized_detailed"] == 7
    assert inv["effective_EI_collapsed"] == 2, "E and one collapsed I per layer"

    monkeypatch.setitem(IZHIKEVICH_CELL_TYPE_DEFAULTS, "Foo", {"sign": -1.0})
    inv = inventory_for(["E", "PV", "Foo"])
    assert inv["effective_EI_collapsed"] == 2, "Foo is inhibitory in the table: must collapse to I"
