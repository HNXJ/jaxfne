"""Tests for backup state and resume equivalence."""

import dataclasses

import jaxfne as jtfne


class TestBackupState:
    """Test backup state creation and serialization."""

    def test_backup_state_initialization(self):
        """Test backup state initializes correctly."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)

        assert isinstance(backup, jtfne.BackupState)
        assert backup.time_ms == 0.0
        assert backup.fixation_counter == 0
        assert backup.runtime_metadata["n_neurons"] == 500

    def test_backup_state_serialization(self):
        """Test backup state can be serialized to dict."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)

        d = backup.to_dict()

        assert d["time_ms"] == 0.0
        assert d["fixation_counter"] == 0
        assert "task_state" in d

    def test_backup_to_manifest(self):
        """Test backup exports to manifest."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)

        manifest = backup.to_manifest()

        assert "checkpoint" in manifest
        assert manifest["checkpoint"]["time_ms"] == 0.0


class TestTaskEpisode:
    """Test task episode and backup/resume."""

    def test_episode_probe_chain(self):
        """Test probe method chaining."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)
        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        result = episode.probe(readouts=("spk", "vm", "lfp_proxy"))
        assert result is episode

    def test_episode_validate_truth_gates(self):
        """Test validation checks truth gates."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)
        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        results = episode.validate(checks=("truth_gates_preserved", "finite_outputs", "strict_json"))

        assert results["truth_gates_preserved"] is True
        assert bool(results["finite_outputs"]) is True
        assert results["strict_json"] is True

    def test_episode_resume_method(self):
        """Test episode can be resumed."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)
        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        resumed = episode.resume(from_segment="d4")
        assert isinstance(resumed, jtfne.TaskEpisode)
        assert resumed is not episode


class TestBackupHistoryUnderrun:
    """A backup whose time_ms implies more history than history_buffer holds must not crash."""

    def test_start_step_beyond_history_buffer_length_does_not_raise(self):
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        # A short history window (100ms -> 1000 steps at dt_ms=0.1) paired with a
        # resume point (200ms -> 2000 steps) further back than that window covers.
        backup = model.initialize_backup(paradigm, history_ms=100.0)
        assert backup.history_buffer.shape[0] == 1000
        short_history_backup = dataclasses.replace(backup, time_ms=200.0)

        episode = model.run_task(
            paradigm=paradigm,
            gate=paradigm.make_fixation_gate(),
            backup=short_history_backup,
            runtime_mode="full",
        )

        assert episode.vm.shape[0] > 0
        # Steps before the available history window fall back to the resting-potential
        # default (-65 mV) rather than being populated from data that doesn't exist.
        assert float(episode.vm[0, 0]) == -65.0


class TestManifest:
    """Test output manifest."""

    def test_manifest_creation(self):
        """Test manifest creates from episode data."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)

        manifest = jtfne.Manifest(
            config=cfg,
            paradigm=paradigm,
            backup=backup,
            episode_metadata={"duration_ms": 12000.0, "dt_ms": 0.1, "n_neurons": 500, "areas": list(cfg.areas)},
            generated_at_utc="2026-06-10T18:00:00Z",
        )

        d = manifest.to_dict()

        assert d["claim_level"] == "computational_scaffold"
        assert d["physical_amplitude_calibrated"] is False
        assert d["total_neurons"] == 500

    def test_manifest_save(self, tmp_path):
        """Test manifest saves to JSON."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)

        manifest = jtfne.Manifest(
            config=cfg,
            paradigm=paradigm,
            backup=backup,
            episode_metadata={"duration_ms": 12000.0, "dt_ms": 0.1, "n_neurons": 500, "areas": list(cfg.areas)},
            generated_at_utc="2026-06-10T18:00:00Z",
        )

        path = tmp_path / "manifest.json"
        manifest.save(path)

        assert path.exists()

        import json
        data = json.loads(path.read_text())
