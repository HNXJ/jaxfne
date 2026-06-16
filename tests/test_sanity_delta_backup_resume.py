"""Tests for backup state and resume equivalence."""

import jax.numpy as jnp
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

    def test_episode_initialization(self):
        """Test episode initializes correctly."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)

        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        assert isinstance(episode, jtfne.TaskEpisode)
        assert episode.spikes.shape[0] == int(12000.0 / 0.1)  # n_steps
        assert episode.vm.shape[0] == int(12000.0 / 0.1)

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

        results = episode.validate(checks=("truth_gates_preserved", "finite_outputs"))

        assert results["truth_gates_preserved"] is True
        assert bool(results["finite_outputs"]) is True

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
