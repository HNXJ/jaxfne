"""Tests for proxy-safe readout naming conventions."""

import jaxfne as jtfne


class TestProxyReadoutNames:
    """Test proxy readout naming follows conventions."""

    def test_proxy_readout_naming_convention(self):
        """Test that proxy readouts use *_proxy suffix."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)
        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        # Probe with proxy readouts - validation succeeds
        result = episode.probe(readouts=("lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy"))

        assert result is episode
        # probe() validates readout names match *_proxy or core forms
        assert hasattr(episode, "spikes")
        assert hasattr(episode, "vm")

    def test_proxy_implies_laminar_proxy_no_pde(self):
        """Test that proxy readouts imply field_solver_status == laminar_proxy_no_pde."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()

        assert cfg.field_solver_status == "laminar_proxy_no_pde"

        # Proxy readouts are only allowed with this status
        paradigm = cfg.make_paradigm()
        model = cfg.construct()

        # No full PDE solves; only laminar proxy
        assert model.config.field_solver_status == "laminar_proxy_no_pde"

    def test_spikes_and_vm_not_proxy(self):
        """Test that spikes and vm readouts do not carry proxy suffix."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)
        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        # These are not proxies, not PDE outputs
        result = episode.probe(readouts=("spk", "vm"))

        assert result is episode
        assert hasattr(episode, "spikes")
        assert hasattr(episode, "vm")

    def test_readout_naming_canonical(self):
        """Test canonical readout name forms."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)
        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        # All proxy forms available
        canonical = ("spk", "vm", "lfp_proxy", "csd_proxy", "eeg_proxy", "meg_proxy")

        for readout in canonical:
            result = episode.probe(readouts=(readout,))
            assert result is episode

    def test_no_biological_field_solve_claim(self):
        """Test that config forbids biological field-solve claim."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()

        # These are computational proxies, proxy-scoped for computational examples
        assert cfg.physical_amplitude_claim_allowed is False
        assert cfg.claim_level == "computational_scaffold"
        assert cfg.field_solver_status == "laminar_proxy_no_pde"

    def test_manifest_preserves_proxy_claim(self):
        """Test that manifest output preserves proxy status."""
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

        assert d["field_solver_status"] == "laminar_proxy_no_pde"
        assert d["claim_level"] == "computational_scaffold"
        assert d["physical_amplitude_claim_allowed"] is False
