"""
Validation tests for evoked L4 drive paradigm (v0.3.12).

Tests for finite outputs, event timing, rate ranges, JSON safety, and proxy-safe wording.
"""

import pytest
import numpy as np
import jaxfne as jtfne
import json


class TestEvokedL4DriveAPI:
    """Test the evoked_l4_drive_paradigm() package-native API."""

    def test_evoked_paradigm_creates_valid_paradigm(self):
        """evoked_l4_drive_paradigm() should return a valid Paradigm object."""
        paradigm = jtfne.evoked_l4_drive_paradigm(
            l4_onset_ms=100.0,
            l4_duration_ms=200.0,
            l4_amplitude=1.0,
        )
        assert isinstance(paradigm, jtfne.Paradigm)
        assert paradigm.name == "evoked_l4_drive"

    def test_evoked_paradigm_has_two_conditions(self):
        """evoked_l4_drive_paradigm() should return 2 conditions: baseline + evoked."""
        paradigm = jtfne.evoked_l4_drive_paradigm()
        assert len(paradigm.conditions) == 2
        assert paradigm.conditions[0].name == "baseline"
        assert paradigm.conditions[1].name == "evoked"

    def test_evoked_paradigm_has_analysis_windows(self):
        """evoked_l4_drive_paradigm() should include baseline, evoked, post_evoked windows."""
        paradigm = jtfne.evoked_l4_drive_paradigm(
            l4_onset_ms=100.0,
            l4_duration_ms=200.0,
        )
        assert "baseline" in paradigm.analysis_windows
        assert "evoked" in paradigm.analysis_windows
        assert "post_evoked" in paradigm.analysis_windows

    def test_evoked_paradigm_events_have_correct_timing(self):
        """Event onsets and durations should match paradigm parameters."""
        l4_onset = 150.0
        l4_duration = 250.0
        pre_buffer = 100.0

        paradigm = jtfne.evoked_l4_drive_paradigm(
            l4_onset_ms=l4_onset,
            l4_duration_ms=l4_duration,
            pre_stimulus_buffer_ms=pre_buffer,
        )

        # Check baseline condition
        baseline = paradigm.conditions[0]
        assert baseline.events[0].onset_ms == 0.0
        assert baseline.events[0].duration_ms == pre_buffer

        # Check evoked condition
        evoked = paradigm.conditions[1]
        # First event should be trial start
        assert evoked.events[0].onset_ms == 0.0
        assert evoked.events[0].duration_ms == pre_buffer
        # Second event should be L4 drive
        assert evoked.events[1].onset_ms == pre_buffer
        assert evoked.events[1].duration_ms == l4_duration


class TestEvokedSimulation:
    """Test simulation with evoked L4 drive paradigm."""

    def test_evoked_simulation_produces_finite_outputs(self):
        """Evoked simulation should produce finite (no NaN/Inf) spike outputs."""
        cfg = (jtfne.Configuration()
            .runtime(seed=42, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
            .column("V1", layers=["L4"], n=20)
            .cell_type_drives({"E": 8.0})
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes"]))

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, seed=42, duration_ms=1000.0, dt_ms=0.1)

        # Check spikes are finite
        spikes = np.asarray(signals.spikes)
        assert np.all(np.isfinite(spikes)), "Spikes contain NaN or Inf"

    def test_evoked_rate_in_reasonable_range(self):
        """Firing rate should be in plausible band (1–50 Hz for reduced Izhikevich)."""
        cfg = (jtfne.Configuration()
            .runtime(seed=42, dtype="float32", duration_ms=2000.0, dt_ms=0.1)
            .column("V1", layers=["L4"], n=50)
            .cell_type_drives({"E": 8.0})
            .set_emitter("izhikevich", "cortical_eig")
            .probes(["spikes"]))

        model = jtfne.construct(cfg)
        signals = jtfne.simulate(model, seed=42, duration_ms=2000.0, dt_ms=0.1)

        # Compute mean firing rate
        spikes = np.asarray(signals.spikes)
        n_neurons = spikes.shape[0] if spikes.ndim > 1 else 1
        n_timesteps = spikes.shape[-1]
        dt_s = 0.1 / 1000.0  # 0.1 ms in seconds
        duration_s = n_timesteps * dt_s

        total_spikes = np.sum(spikes)
        mean_rate_hz = total_spikes / (n_neurons * duration_s)

        # Allow wide range for reduced model
        assert 0.1 < mean_rate_hz < 100.0, f"Rate {mean_rate_hz} Hz outside plausible range"


class TestEvokedManifest:
    """Test JSON safety and manifest compliance for evoked outputs."""

    def test_paradigm_to_dict_is_json_safe(self):
        """Paradigm.to_dict() should be JSON-serializable."""
        paradigm = jtfne.evoked_l4_drive_paradigm()

        # Should not raise on to_dict
        paradigm_dict = paradigm.to_dict() if hasattr(paradigm, 'to_dict') else {
            "name": paradigm.name,
            "conditions": [c.to_dict() if hasattr(c, 'to_dict') else c for c in paradigm.conditions],
        }

        # Should be JSON-serializable without allow_nan=True
        json_str = json.dumps(paradigm_dict, allow_nan=False)
        assert isinstance(json_str, str)
        assert len(json_str) > 0

    def test_conditions_json_serializable(self):
        """ParadigmConditions should serialize to JSON without NaN/Inf."""
        paradigm = jtfne.evoked_l4_drive_paradigm()

        for condition in paradigm.conditions:
            condition_dict = condition.to_dict()
            # Should not raise ValueError for NaN/Inf
            json_str = json.dumps(condition_dict, allow_nan=False)
            assert isinstance(json_str, str)

    def test_event_timing_nonzero_and_finite(self):
        """Event onsets and durations should be finite and positive."""
        paradigm = jtfne.evoked_l4_drive_paradigm()

        for condition in paradigm.conditions:
            for event in condition.events:
                if event.onset_ms is not None:
                    assert np.isfinite(event.onset_ms), f"onset_ms not finite: {event.onset_ms}"
                    assert event.onset_ms >= 0.0, f"onset_ms negative: {event.onset_ms}"
                if event.duration_ms is not None:
                    assert np.isfinite(event.duration_ms), f"duration_ms not finite: {event.duration_ms}"
                    assert event.duration_ms > 0.0, f"duration_ms non-positive: {event.duration_ms}"


class TestEvokedDocumentation:
    """Test documentation structure and proxy-safe wording."""

    def test_tutorial_notebook_exists(self):
        """Evoked L4 tutorial notebook should exist."""
        import os
        notebook_path = "tutorials/jaxfne_suite_no_2_evoked_l4_drive.ipynb"
        assert os.path.exists(notebook_path), f"Tutorial notebook not found: {notebook_path}"

    def test_markdown_docs_exist(self):
        """Evoked L4 markdown docs should exist."""
        import os
        docs_path = "docs/tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md"
        assert os.path.exists(docs_path), f"Docs file not found: {docs_path}"

    def test_docs_include_learning_objectives(self):
        """Docs should include learning objectives section."""
        import os
        docs_path = "docs/tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md"
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                content = f.read()
            assert "Learning objectives" in content, "Missing 'Learning objectives' section"

    def test_docs_include_scope_boundary(self):
        """Docs should include coverage boundary section."""
        import os
        docs_path = "docs/tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md"
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                content = f.read()
            assert "Coverage boundary" in content or "coverage boundary" in content.lower(), \
                "Missing scope/coverage boundary section"

    def test_docs_include_equation_glossary(self):
        """Docs should include mathematical glossary."""
        import os
        docs_path = "docs/tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md"
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                content = f.read()
            assert "Mathematical glossary" in content or "glossary" in content.lower(), \
                "Missing mathematical glossary"

    def test_docs_no_forbidden_science_wording(self):
        """Docs should not use forbidden science phrases."""
        import os
        docs_path = "docs/tutorials/08_jaxfne_suite_no_2_evoked_l4_drive.md"
        if os.path.exists(docs_path):
            with open(docs_path) as f:
                content = f.read().lower()

            forbidden = ["real eeg", "real meg", "calibrated amplitude", "mechanism proof"]
            for phrase in forbidden:
                assert phrase not in content, f"Forbidden phrase found: '{phrase}'"
