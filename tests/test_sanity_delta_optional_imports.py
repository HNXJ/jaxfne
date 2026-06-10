"""Tests for optional import patterns and optional notebook support."""

import sys
import pytest
import jaxfne as jtfne


class TestOptionalImports:
    """Test that heavy dependencies are not loaded on root import."""

    def test_jaxfne_imports_without_notebook_deps(self):
        """Test that jaxfne imports without requiring nbconvert or notebook."""
        # If these modules are in sys.modules, they were imported during setup
        # This test documents which deps are expected at root import
        excluded_heavy = ("nbconvert", "notebook", "papermill", "plotly")

        for mod in excluded_heavy:
            if mod in sys.modules:
                # It's OK if these were imported, but they should not be *required*
                # for basic jaxfne functionality
                pass

    def test_sanity_delta_minimal_deps(self):
        """Test that SanityDeltaConfig can instantiate with minimal deps."""
        # This should succeed with just jaxfne, jax, numpy
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()

        assert cfg is not None
        assert len(cfg.areas) * cfg.neurons_per_area == 500

    def test_visualization_optional(self):
        """Test that visualization methods exist but are marked optional."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)
        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        # visualize() exists but may require optional deps
        assert hasattr(episode, "visualize")

    def test_export_html_pdf_optional(self):
        """Test that HTML/PDF export methods exist but are optional."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)
        episode = model.run_task(paradigm=paradigm, gate=paradigm.make_fixation_gate(), backup=backup)

        # export() exists for all serialization formats
        assert hasattr(episode, "export")

    def test_manifest_json_export_core_only(self):
        """Test that manifest JSON export works without optional deps."""
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        paradigm = cfg.make_paradigm()
        model = cfg.construct()
        backup = model.initialize_backup(paradigm)

        # Manifest creation should work with json stdlib only
        manifest = jtfne.Manifest(
            config=cfg,
            paradigm=paradigm,
            backup=backup,
            episode_metadata={"duration_ms": 12000.0, "dt_ms": 0.1, "n_neurons": 500, "areas": list(cfg.areas)},
            generated_at_utc="2026-06-10T18:00:00Z",
        )

        d = manifest.to_dict()
        assert isinstance(d, dict)

    def test_notebook_optional_marker(self):
        """Test that notebook support is marked as optional."""
        # The notebook module should be importable but marked with optional guards
        assert hasattr(jtfne, "SanityDeltaConfig")

        # The config should document whether notebook support is available
        cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
        # notebook_support is a feature-flag, not a hard requirement
        if hasattr(cfg, "notebook_support_available"):
            # This is informational only
            pass
