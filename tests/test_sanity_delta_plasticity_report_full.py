"""Tests for homeostatic plasticity regularizer and reports in full mode."""

import numpy as np
import jaxfne as jtfne

def test_plasticity_bounds_and_report():
    """Verify that homeostatic updates preserve bounds and generate valid reports."""
    cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball(runtime_mode="full")
    paradigm = cfg.make_paradigm()
    model = cfg.construct()
    
    # Enable plasticity
    model.enable_plasticity(
        target_rate_hz=10.0,
        eta_homeo=0.05
    )
    
    backup = model.initialize_backup(paradigm)
    episode = model.run_task(paradigm, paradigm.make_fixation_gate(), backup)
    
    # Export reports
    import tempfile
    from pathlib import Path
    import json
    
    with tempfile.TemporaryDirectory() as tmpdir:
        episode.export(tmpdir)
        
        report_path = Path(tmpdir) / "plasticity_report.json"
        assert report_path.exists()
        
        report = json.loads(report_path.read_text())
        
        # Verify keys
        assert report["enabled"] is True
        assert "pre_weight_min" in report
        assert "post_weight_min" in report
        assert "clip_count" in report
        assert "updated_connection_count" in report
        assert report["excitatory_sign_preserved"] is True
        assert report["inhibitory_sign_preserved"] is True
        assert report["inhibitory_modified"] is False
        assert report["biological_learning_claim"] is False
        
        # Verify that weights are within valid bounds (inhibitory weights are around -0.08)
        assert report["post_weight_min"] >= -0.08
        assert report["post_weight_max"] <= 1.0
        assert report.get("placeholder_values") is False


def test_plasticity_disabled_export_flags_placeholders():
    """When plasticity is off, export must not present placeholder stats as measurements."""
    cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball(runtime_mode="full")
    paradigm = cfg.make_paradigm()
    model = cfg.construct()
    backup = model.initialize_backup(paradigm)
    episode = model.run_task(paradigm, paradigm.make_fixation_gate(), backup)

    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        episode.export(tmpdir)
        report = json.loads((Path(tmpdir) / "plasticity_report.json").read_text())

    assert report["enabled"] is False
    assert report.get("placeholder_values") is True
    assert report.get("status") == "pass_no_update"
