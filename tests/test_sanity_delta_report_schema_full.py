"""Tests validating the strict JSON schemas of all exported reports."""

import json
import tempfile
from pathlib import Path
import jaxfne as jtfne

def test_reports_strict_json_schema():
    """Verify that all exported reports contain the required keys and are strict JSON compliant."""
    cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball(runtime_mode="full")
    paradigm = cfg.make_paradigm()
    model = cfg.construct()
    model.enable_plasticity()
    backup = model.initialize_backup(paradigm)
    
    episode = model.run_task(paradigm, paradigm.make_fixation_gate(), backup)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        episode.export(tmpdir)
        path = Path(tmpdir)
        
        # 1. Validation Report
        val_path = path / "validation_report.json"
        assert val_path.exists()
        val_data = json.loads(val_path.read_text())
        assert "valid" in val_data
        assert "checks" in val_data
        assert "generated_at" in val_data
        assert val_data["checks"]["strict_json"] is True
        
        # 2. Backup/Resume Report
        br_path = path / "backup_resume_report.json"
        assert br_path.exists()
        br_data = json.loads(br_path.read_text())
        assert "vm_max_abs_error" in br_data
        assert "spk_mismatch_count" in br_data
        assert "source_max_abs_error" in br_data
        assert "lfp_proxy_max_abs_error" in br_data
        assert "csd_proxy_max_abs_error" in br_data
        assert "eeg_proxy_max_abs_error" in br_data
        assert "meg_proxy_max_abs_error" in br_data
        assert "weight_max_abs_error" in br_data
        assert "task_state_match" in br_data
        assert "reward_state_match" in br_data
        assert "status" in br_data
        assert "from_segment" in br_data
        
        # 3. Probe Report
        pr_path = path / "probe_report.json"
        assert pr_path.exists()
        pr_data = json.loads(pr_path.read_text())
        assert pr_data["proxy_safe_names"] is True
        assert pr_data["physical_amplitude_calibrated"] is False
        assert "readouts_present" in pr_data
        assert "shape_by_readout" in pr_data
        assert "finite_by_readout" in pr_data
        
        # 4. Plasticity Report
        pl_path = path / "plasticity_report.json"
        assert pl_path.exists()
        pl_data = json.loads(pl_path.read_text())
        assert pl_data["enabled"] is True
        assert pl_data["biological_learning_claim"] is False
        assert "pre_weight_min" in pl_data
        assert "post_weight_min" in pl_data
