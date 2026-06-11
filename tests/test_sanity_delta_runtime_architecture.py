"""Tests for the architecture of the sanity_delta and sanity_runtime split."""

import inspect
import pytest
import jaxfne as jtfne
from jaxfne import sanity_runtime

def test_sanity_runtime_exists():
    """Verify that sanity_runtime module and its helper functions exist."""
    assert hasattr(jtfne, "sanity_runtime")
    
    helpers = [
        "_build_area_index",
        "_build_stimulus_drive",
        "_build_weight_matrix",
        "_scan_task_runtime",
        "_apply_segment_homeostasis",
        "_restore_runtime_state",
        "_compute_proxy_readouts",
        "_compare_backup_resume",
        "_make_probe_metrics",
        "_make_plasticity_metrics",
    ]
    for h in helpers:
        assert hasattr(sanity_runtime, h), f"sanity_runtime missing {h}"

def test_no_forbidden_calls_in_runtime():
    """Verify that sanity_runtime contains no JSON, filesystem IO, or plotting calls."""
    source = inspect.getsource(sanity_runtime)
    
    # Check that forbidden keywords are not in the file
    forbidden = [
        "json.dump",
        "json.write",
        "Path.write",
        "open(",
        ".savefig",
        "matplotlib",
        "pyplot",
        "plotly",
    ]
    for f in forbidden:
        assert f not in source, f"sanity_runtime has forbidden call: {f}"

def test_invalid_mode_raises_value_error():
    """Verify that an invalid runtime mode raises ValueError."""
    cfg = jtfne.SanityDeltaConfig.hierarchical_global_local_oddball()
    paradigm = cfg.make_paradigm()
    model = cfg.construct()
    backup = model.initialize_backup(paradigm)
    
    with pytest.raises(ValueError):
        model.run_task(paradigm, paradigm.make_fixation_gate(), backup, runtime_mode="invalid_mode")
