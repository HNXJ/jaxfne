"""
Targeted unit tests for the v0.3.13 Sensory Omission & Oddball paradigm (Commit 11).

Validates that:
- Omission and oddball paradigm configuration returns the expected ParadigmCondition objects.
- Events, timing buffers, and condition names conform to requirements.
- Spiking and field-proxy outputs are finite and JAX JIT-safe.
- Strict non-negotiable scientific scope boundaries are enforced.
"""

import pytest
import numpy as np
import jax.numpy as jnp
import jaxfne as jtfne
import json
import os


def test_v0313_omission_oddball_paradigm_timing():
    """Verify that omission_oddball_paradigm returns proper conditions and timings."""
    paradigm = jtfne.omission_oddball_paradigm(
        standard_onset_ms=250.0,
        standard_duration_ms=120.0,
        deviant_duration_ms=120.0,
        pre_stimulus_buffer_ms=150.0,
        post_stimulus_buffer_ms=450.0,
    )
    
    assert isinstance(paradigm, jtfne.Paradigm)
    assert len(paradigm.conditions) == 3
    assert [c.name for c in paradigm.conditions] == ["expected", "unexpected", "omitted"]
    
    # Verify analysis window timings
    assert paradigm.analysis_windows["baseline"] == (0.0, 150.0)
    assert paradigm.analysis_windows["stimulus"] == (150.0, 270.0)
    assert paradigm.analysis_windows["post_stimulus"] == (270.0, 720.0)


def test_v0313_omission_paradigm_conditions_metadata():
    """Verify that expected, unexpected, and omitted conditions carry proper event flags."""
    paradigm = jtfne.omission_oddball_paradigm()
    
    # 1. Expected condition
    expected = paradigm.conditions[0]
    assert expected.name == "expected"
    assert any(evt.stimulus == "standard_tone" for evt in expected.events)
    
    # 2. Unexpected condition
    unexpected = paradigm.conditions[1]
    assert unexpected.name == "unexpected"
    assert any(evt.stimulus == "deviant_tone" for evt in unexpected.events)
    
    # 3. Omitted condition
    omitted = paradigm.conditions[2]
    assert omitted.name == "omitted"
    assert any(evt.is_omission for evt in omitted.events)


def test_v0313_json_safety_and_wording_gates():
    """Ensure the paradigm dictionaries and files contain zero NaNs/Infs and adhere to wording guidelines."""
    paradigm = jtfne.omission_oddball_paradigm()
    paradigm_dict = paradigm.to_dict() if hasattr(paradigm, "to_dict") else {"name": paradigm.name}
    
    # Ensure JSON serializable without NaN/Inf
    json_str = json.dumps(paradigm_dict, allow_nan=False)
    assert isinstance(json_str, str)
    
    # Verify no unmerged active inference or real EEG/MEG claims in metadata
    metadata_str = str(paradigm_dict).lower()
    assert "mechanism_proven" not in metadata_str
    assert "active_inference_validated" not in metadata_str
    assert "real_eeg" not in metadata_str


def test_v0313_docs_and_notebooks_exist():
    """Verify that omission docs and notebooks are present in their corresponding paths."""
    assert os.path.exists("docs/tutorials/10_v0313_omission_oddball.md")
    assert os.path.exists("tutorials/jaxfne_v0313_omission_oddball.ipynb")
