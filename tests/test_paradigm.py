"""Tests for paradigm objects and task-flow schemas."""

import json

import jaxfne as jtfne


def test_standard_visual_omission_paradigm_conditions():
    """Test that standard_visual_omission returns a Paradigm with 12 conditions."""
    paradigm = jtfne.standard_visual_omission()

    assert paradigm.name == "standard_visual_omission"
    assert len(paradigm.conditions) == 12

    condition_names = paradigm.condition_names()
    expected_names = [
        "AAAB", "AXAB", "AAXB", "AAAX",
        "BBBA", "BXBA", "BBXA", "BBBX",
        "RRRR", "RXRR", "RRXR", "RRRX",
    ]
    assert condition_names == expected_names

    # Verify each condition can be retrieved by name
    for name in expected_names:
        cond = paradigm.condition(name)
        assert cond is not None
        assert cond.name == name


def test_condition_number_mapping():
    """Test that condition numbers are correctly mapped to conditions."""
    paradigm = jtfne.standard_visual_omission()

    expected_mapping = {
        "AAAB": (1, 2),
        "AXAB": (3,),
        "AAXB": (4,),
        "AAAX": (5,),
        "BBBA": (6, 7),
        "BXBA": (8,),
        "BBXA": (9,),
        "BBBX": (10,),
        "RRRR": tuple(range(11, 27)),
        "RXRR": tuple(range(27, 35)),
        "RRXR": (35, 37, 39, 41),
        "RRRX": (36, 38, 40) + tuple(range(42, 51)),
    }

    for cond_name, expected_numbers in expected_mapping.items():
        cond = paradigm.condition(cond_name)
        assert cond is not None
        assert cond.condition_numbers == expected_numbers


def test_event_code_mapping():
    """Test that event codes are correctly set."""
    paradigm = jtfne.standard_visual_omission()

    expected_codes = {
        "fx": 10,
        "p1": 101,
        "p2": 103,
        "p3": 105,
        "p4": 107,
        "rw": 96,
    }

    assert paradigm.event_codes == expected_codes
    assert paradigm.alignment_code == 101  # P1
    assert paradigm.alignment_label == "p1"


def test_omission_position_detection():
    """Test that omission positions are correctly detected."""
    paradigm = jtfne.standard_visual_omission()

    # Conditions with omissions
    omission_conds = paradigm.omission_conditions()
    assert len(omission_conds) == 9  # AXAB, AAXB, AAAX, BXBA, BBXA, BBBX, RXRR, RRXR, RRRX

    # Check specific omission positions
    axab = paradigm.condition("AXAB")
    assert axab is not None
    assert axab.has_omission()
    assert axab.omission_position == "p2"
    assert axab.omitted_event_label() == "p2"

    aaax = paradigm.condition("AAAX")
    assert aaax is not None
    assert aaax.has_omission()
    assert aaax.omission_position == "p4"
    assert aaax.omitted_event_label() == "p4"

    # Condition without omission
    aaab = paradigm.condition("AAAB")
    assert aaab is not None
    assert not aaab.has_omission()
    assert aaab.omitted_event_label() is None


def test_analysis_windows():
    """Test that analysis windows are correctly set."""
    paradigm = jtfne.standard_visual_omission()

    assert "baseline" in paradigm.analysis_windows
    assert "event" in paradigm.analysis_windows
    assert "post_event" in paradigm.analysis_windows

    assert paradigm.analysis_windows["baseline"] == (-500.0, 0.0)
    assert paradigm.analysis_windows["event"] == (0.0, 500.0)
    assert paradigm.analysis_windows["post_event"] == (500.0, 1000.0)

    assert paradigm.pre_stimulus_buffer_ms == 1000.0


def test_paradigm_json_safety():
    """Test that Paradigm objects can be safely serialized to JSON."""
    paradigm = jtfne.standard_visual_omission()

    # Test to_dict() for Paradigm
    paradigm_dict = paradigm.to_dict()
    assert isinstance(paradigm_dict, dict)
    assert "name" in paradigm_dict
    assert "conditions" in paradigm_dict
    assert "event_codes" in paradigm_dict
    assert "analysis_windows" in paradigm_dict

    # Test to_dict() for each condition
    for cond in paradigm.conditions:
        cond_dict = cond.to_dict()
        assert isinstance(cond_dict, dict)
        assert "name" in cond_dict
        assert "sequence" in cond_dict
        assert "events" in cond_dict
        assert isinstance(cond_dict["events"], list)

    # Test that paradigm can be JSON serialized
    json_str = json.dumps(paradigm_dict, allow_nan=False)
    assert isinstance(json_str, str)

    # Test that JSON can be loaded back
    loaded = json.loads(json_str)
    assert loaded["name"] == "standard_visual_omission"
    assert len(loaded["conditions"]) == 12


def test_no_truth_gate_expansion():
    """Test that paradigm objects do not expand or mutate truth gates."""
    paradigm = jtfne.standard_visual_omission()

    # Paradigm should not add or modify truth gates
    assert paradigm.metadata.get("truth_mode") is None or paradigm.metadata.get("truth_mode") == "truth_safe_unverified"
    assert paradigm.metadata.get("claim_level") is None or paradigm.metadata.get("claim_level") == "computational_scaffold"

    # Check that no condition tries to add biological/mechanism claims
    for cond in paradigm.conditions:
        cond_meta = cond.metadata
        for key in cond_meta.keys():
            # Ensure no mechanism_proven, active_inference_validated, etc.
            assert "mechanism_proven" not in key.lower()
            assert "active_inference" not in key.lower()
            assert "prediction_error" not in key.lower()

    # Check that event metadata doesn't contain over-claims
    for cond in paradigm.conditions:
        for evt in cond.events:
            evt_meta = evt.metadata
            for key in evt_meta.keys():
                assert "mechanism_proven" not in key.lower()
                assert "active_inference" not in key.lower()


def test_paradigm_event_creation():
    """Test that ParadigmEvent objects are created and serialized correctly."""
    evt = jtfne.ParadigmEvent(
        label="p1",
        onset_ms=100.0,
        duration_ms=50.0,
        code=101,
        stimulus="visual_stimulus_A",
        is_omission=False,
    )

    assert evt.label == "p1"
    assert evt.code == 101
    assert evt.stimulus == "visual_stimulus_A"
    assert not evt.is_omission

    evt_dict = evt.to_dict()
    assert evt_dict["label"] == "p1"
    assert evt_dict["code"] == 101


def test_paradigm_condition_creation():
    """Test that ParadigmCondition objects are created correctly."""
    evt1 = jtfne.ParadigmEvent(label="p1", onset_ms=100.0, code=101)
    evt2 = jtfne.ParadigmEvent(label="p2", onset_ms=200.0, code=103, is_omission=True)

    cond = jtfne.ParadigmCondition(
        name="TEST_AXXX",
        sequence=("A", "X", "B", "C"),
        omission_position="p2",
        condition_numbers=(1, 2, 3),
        events=(evt1, evt2),
    )

    assert cond.name == "TEST_AXXX"
    assert cond.has_omission()
    assert cond.omission_position == "p2"
    assert len(cond.condition_numbers) == 3


def test_paradigm_batch_method():
    """Test that Paradigm.batch() returns proper trial batch metadata."""
    paradigm = jtfne.standard_visual_omission()

    batch = paradigm.batch(n_trials=100, seed=42)
    assert batch["name"] == "standard_visual_omission"
    assert batch["n_trials"] == 100
    assert batch["seed"] == 42

    # Test with condition weights
    weights = {"AAAB": 2.0, "AAAX": 1.0}
    batch_weighted = paradigm.batch(n_trials=100, seed=42, condition_weights=weights)
    assert batch_weighted["condition_weights"] == weights
