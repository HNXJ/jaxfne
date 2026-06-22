"""Tests for the generalized sequential paradigm backbone."""

from __future__ import annotations

import json

import jaxfne as jtfne


def _present_a(x: object) -> object:
    return x


class TestGeneralSequentialOddballParadigm:
    def test_root_api_is_available(self) -> None:
        assert hasattr(jtfne, "general_sequential_oddball_paradigm")
        assert hasattr(jtfne.paradigm, "general_sequential_oddball_paradigm")
        assert (
            jtfne.general_sequential_oddball_paradigm
            is jtfne.paradigm.general_sequential_oddball_paradigm
        )

    def test_variable_length_conditions_and_timing(self) -> None:
        event_windows = {
            "fix": (-500.0, 0.0),
            "p1": (0.0, 500.0),
            "p2": (500.0, 950.0),
            "p3": (950.0, 1450.0),
            "p4": (1450.0, 1950.0),
            "p5": (1950.0, 2450.0),
        }
        paradigm = jtfne.general_sequential_oddball_paradigm(
            name="toy_seq",
            event_windows=event_windows,
            sequence_event_labels=("p1", "p2", "p3", "p4", "p5"),
            conditions={
                "AAAAA": ("A", "A", "A", "A", "A"),
                "AAXXA": ("A", "A", "X", "X", "A"),
            },
            presentations={"A": _present_a, "X": "omission"},
            metadata={"source": "unit_test"},
        )

        assert paradigm.name == "toy_seq"
        assert paradigm.metadata["task_kind"] == "sequential_oddball"
        assert paradigm.metadata["sequence_event_labels"] == ["p1", "p2", "p3", "p4", "p5"]
        assert len(paradigm.conditions) == 2
        assert paradigm.conditions[0].sequence == ("A", "A", "A", "A", "A")
        assert paradigm.conditions[1].omission_position == "p3"
        assert paradigm.conditions[1].events[2].is_omission is True
        assert paradigm.conditions[1].events[2].duration_ms == 500.0
        assert paradigm.conditions[1].events[0].onset_ms == 0.0
        assert paradigm.conditions[1].events[-1].onset_ms == 1950.0
        assert paradigm.event_codes["p1"] == 101
        assert paradigm.event_codes["p5"] == 105

    def test_explicit_event_list_is_supported(self) -> None:
        paradigm = jtfne.general_sequential_oddball_paradigm(
            name="explicit_seq",
            event_windows={
                "fix": (-500.0, 0.0),
                "sample": (0.0, 250.0),
                "delay": (250.0, 1000.0),
                "match": (1000.0, 1250.0),
            },
            conditions=[
                {
                    "name": "sample_match",
                    "events": [
                        {"label": "fix", "onset_ms": -500.0, "duration_ms": 500.0, "stimulus": "fix"},
                        {"label": "sample", "onset_ms": 0.0, "duration_ms": 250.0, "stimulus": "A"},
                        {"label": "delay", "onset_ms": 250.0, "duration_ms": 750.0},
                        {"label": "match", "onset_ms": 1000.0, "duration_ms": 250.0, "stimulus": "A"},
                    ],
                    "sequence": ("A", "A"),
                    "probability": 1.0,
                }
            ],
            static_events={"fix": {"stimulus": "fixation", "metadata": {"target_group": "fixation_neurons"}}},
        )

        cond = paradigm.conditions[0]
        assert cond.name == "sample_match"
        assert cond.sequence == ("A", "A")
        assert cond.events[0].label == "fix"
        assert cond.events[0].metadata["target_group"] == "fixation_neurons"
        assert cond.events[1].label == "sample"
        assert cond.events[2].label == "delay"
        assert cond.events[3].label == "match"

    def test_explicit_event_list_infers_omission_position(self) -> None:
        paradigm = jtfne.general_sequential_oddball_paradigm(
            name="explicit_omission_seq",
            event_windows={
                "p1": (0.0, 500.0),
                "p2": (500.0, 1000.0),
                "p3": (1000.0, 1500.0),
            },
            conditions=[
                {
                    "name": "omit_p2",
                    "events": [
                        {"label": "p1", "onset_ms": 0.0, "duration_ms": 500.0, "stimulus": "A"},
                        {"label": "p2", "onset_ms": 500.0, "duration_ms": 500.0, "is_omission": True},
                        {"label": "p3", "onset_ms": 1000.0, "duration_ms": 500.0, "stimulus": "A"},
                    ],
                }
            ],
        )
        cond = paradigm.conditions[0]
        assert cond.has_omission()
        assert cond.omission_position == "p2"
        assert cond.events[1].is_omission is True

    def test_general_delayed_match_to_sample_wrapper(self) -> None:
        paradigm = jtfne.general_delayed_match_to_sample_paradigm(
            event_windows={
                "sample": (0.0, 200.0),
                "delay": (200.0, 900.0),
                "match": (900.0, 1100.0),
            },
            sequence_event_labels=("sample", "match"),
            conditions={"AA": ("A", "A")},
            presentations={"A": _present_a},
        )
        assert paradigm.metadata["task_kind"] == "delayed_match_to_sample"
        assert paradigm.condition("AA") is not None

    def test_json_safety(self) -> None:
        paradigm = jtfne.general_sequential_oddball_paradigm(
            event_windows={"p1": (0.0, 100.0)},
            sequence_event_labels=("p1",),
            conditions={"A": ("A",)},
            presentations={"A": _present_a},
        )
        payload = paradigm.to_dict()
        json.dumps(payload, allow_nan=False)
        json.dumps(paradigm.conditions[0].to_dict(), allow_nan=False)
