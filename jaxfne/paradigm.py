from dataclasses import dataclass, field, replace
from typing import Any, Optional, Sequence, Mapping
from .io import json_safe

@dataclass(frozen=True)
class ParadigmEvent:
    """Discrete event within a task trial: stimulus, behavioral code, or omission marker."""

    label: str
    onset_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    code: Optional[int] = None
    stimulus: Optional[str] = None
    is_omission: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        return json_safe({
            "label": self.label,
            "onset_ms": self.onset_ms,
            "duration_ms": self.duration_ms,
            "code": self.code,
            "stimulus": self.stimulus,
            "is_omission": self.is_omission,
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class ParadigmCondition:
    """A specific trial condition: sequence of stimuli and associated events."""

    name: str
    sequence: tuple[str, str, str, str]
    omission_position: Optional[str] = None
    probability: Optional[float] = None
    condition_numbers: tuple[int, ...] = ()
    events: tuple[ParadigmEvent, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_omission(self) -> bool:
        """Return True if this condition contains an omission."""
        return self.omission_position is not None

    def omitted_event_label(self) -> Optional[str]:
        """Return the label of the omitted event, or None if no omission."""
        if self.omission_position is None:
            return None
        for evt in self.events:
            if evt.label == self.omission_position:
                return evt.label
        return self.omission_position

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        return json_safe({
            "name": self.name,
            "sequence": self.sequence,
            "omission_position": self.omission_position,
            "probability": self.probability,
            "condition_numbers": self.condition_numbers,
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        })


@dataclass(frozen=True)
class Paradigm:
    name: str = "none"
    blocks: list[dict[str, Any]] = field(default_factory=list)
    conditions: tuple[ParadigmCondition, ...] = ()
    alignment_code: int = 101
    alignment_label: str = "p1"
    pre_stimulus_buffer_ms: float = 1000.0
    analysis_windows: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "baseline": (-500.0, 0.0),
        "event": (0.0, 500.0),
        "post_event": (500.0, 1000.0),
    })
    event_codes: dict[str, int] = field(default_factory=lambda: {
        "fx": 10,
        "p1": 101,
        "p2": 103,
        "p3": 105,
        "p4": 107,
        "rw": 96,
    })
    metadata: dict[str, Any] = field(default_factory=dict)

    def habituation(self, sequence: Sequence[str], n_trials: int) -> "Paradigm":
        return replace(
            self,
            blocks=[*self.blocks, {"kind": "habituation", "sequence": list(sequence), "n_trials": n_trials}],
        )

    def main_block(self, **kwargs: Any) -> "Paradigm":
        return replace(self, blocks=[*self.blocks, {"kind": "main_block", **kwargs}])

    def batch(self, n_trials: int, seed: int = 0, condition_weights: Optional[dict[str, float]] = None) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_trials": n_trials,
            "seed": seed,
            "blocks": self.blocks,
            "condition_weights": condition_weights,
        }

    def condition(self, name: str) -> Optional[ParadigmCondition]:
        """Return ParadigmCondition by name, or None if not found."""
        for cond in self.conditions:
            if cond.name == name:
                return cond
        return None

    def condition_names(self) -> list[str]:
        """Return list of condition names."""
        return [c.name for c in self.conditions]

    def omission_conditions(self) -> list[ParadigmCondition]:
        """Return list of conditions containing omissions."""
        return [c for c in self.conditions if c.has_omission()]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe dictionary representation."""
        return json_safe({
            "name": self.name,
            "blocks": self.blocks,
            "conditions": [c.to_dict() for c in self.conditions],
            "alignment_code": self.alignment_code,
            "alignment_label": self.alignment_label,
            "pre_stimulus_buffer_ms": self.pre_stimulus_buffer_ms,
            "analysis_windows": self.analysis_windows,
            "event_codes": self.event_codes,
            "metadata": self.metadata,
        })


def paradigm(name: str = "none") -> Paradigm:
    return Paradigm(name=name)


def evoked_l4_drive_paradigm(
    l4_onset_ms: float = 100.0,
    l4_duration_ms: float = 200.0,
    l4_amplitude: float = 1.0,
    pre_stimulus_buffer_ms: float = 200.0,
    post_stimulus_buffer_ms: float = 500.0,
    name: str = "evoked_l4_drive",
) -> Paradigm:
    """Create a simple baseline-vs-evoked L4 drive paradigm.

    Parameters
    ----------
    l4_onset_ms : float
        Stimulus onset (ms after trial start).
    l4_duration_ms : float
        Stimulus duration (ms).
    l4_amplitude : float
        L4 drive amplitude (relative units).
    pre_stimulus_buffer_ms : float
        Pre-stimulus baseline window (ms).
    post_stimulus_buffer_ms : float
        Post-stimulus window (ms).
    name : str
        Paradigm name.

    Returns
    -------
    Paradigm
        A Paradigm with baseline and evoked conditions.

    Notes
    -----
    This is a minimal proxy paradigm scaffold. All outputs are simulated.
    No empirical validation against real data.
    """
    # Baseline condition: no L4 drive
    baseline_condition = ParadigmCondition(
        name="baseline",
        sequence=("pre", "stim", "post", "null"),
        events=(
            ParadigmEvent(label="trial_start", onset_ms=0.0, duration_ms=pre_stimulus_buffer_ms),
            ParadigmEvent(label="stim_absent", onset_ms=pre_stimulus_buffer_ms, duration_ms=l4_duration_ms),
            ParadigmEvent(label="post_stim", onset_ms=pre_stimulus_buffer_ms + l4_duration_ms, duration_ms=post_stimulus_buffer_ms),
        ),
        probability=0.5,
    )

    # Evoked condition: with L4 drive
    evoked_condition = ParadigmCondition(
        name="evoked",
        sequence=("pre", "stim", "post", "null"),
        events=(
            ParadigmEvent(label="trial_start", onset_ms=0.0, duration_ms=pre_stimulus_buffer_ms),
            ParadigmEvent(label="l4_drive", onset_ms=pre_stimulus_buffer_ms, duration_ms=l4_duration_ms, stimulus="l4_evoked"),
            ParadigmEvent(label="post_stim", onset_ms=pre_stimulus_buffer_ms + l4_duration_ms, duration_ms=post_stimulus_buffer_ms),
        ),
        probability=0.5,
    )

    paradigm = Paradigm(
        name=name,
        conditions=(baseline_condition, evoked_condition),
        pre_stimulus_buffer_ms=pre_stimulus_buffer_ms,
        analysis_windows={
            "baseline": (0.0, pre_stimulus_buffer_ms),
            "evoked": (pre_stimulus_buffer_ms, pre_stimulus_buffer_ms + l4_duration_ms),
            "post_evoked": (pre_stimulus_buffer_ms + l4_duration_ms, pre_stimulus_buffer_ms + l4_duration_ms + post_stimulus_buffer_ms),
        },
    )
    return paradigm


def omission_oddball_paradigm(
    standard_onset_ms: float = 500.0,
    standard_duration_ms: float = 100.0,
    deviant_onset_ms: Optional[float] = None,
    deviant_duration_ms: float = 100.0,
    deviant_label: str = "deviant",
    omission_position: str = "standard",
    pre_stimulus_buffer_ms: float = 200.0,
    post_stimulus_buffer_ms: float = 500.0,
    name: str = "omission_oddball",
) -> Paradigm:
    """Create an omission/oddball detection paradigm.

    Parameters
    ----------
    standard_onset_ms : float
        Standard stimulus onset (ms after trial start).
    standard_duration_ms : float
        Standard stimulus duration (ms).
    deviant_onset_ms : Optional[float]
        Deviant stimulus onset. If None, use standard_onset_ms.
    deviant_duration_ms : float
        Deviant stimulus duration (ms).
    deviant_label : str
        Label for deviant condition (e.g., "deviant", "unexpected").
    omission_position : str
        Position of omitted stimulus: "standard" or "deviant".
    pre_stimulus_buffer_ms : float
        Pre-stimulus baseline window (ms).
    post_stimulus_buffer_ms : float
        Post-stimulus window (ms).
    name : str
        Paradigm name.

    Returns
    -------
    Paradigm
        A Paradigm with expected, unexpected, omitted, and post-omission conditions.

    Notes
    -----
    This is a minimal omission/oddball scaffold. All outputs are simulated.
    No empirical validation against real data. Event windows are declarative.
    """
    if deviant_onset_ms is None:
        deviant_onset_ms = standard_onset_ms

    # Expected condition: standard stimulus
    expected_condition = ParadigmCondition(
        name="expected",
        sequence=("pre", "standard", "post", "null"),
        events=(
            ParadigmEvent(label="trial_start", onset_ms=0.0, duration_ms=pre_stimulus_buffer_ms),
            ParadigmEvent(label="standard", onset_ms=pre_stimulus_buffer_ms, duration_ms=standard_duration_ms, stimulus="standard_tone"),
            ParadigmEvent(label="post_stimulus", onset_ms=pre_stimulus_buffer_ms + standard_duration_ms, duration_ms=post_stimulus_buffer_ms),
        ),
        probability=0.8,
    )

    # Unexpected condition: deviant stimulus
    unexpected_condition = ParadigmCondition(
        name="unexpected",
        sequence=("pre", "deviant", "post", "null"),
        events=(
            ParadigmEvent(label="trial_start", onset_ms=0.0, duration_ms=pre_stimulus_buffer_ms),
            ParadigmEvent(label=deviant_label, onset_ms=pre_stimulus_buffer_ms, duration_ms=deviant_duration_ms, stimulus="deviant_tone"),
            ParadigmEvent(label="post_stimulus", onset_ms=pre_stimulus_buffer_ms + deviant_duration_ms, duration_ms=post_stimulus_buffer_ms),
        ),
        probability=0.1,
        omission_position=None,
    )

    # Omitted condition: stimulus silent
    omitted_condition = ParadigmCondition(
        name="omitted",
        sequence=("pre", "silence", "post", "null"),
        events=(
            ParadigmEvent(label="trial_start", onset_ms=0.0, duration_ms=pre_stimulus_buffer_ms),
            ParadigmEvent(label="omission", onset_ms=pre_stimulus_buffer_ms, duration_ms=standard_duration_ms, is_omission=True),
            ParadigmEvent(label="post_omission", onset_ms=pre_stimulus_buffer_ms + standard_duration_ms, duration_ms=post_stimulus_buffer_ms),
        ),
        probability=0.1,
        omission_position=omission_position,
    )

    paradigm = Paradigm(
        name=name,
        conditions=(expected_condition, unexpected_condition, omitted_condition),
        pre_stimulus_buffer_ms=pre_stimulus_buffer_ms,
        analysis_windows={
            "baseline": (0.0, pre_stimulus_buffer_ms),
            "stimulus": (pre_stimulus_buffer_ms, pre_stimulus_buffer_ms + standard_duration_ms),
            "post_stimulus": (pre_stimulus_buffer_ms + standard_duration_ms, pre_stimulus_buffer_ms + standard_duration_ms + post_stimulus_buffer_ms),
        },
    )
    return paradigm
