"""Task, condition, and event paradigms for jaxfne simulations.

This module also carries the generic sequential-paradigm builders used to
encode arbitrary event-structured tasks, including oddball and delayed-match
families.
"""

from __future__ import annotations

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
    sequence: tuple[str, ...]
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


def _normalize_window_pair(value: Any, *, label: str) -> tuple[float, float]:
    """Convert a two-value window specification to a float pair."""
    if isinstance(value, Mapping):
        start = value.get("start_ms", value.get("start"))
        end = value.get("end_ms", value.get("end"))
        value = (start, end)
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{label} must be a numeric 2-tuple/list, not a string")
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{label} must be a 2-tuple/list of (start_ms, end_ms)")
    start, end = value
    return float(start), float(end)


def _summarize_presentations(presentations: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Return JSON-safe summaries for token->presentation declarations.

    Presentation values may be callables, numeric gains, model adapters, or any
    other implementation detail. The paradigm object records only a safe summary
    so the task spec remains serializable; the runtime mapping can stay outside
    the manifest.
    """
    summary: dict[str, Any] = {}
    if not presentations:
        return summary
    for token, spec in presentations.items():
        if callable(spec):
            summary[str(token)] = {
                "kind": "callable",
                "name": getattr(spec, "__name__", spec.__class__.__name__),
                "module": getattr(spec, "__module__", None),
            }
            continue
        try:
            summary[str(token)] = json_safe(spec)
        except Exception:
            summary[str(token)] = {"kind": "repr", "repr": repr(spec)}
    return summary


def _coerce_paradigm_event(
    event: Any,
    *,
    default_window: Optional[tuple[float, float]] = None,
    sequence_token: Optional[str] = None,
    omission_tokens: Sequence[str] = ("X", "omit", "omission"),
    presentations: Optional[Mapping[str, Any]] = None,
    static_events: Optional[Mapping[str, Any]] = None,
) -> ParadigmEvent:
    """Coerce one event-like object into a ParadigmEvent."""
    if isinstance(event, ParadigmEvent):
        return event
    if not isinstance(event, Mapping):
        raise TypeError(f"event specifications must be mapping-like or ParadigmEvent, got {type(event).__name__}")
    label = str(event.get("label", sequence_token if sequence_token is not None else "event"))
    onset_ms = event.get("onset_ms")
    duration_ms = event.get("duration_ms")
    if (onset_ms is None or duration_ms is None) and default_window is not None:
        onset_ms = default_window[0] if onset_ms is None else onset_ms
        duration_ms = (default_window[1] - default_window[0]) if duration_ms is None else duration_ms
    stimulus = event.get("stimulus", sequence_token)
    is_omission = bool(event.get("is_omission", False)) or (sequence_token in omission_tokens if sequence_token is not None else False)
    metadata = dict(event.get("metadata", {}))
    if static_events and label in static_events:
        extra = static_events[label]
        if isinstance(extra, Mapping):
            metadata.update({
                str(k): v
                for k, v in extra.items()
                if k not in {"label", "onset_ms", "duration_ms", "stimulus", "is_omission", "code", "metadata"}
            })
            metadata.update(dict(extra.get("metadata", {})))
            if "stimulus" in extra and stimulus is None:
                stimulus = extra.get("stimulus")
            if "is_omission" in extra:
                is_omission = bool(extra["is_omission"])
            if "code" in extra and event.get("code") is None:
                event_code = extra.get("code")
            else:
                event_code = event.get("code")
        else:
            event_code = event.get("code")
    else:
        event_code = event.get("code")
    if sequence_token is not None and presentations and sequence_token in presentations:
        metadata.setdefault("presentation", presentations[sequence_token])
    return ParadigmEvent(
        label=label,
        onset_ms=float(onset_ms) if onset_ms is not None else None,
        duration_ms=float(duration_ms) if duration_ms is not None else None,
        code=int(event_code) if event_code is not None else None,
        stimulus=None if is_omission else (None if stimulus is None else str(stimulus)),
        is_omission=is_omission,
        metadata=metadata,
    )


def _build_sequential_condition(
    *,
    name: str,
    sequence: Sequence[Any],
    event_windows: Mapping[str, tuple[float, float]],
    sequence_event_labels: Sequence[str],
    omission_tokens: Sequence[str],
    presentations: Optional[dict[str, Any]] = None,
    static_events: Optional[Mapping[str, Any]] = None,
    probability: Optional[float] = None,
    condition_numbers: Sequence[int] = (),
    metadata: Optional[dict[str, Any]] = None,
    omission_position: Optional[str] = None,
) -> ParadigmCondition:
    """Build one generic sequential condition from a token sequence."""
    if len(sequence) != len(sequence_event_labels):
        raise ValueError(
            f"condition {name!r} has sequence length {len(sequence)}, "
            f"but {len(sequence_event_labels)} sequential event labels were provided"
        )
    events: list[ParadigmEvent] = []
    for label, token in zip(sequence_event_labels, sequence):
        if label not in event_windows:
            raise KeyError(f"event label {label!r} is missing from event_windows")
        events.append(
            _coerce_paradigm_event(
                {"label": label, "stimulus": token},
                default_window=event_windows[label],
                sequence_token=str(token),
                omission_tokens=omission_tokens,
                presentations=presentations,
                static_events=static_events,
            )
        )
    inferred_omission = omission_position
    if inferred_omission is None:
        for evt in events:
            if evt.is_omission:
                inferred_omission = evt.label
                break
    return ParadigmCondition(
        name=name,
        sequence=tuple(str(x) for x in sequence),
        omission_position=inferred_omission,
        probability=probability,
        condition_numbers=tuple(int(x) for x in condition_numbers),
        events=tuple(events),
        metadata=metadata or {},
    )


def _normalize_condition_specs(
    conditions: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Normalize condition declarations into a list of dict specs."""
    if conditions is None:
        return []
    if isinstance(conditions, Mapping):
        specs: list[dict[str, Any]] = []
        for name, value in conditions.items():
            if isinstance(value, Mapping):
                spec = dict(value)
                spec.setdefault("name", name)
                specs.append(spec)
            else:
                specs.append({"name": name, "sequence": tuple(value)})
        return specs
    return [dict(spec) for spec in conditions]


def general_sequential_oddball_paradigm(
    *,
    name: str = "general_sequential_oddball",
    event_windows: Optional[Mapping[str, Sequence[float]]] = None,
    event_codes: Optional[Mapping[str, Sequence[float]]] = None,
    sequence_event_labels: Optional[Sequence[str]] = None,
    conditions: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]] = None,
    presentations: Optional[Mapping[str, Any]] = None,
    static_events: Optional[Mapping[str, Any]] = None,
    blocks: Optional[Sequence[Mapping[str, Any]]] = None,
    analysis_windows: Optional[Mapping[str, Sequence[float]]] = None,
    comparison_label: str = "p1",
    comparison_code: int = 101,
    pre_stimulus_buffer_ms: float = 1000.0,
    omission_tokens: Sequence[str] = ("X", "omit", "omission"),
    metadata: Optional[dict[str, Any]] = None,
) -> Paradigm:
    """Build a general event-structured sequential paradigm.

    This is the generic backbone for any sequential task family where trial
    structure is expressed as an ordered list of event windows and each
    condition provides either a token sequence or a full explicit event list.

    Parameters
    ----------
    name:
        Paradigm name.
    event_windows:
        Mapping from event label to ``(start_ms, end_ms)`` windows.
        This is the preferred spelling.
    event_codes:
        Alias for ``event_windows`` to match compact user-facing sketches.
        If both are provided, ``event_windows`` wins. Distinct from
        :attr:`Paradigm.event_codes` (the int comparison-code map), which this
        builder still populates separately from ``sequence_event_labels``.
    sequence_event_labels:
        Ordered labels whose windows are populated by each condition's token
        sequence, e.g. ``("p1", "p2", "p3")``.
    conditions:
        Either a mapping of condition name to sequence tokens, or a sequence of
        explicit condition specs containing ``name`` and either ``sequence`` or
        ``events``.
    presentations:
        Optional token -> presentation descriptor mapping. Callables are stored
        only as JSON-safe summaries in the returned paradigm metadata.
    static_events:
        Optional mapping of non-sequence event labels to static metadata,
        stimulus annotations, or explicit event overrides.
    blocks:
        Optional block declarations to carry through unchanged.
    analysis_windows:
        Optional analysis window mapping. Defaults to a canonical
        baseline/event/post-event trio.
    comparison_label / comparison_code:
        Alignment marker metadata.
    pre_stimulus_buffer_ms:
        Alignment buffer metadata (ms).
    omission_tokens:
        Sequence tokens that should be treated as omissions.
    metadata:
        Additional JSON-safe metadata stored on the returned paradigm.

    Returns
    -------
    Paradigm
        Immutable sequential task specification.
    """
    windows_in = event_windows if event_windows is not None else event_codes
    if windows_in is None:
        raise ValueError("general_sequential_oddball_paradigm requires event_windows or event_codes")
    normalized_windows = {str(label): _normalize_window_pair(window, label=str(label)) for label, window in windows_in.items()}

    seq_labels = tuple(str(x) for x in (sequence_event_labels or tuple(normalized_windows.keys())))
    if not seq_labels:
        raise ValueError("general_sequential_oddball_paradigm requires at least one sequence_event_label")

    presentation_summaries = _summarize_presentations(presentations)
    specs = _normalize_condition_specs(conditions)
    if not specs:
        raise ValueError("general_sequential_oddball_paradigm requires at least one condition")

    conds: list[ParadigmCondition] = []
    for spec in specs:
        cond_name = str(spec.get("name", "condition"))
        if "events" in spec and spec["events"] is not None:
            explicit_events = tuple(
                _coerce_paradigm_event(
                    ev,
                    default_window=normalized_windows.get(str(ev.get("label"))) if isinstance(ev, Mapping) else None,
                    sequence_token=str(ev.get("stimulus")) if isinstance(ev, Mapping) and ev.get("stimulus") is not None else None,
                    omission_tokens=omission_tokens,
                    presentations=presentation_summaries,
                    static_events=static_events,
                )
                for ev in spec["events"]
            )
            sequence = spec.get("sequence")
            if sequence is None:
                sequence = tuple(
                    str(evt.stimulus if evt.stimulus is not None else evt.label)
                    for evt in explicit_events
                    if evt.label in seq_labels
                )
            omission_position = spec.get("omission_position")
            if omission_position is None:
                for evt in explicit_events:
                    if evt.is_omission:
                        omission_position = evt.label
                        break
            conds.append(
                ParadigmCondition(
                    name=cond_name,
                    sequence=tuple(str(x) for x in sequence),
                    omission_position=omission_position,
                    probability=spec.get("probability"),
                    condition_numbers=tuple(int(x) for x in spec.get("condition_numbers", ()) or ()),
                    events=explicit_events,
                    metadata=dict(spec.get("metadata", {})),
                )
            )
            continue

        sequence = spec.get("sequence")
        if sequence is None:
            raise ValueError(f"condition {cond_name!r} must provide either 'sequence' or 'events'")
        conds.append(
            _build_sequential_condition(
                name=cond_name,
                sequence=tuple(sequence),
                event_windows=normalized_windows,
                sequence_event_labels=seq_labels,
                omission_tokens=omission_tokens,
                presentations=presentation_summaries if presentation_summaries else None,
                static_events=static_events,
                probability=spec.get("probability"),
                condition_numbers=spec.get("condition_numbers", ()),
                metadata=dict(spec.get("metadata", {})),
                omission_position=spec.get("omission_position"),
            )
        )

    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("task_kind", "sequential_oddball")
    merged_metadata["sequence_event_labels"] = list(seq_labels)
    merged_metadata["event_windows"] = {label: [start, end] for label, (start, end) in normalized_windows.items()}
    if presentation_summaries:
        merged_metadata["presentations"] = presentation_summaries
    if static_events is not None:
        try:
            merged_metadata["static_events"] = json_safe(static_events)
        except Exception:
            merged_metadata["static_events"] = {str(k): repr(v) for k, v in static_events.items()}

    if analysis_windows is None:
        analysis_windows = {
            "baseline": (-500.0, 0.0),
            "event": (0.0, 500.0),
            "post_event": (500.0, 1000.0),
        }

    event_code_map = {
        "fx": 10,
        **{str(label): int(100 + idx) for idx, label in enumerate(seq_labels, start=1)},
    }
    event_code_map[str(comparison_label)] = int(comparison_code)

    return Paradigm(
        name=name,
        blocks=list(blocks or []),
        conditions=tuple(conds),
        comparison_code=comparison_code,
        comparison_label=comparison_label,
        pre_stimulus_buffer_ms=pre_stimulus_buffer_ms,
        analysis_windows={str(k): _normalize_window_pair(v, label=str(k)) for k, v in analysis_windows.items()},
        event_codes=event_code_map,
        metadata=merged_metadata,
    )


def general_delayed_match_to_sample_paradigm(
    *,
    name: str = "general_delayed_match_to_sample",
    event_windows: Optional[Mapping[str, Sequence[float]]] = None,
    event_codes: Optional[Mapping[str, Sequence[float]]] = None,
    sequence_event_labels: Optional[Sequence[str]] = None,
    conditions: Optional[Mapping[str, Any] | Sequence[Mapping[str, Any]]] = None,
    presentations: Optional[Mapping[str, Any]] = None,
    static_events: Optional[Mapping[str, Any]] = None,
    blocks: Optional[Sequence[Mapping[str, Any]]] = None,
    analysis_windows: Optional[Mapping[str, Sequence[float]]] = None,
    comparison_label: str = "sample",
    comparison_code: int = 101,
    pre_stimulus_buffer_ms: float = 1000.0,
    omission_tokens: Sequence[str] = ("X", "omit", "omission"),
    metadata: Optional[dict[str, Any]] = None,
) -> Paradigm:
    """Build a general delayed-match-to-sample paradigm on the same backbone."""
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("task_kind", "delayed_match_to_sample")
    return general_sequential_oddball_paradigm(
        name=name,
        event_windows=event_windows,
        event_codes=event_codes,
        sequence_event_labels=sequence_event_labels,
        conditions=conditions,
        presentations=presentations,
        static_events=static_events,
        blocks=blocks,
        analysis_windows=analysis_windows,
        comparison_label=comparison_label,
        comparison_code=comparison_code,
        pre_stimulus_buffer_ms=pre_stimulus_buffer_ms,
        omission_tokens=omission_tokens,
        metadata=merged_metadata,
    )


@dataclass(frozen=True)
class Paradigm:
    """Immutable experimental-paradigm specification (blocks, conditions, windows).

    ``name`` labels the paradigm; ``blocks`` and ``conditions`` declare the trial
    structure; ``alignment_code``/``alignment_label`` set the event-alignment
    marker; ``pre_stimulus_buffer_ms`` is the pre-event buffer (ms); and
    ``analysis_windows`` maps window names to ``(start_ms, end_ms)`` ranges (all
    in milliseconds, relative to the alignment event).
    """

    name: str = "none"
    blocks: list[dict[str, Any]] = field(default_factory=list)
    conditions: tuple[ParadigmCondition, ...] = ()
    comparison_code: int = 101
    comparison_label: str = "p1"
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
        """Documented public function `habituation`."""
        return replace(
            self,
            blocks=[*self.blocks, {"kind": "habituation", "sequence": list(sequence), "n_trials": n_trials}],
        )

    def main_block(self, **kwargs: Any) -> "Paradigm":
        """Documented public function `main_block`."""
        return replace(self, blocks=[*self.blocks, {"kind": "main_block", **kwargs}])

    def batch(self, n_trials: int, seed: int = 0, condition_weights: Optional[dict[str, float]] = None) -> dict[str, Any]:
        """Documented public function `batch`."""
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
            "comparison_code": self.comparison_code,
            "comparison_label": self.comparison_label,
            "pre_stimulus_buffer_ms": self.pre_stimulus_buffer_ms,
            "analysis_windows": self.analysis_windows,
            "event_codes": self.event_codes,
            "metadata": self.metadata,
        })


def paradigm(name: str = "none") -> Paradigm:
    """Documented public function `paradigm`."""
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


def coop_omission_oddball_paradigm(
    duration_ms: float = 10000.0,
    dt_ms: float = 0.5,
    freq_hz: float = 6.0,
    omission_prob: float = 0.1,
    standard_amplitude: float = 5.0,
    pre_stimulus_buffer_ms: float = 200.0,
    target_indices: Optional[Sequence[int]] = None,
    seed: int = 42,
    name: str = "coop_omission_oddball"
) -> Paradigm:
    """Create a Continuous Omission Oddball Paradigm (COOP) stimulus sequence.
    
    Chains standard triangular/pulse drive events and omission events continuously.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    
    period_ms = 1000.0 / freq_hz
    event_dur_ms = 50.0
    
    onsets = np.arange(pre_stimulus_buffer_ms, duration_ms - period_ms, period_ms)
    events = []
    
    events.append(ParadigmEvent(
        label="trial_start",
        onset_ms=0.0,
        duration_ms=pre_stimulus_buffer_ms,
        metadata={}
    ))
    
    for idx, onset in enumerate(onsets):
        is_omitted = rng.uniform() < omission_prob
        label = "omission" if is_omitted else f"pulse_{idx}"
        meta = {
            "drive_amplitude": 0.0 if is_omitted else standard_amplitude,
            "event_duration_ms": event_dur_ms
        }
        if target_indices is not None:
            meta["target_indices"] = list(target_indices)
            
        events.append(ParadigmEvent(
            label=label,
            onset_ms=float(onset),
            duration_ms=event_dur_ms,
            is_omission=is_omitted,
            metadata=meta
        ))
        
    coop_condition = ParadigmCondition(
        name="coop_condition",
        sequence=("pre", "coop", "post", "null"),
        events=tuple(events),
        probability=1.0,
        omission_position="standard" if omission_prob > 0.0 else None
    )
    
    return Paradigm(
        name=name,
        conditions=(coop_condition,),
        pre_stimulus_buffer_ms=pre_stimulus_buffer_ms,
        analysis_windows={
            "baseline": (0.0, pre_stimulus_buffer_ms),
            "continuous_coop": (pre_stimulus_buffer_ms, duration_ms)
        }
    )

import sys
from types import ModuleType as _ModuleType

class _ParadigmModuleWrapper(_ModuleType):
    def __call__(self, *args, **kwargs):
        return paradigm(*args, **kwargs)

# Replace the module in sys.modules
_current_module = sys.modules[__name__]
_wrapper = _ParadigmModuleWrapper(__name__)
_wrapper.__dict__.update(_current_module.__dict__)
_wrapper.__file__ = __file__
sys.modules[__name__] = _wrapper


