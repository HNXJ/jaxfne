---
name: jaxfne-paradigm-design
description: >-
  Build jaxfne task paradigms (omission, global/local oddball, delayed-match-
  to-sample, MonkeyLogic-style explicit event lists, or any other sequential
  task) using the verified Paradigm/ParadigmCondition/ParadigmEvent grammar
  and the general_sequential_oddball_paradigm backbone. Use whenever a task
  asks to design, encode, or extend a trial/event structure, an oddball
  family, or a paradigm passed into jtfne.run_trials/Model.tune.
---

# jaxfne Paradigm Design

USE FIRST: `catalog-glossary-jaxfne` (confirm the exact builder still exists
at the names below before relying on this skill — it is a snapshot).

## The real grammar (verified 2026-06-22, jaxfne `dev`/`main` post-e90d87b)

Three frozen dataclasses in `jaxfne/paradigm.py`, all JSON-safe via `.to_dict()`:

```text
ParadigmEvent      label, onset_ms, duration_ms, code, stimulus, is_omission, metadata
ParadigmCondition  name, sequence: tuple[str, ...], omission_position, probability,
                   condition_numbers, events: tuple[ParadigmEvent, ...], metadata
Paradigm           name, blocks, conditions, comparison_code/label,
                   pre_stimulus_buffer_ms, analysis_windows, event_codes, metadata
```

`Paradigm.event_codes` (the dataclass field) is an **int** label→comparison-
code map (`{"p1": 101, ...}`) — do not confuse it with the builder kwarg
`event_codes=` below, which is a *different thing* (see caveat).

## Builders, narrowest to most general

- `jtfne.paradigm(name)` — empty `Paradigm` shell.
- `jtfne.evoked_l4_drive_paradigm(...)` — fixed baseline-vs-evoked L4 drive, 2 conditions.
- `jtfne.omission_oddball_paradigm(...)` — fixed expected/unexpected/omitted, 3 conditions.
- `jtfne.coop_omission_oddball_paradigm(duration_ms=..., freq_hz=..., omission_prob=...)` —
  Continuous Omission Oddball Paradigm: one long condition, randomly-omitted
  periodic pulses (the original, narrower COOP).
- **`jtfne.general_sequential_oddball_paradigm(...)`** — the generic backbone.
  Covers omission, global, local, synchronous, asynchronous, active, passive,
  and any other sequential family via declared event windows + per-condition
  token sequences or explicit event lists. **Prefer this over hand-rolling a
  new fixed-shape builder** for any task family not already covered above.
- `jtfne.general_delayed_match_to_sample_paradigm(...)` — thin wrapper over the
  backbone with DMS-flavored defaults (`comparison_label="sample"`).

## `general_sequential_oddball_paradigm` — two ways to specify a condition

**1. Token sequence** (when every condition shares the same event windows):

```python
paradigm = jtfne.general_sequential_oddball_paradigm(
    name="global_local_oddball",
    event_windows={"p1": (0, 200), "p2": (200, 400), "p3": (400, 600), "p4": (600, 800)},
    sequence_event_labels=("p1", "p2", "p3", "p4"),
    conditions={"AAAA": ("A", "A", "A", "A"), "AAAB": ("A", "A", "A", "B")},
)
```

Any token in `omission_tokens` (default `("X", "omit", "omission")`) at any
position marks that event `is_omission=True` and auto-sets
`ParadigmCondition.omission_position` to that event's label:

```python
jtfne.general_sequential_oddball_paradigm(
    event_windows={"p1": (0, 500), "p2": (500, 1000), "p3": (1000, 1500)},
    sequence_event_labels=("p1", "p2", "p3"),
    conditions={"omit_p2": ("A", "X", "A")},
)  # -> condition.has_omission() is True, omission_position == "p2"
```

**2. Explicit event list** (variable-length trials, non-stimulus events like
fixation/delay, or per-event metadata/codes) — pass `"events"` instead of
`"sequence"` in a condition spec:

```python
paradigm = jtfne.general_delayed_match_to_sample_paradigm(
    event_windows={"sample": (0, 200), "delay": (200, 700), "match": (700, 900)},
    sequence_event_labels=("sample", "match"),
    conditions=[{
        "name": "AA_match",
        "events": [
            {"label": "sample", "onset_ms": 0.0, "duration_ms": 200.0, "stimulus": "A"},
            {"label": "delay", "onset_ms": 200.0, "duration_ms": 500.0},
            {"label": "match", "onset_ms": 700.0, "duration_ms": 200.0, "stimulus": "A"},
        ],
    }],
)
```

MonkeyLogic-style trial scripts (an ordered label list with no shared
sequence shape across conditions) are exactly this explicit-event-list form —
one condition per script, one event dict per item:

```python
jtfne.general_sequential_oddball_paradigm(
    event_windows={"fix": (-200, 0), "p1": (0, 200), "d1": (200, 400), "p2": (400, 600)},
    conditions=[{"name": "trial1", "events": [
        {"label": "fix", "onset_ms": -200.0, "duration_ms": 200.0},
        {"label": "p1", "onset_ms": 0.0, "duration_ms": 200.0, "stimulus": "A"},
        {"label": "d1", "onset_ms": 200.0, "duration_ms": 200.0},
        {"label": "p2", "onset_ms": 400.0, "duration_ms": 200.0, "stimulus": "A"},
    ]}],
)
```

The explicit-events branch auto-infers `omission_position` from any
`is_omission=True` event the same way the token-sequence branch does —
verified in `tests/test_general_sequential_paradigm.py::test_explicit_event_list_infers_omission_position`.

## Other kwargs worth knowing

- `presentations={"A": some_callable_or_value}` — token→presentation map.
  Callables are stored as a JSON-safe `{kind, name, module}` summary in
  `paradigm.metadata["presentations"]`, never executed by the builder itself
  and never embedded as raw code — the runtime mapping stays outside the
  manifest.
- `static_events={"fix": {"stimulus": "fixation", "metadata": {...}}}` —
  per-label metadata/stimulus/code overrides merged into matching events,
  e.g. for a fixation window that every condition shares.
- `omission_tokens=(...)` — override which sequence tokens count as omissions.

## Caveat — `event_codes=` kwarg is NOT `Paradigm.event_codes`

The builder kwarg `event_codes=` is **only** an alias for `event_windows=`
("to match compact user-facing sketches" — if both are given,
`event_windows` wins). It does **not** set the returned `Paradigm.event_codes`
int map, which the builder always derives itself from `sequence_event_labels`
(`{"fx": 10, label_i: 100+i, ...}` plus `comparison_label: comparison_code`).
If you need specific int codes per label, set them after the call:
`paradigm = replace(paradigm, event_codes={...})` (or just rely on the
auto-derived ones — most readouts key on `label`/`onset_ms`, not the int code).

## Known footguns (verified 2026-07-01 — read before wiring a new paradigm into `simulate()`)

1. **`simulate(paradigm=...)` silently no-ops on the wrong type.**
   `Model._resolve_stimulus_schedule` only recognizes `StimulusSchedule` and
   `ParadigmCondition` — anything else (a bare list of event dicts, a raw
   `Paradigm`) resolves to `None` with **no error or warning**. Always pass
   a `StimulusSchedule` (built by hand or via `jaxfne.core.stimulus_schedule(...)`)
   or a `ParadigmCondition` (e.g. `paradigm.conditions[0]` from any builder
   above) — never the bare `Paradigm` object itself, and never a plain list.
2. **`stimulus_schedule()`'s drive heuristic isn't stimulus-aware.**
   `is_drive = not e.is_omission and e.onset_ms is not None` — this injects
   the default `drive_amplitude` (5.0) into *every* non-omission event,
   including pure timing markers like `trial_start`/`post_stimulus`/
   `post_omission` that `omission_oddball_paradigm` emits, not just the
   labeled stimulus. Harmless for a paired same-marker-structure contrast
   (both conditions share the markers, so it cancels) but will corrupt any
   comparison across conditions with differently-timed markers. Set
   `metadata={"drive_amplitude": 0.0}` on marker-only events if you need
   them genuinely silent (`coop_omission_oddball_paradigm` already does
   this correctly for its own omitted-pulse events — copy that pattern).
3. **`HierarchicalOddballParadigm` (in `jaxfne/sanity_delta.py`, not
   `paradigm.py`) is a different, heavier tool than it sounds.** It's a
   fixed-AAAB task-schedule + PFC-fixation-gate + reward-eligibility
   framework (`SanityDeltaConfig`, `BehaviorGate`), not a lightweight
   paradigm builder. Don't reach for it just because a task is named
   "hierarchical" or "global/local" — check `jaxfne-worker-context-router`
   or just verify the class's actual fields before assuming it fits; for a
   simple global/local oddball étude, `general_sequential_oddball_paradigm`
   with 4 labeled conditions (LSGS/LDGD/LDGS/LSGD-style) is the right tool.

## Truth-gate reminder

Paradigms are declarative trial structure only — no claim about behavioral
validity, biological plausibility, or empirical correspondence. `metadata`
carries `task_kind` (`"sequential_oddball"` / `"delayed_match_to_sample"`)
for downstream readout routing, not a claim of task validation.

Full implementation + test reference: `jaxfne/paradigm.py`,
`tests/test_general_sequential_paradigm.py` (jaxfne repo, commit `e90d87b`).
