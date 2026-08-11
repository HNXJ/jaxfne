---
name: jaxfne-paradigm-design
description: >-
  Build jaxfne paradigms, conditions, events, schedules, oddball tasks, and
  delayed-match-to-sample structures using the current public grammar.
---

# jaxfne paradigm procedure

Read `catalog-glossary-jaxfne` before naming a builder. Paradigms declare trial
structure; they do not establish behavioral validity or biological claims.

## Public objects

Verify the current dataclasses:

```text
ParadigmEvent
ParadigmCondition
Paradigm
```

Use JSON-safe event metadata and explicit onset/duration/stimulus/omission
fields. Preserve condition labels and analysis windows.

## Builder choice

- `paradigm(...)` — empty shell when constructing explicitly.
- `omission_oddball_paradigm(...)` — fixed omission structure.
- `coop_omission_oddball_paradigm(...)` — continuous omission pulses.
- `general_sequential_oddball_paradigm(...)` — general token or explicit-event
  backbone.
- `general_delayed_match_to_sample_paradigm(...)` — DMS-flavored wrapper.

Use the general backbone rather than creating another fixed-shape builder when
the existing event grammar represents the task.

## Event procedure

Choose one representation per condition:

- token sequences when event windows are shared;
- explicit event lists for variable-length trials, markers, delays, or
  per-event metadata.

For per-neuron drive targeting, place `target_indices` on the event dictionary
and derive indices from `model.neuron_table()`. It is not a schedule constructor
argument.

Pass a supported `StimulusSchedule` or `ParadigmCondition` into simulation.
Verify how the current model resolves a paradigm; do not pass a bare list or
unsupported wrapper by assumption.

## Validation

- Check event windows and condition structure.
- Check marker-only events do not receive unintended drive.
- Check omission flags and inferred omission positions.
- Check JSON serialization and finite schedules.
- Add a targeted public-builder test for new behavior.

Implementation truth is in `jaxfne/paradigm.py`, `jaxfne/stimulus.py`, and
tests. Mathematical/task interpretation belongs to project source documents.
