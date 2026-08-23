# Canonical v2 acceptance source — location & ingestion contract

## Status

The current project-goal source for seal evaluation is the **TFNE submission
project-source set v2**, whose final acceptance list contains goals 1–100.
That file is maintained outside this checkout as of 2026-08-22. This document
fixes its canonical in-repository destination and ingestion rules so every
future reviewer evaluates the same 100 rows.

## Canonical location

```
artifacts/private_acceptance/jaxfne_v0_4_17_final_goals_v2.md
```

## Ingestion requirements (W1)

1. Content must be inserted verbatim from the owner-supplied source.
2. The file must carry this header block, preserving provenance semantics:

   > **Status:** SUBMISSION TARGET / SPECIFICATION — not executable scientific
   > evidence. This list defines what the candidate is evaluated against; it
   > confers no truth on any claim and never overrides frozen executable
   > behavior or frozen results.
   > Precedence: frozen executable/evidence truth > matching definitions >
   > this goal specification.

3. The superseded 95-goal snapshot (`jaxfne_v0_4_17_final_100_goals.md`)
   stays untouched beneath its SUPERSEDED banner.
4. After ingestion, re-run the row-based matrix scorer; rows 96–100 are scored
   only from this canonical copy.

## Non-authority clause

No entry in the canonical v2 list may be cited as evidence of implementation,
validation, or observed behavior. Evidence vocabulary remains:
SPECIFIED / IMPLEMENTED / TESTED / OBSERVED (see AGENTS.md).
