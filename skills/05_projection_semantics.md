# Projection Semantics Skill

## Purpose
Make proxy readouts interpretable and explicit.

## Rules
- Any projection that normalizes must expose the normalization choice explicitly.
- Provide a density-preserving mode wherever row-normalized or unit-sum projection exists.
- Do not let a default projection erase absolute scale without making that choice visible.
- Keep proxy labels accurate in names, docs, and plots.
- Every projection report should state what is preserved and what is discarded.

## Acceptance checks
- Projection mode is part of the API contract.
- Tests can verify density-preserving and normalized behavior separately.
- Documentation states exactly what the projection preserves and what it discards.
