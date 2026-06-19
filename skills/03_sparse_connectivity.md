# Sparse Connectivity Skill

## Purpose
Replace dense O(N^2) connectivity construction with sparse or block-sparse construction.

## Rules
- Do not materialize dense `n x n` matrices when the actual graph is sparse or local.
- Generate edge lists directly from layer, area, or block rules.
- Prefer block-sparse, CSR-like, or edge-list representations for large networks.
- Preserve semantics; only the representation should change.
- Use dense construction only for small tests or explicit debugging.

## Acceptance checks
- Large-N builds do not scale quadratically in memory by default.
- Connectivity code has a sparse path that is the default for large models.
- Unit tests cover identical connectivity semantics between dense and sparse representations on small cases.
