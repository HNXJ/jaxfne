# Batch-First Simulation Skill

## Purpose
Make multi-seed and multi-trial simulation fast by default.

## Rules
- Prefer `vmap` or other vectorized execution when more than one seed, candidate, or trial is present.
- Keep Python loops only as a deliberate fallback when vectorization is not viable.
- Reuse compiled models; rebuild only when structure changes.
- Keep hot numerical paths in JAX arrays and JAX control flow.

## Acceptance checks
- Batched runs use vectorized execution on the common path.
- Python loops are not the default path for repeated trials.
- Performance-sensitive tests confirm that the batch path is the fast path.
