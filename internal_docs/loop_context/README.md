# jaxfne loop context bundle

Canonical, contract-first context for autonomous `/loop` and `/pr-maintain` ticks
and for any downstream Claude Code agent doing jaxfne work. Read `00_MANIFEST.md`
first — it carries the **live reconciliation** block and the trust order.

## Trust order (non-negotiable)

**live git > this reconciled bundle > original ZIP/PDF/checklist/assessment** (context only).

Always `git fetch --all --prune` and re-freeze `main/dev/tag` before any mutation.
Never name a symbol, path, or line from this bundle without re-verifying it against
live git — the bundle was seeded from a stale `0.3.27` ZIP and reconciled to
`fab4c9c` (`jaxfne 0.3.29`) on 2026-06-04.

## Files

| file | use |
|---|---|
| `00_MANIFEST.md` | purpose, live reconciliation, input/source hashes, freshness |
| `01_REPO_MAP.md` | module-by-module map (role, JAX-critical, public, risk, tests) |
| `02_PUBLIC_API_CONTRACT.md` | root `__all__` surface, signatures, stubs/leaks |
| `03_JAX_RUNTIME_CONTRACT.md` | PRNG/scan/vmap/jit/dtype/PyTree contract + live grep counts |
| `04_TRUTH_GATES_AND_CLAIMS.md` | truth gates + proxy-vs-solver claim contract |
| `05_BACKLOG.md` | ranked backlog with live status (B01 DONE; B02 next) |
| `06_VALIDATION_LADDER.md` | copy-paste validation commands + expected receipts |
| `07_V0330_ARCHITECTURE_NOTES.md` | v0.3.30 connectivity/FlatNet/PyNWB plans (RED/gated) |
| `08_RISKS_AND_FRAGILITIES.md` | fragile spots + smallest safe mitigations |

## Current state (as reconciled)

- Release: `v0.3.29` @ `fab4c9c`; `main == dev`; `agy` untouched.
- B01 (objective null RNG reproducibility): **DONE** — PR #22, green, awaiting human merge go.
- Next ready GREEN item: **B02** (release-clean script hardening).
- RED/gated (need human design): B07 connectivity compiler, B08 FlatNet boundary, B09 PyNWB export.

## Maintenance

When a backlog item ships, update `05_BACKLOG.md`'s status table and the matching
risk row in `08_RISKS_AND_FRAGILITIES.md`. Keep this bundle reconciled to the
latest released SHA; stale context is worse than no context.
