# W17 downstream mechanism semantics (design surfaces)

Status: **PLAN** — v0.4.22 semantics and surface specification. Evidence receipt:
`artifacts/audit/jomission_downstream_lessons.md`.

Source: `artifacts/roadmap/ROADMAP_0422_0424.md` W17 / §9.

## Sub-items and release targets

| ID | Goal | 0.4.22 | 0.4.23 |
| --- | --- | --- | --- |
| W17.1 | configured → realized → executed → effective queryable | semantics; low-risk prototype | deeper integration |
| W17.2 | memory preflight (persistent/dynamic/delay/recording/temp) | extends W10; design `memory_report` | implement + streaming |
| W17.3 | operating-point system ID $S_{ij}=dY_i/du_j$ | benchmark/probe interface | utility if evidenced |
| W17.4 | run finalization (execution complete ≠ evidence durable) | audit surfaces; define semantics | minimal solution if justified |

## Provisional public surfaces (syntax not finalized)

- `mechanism_report(selector, intervention=None)` — effective mechanism query.
- `memory_report(model, runtime, recorder)` — component memory preflight.
- Perturbation hooks for local susceptibility at operating point.
- Transactional finalize: progress, atomic persist, hash, manifest, terminal state.

## Authority rules

- Edge transforms align with W16 structural $G$ — one mutation authority.
- Do not upstream Jomission-specific controllers or substrate gates.
- Evidence priority: OBSERVED_DOWNSTREAM_NEED > GENERALIZABLE_INFERENCE >
  PROJECT_SPECIFIC (rejected).

## v0.4.22 deliverable

Semantics and surface names documented; no public implementation claims beyond
existing audit/instrumentation receipts (W10, W8).
