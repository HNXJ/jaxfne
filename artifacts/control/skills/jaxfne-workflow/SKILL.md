---
name: jaxfne-workflow
description: Route a scientific modeling task through the canonical intent-to-evidence workflow. Use when executing or guiding any end-to-end JaxFNE question.
metadata:
  audience: agents
---
# jaxfne canonical workflow

## WHEN
Any end-to-end scientific task: intent → TFNE/JDNA specification →
canonical JaxFNE path → realization → execution → observation → claim →
V(claim) → evidence. This skill routes and gates; it implements nothing.

## AUTHORITIES
1. Repository `artifacts/AGENTS.md` (evidence discipline, work loop).
2. Lifecycle semantics: `docs/doctrine/tfne_algebra.md`,
   `docs/doctrine/tfne_jdna_boundary.md`.
3. Specialized skills below (invoked, never copied).

## ROUTING (invoke, do not reimplement)
| Stage | Skill | Gate before advancing |
|---|---|---|
| Intent → TFNE/JDNA spec | jaxfne-core | question stated; TFNE/JDNA chosen with provenance |
| Canonical entry selection | jaxfne-core | parallel paths named; chosen one justified |
| Realization → execution | jaxfne-repo | symbols live-verified; smallest scoped delta |
| Observation → claim | jaxfne-science | protocol + nulls declared; proxy status explicit |
| V(claim) → evidence | jaxfne-science, jaxfne-audit | strength matches claim scope |
| Release/seal | jaxfne-release, jaxfne-seal | authorities + gates green |

## RULES
- One canonical path per capability (see jaxfne-core pinning rule).
- TFNE/JDNA → JaxFNE handoff preserves semantic boundaries: no second
  simulator, no silent flattening, no invented defaults at boundaries.
- `configured → realized → executed → effective` stay distinct; compare
  across stages, never collapse them.
- Verification is selected from the actual claim (V=V(claim)).
- Unsupported routes, ambiguous intent, or insufficient evidence →
  STOP/ask, never improvise a parallel implementation.

## STOP
- Ambiguous scientific intent; unresolvable entry selection; refused
  capability needed by the task; evidence weaker than the claim.

## VERIFY
- Each stage's gate recorded (chosen entry, claim, evidence strength).

## DONE
- Evidence bundle traceable intent→claim with per-stage gates logged.
