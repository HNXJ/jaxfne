---
name: jaxfne-frozen-use
description: Operating guidelines during the 40-day frozen-use period.
metadata:
  audience: agents
---
# jaxfne frozen-use procedure

## WHEN
All daily development, simulation experiments, documentation, and tooling during Delta C_core = 0 period.

## AUTHORITIES
1. `artifacts/issue_log/ISSUE_LOG.md`.
2. `scratch/CURRENT_TASK.md`.

## RULES
- Invariant: Delta C_core = 0 (jaxfne/ core physics and dynamics are FROZEN).
- Operational mode: observe -> reproduce -> log (NOT observe -> patch core).
- Writable areas: docs/, skills/, artifacts/figures/publication/final/, artifacts/issue_log/, scratch/.
- Structured observation logging required for all findings.

## STEPS
1. Perform simulations, analysis, tutorial runs, or doc improvements.
2. When encountering an anomaly, write a minimal reproducer.
3. Record entry in `artifacts/issue_log/ISSUE_LOG.md` (BUG, FRICTION, DOC, PERF, SCIENCE, IDEA, HARNESS, FACT).
4. Do not touch `jaxfne/` core files without explicit emergency authorization.

## STOP
- Core bug encountered: log it and stop, do not patch.

## VERIFY
- Minimal reproduction verified and recorded in issue log.

## DONE
- Observation recorded and development continued in allowed editable scope.
