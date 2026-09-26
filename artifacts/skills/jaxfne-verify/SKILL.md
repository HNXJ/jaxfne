---
name: jaxfne-verify
description: Check configured vs realized vs executed identity and named properties before any claim.
metadata:
  audience: agents
---
# jaxfne verify

## WHEN
Before reporting a result, and after any change to a spec, builder or runner.

## AUTHORITIES
1. Repository `artifacts/AGENTS.md` (Evidence; H-series review rules).
2. Live code: `jaxfne.agent.compare`, `jaxfne.agent.verify`, `jaxfne.agent.PROPERTIES`.

## RULES
- Identity is checked per semantic class (time, weight, mechanism, delay); a count match is not identity.
- A property not in `PROPERTIES` is refused; do not substitute a nearby check.
- A verification needs a counterexample that fails it (swap a realization, perturb a value).
- SPECIFIED, IMPLEMENTED, TESTED and OBSERVED are distinct evidence levels.

## STEPS
1. `jaxfne.agent.compare(run)`: read each class verdict and its missing/extra samples.
2. `jaxfne.agent.verify(run, p)` for every property the claim relies on.
3. Build one counterexample run and confirm the property FAILs on it.

## STOP
- Any FAIL; `NOT_APPLICABLE` where the claim needs the configured side.

## VERIFY
- PASS on the run, FAIL on the counterexample, both recorded.

## DONE
- Claim tied to property verdicts with evidence, plus the counterexample that fails.
