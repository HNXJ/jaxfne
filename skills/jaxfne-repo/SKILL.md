---
name: jaxfne-repo
description: Bounded implementation, debugging, testing, API changes, and repository verification for jaxfne.
metadata:
  audience: agents
---
# jaxfne repository procedure

## WHEN TO USE
CODE work on jaxfne: bounded implementation, debugging, testing, API changes,
and repository verification.

## AUTHORITIES TO READ
1. Repository AGENTS.md (branches, root freeze, step completion rule).
2. Live code and the executable parameter/storage/path bindings of the
   selected backend.

## INVARIANTS
- Pre-existing user changes are preserved.
- The smallest semantic boundary is patched; compatibility is preserved
  unless the task explicitly authorizes contraction.
- Cheaper property/unit tests are preferred; assertions attach to shared
  minimal-circuit executions rather than multiplying simulations.
- Relevant nulls, negative controls, shape rejection, finiteness,
  deterministic/PRNG semantics, and continuation are verified when affected.
- Targeted tests and exact receipts are run; broader gates only when
  proportional to the checkpoint.
- Git remote mutation requires explicit authorization.

## PROCEDURE
Before mutation:
```bash
git branch --show-current
git rev-parse HEAD
git status --short
```
1. Preserve pre-existing user changes.
2. Identify the executable parameter/storage/path actually consumed by the
   selected backend before modifying optimization or configuration code.
3. Patch the smallest semantic boundary. Preserve compatibility unless the
   task explicitly authorizes contraction.
4. Add cheap property/unit tests first; attach many assertions to shared
   minimal-circuit executions rather than multiplying simulations.
5. Verify relevant nulls, negative controls, shape rejection, finiteness,
   deterministic/PRNG semantics, and continuation when affected.
6. Run targeted tests and exact receipts. Run broader gates only when
   proportional to the checkpoint.
7. If tests become slow, distinguish test-selection drift, repeated
   execution, compilation/runtime regression, and environment variance
   before optimizing.
8. Update directly affected docs only. Public docs describe
   equations/contracts, not debugging history.
9. Report API, mathematical, numerical, evidence, documentation, and
   compatibility deltas when material.

## STOP CONDITIONS
- Dirty tree ambiguity; unverified symbol or binding; remote mutation
  without explicit authorization; identity or integrity failure.

## REQUIRED VERIFICATION
- Targeted tests and exact receipts with output traces; the project
  validation suite; delta report when material.

## FORBIDDEN INFERENCES
- Capability claims without live verification; un-scoped documentation
  edits; silent remote mutations; bundling unfinished next-step work.

## COMPLETION
- Scoped delta committed under the step completion rule, dev == origin/dev
  verified, next checkpoint updated.

## AUTHORIZED REMOTE OPERATIONS
10. Git remote mutation requires explicit authorization. Routine non-force
    `git push origin dev` under the standing completion rule is part of step
    completion; tagging, main merge, release, and force operations need
    separate authorization.