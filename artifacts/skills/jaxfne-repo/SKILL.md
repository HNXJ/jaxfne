---
name: jaxfne-repo
description: Bounded implementation, debugging, testing, API changes, and repository verification for jaxfne.
metadata:
  audience: agents
---
# jaxfne repository procedure

## WHEN
CODE work on jaxfne: bounded implementation, debugging, testing, API changes, and repository verification.

## AUTHORITIES
1. Repository `artifacts/AGENTS.md` (branches, root freeze, step completion rule).
2. Live code and executable bindings.

## RULES
- Pre-existing user changes are preserved.
- Patch the smallest semantic boundary; preserve compatibility.
- Delta C_core = 0 during frozen period; no core patches without explicit authorization.
- Git remote mutation requires explicit authorization.

## STEPS
1. Execute Gate 0 (`scripts/harness/gate0_git_reality.py`).
2. Identify parameter/storage bindings in selected backend.
3. Write cheap property/unit tests; attach assertions to shared minimal-circuit runs.
4. Run targeted tests and exact receipts.
5. Update affected docs and register in `mkdocs.yml` nav.

## STOP
- Dirty tree ambiguity; unverified symbol; core freeze breach.

## VERIFY
- Targeted tests and exact receipts with output traces; delta report when material.

## DONE
- Scoped delta committed under step completion rule, dev == origin/dev verified.
