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
1. Repository `artifacts/AGENTS.md` (branches, step completion rule).
2. Live code and executable bindings.

## RULES
- Pre-existing user changes are preserved.
- Patch the smallest semantic boundary; preserve compatibility and current declared invariants.
- Git remote mutation requires explicit authorization unless the active task authorizes it.

## STEPS
1. Execute Gate 0 (`scripts/harness/gate0_git_reality.py`).
2. Identify parameter/storage bindings in the selected backend.
3. Write cheap property/unit tests; attach assertions to shared minimal-circuit runs.
4. Run targeted tests and exact receipts.
5. Update affected docs and register in `mkdocs.yml` nav when public surfaces change.

## STOP
- Dirty tree ambiguity; unverified symbol; contradiction between authority and live code.

## VERIFY
- Targeted tests and exact receipts with output traces; delta report when material.

## DONE
- Scoped delta committed under step completion rule; `dev == origin/dev` verified when push is authorized.
