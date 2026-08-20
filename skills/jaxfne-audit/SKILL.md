---
name: jaxfne-audit
description: Independent scientific and technical audit for jaxfne.
metadata:
  audience: agents
---
# jaxfne audit procedure

## WHEN
Independent measurement of code, mathematics, claims, figures, or evidence.

## AUTHORITIES
1. Gate 0 Git reality: `scripts/harness/gate0_git_reality.py`.
2. Frozen evidence: `artifacts/publication/publication_evidence_index.json`.
3. Doctrine: `docs/doctrine/tfne_containment_architecture.md`.

## RULES
- Gate 0 before reporting any project truth.
- Review before Progress; never silently repair.
- Do not inherit executor scores or claims.
- Epistemic invariants: NEGATIVE != UNRESOLVED; relative != calibrated; H != homeostasis.
- Zero public release/tag/push actions during audit.

## STEPS
1. Run Gate 0 to verify repository root, remote, branch, and clean fetch state.
2. Query live code/receipts before forming a conclusion.
3. Formulate falsifiable findings: F = (claim, severity, evidence, expected, actual, reproduction).
4. Score out of 100 based on reproducible defects.

## STOP
- Unresolved workspace identity, STALE_LOCAL_STATE, or missing evidence.

## VERIFY
- Reproduce findings with exact shell/python receipts.

## DONE
- Audit report returned with explicit findings and verified score.
