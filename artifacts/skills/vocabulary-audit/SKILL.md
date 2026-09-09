---
name: vocabulary-audit
description: Audit JaxFNE prose against the canonical vocabulary table.
metadata:
  audience: agents
---
# JaxFNE vocabulary audit

## WHEN
Reviewing or editing JaxFNE public docs, README, skills, or agent context for
terminology consistency.

## AUTHORITY
`artifacts/vocabulary/JAXFNE_VOCABULARY.md` — project source, not a procedure.

## PROCEDURE
1. Read `JAXFNE_VOCABULARY.md`.
2. For each relevant noncanonical occurrence: classify meaning (see critic labels).
3. Replace only if canonical term preserves meaning.
4. Preserve API identifiers, equations, citations, historical text, technically
   precise controlled words.
5. Report `UNCERTAIN` cases; never blind global find-and-replace.
6. Run `python scripts/audit_vocabulary.py --check` on doc changes.

## DELEGATION
For large passes, follow `artifacts/subagents/vocabulary_critic.md` and assign
non-overlapping file sets.

## STOP
- Semantic conflict between canonical term and required technical precision.
- Replacement would change scientific meaning.
