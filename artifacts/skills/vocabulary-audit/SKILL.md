---
name: vocabulary-audit
description: Audit JaxFNE prose against the canonical vocabulary table.
metadata:
  audience: agents
---
# JaxFNE vocabulary audit

## WHEN
Reviewing or editing JaxFNE public docs, README, skills, or agent context for terminology consistency.

## AUTHORITIES
1. `artifacts/vocabulary/JAXFNE_VOCABULARY.md` — project source, not a procedure.
2. `artifacts/subagents/vocabulary_critic.md` — review procedure for large passes.

## RULES
- Replace only if the canonical term preserves meaning.
- Preserve API identifiers, equations, citations, historical text, and technically precise controlled words.
- Report `UNCERTAIN` cases; never blind global find-and-replace.

## STEPS
1. Read `JAXFNE_VOCABULARY.md`.
2. For each relevant noncanonical occurrence: classify meaning (see critic labels).
3. Run `python scripts/audit_vocabulary.py --check` on doc changes.
4. For large passes, delegate file sets via `vocabulary_critic.md`.

## STOP
- Semantic conflict between canonical term and required technical precision.
- Replacement would change scientific meaning.

## VERIFY
- `python scripts/audit_vocabulary.py --check` passes for touched prose surfaces.

## DONE
- Terminology changes recorded with classification notes when non-obvious.
