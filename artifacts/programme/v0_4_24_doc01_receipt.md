# 24-DOC-01 receipt — docs minimization (relocation + placement)

**Branch:** `dev`. No information deleted; 8 files relocated byte-identical.

## Inventory findings (verified, not assumed)
- Published nav was already tight (7 sections); all working files
  (drafts, reports, audits) were already `exclude_docs`-only. No
  published→excluded links (checked programmatically: 0).
- Glossaries complementary (`mathematical_glossary_flow` math-flow vs
  `reference/glossary_of_methods` methods; 2 shared headings only).
- RBS/HDP/proxy/continuation mentions outside canonical homes are
  one-line references, not competing definitions (spot-checked
  `guides/hdp.md`, `api/emitters.md`, `api/fields.md`).
- H/state protocol docs complementary (RBD vs HDP memory).
- `results_reconstruction/` drafts: parallel non-identical tree to
  `publication/manuscript/` (Aug 26 vs Aug 27). Publication evidence —
  DEFERRED, needs human confirm before archival.

## Executed (all mechanical, gate-verified)
- 5× `docs/STDP_*_REPORT.md` + `docs/HDP_REPORT.md` → 
  `artifacts/legacy/reports/` (v047 demand; MERGE-into-stdp.md plan
  superseded — merging qualified per-report verdicts would lose
  information; relocation keeps bytes; sibling links intact).
- `docs/subagent-pool.md` → `artifacts/subagent-pool.md` (active agent
  ops belong with `artifacts/AGENTS.md`, not user docs).
- `docs/v047_refactor_audit.md` → `artifacts/legacy/` (contract script
  ARCHIVE action).
- Referrers updated: CORTEX checklist (5), showcases (2), doctrine
  inventory backticks (3), `DOC_OVERRIDES` paths (7), `exclude_docs`
  pruned (8 entries). `for_ai_agents.md` (test-required) and versioned
  `releases/` history deliberately untouched.

## Preferred terms (canonical homes; references stay one-line)
RBS/HDP/RBD → `doctrine/rbs_rbd_hdp.md`; proxy vs calibrated →
`api/fields.md` + relative-quantity doctrine; continuation →
`api/runtime.md`; configured/realized/executed/effective → TFNE
containment doctrine; source/field/probe → tensor-field workflows guide;
relative/calibrated → relative-quantity doctrine; JDNA/receipt keep
exact meanings.

## Measures
docs/: 152 files / 27,860 lines → 144 / 26,924 (−936 moved, 0 deleted).
Nav entries unchanged (none moved were navigated). Vocabulary: pass
before/after, 0 findings.

## Gates
orphans PASS · vocabulary PASS · mkdocs `--strict` PASS · version/docs/
contract/router suites (64) PASS.
