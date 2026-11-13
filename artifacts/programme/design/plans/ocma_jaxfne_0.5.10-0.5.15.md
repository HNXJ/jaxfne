# JaxFNE x OCMA: unified programme through 0.5.15

Status: planning proposal, 2026-11-12. Owner-authorized planning scope; **not** authorization to change sealed TFNE/2, signed preregistrations, frozen evidence, release gates, or publication status. Source baseline: HNXJ/jaxfne dev 342279db (2026-11-10), HNXJ/ocma dev `cases/jaxfne/{README,PLAN,glossary}.md`. Refresh at execution. This document is the single long-term programme home; mutable execution remains in `artifacts/todo_stack.md`.

## Goal and boundaries

Deliver a simpler, more reliable, scientifically defensible JaxFNE: correct semantics, reproducible execution, quantified performance, observation-specific model reduction, explicit negative results, and evidence-backed paper claims. Apply OCMA as an independent **audit and prioritization** process, not an oracle for correctness or a license to generate new features. Optimize verified defect removal and information gain per cost, not counts of commits, tests, references, or LOC.

Existing authority remains: 0.5.10 release first; 0.5.11 arXiv submission; 0.6.0 post-review acceptance, subject to existing human release/signature gates. 0.5.12–0.5.15 are **proposed pre-0.6.0 development milestones**; do not delay an otherwise authorized 0.6.0 acceptance solely to satisfy this roadmap. Keep the paper plan's 2026-10-28 removal of the reference cap; its older "at most 80" heading is stale. Do not rewrite G-0 without Hamm's decision.

## OCMA evidence method

OCMA-J24.1 case: 117 reference rows inventoried, 88 in scope, 29 excluded; 24 factors in three groups: H1 mechanism in core, H2 reproduction evidence, H3 dashboard readiness. Scales distinguish verified (+1), partial (+0.6), indirect (+0.2), searched absent (0), disagreement (-0.6), refusal/failed criterion (-1), and not-applicable (null). A nonzero score requires existing path:line evidence. Three independent raters, mechanical validation, shuffled-paper baseline, adjudication, and sensitivity rerating. Independent round-1 alpha H1/H2/H3 = 0.844/0.880/0.815 versus shuffled p95 = 0.283/0.106/0.284; feedback alpha 0.939/0.971/0.941 is **not independent**. Initial consensus: 16 existing dashboard rows, one existing core-check (Jansen-Rit), one to build (Geweke), 68 study-list. Round-2 sensitivity changed action on 2/25 rows; second model family not run. OCMA agreement establishes rating consistency, **not** biological truth, implementation parity, or replication. The rater tooling must remain outside the rated tree.

Per target: freeze question/estimand, units, source version, positive/negative/null controls, signed protocol, thresholds, independent reference/oracle, seeds/backend/dtype, raw outputs, score JSON, uncertainty, selection history, and criterion-level verdict. Distinguish paper verdict, core-path verdict, and numerical-method check. Keep negative and failed results; never promote integration PASS into paper replication. No target-number fitting after confirmatory signature. Prefer cheap mechanistic core-checks before full-paper reproductions.

## Release milestones and closing gates

| Milestone | Work | Closing evidence |
|---|---|---|
| **0.5.10 — finish existing programme** | PCL R5-approximate and first-divergence attribution P-055; B1 HH/OU unit and protocol reconciliation before scored CPU/A4000/Brian2/NEST/Jaxley comparisons; F2/F4, S3/S4/S6/S7/S16; CAMINOS PLoS panels, conformance, dashboard freshness. | Existing 0.5.10 checklist and signed protocols, no rewritten failures; exact current-SHA broad/RC gate and human release go. |
| **0.5.11 — submit and stabilize** | Existing F8/S15, C1/C2/C9–C12 arXiv plan; signed prereg and held-out loss test for P-052 surrogate fitness; order-independent seeded realization and permutation mutant; Lee 2025/Mazzoni 2015 stage-zero and signed P3 only if evidence supports. | Paper checklist 100/100 on its declared checks; clean-figure regeneration; independent review; submission by Hamm. Training claim remains unvalidated unless scored. |
| **0.5.12 — OCMA coverage** | Refresh OCMA inventory against current code; resolve 24-factor evidence citations; separate tool/rated tree; run independent second-model-family sensitivity; prioritize dashboard core-check and roadmap entries, including Geweke only if its estimand is appropriate. | Validated factor rows; alpha and shuffled null with uncertainty; independently adjudicated disagreements; dashboard provenance and criteria; explicit unknowns and exclusions. |
| **0.5.13 — correctness and semantics** | Audit specification-to-realization-to-execution consumers, RNG identity, state continuation, delayed/zero-delay updates, source/field units, gradients, estimators, numerical stability and CPU/GPU differences. Focus on known issues before speculative defects. | For each confirmed defect: minimal red case on old code, one causal repair, positive/negative/boundary and mutant controls, sibling checks, broad gate; no vacuous count-only passes. |
| **0.5.14 — thinning and slop removal** | Inventory duplicate pathways, stored-only parameters, dead flags, compatibility shims, duplicated defaults, unowned state, stale docs, redundant generators, and unnecessary dependency/compilation cost. Preserve stable public API or explicitly deprecate. | One mechanism/class per change; behavioral equivalence of supported paths; continuation/RNG/provenance equivalence where required; measured LOC/API/compile/RSS deltas and performance; no semantic regression. |
| **0.5.15 — integration and qualification** | Full Atlas observation-specific reduction matrix, source-field approximation validation, cross-backend performance, fresh-clone reproducibility, dashboard/OCMA evidence closure and independent review. | Predeclared observation errors/tolerances and failures, calibrated-vs-proxy labels, uncertainty, complete receipts; broad/RC and external review on exact candidate; no known in-scope P0/P1 without explicit disposition. |

0.6.0 remains the separate post-peer-review acceptance release with its original owner gates. No milestone promises that every paper will reproduce or that every field output is physically calibrated.

## Continuous defect and complexity protocol

1. **Mine facts**: inspect current branch/HEAD, state, open/goal/todo stacks, issues, current code, tests and receipts. OCMA ratings and agent findings are hypotheses; verify against consumers. Maintain one authoritative home per fact.
2. **Rank** by scientific claim severity, reproducible failure, blast radius, and cost. Prefer falsifying tests: nonneutral parameters, asymmetric geometry, sparse weights, negative/null/shuffle, cross-backend, restart and serialization.
3. **Repair**: smallest failing case; show red on prior behavior; fix the cause; test siblings and independent observation. Two unsuccessful fixes require a new hypothesis. Preserve unrelated edits and frozen records.
4. **Thin** only after semantics and usage are established. Remove duplicates/dead internals; classify public parameters as active, compatibility, ignored-by-design or defective. Avoid speculative refactors, metric gaming, and unmeasured performance claims.
5. **Audit** with OCMA where evidence classification is useful; second model family, shuffled null, and keeper adjudication. Re-score only changed evidence, keep signed history immutable. Dashboard must show criterion and paper verdicts separately.
6. **Seal** with exact SHA/config/seed, tests, resource measurements, receipt paths, and independent review. Git/CI success is not biological validation. Run broad gates before authorized push; human retains tag/main/release/submission approval.

## Priorities and stop conditions

Immediate path: PCL R5 approximate result and P-055 attribution -> B1 unit/protocol reconciliation -> scored benchmark -> current 0.5.10 gates. Parallel, within existing three-lane limit: CAMINOS panel completion, conformance/reproducibility audit, NeuroAI submission preparations (owner submits). No additional writer lane without capacity.

Stop affected work on: conflict with sealed TFNE/2 or frozen evidence; unsigned confirmatory protocol; missing oracle/units/estimand; unknown license; stale SHA; unexplained numerical discrepancy; proposed public-API or release-gate change; unapproved main/tag/Release/PyPI/submission. Ask Hamm one bounded decision, preserve other ready work.

## Acceptance and measurement

Track per release: confirmed defects opened/closed (severity and reproducer), failed controls caught, coverage by *behavioral* gate rather than test count, unresolved OCMA factors, signed criterion verdicts, false-positive reviewer findings, reproducible resource benchmarks, API/dependency/LOC deltas, fresh-clone CI, and exact claim-to-evidence coverage. No aggregate "scientific accuracy" score from heterogeneous papers. A null/negative finding is a result; missing data remains unknown.

## Execution routing

- Current decisions: `artifacts/open_stack.md`; owner goals: `artifacts/goal_stack.md`; remaining executable tasks: `artifacts/todo_stack.md`; open defects: `artifacts/assurance/issue_log/ISSUE_LOG.md`.
- Current release gates: `artifacts/etudes/meta/plans/release_plan_0.5.9-0.6.0.md`; paper: `artifacts/programme/design/plans/arxiv_paper_plan.md`.
- OCMA case method, ontology, outputs: HNXJ/ocma `cases/jaxfne/`; JaxFNE dashboard-facing result: `artifacts/etudes/meta/literature/ocma/`.
- This note holds **long-term scope and gates**, not duplicated mutable task status. On each release transition, derive small actionable todo rows from this note; delete completed rows and retain historical evidence in Git/receipts.
