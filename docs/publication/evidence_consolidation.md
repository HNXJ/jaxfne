# Publication Evidence Consolidation (PEC)

**Status:** FROZEN @ PEC checkpoint  
**Authority:** `artifacts/publication/publication_evidence_index.json`  
**Spec:** `artifacts/publication/pec_consolidation_spec.json`  
**Receipt:** `artifacts/publication/pec_consolidation_receipt.json`

## Purpose

Immutable claim → protocol → receipt → arrays → analysis → figure-panel index for the **0.4.17 hard feature freeze**. Figures 1–7 must be generated from this index, not from memory or ad hoc scripts.

## Evidence ladder (claim level)

```text
DEMONSTRATED > MECHANISTICALLY_SUPPORTED > REPRESENTATIONAL > PROSPECTIVE
```

## Polarity axis (orthogonal)

```text
POSITIVE | NEGATIVE | UNRESOLVED
```

Examples: H4 = `DEMONSTRATED + NEGATIVE`; C3 = `DEMONSTRATED + NEGATIVE`; D3 = `DEMONSTRATED + NEGATIVE`; W3b = `DEMONSTRATED + UNRESOLVED`; E5 = `DEMONSTRATED + POSITIVE`.

## Development ledger (closed)

| ID | Scope |
|----|-------|
| A | Publication capability audit |
| B | Single-source multiscale observation (Experiment A) |
| C | Wave estimator + prospective NO_WAVE |
| D | Biological RBS containment + NO_ADAPTATION |
| E1–E5 | Hierarchy → delays → local RBD → observation → causal perturbation |

## Publication sequence (authorized next steps)

1. Evidence consolidation (**complete @ PEC**)
2. Figure 1 grammar map (**complete @ fig01_generation_receipt**)
3. Figures 2–4 (Experiment A) — **next authorized checkpoint**
4. Figure 5 (Protocol C)
5. Figure 6 (H / W / D)
6. Figure 7 (E1–E5)
7. Main-text reconstruction
8. Supplement

### Figure 1 authority

| Artifact | Path |
|----------|------|
| Spec | `artifacts/publication/fig01_grammar_spec.json` |
| Generator | `scripts/publication_figures/fig01_grammar.py` |
| Output | `figures/publication/fig01_tfne_grammar.png` |
| Semantic audit | `artifacts/publication/fig01_semantic_audit.json` |
| Receipt | `artifacts/publication/fig01_generation_receipt.json` |

Provenance: `artifact_introduction_commit` = PEC introduction (`c6d4c89`); verified checkpoint advances on `dev` without rewriting frozen PEC receipt.

## Feature freeze invariant

No E6, D4, C follow-up, W3c on the publication critical path, or new biological mechanism because a figure looks sparse. New science → post-0.4.17 backlog.

## Private manuscript organization

Panel prose and aesthetics remain in gitignored `scratch/` (e.g. `figure_requirements_matrix_v2_0417.md`). This repository carries **reproducibility index only**.

## Provenance

- `execution_parent_sha` — HEAD at simulate/execution time (legacy receipts may label this `package_head`)
- `artifact_commit_sha` — commit containing frozen index/receipt artifacts

Do not retroactively rewrite write-once E3/E4/E5 receipts.
