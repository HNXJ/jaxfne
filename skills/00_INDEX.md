# jaxfne skills bundle

This bundle turns repo-review findings into enforceable operating rules.

Use the files in this order:
1. `README.md`
2. `FRICTIONS_STACK.md` (open contradictions — check before ground-truth claims)
3. `01_repo_orientation.md`
4. `10_objective_grammar.md`
5. `11_catalog_glossary.md`
6. `02_analysis_integrity.md`
7. `03_sparse_connectivity.md`
8. `04_batch_first_simulation.md`
9. `05_projection_semantics.md`
10. `06_runtime_fallback_transparency.md`
11. `07_api_contracts.md`
12. `08_parameter_semantics.md`
13. `09_experimental_fence.md`
14. `PATCH.md`
15. `ANTIGRAVITY_PROMPT.md`

First-class `*/SKILL.md` folders (full guidance): see `README.md`.

Bundle rules:
- flat markdown + `*/SKILL.md` folders under `skills/` only
- no executable skill scripts (legacy extensionless `.py` skills removed 2026-06-27)
- no invented APIs
- preserve proxy/scaffold wording unless code and tests prove a stronger claim
- prefer sparse, vectorized, JAX-native paths over dense Python loops
