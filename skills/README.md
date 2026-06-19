# jaxfne skills

Versioned copies of the jaxfne-specific agent skills (the authoritative working
copies live in each contributor's `~/.claude/skills/`). They are committed here
so the guidance that governs jaxfne code travels with the repository.

| skill | role |
|-------|------|
| `catalog-glossary-jaxfne` | flat catalog of the public jaxfne API — check before writing any helper |
| `jaxfne-objective-grammar` | the mandatory fluent grammar: Configuration → Model → Signals → Probe → Objective → Optimizer → Manifest |
| `jaxfne-configuration-fluent-api` | Configuration chaining (`.runtime/.geometry/.population/.cell_types/.connectivity/.probes`) |
| `jaxfne-cortical-column-default` | canonical 1K-neuron laminar column template |
| `jaxfne-signals-probe-objective-chain` | Signals operators, probe extraction, objective composition |
| `jaxfne-modeling-optimization-schema` | Config/Net/Paradigm/Objective/Trainer/Signals dataclass truth-gate checks |
| `jaxfne-notebook-release-gate` | validate notebooks/tutorials/docs/artifacts before any "done" claim |
| `jaxfne-release-mutation-guard` | guard every remote mutation (push, tag, GitHub Release, PyPI) |
| `jaxfne-sha256-artifact-integrity` | SHA256 content identity for configs/notebooks/models/figures/wheels |
| `jaxfne-visualization-schema` | design/fix/audit `jaxfne.vis`; keep plotting package-level and proxy-safe |
| `jaxfne-worker-context-router` | route a jaxfne task to the right module/API/validation lane |
| `jaxfne-spectrolaminar-suite` | scalable spectrolaminar suite + Etude No.3 / TCM etudes; LFP-proxy/density/size/kappa caveats and the crossover-needs-oscillations result |

These are reference guidance, not importable package code. The shipped Python
package is `jaxfne/`.

## Repo-hardening enforcement skills (flat)

A second, flat set of enforcement skills lives at the root of this folder as
numbered markdown files (`00_INDEX.md`, `01_*.md` … `11_*.md`). They turn
repo-review findings into enforceable operating rules rather than descriptions.
Start at `00_INDEX.md`; `PATCH.md` and `ANTIGRAVITY_PROMPT.md` record the bundle's
provenance and apply-prompt.

| file | enforces |
|------|----------|
| `01_repo_orientation.md` | understand the repo/module map before editing |
| `02_analysis_integrity.md` | no synthetic fallback masking a real failure |
| `03_sparse_connectivity.md` | sparse/edge-list construction, no default dense O(N²) |
| `04_batch_first_simulation.md` | vectorized (vmap/scan) on the common repeated-run path |
| `05_projection_semantics.md` | projection normalization is explicit and testable |
| `06_runtime_fallback_transparency.md` | fallbacks report themselves; strict mode fails loudly |
| `07_api_contracts.md` | public helpers implemented, wrapped, or explicitly fenced |
| `08_parameter_semantics.md` | parameter scope (per-area/layer/column/global) is explicit |
| `09_experimental_fence.md` | incomplete bridges/solvers stay clearly experimental |
| `10_objective_grammar.md` | follow the Configuration → … → Manifest chain |
| `11_catalog_glossary.md` | verify a symbol before using it; never invent APIs |

These overlap, at a shorter "rule" altitude, with the first-class skills above
(`jaxfne-objective-grammar`, `catalog-glossary-jaxfne`, `jaxfne-worker-context-router`);
the flat set is the enforceable checklist, the folders are the full guidance.

Keep this set flat — do not recreate nested skill folders for it in the repo.
