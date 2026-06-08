# jaxfne Context Files Review Scorecard

Scope: active agent/context markdowns from `jaxfne_md_context_files.zip`, not the full public documentation tree. Public docs and historical release notes were not treated as agent context.

## Scoring rubric

| Score band | Meaning |
|---|---|
| 90-100 | Highly usable, low ambiguity, strong triggers, current API/roadmap alignment. |
| 80-89 | Useful, but needs sharper triggers, less overlap, or current-version cleanup. |
| 70-79 | Partly useful, but stale, too long, or too narrow/passive. |
| 60-69 | Risky as active context; should be archived or merged. |
| <60 | Remove from active context. |

## Scores

| File | Score | Keep? | Main issue | Update action |
|---|---:|---:|---|---|
| `AGENTS.md` | 78 | Yes | Good release gates, but stale release line and lacks Config/Net roadmap triggers. | Replace with concise agent operating contract. |
| `README.md` | 82 | Yes | Good user intro, but still uses old `Configuration/Model` vocabulary only. | Update to Config-first public story with aliases. |
| `internal_docs/agent_context/claude/CLAUDE.md` | 64 | Replace | Stale version discipline, v0.3.5-era contracts, too much old tutorial language. | Replace with current Claude/Gemini shared context. |
| `internal_docs/source_doctrine/README.md` | 84 | Yes | Good index, but does not mention 0.3.28+ Config/Net/FlatNet decisions. | Update as doctrine map. |
| `internal_docs/source_doctrine/CLAUDE.md` | 86 | Yes | Strong compact contract, but lacks current Config/Net naming and agent trigger terms. | Update as active worker contract. |
| `internal_docs/source_doctrine/jaxfne-core-doctrine.md` | 87 | Yes | Strong truth gates, but old pipeline-only framing; needs five-object architecture. | Update with module ownership map. |
| `internal_docs/source_doctrine/jaxfne-equations-runtime-validation.md` | 88 | Yes | Strong equations/JAX rules; needs FlatNet/JIT/pmap runtime split. | Update with tensor shape and JIT contracts. |
| `internal_docs/source_doctrine/jaxfne-longterm-plan.md` | 83 | Yes | Good high-level plan, but too generic for 0.3.28-0.3.34 execution. | Update with release ladder and solver entry criteria. |
| `internal_docs/source_doctrine/jaxfne-tutorial-etude-atlas.md` | 80 | Yes | Useful gates, but stale Etude naming and unsupported knob warning. | Update to Tutorial/Etude atlas and current trainer scope. |
| `internal_docs/skills/README.md` | 72 | Merge | Many passive skills compete; lacks trigger frontmatter. | Replace with 3 super-skill registry. |
| `internal_docs/skills/skill_repo_orientation.md` | 76 | Merge | Useful commands, but overlaps with AGENTS and release guard. | Fold into notebook/release super-skill. |
| `internal_docs/skills/skill_tutorial_smoke.md` | 79 | Merge | Good validation content, but narrow and passive. | Fold into notebook/release super-skill. |
| `internal_docs/skills/skill_visual_outputs.md` | 81 | Merge | Good visual checks, but should be in notebook/release or visualization schema. | Fold into notebook/release super-skill. |
| `internal_docs/skills/skill_probe_reports.md` | 84 | Merge | Strong readout metadata, but duplicates truth gates and field metadata. | Fold into modeling/schema super-skill. |
| `internal_docs/skills/skill_field_solution_metadata.md` | 82 | Archive/Merge | Deep but too long for frequent triggering; mostly future solver. | Fold essential gates into modeling/schema; archive full version. |
| `internal_docs/skills/skill_physical_field_admissibility.md` | 76 | Archive | Accurate but future-solver-heavy; can over-trigger on current proxy work. | Archive until 0.4.x solver phase. |

## Recommended active context set

1. `AGENTS.md`
2. `README.md`
3. `internal_docs/agent_context/claude/CLAUDE.md`
4. `internal_docs/agent_context/gemini/GEMINI.md`
5. `internal_docs/source_doctrine/*`
6. Three super-skills:
   - `jaxfne_modeling_and_optimization_schema.md`
   - `jax_jit_pmap_performance_guard.md`
   - `jaxfne_notebook_release_gate.md`

## Archive or remove from active trigger pool

Archive these as historical references instead of active context:

```text
internal_docs/skills/skill_repo_orientation.md
internal_docs/skills/skill_tutorial_smoke.md
internal_docs/skills/skill_probe_reports.md
internal_docs/skills/skill_field_solution_metadata.md
internal_docs/skills/skill_physical_field_admissibility.md
internal_docs/skills/skill_visual_outputs.md
internal_docs/agent_context/claude/CLAUDE.md.old
```

The replacement files below preserve their usable content in fewer, more invokable documents.
