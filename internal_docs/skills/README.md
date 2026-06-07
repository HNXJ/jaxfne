# jaxfne Skills Registry (Cursor)

Portable skill routing for jaxfne. This repo keeps **three super-skill documents** (repo-relative symlinks to `.legacy/internal_docs/skills/`) plus **thin adapters** under `adapters/`. Full SKILL.md bodies are not copied here.

## Super-skills (tracked, repo-local)

| File | Skill name | Use when |
|---|---|---|
| `jaxfne_modeling_and_optimization_schema.md` | jaxfne-modeling-optimization-schema | Config, Net, connectivity, weld, Paradigm, Objective, Trainer, AGSDR |
| `jax_jit_pmap_performance_guard.md` | jax-jit-pmap-performance-guard | JAX, JIT, vmap, pmap, scan, FlatNet, GPU |
| `jaxfne_notebook_release_gate.md` | jaxfne-notebook-release-gate | notebooks, tutorials, release, manifests, figures |

## Thin adapters (`adapters/`)

Minimal triggers + gates for companion skills. Canonical full text: user-installed skill of the same name.

**Tier 1:** `jaxfne-worker-context-router`, `jax-neuro-diffsim-guard`, `neuro-biophysics-units-sanity`, `jaxfne-release-mutation-guard`

**Tier 2:** `jaxfne-api-truth`, `jaxfne-code-orientation`, `jaxfne-test-runner`, `jaxfne-repo-audit`, `jaxfne-evidence-validator`, `jaxfne-style-conformance`, `jaxfne-tutorial-executor`, `jaxfne-theta-tutorial-validator`, `jaxfne-worker-handoff`, `jaxfne-jax-lint`, `jaxfne-visualization-schema`, `jaxfne-sha256-artifact-integrity`

## Canonical skill install (outside repo)

Skills are maintained in the developer's user-level agent skill directories. Typical layout:

```text
<user-skill-root>/<skill-name>/SKILL.md
```

Common roots: Claude Code skills directory, Cursor agent skills directory. Use skill **names** from this registry — do not hardcode machine paths in tracked repo files.

## Local Cursor wiring (optional, gitignored)

`.cursor/skills/` and `.cursor/local/` are gitignored. Developers may place local symlinks or copies there; nothing under those paths is committed.

## Cursor rule entry point

`.cursor/rules/jaxfne-super-skills.mdc`

## Doctrine

- `import jaxfne as jtfne`
- Package-native engine (no tutorial-local scientific engines)
- Proxy readout language
- Lazy optional dependencies
- No calibrated EEG/MEG/PDE claims without evidence

See `internal_docs/loop_context/04_TRUTH_GATES_AND_CLAIMS.md`.
