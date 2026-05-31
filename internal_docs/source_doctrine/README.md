# JAXFNE Source Doctrine

This folder is the compact source doctrine for `jaxfne` / TFNE work. It replaces the longer scattered planning bundle with six durable files.

Default posture:

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Canonical pipeline:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

## Files

| File | Purpose |
|---|---|
| `README.md` | Bundle index and active doctrine map. |
| `CLAUDE.md` | Worker/assistant operating contract. |
| `jaxfne-core-doctrine.md` | Package identity, public API rules, truth gates, source/probe/report doctrine. |
| `jaxfne-equations-runtime-validation.md` | Equations, JAX/JIT rules, optimizer status, schemas, and validation gates. |
| `jaxfne-tutorial-etude-atlas.md` | Suite/Etude rules, notebook gates, Etude No. 1 standard, docs/artifact requirements. |
| `jaxfne-longterm-plan.md` | v0.3.x/v0.4.x/v0.5+ direction, solver ladder, refactor plan. |

## Immediate rules

- Canonical import: `import jaxfne as jtfne`.
- Tutorials and Etudes use package APIs as the engine.
- Reusable scientific logic belongs in `jaxfne`, not notebooks.
- Notebook plotting glue is allowed; simulator/readout/objective/solver logic is package code.
- Release-facing notebooks export finite, JSON-safe reports and stable PNG figures.
- Plotly HTML may augment figures; PNG remains required.
- Final audits use public raw GitHub URLs and immutable SHA URLs when possible.