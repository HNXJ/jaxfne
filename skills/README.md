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
