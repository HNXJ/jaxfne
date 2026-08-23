# Guides

**How-to articles** for building circuits, running simulations, reading out
field proxies, tuning objectives, and exporting manifests.

## Core grammar

- **[Configuration Grammar](configuration_grammar.md)** — `Configuration`, the primary chainable abstraction (and how `NeuronalTensor` converges on the same `Model` via `construct()`)
- **[Objective Grammar](objective_grammar.md)** — the user-facing run sequence: the literal chain of calls from setup to manifest

## Plasticity and homeostasis

- **[Homeostasis](homeostasis.md)** — the minimal computational excitability controller (single `k_gain` dial)
- **[RBS, RBD, and HDP](hdp.md)** — Relative Biophysical State, dynamics, and optional hidden-state dependent plasticity; population locality and continuation scope

## Showcases

- **Showcases (`docs/guides/showcases.md` — repository-internal reference, excluded from the built site)** — runnable demonstrations: interactive 3D multi-area network, homeostasis firing-rate change + full raster, closed-loop plasticity under random stimulation, spectrolaminar motif with depth-graded homeostasis

## Probe and readout workflows

- **[Probe operators](probe_operators.md)** — Using the eight readout channels
- **[Output bundles](output_bundles.md)** — Understanding JSON manifests and validation
- **Plotly Visualization (`docs/guides/plotly_visualization.md` — repository-internal reference, excluded from the built site)** — interactive `jaxfne.vis` plots and 3D network views

## Advanced workflows

- **[Operator Composition](operator_composition.md)** — composing field/probe/objective operators
- **[Tensor-field workflows](tensor_field_workflows.md)** — Pipeline overview and organization (source/field tensors — not to be confused with `NeuronalTensor`, the circuit-definition data model covered in Configuration Grammar above)
- **[Jaxley interoperability](jaxley_interop.md)** — Using Jaxley-style models with jaxfne
- **[Calibration](calibration.md)** — Preparing workflows for empirical validation
- **Poisson Admissibility (`docs/guides/poisson_admissibility.md` — repository-internal reference, excluded from the built site)** — admissibility conditions for the elliptic field equation specification

## Next steps

- **[API reference](../api/index.md)** for class/function documentation
- **[Tutorials](../tutorials/index.md)** for progressively detailed examples
- **[Études](../etudes/index.md)** for frozen scientific demonstrations
