# jaxfne Documentation, Tutorials, Etudes, and Suites

## 1. Role

Documentation is part of the executable evidence system. A tutorial teaches a supported path; an Etude demonstrates a full scientific workflow with diagnostics and receipts; a Suite groups coherent release/publication scenarios.

## 2. Canonical teaching order

Teach concepts in increasing complexity:

```text
Emitter
-> Source
-> Field/Probe proxy
-> Objective
-> Optimization
-> structured circuit specification
-> HDP/adaptation
-> experimental physical solver only after proxy semantics are clear
```

For circuit construction, present both supported tiers explicitly:

```text
Configuration -> construct -> Model
NeuronalTensor -> construct -> Model
```

Do not mix builders mid-example unless the tutorial is specifically about interoperability/migration.

## 3. Tutorial execution contract

Release-facing full tutorials should use:

```yaml
duration_ms: ">=1000 where scientifically appropriate for the full tutorial"
dt_ms: 0.1 unless the documented model requires another validated value
dtype: float32 by default
seed: deterministic
canonical_import: "import jaxfne as jtfne"
package_native_scientific_path: true
finite_outputs: true
strict_json: true
png_figures: required
interactive_html: optional
proxy_safe_labels: required
```

Do not make `dt_ms=0.1` a universal mathematical law: alternative steps are allowed when explicitly justified and validated for the selected kernel.

## 4. Etude structure

A publication-grade Etude should contain:

```text
1. question/hypothesis
2. equations and operator glossary
3. complete configuration
4. circuit/tensor construction
5. sanity checks
6. simulation
7. source/readout construction
8. quantitative metrics
9. nulls/ablations
10. objective/optimization if applicable
11. figures
12. interpretation with claim gates
13. failure modes
14. manifest + validation + hashes
```

## 5. Notebook boundary

Allowed notebook-local logic:

```text
paths
panel layout
plot display
small formatting helpers
narrative/explanation
artifact export orchestration
```

Package responsibility:

```text
simulation kernels
source operators
field/probe operators
PSD/spectral computation
objective metrics
optimizers
validation
JSON-safe scientific reports
shared visualization transforms
```

If reusable scientific logic appears in a second notebook, move it into the package and preserve compatibility where needed.

## 6. Evidence artifacts

A full release-facing workflow should emit as applicable:

```text
manifest.json
validation_report.json
metrics.json
asset_hashes.json
figures/*.png
optional interactive/*.html
execution receipt
```

The manifest should include version/SHA, runtime, seeds, circuit specification reference/hash, operator/readout statuses, truth gates, objective settings, and artifact paths.

## 7. Required tutorial families for the method paper

Do not expand breadth indefinitely. Publication closure can be achieved with four canonical evidence families:

### T1 — Minimal EI operator closure

Demonstrate deterministic neural dynamics -> source -> proxy field -> probe -> objective. Include dense/edge or equivalent implementation checks where relevant.

### T2 — Structured laminar NeuralTensor

Demonstrate Areas/Layers/NeuronTypes, geometry/connectivity, laminar readouts, and layer/cell-type accounting. Include structure-shuffle controls.

### T3 — HDP adaptation and omission

Demonstrate H/weight trajectories, perturbation, recovery, omission/oddball conditions, HDP-off null, parameter/time-scale sensitivity, repeated seeds, and full-state continuation where segmented paradigms are used.

### T4 — Objective/optimization recovery

Use a synthetic target with known generating parameters or a controlled target regime. Show that the objective and optimizer recover or approach the known solution under bounded search.

These four families provide a coherent methods narrative. Additional tutorials are useful only if they cover a distinct supported capability.

## 8. Publication figure receipts

Every paper figure must be generated from a frozen SHA and have:

```text
script/notebook
configuration/manifest
input hashes
seed set
analysis parameters
PNG hash
reported numeric table
```

Numbers in the manuscript must be generated from the same result tables as the figures.

## 9. Documentation coverage rule

Do not require every function docstring to embed a documentation URL. Instead:

- every stable public root/module API is indexed/documented;
- every tutorial links to its conceptual/API documentation;
- every changed public scientific behavior updates its linked docs;
- API coverage is audited mechanically where possible.

## 10. Tutorial acceptance

A tutorial is release-ready when:

- it executes from a clean supported environment;
- no local scientific engine duplicates package APIs;
- arrays are finite and shapes checked;
- random state is explicit;
- JSON is strict;
- required PNGs exist;
- readout labels respect claim gates;
- quantitative claims are computed, not visually inferred;
- nulls/controls exist for causal/mechanistic-facing claims;
- execution receipt records exact command, SHA, Python/platform and result.
