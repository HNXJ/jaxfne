# jaxfne

JaxFNE is a Python package for biophysical source-field modeling. It couples
neural activity and biophysical state with plasticity, geometry, and population-
and field-scale dynamics, and each of these can be changed within one model.

**Workflow:** change biology → change dynamics → simulate → measure

A model is declared once and executed by one compiler:

<div class="mermaid">
flowchart LR
  C["CircuitSpec<br/>Configuration or NeuronalTensor"] -->|construct| M["Model"] -->|simulate| S["Signals"]
</div>

and its science reads from emitters to a manifest:

<div class="mermaid">
flowchart LR
  E["Emitter<br/>neural dynamics"] --> Q["Source<br/>currents Q"] --> F["Field<br/>Φ(r, t)"] --> P["Probe<br/>LFP, CSD, EEG"]
  P --> O["Objective"] --> Op["Optimizer"] --> Mf["Manifest"]
</div>

- **State**: add or change biophysical state $H$.
- **Mechanism**: change dynamics, plasticity, connectivity, geometry, or model detail.
- **Observation**: measure spikes, population activity, or fields.
- **Reduction**: develop or reduce models with [JDNA](guides/jdna.md).

[Scope & status](scope_and_status.md) · [Quickstart](quickstart.md)

## Install

```bash
pip install -U jaxfne
pip install "jaxfne[viz]"
```

## Minimal example

```python
import jaxfne as jtfne

jtfne.enable_x64()
tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
```

> Canonical example uses a qualitative laminar scaffold — see [Scope & status](scope_and_status.md#biological-calibration-status-canonical-v1-column).

## Main pages

- [Quickstart](quickstart.md) — build paths, paradigms, H-state adaptation
- [Tutorials](tutorials/index.md) — worked examples
- [Canonical Atlas Suite](guides/atlas_suite.md) — 7-panel interactive atlas
- [Études](etudes/index.md) — demonstrated scientific propositions
- [API reference](api/index.md)
- [H-state / HDP guide](guides/hdp.md)

## Canonical Visualization Atlas

Seven linked panels (`schema`, `network_3d`, `raster`, `lfp`, `h_dynamics`, `hdp`, `oscillatory`) label each quantity **OBSERVED** or **DERIVED** and carry
manifest provenance. A panel whose declared inputs are missing renders an omission card. Representative preview from realized `canonical-v1-column-1000n` output:

<a href="guides/atlas_suite.md">
  <img src="assets/readme/network_3d.png" alt="Network 3D atlas panel" width="100%">
</a>

Full panel set, generation workflow, and provenance rules:
[Canonical Atlas Suite guide](guides/atlas_suite.md).

Canonical 1000-neuron column, pinned 1000 ms reference run:
([index](_static/atlas/index.html) ·
[schema](_static/atlas/schema.html) ·
[network_3d](_static/atlas/network_3d.html) ·
[raster](_static/atlas/raster.html) ·
[lfp](_static/atlas/lfp.html) ·
[h_dynamics](_static/atlas/h_dynamics.html) ·
[hdp](_static/atlas/hdp.html) ·
[oscillatory](_static/atlas/oscillatory.html)).

Twenty-area hierarchy (Atlas AT-10-N20), areas in hierarchy order with
inter-area edge counts: [interactive](_static/visuals/area_graph_n20.html).

<a href="_static/visuals/area_graph_n20.html">
  <img src="assets/visuals/area_graph_n20.png" alt="Twenty-area hierarchy graph" width="100%">
</a>

Three-area hierarchy (`V1–V4–PFC`, 100 neurons/area, bidirectional
feedforward/feedback): [Gallery 09](gallery.md#09-three-area-hierarchy-v1v4pfc)
with the full six-panel atlas
([index](_static/atlas_three_area/index.html) ·
[schema](_static/atlas_three_area/schema.html) ·
[network_3d](_static/atlas_three_area/network_3d.html) ·
[raster](_static/atlas_three_area/raster.html) ·
[lfp](_static/atlas_three_area/lfp.html) ·
[h_dynamics](_static/atlas_three_area/h_dynamics.html) ·
[hdp](_static/atlas_three_area/hdp.html) ·
[oscillatory](_static/atlas_three_area/oscillatory.html)).
