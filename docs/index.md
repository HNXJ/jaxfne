# jaxfne

JaxFNE is a Python package for biophysical source-field modeling, coupling neural
activity and biophysical state with plasticity, geometry, and population- and
field-scale dynamics.

JaxFNE is designed for models that need to change easily. Biology, dynamics,
connectivity, geometry, and observations can be modified within the same model.

**Workflow:** change biology → change dynamics → simulate → measure

You can, for example:

- add or change biophysical state $H$;
- change dynamics, plasticity, connectivity, or geometry;
- change model detail;
- measure spikes, population activity, or fields;
- develop or reduce models with [JDNA](guides/jdna.md).

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
- [Canonical Atlas Suite](guides/atlas_suite.md) — 6-panel interactive atlas
- [Études](etudes/index.md) — demonstrated scientific propositions
- [API reference](api/index.md)
- [H-state / HDP guide](guides/hdp.md)

## Canonical Visualization Atlas

Six linked panels (`network_3d`, `connectivity`, `raster`, `traces`, `spectral`,
`state_summary`) separate **OBSERVED** from **DERIVED** quantities and attach manifest
provenance. Representative preview from realized `canonical-v1-column-1000n` output:

<a href="guides/atlas_suite.md">
  <img src="assets/readme/network_3d.png" alt="Network 3D atlas panel" width="100%">
</a>

Full panel set, generation workflow, and provenance rules:
[Canonical Atlas Suite guide](guides/atlas_suite.md).
