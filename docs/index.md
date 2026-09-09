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

The Canonical Atlas links six panels (`network_3d`, `connectivity`, `raster`, `traces`, `spectral`, `state_summary`) with **OBSERVED** vs. **DERIVED** labeling and manifest provenance.

Every preview below is generated directly from realized JaxFNE simulation outputs (`canonical-v1-column-1000n` scaffold):

<table>
  <tr>
    <td align="center" width="50%">
      <a href="guides/atlas_suite.md">
        <img src="assets/readme/network_3d.png" alt="3D Realized Architecture" width="100%">
      </a><br>
      <sub><b>1. Network 3D</b> (OBSERVED): Realized 3D geometry, lamina, cell classes, sampled synaptic edges</sub>
    </td>
    <td align="center" width="50%">
      <a href="guides/atlas_suite.md">
        <img src="assets/readme/connectivity.png" alt="Realized Connectivity" width="100%">
      </a><br>
      <sub><b>2. Connectivity</b> (OBSERVED): Realized synaptic weight matrix & adjacency</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="guides/atlas_suite.md">
        <img src="assets/readme/raster.png" alt="Spike Raster" width="100%">
      </a><br>
      <sub><b>3. Spike Raster</b> (OBSERVED): Microsecond spike timestamps across realized neuronal populations</sub>
    </td>
    <td align="center" width="50%">
      <a href="guides/atlas_suite.md">
        <img src="assets/readme/traces.png" alt="Membrane Traces" width="100%">
      </a><br>
      <sub><b>4. Membrane Traces</b> (OBSERVED): Somatic membrane potential $V_m$ trajectories</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="guides/atlas_suite.md">
        <img src="assets/readme/spectral.png" alt="Spectral Dynamics" width="100%">
      </a><br>
      <sub><b>5. Spectral</b> (DERIVED): Welch PSD & spectrolaminar power estimates</sub>
    </td>
    <td align="center" width="50%">
      <a href="guides/atlas_suite.md">
        <img src="assets/readme/state_summary.png" alt="State Summary" width="100%">
      </a><br>
      <sub><b>6. State Summary</b> (DERIVED): Cell-type rate distributions and silence fractions</sub>
    </td>
  </tr>
</table>

Generate the complete standalone HTML atlas suite locally with one line:

```python
import jaxfne as jtfne
from jaxfne.vis import build_atlas

# Run simulation
tensor = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=500.0, dt_ms=0.5))
signals = jtfne.simulate(model)

# Build canonical atlas with manifest & standalone interactive HTML panels
manifest = build_atlas(model, signals, out_dir="docs/_static/atlas")
print(f"Atlas generated with SHA256: {manifest['sha256']}")
```

For detailed documentation, see the [Canonical Atlas Suite Guide](guides/atlas_suite.md).
