# jaxfne

JaxFNE is a Python package for biophysical source-field modeling, coupling neural
activity and biophysical state with plasticity, network geometry, and population-
and field-scale dynamics. Neural models can be defined at different levels of
biological detail and reduced when computational efficiency is required.

## What this enables

$$
\text{Model} = \mathrm{JaxFNE}(\text{specification}, \text{dynamics},
\text{biophysical state}, \text{plasticity}, \text{geometry})
$$

$$
\text{Signal} = \mathrm{Probe}(\text{source}, \text{modality}, \text{geometry})
$$

Detail can range from reduced emitters through compartmental models attached via
interoperability bridges, to population- and column-scale networks. Reduction is
an explicit modeling choice.

## Principal capabilities

- **Flexible biophysical state** — $H$/RBS/RBD adds ionic, energetic, synaptic,
  modulatory, or other declared coordinates without a new simulator architecture
  per extension.
- **Source to field to observation** — dynamics → sources → fields → probes for
  spikes, population signals, LFP-like proxies, and calibrated physical modalities
  where a forward model and calibration are defined. Default readouts are
  **relative computational proxies** — see [Scope & status](scope_and_status.md).
- **Models that can develop** — JDNA generative construction (`develop` →
  `NeuronalTensor`) is available; explicit evolution and structural development
  are **planned, not yet implemented** as runtime dynamics.

JaxFNE uses JAX as an efficient numerical substrate; the scientific contribution
is the biophysical source-field modeling framework, not JAX itself.

## TFNE (Tensor-Field Neural Equations)

TFNE represents neural dynamics, biophysical state, plasticity, geometry,
sources, and fields in a common mathematical form.

**Scientific pipeline:** Emitter → Source → Field → Probe → Objective → Optimizer → Manifest

**Software pipeline:** CircuitSpec → `construct` → `Model` → `simulate` → `Signals`

`CircuitSpec` names the conceptual input to `construct` (`Configuration` or
`NeuronalTensor`). It is not a separate public class and is unrelated to
`experimental_hpc.CircuitSpec`.

**Ecosystem.** [Jaxley](https://jaxley.readthedocs.io) focuses on compartmental
biophysical detail; JaxFNE focuses on coupling neural dynamics, geometry, and
field readouts in JAX. Compartmental Jaxley models can attach as emitters via
[Jaxley interoperability](guides/jaxley_interop.md). Other simulators address
different execution models and file formats; compare purpose, representation,
and interoperability rather than ranking.

[Scope & status](scope_and_status.md) · [Public API surface](public_surface_contract.md) (0.4.13)

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

> `canonical-v1-column-1000n` fractions (`L1 E 0.50` etc.) and typed motifs are scaffold values (`value_tag="relative"`), not quantitatively calibrated; `E`/`PV`/`SST`/`VIP` are reduced-emitter scaffold identities — see [Scope & status](scope_and_status.md#biological-calibration-status-canonical-v1-column). `qualitative_laminar_scaffold = true`, `quantitative_cell_fraction = false`, `quantitative_connectivity = false`.

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
