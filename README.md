<p align="center">
  <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/jaxfne-itxt.png" alt="jaxfne" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/v/jaxfne?color=brightgreen" alt="PyPI"></a>
  <a href="https://jaxfne.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/jaxfne/badge/?version=latest" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

# jaxfne

**Tensor-Field Neural Equations (TFNE)** expressed as a JAX simulation engine for
layer-resolved neural circuits, source operators, field proxies, probes, objectives,
and evidence receipts.

**Scientific grammar:** `Emitter → Source → Field → Probe → Objective → Optimizer → Manifest`

**Execution grammar:** `CircuitSpec → construct → Model → simulate → Signals`

`CircuitSpec` is a **conceptual category** (`Configuration | NeuronalTensor`),
not a concrete public production class — `construct` accepts either a
`Configuration` (original path) or a `NeuronalTensor` with a
`RuntimeConfiguration` (tensor-first path). It is unrelated to the
experimental `jaxfne.experimental_hpc.CircuitSpec` type, which is
not accepted by production `construct`.

**Adaptation** (optional HDP family): finite-dimensional hidden biophysical state
$H$ and adaptive parameter coordinates $\Theta$ (synaptic and intrinsic),
mediated by

$$
\dot X = F_X(X,H,\Theta,U),\quad
\dot H = F_H(H,X,\Theta,U),\quad
\dot\Theta = F_\Theta(H,X,\Theta).
$$

RBS represents $H$. RBD defines $H$ dynamics. HDP defines parameter dynamics
and may depend on $H$. $H$ can exist and evolve without HDP. See [H-state / HDP guide](docs/guides/hdp.md).

## Install

```bash
pip install jaxfne
pip install "jaxfne[viz]"   # optional plotting
```

Development: `pip install -e ".[dev,viz]"` after cloning.

## Minimal example

```python
import jaxfne as jtfne

jtfne.enable_x64()
tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
```

### Optional: generative development (JDNA)

Instead of loading a fixed tensor, generate one from a PseudoGenome:

```python
genome  = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor  = jtfne.develop(genome, seed=0)   # K_D: development seed
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
```

JDNA is an optional path; the direct `Configuration`/`NeuronalTensor` paths
remain first-class. See [JDNA guide](docs/guides/jdna.md).

JaxFNE composes established neural dynamical models, source/readout operators, and `JAX-native` execution under the TFNE typed state/map grammar. See [TFNE theory](docs/doctrine/tfne_containment_architecture.md) and [References](docs/reference/references.md).

Import convention: `import jaxfne as jtfne`. Builder paths, paradigms, and
optimization: [Quickstart](docs/quickstart.md).

## Relative and calibrated outputs

Simulated quantities are **relative** by default. **Calibrated** claims require an
explicit transformation with evidence. See [Scope & status](docs/scope_and_status.md).

## Documentation

| Resource | Link |
|----------|------|
| Quickstart | [docs/quickstart.md](docs/quickstart.md) |
| Site | [jaxfne.readthedocs.io](https://jaxfne.readthedocs.io/) |
| Tutorials | [docs/tutorials/](docs/tutorials/) |
| Études | [docs/etudes/](docs/etudes/) |
| Frozen compatibility contract (0.4.13) | [docs/public_surface_contract.md](docs/public_surface_contract.md) |
| Changelog | [docs/changelog.md](docs/changelog.md) |

## Citation

[`CITATION.cff`](CITATION.cff) · [citation guide](docs/citation.md)

## Canonical Visualization Atlas

The JaxFNE Canonical Atlas provides a unified 6-panel visual grammar (`network_3d`, `connectivity`, `raster`, `traces`, `spectral`, `operating_point`) with strict evidence-level separation (**OBSERVED** vs. **DERIVED**), deterministic degradation tracking, and cryptographic manifest provenance.

Every preview below is generated directly from realized JaxFNE simulation outputs (`canonical-v1-column-1000n` scaffold):

<table>
  <tr>
    <td align="center" width="50%">
      <a href="https://jaxfne.readthedocs.io/en/latest/">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/readme/network_3d.png" alt="3D Realized Architecture" width="100%">
      </a><br>
      <sub><b>1. Network 3D</b> (OBSERVED): Realized 3D geometry, lamina, cell classes, sampled synaptic edges</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://jaxfne.readthedocs.io/en/latest/">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/readme/connectivity.png" alt="Realized Connectivity" width="100%">
      </a><br>
      <sub><b>2. Connectivity</b> (OBSERVED): Realized synaptic weight matrix & adjacency</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://jaxfne.readthedocs.io/en/latest/">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/readme/raster.png" alt="Spike Raster" width="100%">
      </a><br>
      <sub><b>3. Spike Raster</b> (OBSERVED): Microsecond spike timestamps across realized neuronal populations</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://jaxfne.readthedocs.io/en/latest/">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/readme/traces.png" alt="Membrane Traces" width="100%">
      </a><br>
      <sub><b>4. Membrane Traces</b> (OBSERVED): Somatic membrane potential $V_m$ trajectories</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <a href="https://jaxfne.readthedocs.io/en/latest/">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/readme/spectral.png" alt="Spectral Dynamics" width="100%">
      </a><br>
      <sub><b>5. Spectral</b> (DERIVED): Welch PSD & spectrolaminar power estimates</sub>
    </td>
    <td align="center" width="50%">
      <a href="https://jaxfne.readthedocs.io/en/latest/">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/readme/operating_point.png" alt="Operating Point" width="100%">
      </a><br>
      <sub><b>6. Operating Point</b> (DERIVED): Cell-type rate distributions and silence fractions</sub>
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

For extended case studies, laminar field readouts, and homeostasis dynamics, see [Showcases](docs/guides/showcases.md).

