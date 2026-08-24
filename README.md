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

$H$-state is the latent representation; HDP is the adaptive dynamical formulation
that uses it. See [H-state / HDP guide](docs/guides/hdp.md).

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

JaxFNE composes established neural dynamical models, source/readout operators, and JAX-native execution under the TFNE typed state/map grammar. See [TFNE theory](docs/doctrine/tfne_containment_architecture.md) and [References](docs/reference/references.md).

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

## Visual examples

Real package outputs (simulated proxy scaffolds; calibration requires an
explicit transformation with evidence):
circuit geometry from `Model.neuron_table()`, a 10 s spike raster, a
laminar spectrolaminar readout, and long-timescale firing-rate dynamics
with homeostasis on/off. More: [showcases](docs/guides/showcases.md).

<table>
  <tr>
    <td align="center">
      <a href="docs/guides/showcases.md">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/showcases/circuit_geometry_column.png" alt="Circuit geometry" width="340">
      </a><br>
      <sub>Canonical laminar column — geometry from `neuron_table()`</sub>
    </td>
    <td align="center">
      <a href="docs/guides/showcases.md">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/showcases/homeostasis_full_raster_10s.png" alt="Spike raster" width="340">
      </a><br>
      <sub>10 s spike raster — homeostasis on (proxy)</sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <a href="docs/guides/showcases.md">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/showcases/spectrolaminar_slow_homeostasis_suite.png" alt="Spectrolaminar readout" width="340">
      </a><br>
      <sub>Spectrolaminar proxy readout — depth-graded homeostasis</sub>
    </td>
    <td align="center">
      <a href="docs/guides/showcases.md">
        <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/showcases/homeostasis_rate_change_10s.png" alt="Long-timescale dynamics" width="340">
      </a><br>
      <sub>Population rate over 10 s — homeostasis off vs on (r*=10 Hz)</sub>
    </td>
  </tr>
</table>
