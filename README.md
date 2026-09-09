<p align="center">
  <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/jaxfne-itxt.png" alt="jaxfne" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/v/jaxfne?color=brightgreen" alt="PyPI"></a>
  <a href="https://jaxfne.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/jaxfne/badge/?version=latest" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
</p>

# jaxfne

JaxFNE is a Python package for biophysical source-field modeling, coupling neural
activity and biophysical state with plasticity, network geometry, and population-
and field-scale dynamics. Neural models can be defined at different levels of
biological detail and reduced when computational efficiency is required.

## What this enables

At the level of specification and readout:

$$
\text{Model} = \mathrm{JaxFNE}(\text{specification}, \text{dynamics},
\text{biophysical state}, \text{plasticity}, \text{geometry})
$$

$$
\text{Signal} = \mathrm{Probe}(\text{source}, \text{modality}, \text{geometry})
$$

Detail can range from reduced point-neuron emitters through compartmental models
attached via interoperability bridges, to population- and column-scale networks.
Reduction is an explicit modeling choice, not a hidden approximation.

## Principal capabilities

**Flexible biophysical state.** $H$ through RBS/RBD lets models add the ionic,
energetic, synaptic, modulatory, or other biophysical coordinates required by a
question without requiring a new simulator architecture for each extension.

**Source to field to observation.** Neural dynamics produce sources, fields, and
probes for spikes, population signals, LFP-like proxies, and calibrated physical
modalities (EEG/MEG-style projections and similar) where an appropriate forward
model and calibration are defined. By default field readouts are **relative
computational proxies**, not calibrated physical measurements — see
[Scope & status](docs/scope_and_status.md).

**Models that can develop.** [JDNA](docs/guides/jdna.md) describes generative
model construction from a `PseudoGenome` (`develop` → `NeuronalTensor` today).
Planned extensions toward explicit evolution, structural development, and model
reduction are **not yet implemented** as runtime developmental dynamics; see
[Scope & status](docs/scope_and_status.md) and the private roadmap programme.

## Implementation substrate

JaxFNE is implemented in JAX for efficient, composable numerical execution on CPU
and accelerator hardware. JAX is the numerical substrate; the scientific
contribution is the biophysical source-field model above.

## TFNE (Tensor-Field Neural Equations)

TFNE represents neural dynamics, biophysical state, plasticity, geometry,
sources, and fields in a common mathematical form. Individual emitters, source
maps, fields, and probes are replaceable implementations of typed roles within
that form.

**Scientific pipeline:** `Emitter → Source → Field → Probe → Objective → Optimizer → Manifest`

**Software pipeline:** `CircuitSpec → construct → Model → simulate → Signals`

`CircuitSpec` names the conceptual input to `construct` (`Configuration` or
`NeuronalTensor` with `RuntimeConfiguration`). It is not a separate public
production class and is unrelated to `jaxfne.experimental_hpc.CircuitSpec`.

**Adaptation** (optional HDP family): finite-dimensional hidden biophysical
state $H$ and adaptive parameter coordinates $\Theta$, with dynamics

$$
\dot X = F_X(X,H,\Theta,U),\quad
\dot H = F_H(H,X,\Theta,U),\quad
\dot\Theta = F_\Theta(H,X,\Theta).
$$

RBS represents $H$; RBD defines $H$ dynamics; HDP defines parameter dynamics.
See [H-state / HDP guide](docs/guides/hdp.md).

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

See [TFNE theory](docs/doctrine/tfne_containment_architecture.md) and [References](docs/reference/references.md) for the mathematical structure.

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
| Public API surface (0.4.13) | [docs/public_surface_contract.md](docs/public_surface_contract.md) |
| Changelog | [docs/changelog.md](docs/changelog.md) |

## For AI agents

Read and inspect [`artifacts/context.md`](https://github.com/HNXJ/jaxfne/blob/main/artifacts/context.md).

It is a router rather than a specification: it names the single public import, the two
invariant grammars, where authority actually lives, and the gate commands that settle a claim.
Reading and inspecting that one file is enough to orient a coding agent — one working in a
terminal, in a sandbox, or under any memory- or skill-based harness — without re-deriving the
public API from source or guessing at the internal layout.

The file is present in the repository and in the source distribution. It is not installed
alongside the wheel, so an agent working from an installed package should read it at the link
above.

## Citation

[`CITATION.cff`](CITATION.cff) · [citation guide](docs/citation.md)

## Canonical Visualization Atlas

Six linked panels (`network_3d`, `connectivity`, `raster`, `traces`, `spectral`, `state_summary`) separate **OBSERVED** from **DERIVED** quantities and attach
manifest provenance. Previews and generation code:
[documentation site](https://jaxfne.readthedocs.io/en/latest/) and
[Atlas guide](docs/guides/atlas_suite.md).

