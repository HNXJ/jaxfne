# Welcome to jaxfne

**jaxfne** is a JAX-native source-to-field/readout layer for building multimodal tensor-field workflows from emitter or network model outputs.

Given model outputs from your own emitters, [Jaxley](https://github.com/google/jaxley)-style models, or other JAX workflows, jaxfne organizes them into:

- **Source tensors** — spatially localized currents or emitter states
- **Field operators** — extracellular potential and current-source density proxies
- **Probe readouts** — multimodal outputs: spikes, voltage, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy
- **Output bundles** — JSON-safe manifests with validation metadata

## Key features

- **JAX-native** — vmap, jit, scan-based kernels; CPU-safe by default
- **Multimodal readouts** — Eight probe operators (SPK, Vm, source, LFP-proxy, CSD-proxy, EEG-proxy, MEG-proxy, EMM-proxy)
- **Tensor-field workflows** — Source-to-field organization for local and global interaction summaries
- **Calibration-ready** — Design supports future empirical calibration and validation workflows
- **Reproducible** — JSON-safe output bundles with metadata and receipt tracking
- **CPU-first examples** — Validation on CPU; optional accelerator execution through JAX

## Quick example

```python
import jaxfne as jtfne

# Configure a 100-neuron network with multimodal probes
cfg = (
    jtfne.configuration()
    .network(n=100)
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy")
    .probe(name="probe", n_contacts=16)
)

# Build and simulate
model = jtfne.construct(cfg)
signals = model.simulate(jtfne.simulation(duration_ms=100.0))

# Compute readouts
readouts = model.compute_readout(signals, [
    jtfne.readout_spec("rate", "spike_rate_hz"),
    jtfne.readout_spec("lfp", "lfp_abs_mean"),
])

# Inspect results
for result in readouts:
    print(f"{result.name}: {result.value} [{result.status}]")
```

## Getting started

**New to jaxfne?** Start here:

- **[Installation](install.md)** — Install from PyPI or GitHub
- **[Quickstart](quickstart.md)** — Minimal working example with explanations
- **[FAQ](faq.md)** — Common questions and troubleshooting

## Explore

**Learn by doing:**

- **[Tutorials](tutorials/index.md)** — Six progressively detailed tutorials from single neurons to whole cortical circuits
- **[Guides](guides/index.md)** — How-to articles for specific workflows
  - [Probe operators](probe_operators.md) — Using the eight readout channels
  - [Tensor-field workflows](tensor_field_workflows.md) — Organizing sources and fields
  - [Jaxley interoperability](jaxley_interop.md) — Working with Jaxley-style models
  - [Calibration](calibration.md) — Preparing workflows for empirical validation
  - [Output bundles](output_bundles.md) — Understanding JSON manifests and metadata

**API reference:**

- [Core API](api/core.md) — Configuration, model, simulation, and readout specs
- [Emitters](api/emitters.md) — Izhikevich and other neuron models
- [Fields](api/fields.md) — Source projection and field operators
- [Probes](api/probes.md) — Eight probe readout operators
- [Objectives](api/objectives.md) — Objective and evaluation scaffolds
- [Validation](api/validation.md) — Invariant and claim-gate checks

## About jaxfne

- **[Scope and limitations](scope_and_limitations.md)** — What jaxfne is and is not designed for
- **[Changelog](changelog.md)** — Release history and updates
- **[Citation](citation.md)** — How to cite jaxfne in your work
- **[Contributing](contributing.md)** — Contributing code, examples, or feedback

## Installation

Install from PyPI:

```bash
pip install jaxfne
```

Or from source:

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e .
```

See [Installation](install.md) for more options.

## Feedback and contributions

Questions or suggestions? Open an issue or discussion on [GitHub](https://github.com/HNXJ/jaxfne).

We welcome tutorials, examples, and contributions to guides and API documentation.

## License

MIT License. See [LICENSE](https://github.com/HNXJ/jaxfne/blob/main/LICENSE) in the repository.
