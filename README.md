<p align="center">
  <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/jaxfne-itxt.png" alt="jaxfne" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/v/jaxfne?color=brightgreen" alt="PyPI package"></a>
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/pyversions/jaxfne" alt="Python versions"></a>
  <a href="https://jaxfne.readthedocs.io/en/latest/"><img src="https://readthedocs.org/projects/jaxfne/badge/?version=latest" alt="Documentation"></a>
  <a href="https://github.com/HNXJ/jaxfne/actions/workflows/ci.yml"><img src="https://github.com/HNXJ/jaxfne/actions/workflows/ci.yml/badge.svg?branch=dev" alt="CI (fast)"></a>
  <a href="https://github.com/HNXJ/jaxfne/actions/workflows/release_ci.yml"><img src="https://github.com/HNXJ/jaxfne/actions/workflows/release_ci.yml/badge.svg?branch=main" alt="Release CI"></a>
  <a href="https://github.com/HNXJ/jaxfne/actions/workflows/ci.yml"><img src="https://img.shields.io/badge/coverage-reported%20in%20CI-blue" alt="Coverage"></a>
  <a href="docs/contributing.md"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg" alt="Contributions welcome"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
</p>

# jaxfne

**Simulate laminar cortical circuits in JAX** — from point-neuron populations to
depth-resolved LFP/CSD/EEG proxy readouts, spectrolaminar summaries, and
population-level optimization. Define a column (or multi-area hierarchy), run it,
and inspect rasters, layer-targeted drive, and field-proxy traces in one pipeline.

## Install

```bash
pip install jaxfne
pip install "jaxfne[viz]"   # matplotlib/plotly readouts
```

Development checkout: `pip install -e ".[dev,viz]"` after cloning.

## Minimal example

```python
import jaxfne as jtfne

jtfne.enable_x64()
tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)

jtfne.vis.raster(signals)                    # population raster
jtfne.vis.spectrolaminar_suite(signals)      # laminar PSD readout
```

Canonical import: `import jaxfne as jtfne`. More paths (fluent `Configuration`,
multi-trial sweeps, HDP plasticity, Jaxley bridge): **[Quickstart](docs/quickstart.md)**.

## Scope & status

jaxfne is a **computational scaffold**, not a calibrated physical solver.
Field and probe outputs are **proxy readouts** — useful for method development,
not calibrated biophysical recordings. Full gate table:
[Scope & status](docs/scope_and_status.md).

Exported but not yet implemented: `GLIFEmitter`, `LIFEmitter`, `write_nwb`, `read_nwb`.

## Documentation

| Resource | Link |
|----------|------|
| Quickstart (three build paths, canonical column, Jaxley) | [docs/quickstart.md](docs/quickstart.md) |
| Full docs site | [jaxfne.readthedocs.io](https://jaxfne.readthedocs.io/) |
| Tutorials & études | [docs/tutorials/](docs/tutorials/) |
| Changelog | [docs/changelog.md](docs/changelog.md) |
| Contributing | [docs/contributing.md](docs/contributing.md) |

## Built for AI agents too

jaxfne ships **`skills/`** and **`AGENTS.md`** as first-class agent documentation —
verified against the same package source as the human docs, not a parallel spec.
See **[Documentation for AI agents](docs/for_ai_agents.md)** and the [`skills/`](skills/) index.

## Citation

Machine-readable metadata: [`CITATION.cff`](CITATION.cff) (GitHub **Cite this repository**).
BibTeX and Zenodo DOI setup: [docs/citation.md](docs/citation.md).
