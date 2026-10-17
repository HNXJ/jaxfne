<p align="center">
  <img src="https://raw.githubusercontent.com/HNXJ/jaxfne/main/docs/assets/jaxfne-itxt.png" alt="jaxfne" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/jaxfne/"><img src="https://img.shields.io/pypi/v/jaxfne?color=brightgreen" alt="PyPI"></a>
  <a href="https://jaxfne.readthedocs.io/en/stable/"><img src="https://readthedocs.org/projects/jaxfne/badge/?version=stable" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <a href="https://doi.org/10.5281/zenodo.21448155"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.21448155.svg" alt="DOI"></a>
</p>

# jaxfne

JaxFNE is a Python package for biophysical source-field modeling. It couples
neural activity and biophysical state with plasticity, geometry, and population-
and field-scale dynamics, and each of these can be changed within one model.

**Workflow:** change biology → change dynamics → simulate → measure

- **State**: add or change biophysical state $H$.
- **Mechanism**: change dynamics, plasticity, connectivity, geometry, or model detail.
- **Observation**: measure spikes, population activity, or fields.
- **Reduction**: develop or reduce models with [JDNA](docs/guides/grammar/jdna.md).

[Quickstart](docs/start/quickstart.md) · [Scope & status](docs/reference/project/scope_and_status.md)

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

Import convention: `import jaxfne as jtfne`.

## Documentation

| Resource | Link |
|----------|------|
| Quickstart | [docs/start/quickstart.md](docs/start/quickstart.md) |
| Site | [jaxfne.readthedocs.io](https://jaxfne.readthedocs.io/) |
| Tutorials | [docs/tutorials/](docs/tutorials/) |
| Études | [docs/etudes/](docs/etudes/) |
| Public API surface | [docs/reference/public_surface_contract.md](docs/reference/public_surface_contract.md) |
| Changelog | [docs/reference/changelog.md](docs/reference/changelog.md) |

If you are an AI agent, read [`artifacts/AGENTS.md`](artifacts/AGENTS.md).

## Citation

Concept DOI (all versions): [10.5281/zenodo.21448155](https://doi.org/10.5281/zenodo.21448155) ·
[`CITATION.cff`](CITATION.cff) · [citation guide](docs/reference/citation.md)

## Canonical Visualization Atlas

Seven linked panels (`schema`, `network_3d`, `raster`, `lfp`, `h_dynamics`, `hdp`, `oscillatory`) label each quantity `OBSERVED` or `DERIVED` and carry
manifest provenance. A panel whose declared inputs are missing renders an omission card. Previews and generation code:
[documentation site](https://jaxfne.readthedocs.io/en/latest/) and
[Atlas guide](docs/guides/interop/atlas_suite.md).

