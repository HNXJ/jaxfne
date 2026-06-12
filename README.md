# jaxfne

[![PyPI version](https://img.shields.io/pypi/v/jaxfne.svg)](https://pypi.org/project/jaxfne/)
[![Docs](https://readthedocs.org/projects/jaxfne/badge/?version=latest)](https://jaxfne.readthedocs.io/en/latest/)
[![Release](https://img.shields.io/github/v/release/HNXJ/jaxfne)](https://github.com/HNXJ/jaxfne/releases)

`jaxfne` is a compact JAX-first scaffold for **Tensor-Field Neural Equations (TFNE)**: a source-to-field/readout grammar for computational neurophysiology.

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

The package is built for tutorial-scale evidence generation with explicit
**scope**, **status**, and **metadata** at every stage: laminar and multi-area
neural emitters, source bookkeeping, proxy field/readout operators, objective
reports, optimization workflows, and strict JSON-safe run manifests. Kernels are
JAX-first, with deterministic seeds and a `float32` default.

## Why it exists

Neural models often stop at spikes, voltages, or population rates. TFNE adds an
explicit bridge from emitter dynamics to declared source tensors, field/proxy
maps, probe operators, objective reports, and reproducible metadata.

`jaxfne` focuses on that bridge. It keeps emitters, source maps, field/proxy
operators, probes, objectives, optimizers, and validation reports separated
enough that each assumption can be inspected, named, and tested.

## Scientific framing

Emitters generate state trajectories. Source maps convert those trajectories
into declared current / source tensors under explicit bookkeeping
(one source mode per run, with a synaptic double-count guard). Field/proxy
layers map source tensors into extracellular observable variables using a
quasi-static resistive framing — a row-normalized laminar projection, not a
Maxwell or boundary-value field solve. Probe operators then sample named
readouts: spike rasters, voltage traces, source traces, LFP-proxy and
CSD-proxy laminar proxies, EEG-proxy and MEG-proxy linear-projection proxies, and
an EMM-proxy within-run activity/source/field cost summary.

The v0.3.x line uses simulated / proxy readouts designed for computational scaffolds.
Readout amplitudes are normalized proxy units; the EMM-proxy is an internal normalized
activity summary.

## Install

```bash
pip install jaxfne
```

Visualization extras, when needed:

```bash
pip install "jaxfne[viz]"
```

Development checkout:

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e ".[dev,viz]"
```

The canonical import is:

```python
import jaxfne as jtfne
```

## 0.3.28+ object model

| Object | Role |
|---|---|
| `Config` | Declarative circuit/task/training spec; the bio-circuit PCB sketch. |
| `Net` | Compiled biophysical circuit from Config. |
| `Paradigm` | Task/trial/stimulus schedule, from constant DC to sequence tasks. |
| `Objective` | Metrics, gates, and scores computed from Signals. |
| `Trainer` | AGSDR or other tuning loop over declared trainables. |
| `Signals` | Tensor outputs and query/layout API. |
| `FlatNet` | JAX/JIT/pmap-friendly array form with tracking maps. |

Compatibility aliases may remain during migration:

```text
Configuration -> Config
Model -> Net
FlatModel -> FlatNet
```

## Minimal workflow

```python
import jaxfne as jtfne

cfg = (
    jtfne.Config(schema_version="0.3.28")
    .runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
    .areas(["V1"])
    .layers(["L2/3"])
    .cells({"E": 1.0})
    .cell_params({"E": {"drive": 4.5, "noise": 0.5}})
    .mechanisms({})
    .connections([])
    .probes(["spk", "vm", "source", "lfp_proxy", "csd_proxy"])
)

net = jtfne.construct(cfg)
paradigm = jtfne.Paradigm.constant_dc(target={"area": "V1"}, amplitude=1.0)
signals = net.simulate(paradigm=paradigm, seed=7)
print(signals.get("spikes", layout="time_node").shape)
```

## Querying signals

Named readouts are addressable by alias, and neuron-indexed signals (Vm,
spikes, sources) can be filtered with semantic selectors over the declared
area / layer / cell-type / id metadata. Selection requires a run that carries a
neuron-identity table; otherwise the call raises rather than guessing.

```python
import jaxfne as jtfne

cfg = jtfne.suite2_four_celltype_config(seed=0)
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=10, dt_ms=0.1, seed=0)

vm = signals.get("vm")                      # alias for V_m
vm_excit = signals.get("vm", cell_type="E") # selector over neuron axis
idx = model.select(cell_type="E")           # resolve selector to indices
```

## Readout status

| Readout     | Role                                                   | Image Reference |
| ----------- | ------------------------------------------------------ | --------------- |
| `spk`       | spike / event readout                                  | ![spk](outputs/v034_plasticity/figures/spk.png) |
| `vm`        | voltage / emitter state trace                          | ![vm](outputs/v034_plasticity/figures/vm.png) |
| `source`    | declared source tensor or source proxy                 | ![raster](outputs/v034_plasticity/figures/raster.png) |
| `lfp_proxy` | local potential laminar proxy                          | ![lfp_proxy](outputs/v034_plasticity/figures/lfp_proxy.png) |
| `csd_proxy` | laminar source-profile / second-derivative proxy       | ![csd_proxy](outputs/v034_plasticity/figures/csd_proxy.png) |
| `eeg_proxy` | linear scalp-channel projection proxy                  | ![eeg_proxy](outputs/v034_plasticity/figures/eeg_proxy.png) |
| `meg_proxy` | current-orientation / magnetic projection proxy        | ![meg_proxy](outputs/v034_plasticity/figures/meg_proxy.png) |
| `spectrolaminar_proxy` | depth x frequency power proxy                       | ![spectrolaminar_proxy](outputs/v034_plasticity/figures/spectrolaminar_proxy.png) |
| `agsdr`     | synaptic gain matrix tuning                            | ![agsdr](outputs/v034_plasticity/figures/agsdr_matrix_gain.png) |
| `emm_proxy` | normalized within-run activity / source / field cost   | (derived cost summary) |

## Validate a checkout

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Install](docs/install.md)
- [Probe operators](docs/probe_operators.md)
- [Tutorials](docs/tutorials/index.md)
- [API reference](docs/api/index.md)
- [PyPI](https://pypi.org/project/jaxfne/)
- [Releases](https://github.com/HNXJ/jaxfne/releases)
