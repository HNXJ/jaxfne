# jaxfne

[![PyPI version](https://img.shields.io/pypi/v/jaxfne.svg)](https://pypi.org/project/jaxfne/)
[![Docs](https://readthedocs.org/projects/jaxfne/badge/?version=latest)](https://jaxfne.readthedocs.io/en/latest/)
[![Release](https://img.shields.io/github/v/release/HNXJ/jaxfne)](https://github.com/HNXJ/jaxfne/releases)

`jaxfne` is a compact JAX-first scaffold for **Tensor-Field Neural Equations (TFNE)**: a source-to-field/readout grammar for computational neurophysiology.

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
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
into declared current-like / source-like tensors under explicit bookkeeping
(one source mode per run, with a synaptic double-count guard). Field/proxy
layers map source tensors into extracellular observable-like variables using a
quasi-static resistive framing — a row-normalized laminar projection, not a
Maxwell or boundary-value field solve. Probe operators then sample named
readouts: spike rasters, voltage-like traces, source traces, LFP-like and
CSD-like laminar proxies, EEG-like and MEG-like linear-projection proxies, and
an EMM-proxy within-run activity/source/field cost summary.

The v0.3.x line uses **simulated / proxy** readouts unless a run supplies
physical geometry, calibration, boundary and gauge handling, solver residuals,
units, and validation evidence. Readout amplitudes are uncalibrated proxy units;
the EMM-proxy is an internal normalized activity summary, not a metabolic measurement.

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

## Minimal workflow

```python
import jaxfne as jtfne

cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=7, dtype="float32", duration_ms=1000.0, dt_ms=0.1)
cfg = cfg.column("single_neuron", layers=["L2/3"], n=1)
cfg = cfg.cell_types({"E": 1.0})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["MUA-proxy", "source-proxy", "LFP-proxy"])

model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=7)
print(signals.V_m.shape, float(signals.spikes.sum()))
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

## Multi-objective tuning

```python
objectives = jtfne.rate_targets(
    groups={"first_half": range(24), "second_half": range(24, 48)},
    targets_hz={"first_half": 5.0, "second_half": 10.0},
)
optimizer = jtfne.agsdr(
    parameters={"drive_scale_a": (0.35, 2.25), "drive_scale_b": (0.35, 2.25)},
    generations=8,
    population_size=6,
    seed=42,
)
result = model.tune(objectives=objectives, optimizer=optimizer)
print(result.best_score, result.best_parameters)
```

## Readout status

| Readout     | Role                                                   |
| ----------- | ------------------------------------------------------ |
| `spk`       | spike / event readout                                  |
| `vm`        | voltage / emitter state trace                          |
| `source`    | declared source tensor or source proxy                 |
| `lfp_like`  | local potential-like laminar proxy                     |
| `csd_like`  | laminar source-profile / second-derivative-like proxy  |
| `eeg_like`  | linear scalp-channel projection proxy                  |
| `meg_like`  | current-orientation / magnetic projection proxy        |
| `emm_proxy` | normalized within-run activity / source / field cost   |

All readouts are simulated/proxy under `truth_safe_unverified` status unless a
run supplies geometry, calibration, solver residuals, units, and validation.

## Validate a checkout

```bash
python -m compileall -q jaxfne tests examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
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
