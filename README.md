# jaxfne

JAX-native tools for compact Tensor-Field Neural Equation workflows.

```text
Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export
```

`jaxfne` is a computational scaffold for emitter/source/field/probe modeling, tutorials, validation reports, and optimizer workflows. Current tutorial readouts are simulated/proxy-scale unless a run provides solver, calibration, boundary, gauge, units, residual, convergence, and validation evidence.

## Install

```bash
pip install -U jaxfne
```

Development checkout:

```bash
git clone https://github.com/HNXJ/jaxfne.git
cd jaxfne
pip install -e .[dev,viz,opt]
```

## Canonical import

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
    .probes(["spk", "vm", "source", "lfp_like", "csd_like"])
)

net = jtfne.construct(cfg)
paradigm = jtfne.Paradigm.constant_dc(target={"area": "V1"}, amplitude=1.0)
signals = net.simulate(paradigm=paradigm, seed=7)
print(signals.get("spikes", layout="time_node").shape)
```

## Core readouts

| Readout | Role |
|---|---|
| `spk` | spike events or matrix |
| `vm` | emitter voltage/state trace |
| `source` | source/current proxy |
| `lfp_like` | laminar LFP-like proxy |
| `csd_like` | CSD-like proxy |
| `eeg_like` | linear EEG-like proxy |
| `meg_like` | linear MEG-like proxy |
| `emm_proxy` | normalized activity-cost proxy |

## Validate a checkout

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
PYTHONPATH=. python scripts/audit_notebooks_and_assets.py --check
mkdocs build --strict
```

MIT License.
