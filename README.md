# jaxfne

JAX-based tools for TFNE source, field, probe, objective, and optimizer workflows.

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

`jaxfne` is built for compact computational biophysics tutorials and package-level experiments. Current tutorial readouts use simulated/proxy scales with JSON-safe reports and deterministic seeds.

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
print(signals.V_m.shape, signals.spikes.sum())
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

## Core readouts

| Readout | Role |
|---|---|
| SPK | spike matrix or events |
| Vm | emitter voltage/state trace |
| Source | source/current proxy |
| LFP-proxy | laminar field proxy |
| CSD-proxy | spatial source/divergence proxy |
| EEG-proxy | linear scalp-channel proxy |
| MEG-proxy | linear magnetic-channel proxy |
| EMM-proxy | normalized activity-cost proxy |

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

MIT License.
