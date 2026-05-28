# jaxfne

JAX-native tools for TFNE source, field, probe, objective, and optimizer workflows.

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

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

## Minimal example

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

## Main pages

- [Install](install.md)
- [Quickstart](quickstart.md)
- [Probe operators](probe_operators.md)
- [Tutorials](tutorials/index.md)
- [API reference](api/index.md)
- [Changelog](changelog.md)
