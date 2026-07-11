# Quickstart

Study **laminar population dynamics** — spikes, membrane traces, and depth-resolved
proxy readouts (LFP, CSD, spectrolaminar PSD) — from a JAX-native circuit you
define once and simulate reproducibly.

> Verified against `jaxfne==0.4.5`. `NeuronalTensor` is the preferred path for new code.

## Install & import

```bash
pip install jaxfne
pip install "jaxfne[viz]"
```

```python
import jaxfne as jtfne
jtfne.enable_x64()   # before array construction
```

## Jaxley interoperability

[Jaxley](https://jaxley.readthedocs.io) = biophysical detail (compartments, HH channels).
jaxfne = population/field scale + proxy readouts. Compose them:

```python
import jaxley as jx
from jaxley.channels import HH
import jaxfne as jtfne

cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
cell.insert(HH())
cell.record("v")
cell.stimulate(jx.step_current(i_delay=10, i_dur=50, i_amp=0.1, delta_t=0.025, t_max=100))

sig = jtfne.JaxleyBridge(model=cell).simulate(duration_ms=100.0, dt_ms=0.025)
fig = jtfne.vis.vm(sig)
```

See [Jaxley interoperability guide](guides/jaxley_interop.md) for homeostatic
wrappers and laminar field projection.

## Which workflow should I use?

Pick **one path per script** — they return different types:

| Goal | Path | Returns |
|------|------|---------|
| New code, JSON circuits, HDP, multi-area | `NeuronalTensor` → `construct` → `simulate` | `Model` / `Signals` |
| Fluent builder, AGSDR, homeostasis, per-neuron drive | `laminar_cortex_config` / `build_laminar_column` → `construct` → `simulate` | `Model` / `Signals` |
| Multi-trial spectrolaminar sweeps | `tutorial_utils.make_laminar_column_config` → `simulate_laminar_trials` | plain `dict` |

Per-event subset drive (e.g. L4-E only): put `target_indices` on each **event dict**
in `StimulusSchedule`, built from `model.neuron_table()`.

## NeuronalTensor (preferred)

```python
tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5)
model   = jtfne.construct(tensor, runtime)
signals = jtfne.simulate(model)
spk = signals.get("spk")
vm_e = signals.get("vm", cell_type="E")
```

Inline circuit:

```python
E  = jtfne.NeuronType.make("E", fraction=0.9)
PV = jtfne.NeuronType.make("PV", fraction=0.1)
L4 = jtfne.Layer(name="L4", n_neurons=100, neuron_types=[E, PV])
tensor = jtfne.NeuronalTensor(areas=[jtfne.Area(name="V1", layers=[L4])])
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

**HDP plasticity** — pass `RuntimeConfig` (not `RuntimeConfiguration`) to `simulate()`:

```python
runtime_hdp = jtfne.RuntimeConfig(enable_hdp=True, hdp_params={
    "K_HDP": 0.01, "tau_0_ms": 200.0, "K_ctrl": 5.0,
    "size_scale_by_cell_type": {"E": 2.0, "PV": 1.0},
})
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0, runtime=runtime_hdp)
```

API: [neuronal_tensor.md](api/neuronal_tensor.md) · Migration: [migration_guide.md](migration_guide.md)
· Example: [examples/08_neuronal_tensor_first.py](https://github.com/HNXJ/jaxfne/blob/main/examples/08_neuronal_tensor_first.py)

## Configuration (supported, compatibility)

```python
cfg = (jtfne.build_laminar_column(n=1000, ei_profile="canonical")
          .set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
          .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
model   = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

`ei_profile="canonical"` applies the verified laminar E:I gradient (E peaks deep,
I peaks superficial, PV at L4, ≈77E:23I). `ei_profile="flat"` keeps legacy
depth-invariant composition.

Multi-area: `jtfne.build_multi_area_columns(["V1", "V4", "PFC"], ei_profile="canonical")`.

Full fluent API: [Configuration Grammar](guides/configuration_grammar.md).

### Layers, connectivity, homeostasis

```python
cfg = jtfne.build_laminar_column(
    "V1", n=2000,
    layers=["L1", "L2/3", "L4", "L5", "L6"],
    layer_fractions={"L1": (0.00, 0.08), "L2/3": (0.08, 0.40), "L4": (0.40, 0.55),
                      "L5": (0.55, 0.80), "L6": (0.80, 1.00)},
)
cfg = (cfg
    .connections(name="L4_recurrent", source={"layer": "L4"}, target={"layer": "L4"},
                 probability=0.2, weight=0.5)
    .homeostasis(relative_baseline=1.0, r_star=8.0, k_gain=1.0)
)
```

`plasticity()` records manifest intent only — STDP runs via separate `run_stdp_stream`, not `simulate()`.

## Object grammar

```text
setup → config → construct → simulate → visualize → tune/objective → optimize → export
```

Typed chain: `Config → Emitter → SourceMap → FieldProxy → Probe → Signals → Objective → Optimizer → Manifest`.

## Interactive 3D circuit view

```python
cfg = jtfne.build_laminar_column(n=1000, ei_profile="canonical")
model = jtfne.construct(cfg.set_emitter("izhikevich", "cortical_eig"))
jtfne.vis.visualize_network_3d(model, output_html="network3d.html")
```

## Tune toward a target

```python
obj = jtfne.rate_synchrony_targets()
result = model.tune(obj, optimizer="AGSDR", steps=50)
manifest = jtfne.manifest(cfg, signals=signals)
```

## Scope & status

Proxy readouts only — see [Scope & status](scope_and_status.md).

## Validate locally

```bash
python3 -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest \
  tests/test_api_smoke.py tests/test_root_import_lightweight.py \
  tests/test_signals_get_v0329.py -q --tb=short
python3 -m mkdocs build --strict
```

Contributing: [CONTRIBUTING.md](https://github.com/HNXJ/jaxfne/blob/main/CONTRIBUTING.md).
