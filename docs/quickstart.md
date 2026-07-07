# Quickstart

> Verified against `jaxfne==0.4.5` (editable source checkout) for the
> `Configuration` sections below. `NeuronalTensor` (the preferred path) is
> available on `dev` ahead of the next tagged release and will get its own
> version pin once tagged.

The pipeline is one linear chain — each step returns the input to the next:

```text
setup -> config -> construct -> simulate -> visualize -> tune/objective -> optimize -> export
```

## NeuronalTensor (tensor-first) in three lines — preferred

`NeuronalTensor` is the preferred, declarative `Areas x Layers x NeuronTypes`
data model for defining a circuit (0.4.7+). Load a packaged canonical circuit
or build one inline:

```python
import jaxfne as jtfne
jtfne.enable_x64()

tensor  = jtfne.load_canonical_neuronal_tensor("canonical-v1-column-1000n")
runtime = jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5)
model   = jtfne.construct(tensor, runtime)
signals = jtfne.simulate(model)
```

Or define your own:

```python
E, PV = jtfne.NeuronType.make("E", fraction=0.9), jtfne.NeuronType.make("PV", fraction=0.1)
tensor = jtfne.NeuronalTensor(areas=[
    jtfne.Area(name="V1", layers=[jtfne.Layer(name="L4", n_neurons=100, neuron_types=[E, PV])])
])
model   = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

To enable HDP homeostatic plasticity (synaptic + per-cell-type H-factor
adaptation, cube-law time constant `tau_i = tau_0_ms * size_i**3`) on this
same `model`, pass an explicit `runtime=` override — no new API needed:

```python
runtime_hdp = jtfne.RuntimeConfig(enable_hdp=True, hdp_params={
    "K_HDP": 0.01, "tau_0_ms": 200.0, "K_ctrl": 5.0,
    "size_scale_by_cell_type": {"E": 2.0, "PV": 1.0},
})
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0, runtime=runtime_hdp)
diag = model.last_hdp_diagnostics()   # H_trace, weight trace
```

API reference: [`docs/api/neuronal_tensor.md`](api/neuronal_tensor.md).
Coming from `Configuration`? See the [migration guide](migration_guide.md).

## Configuration: canonical cortex in three lines — supported, compatibility

`Configuration` remains fully supported for existing code and the fluent
method-chaining style. The fastest path to a realistic laminar column this
way: no arguments are required; the defaults give a 1000-neuron V1 column,
and `ei_profile="canonical"` applies the verified ground-truth E:I gradient
(E peaks deep ≈90%, I peaks superficial 50%, PV at L4, ≈77E:23I) with proper
laminar placement.

```python
import jaxfne as jtfne
jtfne.enable_x64()

cfg = (jtfne.build_laminar_column(n=1000, ei_profile="canonical")
          .set_emitter("izhikevich", "cortical_eig")
          .probes(["spikes", "V_m", "LFP", "CSD"], n_contacts=16)
          .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann"))
model   = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
```

`ei_profile="flat"` (the default) keeps the legacy depth-invariant composition
and `uniform3d` placement unchanged. Use `jtfne.build_multi_area_columns(...)`
for a V1→V4→PFC hierarchy with the same prior in each area.

## Configure, construct, simulate (Configuration, supported)

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

## Tune

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
print(result.summary)
```

## Validate

```bash
python -m compileall -q jaxfne tests examples
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
```
