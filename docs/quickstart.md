# Quickstart

> Verified against `jaxfne==0.4.1` (`pip install "jaxfne==0.4.1"`).

The pipeline is one linear chain — each step returns the input to the next:

```text
setup -> config -> construct -> simulate -> visualize -> tune/objective -> optimize -> export
```

## Canonical cortex in three lines

The fastest path to a realistic laminar column. No arguments are required; the
defaults give a 1000-neuron V1 column, and `ei_profile="canonical"` applies the
verified ground-truth E:I gradient (E peaks deep ≈90%, I peaks superficial 50%,
PV at L4, ≈77E:23I) with proper laminar placement.

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

## Configure, construct, simulate

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
