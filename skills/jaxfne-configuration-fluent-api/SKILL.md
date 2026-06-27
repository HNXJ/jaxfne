---
name: jaxfne-configuration-fluent-api
description: >-
  Verified fluent chaining on jaxfne.Configuration and builder entry points.
  USE when composing configs. Documents only methods that exist on disk — not
  aspirational cfg.geometry() / cfg.multi_area() from legacy drafts.
---

# jaxfne Configuration Fluent API

USE FIRST: `catalog-glossary-jaxfne`, `jaxfne-cortical-column-default`.

## Pattern

Each fluent call returns a **new** `Configuration` (immutable-style chaining):

```python
cfg = (jtfne.build_laminar_column("V1", n=1000, ei_profile="canonical",
                                  layers=jtfne.CANONICAL_LAYERS_6L)
       .runtime(seed=0, duration_ms=1000.0, dt_ms=0.5, recurrent_backend="edge_list")
       .connectivity(within_area="all_to_all_uniform_random", p_connect=0.1)
       .set_emitter("izhikevich", "cortical_eig")
       .set_probes(["spikes", "V_m", "LFP", "CSD", "source"], n_contacts=32)
       .field(domain="laminar_column", conductivity="proxy"))
```

Alternative entry: `jtfne.laminar_cortex_config(...)` then `.layer_fractions(...)`,
`.area_layer_cell_types(...)`, `.runtime(...)`.

## Verified `Configuration` methods

| Method | Role |
|--------|------|
| `.runtime(**kwargs)` | seed, duration_ms, dt_ms, jit, recurrent_backend, … |
| `.field(**kwargs)` | laminar proxy field domain (not PDE solve) |
| `.probe(**kwargs)` / `.set_probes(modes, **kwargs)` | readout declarations |
| `.column(name, layers, n)` / `.add_column(...)` | column spec |
| `.layer_fractions(layer_fractions)` | z-bands; width ∝ neuron count |
| `.area_layer_cell_types(area, {...})` | **per-layer** E/PV/SST/VIP fractions |
| `.cell_types(fractions)` | **global** cell-type weights only |
| `.connectivity(**kwargs)` | within/inter connectivity rules |
| `.set_emitter(family, preset)` | emitter family |
| `.plasticity(...)` | declarative only (`declared_not_wired_to_simulate`) |
| `.homeostasis(...)` | wired synaptic homeostasis when eta≠0 |
| `.hdp(...)` | HDP metadata for simulate dispatch |
| `.cell_params(selector, params)` | targeted parameter overrides |
| `.areas(area_names)` | multi-area names |
| `.cell_type_drives(drives)` | baseline drive by cell type |
| `.validate()` | structural validation dict |

## Common mistakes

| Wrong | Right |
|-------|-------|
| `.cell_types({ "L2": {...}, "L3": {...} })` | `.area_layer_cell_types("V1", {...})` |
| `.cell_types("default")` | explicit fractions or `ei_profile="canonical"` builder |
| `.field(mode="pde")` | proxy scaffold only; `field_solver_status=linear_solver` |
| `.geometry(...)`, `.multi_area(...)` | not verified fluent methods — use `laminar_cortex_config` / builders |

## Layer naming

- `ei_profile="canonical"` requires **6 layers** (`CANONICAL_LAYERS_6L`).
- Default `laminar_cortex_config` often uses **5 layers** with `L2/3` merged.

See `skills/FRICTIONS_STACK.md` F-002.

## Inspect before simulate

```python
model = jtfne.construct(cfg)
nt = model.neuron_table()  # authoritative counts — not layer_celltype_count_table on raw cfg
```

## Related skills

- `jaxfne-objective-grammar` — simulate → Signals chain
- `jaxfne-spectrolaminar-suite` — scale path with `recurrent_backend="edge_list"`
