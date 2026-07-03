---
name: jaxfne-config
description: >-
  Verified Configuration fluent API + the canonical 1K-neuron laminar column
  template (E-deep/I-superficial gradient). USE when composing, building, or
  auditing a Configuration. Merged from jaxfne-configuration-fluent-api +
  jaxfne-cortical-column-default (2026-06-30) — documents only methods and
  fractions verified on disk, not aspirational cfg.geometry()/multi_area()/
  weld() from legacy drafts.
---

# jaxfne Config

USE FIRST: `catalog-glossary-jaxfne`.

## Which config tier — `Configuration` is the quick-start tier

`Configuration` is one of three tiers of the same pipeline (all compile to
`Model`/`IzhikevichParams`/`EdgeList`): `Configuration` is the quick-start
single-area/column tier; `HDPColumnConfig` (`jaxfne-neural-tensor` skill /
`jaxfne/hdp_network.py`) is a thin canonical-6-layer convenience wrapper
around `Configuration`; `NeuronalTensor` (`jaxfne-neural-tensor` skill) is
the structured "IC/PCB schematic" tier for multi-area/typed circuits. Use
`Configuration` directly (not `HDPColumnConfig`) whenever your layer names
or cell-type-fraction shape don't fit `HDPColumnConfig`'s canonical
`L1..L6` assumption — `Configuration.area_layer_cell_types(area, dict)`
accepts arbitrary layer names and per-layer fraction dicts. See
`AGENTS.md` § "Config complexity tiers" for the full picture.

## What `Configuration` actually is (verified via `dataclasses.fields`)

```python
Configuration(networks, emitters, fields, probes, metadata)
```

**Not real** (do not invent): `.circuit`, `.paradigm`, `.objective`, `.optimizer`
typed sub-specs, `jtfne.weld(...)`. Those read as aspirational drafts, not the
disk API — verify any unfamiliar `Configuration`/top-level name against
`dataclasses.fields(Configuration)` / `dir(jtfne)` before trusting it.

## Fluent pattern

Each call returns a **new** `Configuration` (immutable-style chaining):

```python
cfg = (jtfne.build_laminar_column("V1", n=1000, ei_profile="canonical",
                                  layers=jtfne.CANONICAL_LAYERS_6L)
       .runtime(seed=0, duration_ms=1000.0, dt_ms=0.5, recurrent_backend="edge_list")
       .connectivity(within_area="all_to_all_uniform_random", p_connect=0.1)
       .set_emitter("izhikevich", "cortical_eig")
       .set_probes(["spikes", "V_m", "LFP", "CSD", "source"], n_contacts=32)
       .field(domain="laminar_column", conductivity="proxy"))
model = jtfne.construct(cfg)
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
| `.cell_types(fractions)` | **global** cell-type weights only (wrong for laminar E:I gradient — see below) |
| `.connectivity(**kwargs)` | within/inter connectivity rules |
| `.set_emitter(family, preset)` | emitter family |
| `.plasticity(...)` | declarative only (`declared_not_wired_to_simulate`) |
| `.homeostasis(eta=..., r_star=...)` | wired synaptic homeostasis when eta≠0 (excitability damper, not bidirectional setpoint) |
| `.hdp(...)` | HDP metadata for simulate dispatch — see `jaxfne-neural-tensor` for the real HDP mechanics |
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
| `.geometry(...)`, `.multi_area(...)`, `jtfne.weld(...)` | not verified — use `laminar_cortex_config` / builders |
| `load_config`/`validate_config`/`JaxFNEConfig`/`.jcfg.json` | **DELETED 2026-06-30** — legacy format lived only in tests, never a real asset. Use `Configuration` builders (this skill) or the `NeuronalTensor` path (`jaxfne-neural-tensor`) instead. |

## Layer naming (5 vs 6 layer — pick one)

| Builder | Layer names | When |
|---------|-------------|------|
| `laminar_cortex_config(..., layers=[...])` | Often 5-layer: `L1,L2/3,L4,L5,L6` | Multi-area config path |
| `build_laminar_column(..., ei_profile="canonical")` | **Requires 6-layer:** `L1,L2,L3,L4,L5,L6` | Fluent builder + canonical biophysics |

Constants: `jtfne.CANONICAL_LAYERS_6L`, `DEFAULT_LAYERS` in `builders.py`.

## Canonical 1K-neuron column (default prior unless user specifies otherwise)

**Ground truth** (verified via `construct(cfg).neuron_table()`), two laws:

1. **E peaks DEEP** — excitatory fraction rises with depth (L6 ≈ 90% E).
2. **I peaks SUPERFICIAL** — inhibitory fraction highest in L1 (50% I); largest
   inhibitory **neuron count** in dense L2.

Overall ≈ **77E : 23I**. NOT the wrong global `cell_types=` weight (≈41:59 over-inhibitory).

| Layer | z-band (width ∝ count) | Neurons | I-fraction |
|-------|------------------------|--------:|-----------:|
| L1 | 0.00–0.10 | 100 | 50% |
| L2 | 0.10–0.35 | 250 | 30% |
| L3 | 0.35–0.55 | 200 | 25% |
| L4 | 0.55–0.65 | 100 | 20% |
| L5 | 0.65–0.85 | 200 | 12% |
| L6 | 0.85–1.00 | 150 | 10% |

Per-layer fractions (use `.area_layer_cell_types`, NOT global `cell_types=`):

```python
LAYER_CELL_TYPE_FRAC = {
    "L1": {"E": 0.50, "PV": 0.00, "SST": 0.15, "VIP": 0.35},
    "L2": {"E": 0.70, "PV": 0.15, "SST": 0.10, "VIP": 0.05},
    "L3": {"E": 0.75, "PV": 0.13, "SST": 0.08, "VIP": 0.04},
    "L4": {"E": 0.80, "PV": 0.12, "SST": 0.05, "VIP": 0.03},
    "L5": {"E": 0.88, "PV": 0.06, "SST": 0.04, "VIP": 0.02},
    "L6": {"E": 0.90, "PV": 0.05, "SST": 0.03, "VIP": 0.02},
}
```

PV concentrates in L4; **absent in L1** (VIP/SST only).

### Quickstart — config path

```python
LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6"]
ZBANDS = {"L1": (0.00, 0.10), "L2": (0.10, 0.35), "L3": (0.35, 0.55),
          "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00)}

cfg = (jtfne.laminar_cortex_config(
           seed=0, duration_ms=1000.0, dt_ms=0.5, areas=["V1"], layers=LAYERS,
           n=1000, emitter="izhikevich",
           baseline_drive_by_cell_type={"E": 5.0, "PV": 5.0, "SST": 5.0, "VIP": 5.0})
       .layer_fractions(layer_fractions=ZBANDS)
       .area_layer_cell_types("V1", LAYER_CELL_TYPE_FRAC))
model = jtfne.construct(cfg)
nt = model.neuron_table()  # list[dict]: neuron_id, area, layer, cell_type, x, y, z
```

### Quickstart — fluent builder with canonical biophysics

```python
cfg = (jtfne.build_laminar_column("V1", n=1000, ei_profile="canonical",
                                  layers=jtfne.CANONICAL_LAYERS_6L)
       .runtime(seed=0, duration_ms=1000.0, dt_ms=0.5)
       .set_emitter("izhikevich", "cortical_eig"))
model = jtfne.construct(cfg)
```

`ei_profile="canonical"` applies verified fractions + biophysics at construct time
(`core._apply_canonical_biophysics`): SST a=0.05, deep-E source scaling, PV↔E
local gain, random v0 ~ U(-70, 0) unless `random_v0=False`.

## Drive / sanity

- `baseline_drive_by_cell_type=5.0` → mean rate ≈ 18 Hz (8–25 Hz band) at n=1000.
- Vm rest ≈ −66 to −67 mV; spike peak +30 mV; `|Vm| > 150` or NaN = blowup, not science.
- Per-neuron drive: `model.with_emitter_parameters(drive_per_neuron=array)` from `neuron_table()`.

## Projection default

`project_laminar_sources` defaults to **`mode="density_preserving"`**. Use
`row_normalize` only when explicitly intended (density-erasing for out-of-population
contacts — see `skills/FRICTIONS_STACK.md` F-003).

## Inspect before simulate

```python
model = jtfne.construct(cfg)
nt = model.neuron_table()  # authoritative counts — not layer_celltype_count_table on raw cfg
```

## Override protocol

If the user specifies another architecture, state it explicitly. Otherwise the
canonical 1K-neuron column above is the prior.

## Related skills

- `jaxfne-neural-tensor` — the alternative `NeuronalTensor` build path (Areas × Layers × NeuronTypes), converges on the same `construct()`
- `jaxfne-neural-network` — `construct → simulate → Signals → probe/objective/tune` chain
- `jaxfne-vis-modules` — spectrolaminar readout, raster/trace plotting
