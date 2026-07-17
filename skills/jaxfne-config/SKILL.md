---
name: jaxfne-config
description: >-
  Compose, build, or audit a jaxfne Configuration — the fluent builder chain
  (.runtime, .connectivity, .set_emitter, .set_probes, .layer_fractions,
  .area_layer_cell_types) and the canonical 1K-neuron laminar cortical column
  (E-deep / I-superficial gradient, PV peaks at L4). Use whenever a task builds
  or edits a Configuration, a laminar or cortical column, build_laminar_column
  / laminar_cortex_config, layer fractions, per-layer E/PV/SST/VIP cell-type
  composition, connectivity rules on a Configuration, emitter or probe
  declarations, or asks for the default column prior.
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
| `.homeostasis(relative_baseline=1.0, **kwargs)` | wired synaptic homeostasis when `k_gain≠0` (derived as `relative_baseline-1.0`, overridable via explicit `k_gain=`; excitability damper, not bidirectional setpoint) |
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

**Ground truth** (re-verified 2026-07-13 directly against `jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS`
and a live `construct(cfg).neuron_table()` — the previous table below this
line was stale and off by up to 25 points per layer; corrected here, not
just flagged), two laws hold:

1. **E peaks DEEP** — excitatory fraction rises with depth (L6 = 95% E).
2. **I peaks SUPERFICIAL** — inhibitory fraction highest in L1/L2/L3 (50% I
   each, not just L1); lowest in L6 (5% I).

Overall ≈ **66E : 34I** (verified: 657–658 E of 1000 at n=1000, seed 0). NOT
77:23 (a stale figure from an earlier fraction table) and NOT the wrong
global `cell_types=` weight (≈41:59 over-inhibitory).

| Layer | z-band (width ∝ count) | Neurons | I-fraction |
|-------|------------------------|--------:|-----------:|
| L1 | 0.00–0.10 | 100 | 50% |
| L2 | 0.10–0.35 | 250 | 50% |
| L3 | 0.35–0.55 | 200 | 50% |
| L4 | 0.55–0.65 | 100 | 30% |
| L5 | 0.65–0.85 | 200 | 15% |
| L6 | 0.85–1.00 | 150 | 5% |

Per-layer fractions — this IS `jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS`
verbatim (query the live constant instead of copying this table if in doubt,
it can drift again):

```python
jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS == {
    "L1": {"E": 0.50, "PV": 0.05, "SST": 0.10, "VIP": 0.35},
    "L2": {"E": 0.50, "PV": 0.25, "SST": 0.10, "VIP": 0.15},
    "L3": {"E": 0.50, "PV": 0.25, "SST": 0.15, "VIP": 0.10},
    "L4": {"E": 0.70, "PV": 0.20, "SST": 0.05, "VIP": 0.05},
    "L5": {"E": 0.85, "PV": 0.05, "SST": 0.05, "VIP": 0.05},
    "L6": {"E": 0.95, "PV": 0.00, "SST": 0.05, "VIP": 0.00},
}
```

PV **peaks at L2/L3 (25% each)**, is present but lower at L1/L4 (5%/20%),
and is **absent at L6** (0%) — not "concentrates in L4, absent in L1" as an
earlier pass of this skill claimed.

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
