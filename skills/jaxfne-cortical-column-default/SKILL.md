---
name: jaxfne-cortical-column-default
description: >-
  Canonical 1K-neuron laminar column template — verified E-deep / I-superficial
  gradient, layer fractions, drive calibration, and spectrolaminar readout entry
  points. USE when building a default V1 column unless the user specifies another
  architecture. Cross-check layer naming (5 vs 6 layer) via FRICTIONS_STACK F-002.
---

# jaxfne Cortical Column Default

USE FIRST: `catalog-glossary-jaxfne`, `jaxfne-spectrolaminar-suite` (crossover regime).

## Ground truth (verified via `construct(cfg).neuron_table()`)

Two laws:

1. **E peaks DEEP** — excitatory fraction rises with depth (L6 ≈ 90% E).
2. **I peaks SUPERFICIAL** — inhibitory fraction highest in L1 (50% I); largest
   inhibitory **neuron count** in dense L2.

Overall ≈ **77E : 23I**. NOT the wrong global `cell_types=` weight (≈41:59 over-inhibitory).

## Layer sets — pick one (see `skills/FRICTIONS_STACK.md` F-002)

| Builder | Layer names | When |
|---------|-------------|------|
| `laminar_cortex_config(..., layers=[...])` | Often 5-layer: `L1,L2/3,L4,L5,L6` | Multi-area config path |
| `build_laminar_column(..., ei_profile="canonical")` | **Requires 6-layer:** `L1,L2,L3,L4,L5,L6` | Fluent builder + canonical biophysics |

Constants in code: `jtfne.CANONICAL_LAYERS_6L`, `DEFAULT_LAYERS` in `builders.py`.

## 6-layer architecture (1000 neurons)

| Layer | z-band (width ∝ count) | Neurons | I-fraction |
|-------|------------------------|--------:|-----------:|
| L1 | 0.00–0.10 | 100 | 50% |
| L2 | 0.10–0.35 | 250 | 30% |
| L3 | 0.35–0.55 | 200 | 25% |
| L4 | 0.55–0.65 | 100 | 20% |
| L5 | 0.65–0.85 | 200 | 12% |
| L6 | 0.85–1.00 | 150 | 10% |

Per-layer cell-type fractions (use `.area_layer_cell_types`, NOT global `cell_types=`):

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

## Quickstart — config path (verified)

```python
import jaxfne as jtfne

LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6"]
ZBANDS = {
    "L1": (0.00, 0.10), "L2": (0.10, 0.35), "L3": (0.35, 0.55),
    "L4": (0.55, 0.65), "L5": (0.65, 0.85), "L6": (0.85, 1.00),
}

cfg = (jtfne.laminar_cortex_config(
           seed=0, duration_ms=1000.0, dt_ms=0.5, areas=["V1"], layers=LAYERS,
           n=1000, emitter="izhikevich",
           baseline_drive_by_cell_type={"E": 5.0, "PV": 5.0, "SST": 5.0, "VIP": 5.0})
       .layer_fractions(layer_fractions=ZBANDS)
       .area_layer_cell_types("V1", LAYER_CELL_TYPE_FRAC))
model = jtfne.construct(cfg)
nt = model.neuron_table()  # list[dict]: neuron_id, area, layer, cell_type, x, y, z
```

## Quickstart — fluent builder with canonical biophysics

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

## Spectrolaminar readout

```python
sig = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
fig = jtfne.vis.spectrolaminar_suite(sig)  # preferred single-run readout
```

**Crossover:** depth×frequency alpha/beta vs gamma is **not** guaranteed by column
geometry alone. In asynchronous-irregular regimes the LFP is broadband at every
depth. See `jaxfne-spectrolaminar-suite` — crossover requires band-limited
**oscillatory** layer-local regimes while global kappa stays low.

## Projection default

`project_laminar_sources` defaults to **`mode="density_preserving"`**. Use
`row_normalize` only when explicitly intended (see FRICTIONS_STACK F-003).

## Override protocol

If the user specifies another architecture, state it explicitly. Otherwise this
template is the prior.
