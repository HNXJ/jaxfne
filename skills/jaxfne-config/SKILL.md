---
name: jaxfne-config
description: >-
  Compose, build, or audit a jaxfne Configuration using its fluent builders,
  runtime, connectivity, emitter, probe, field, and layer-composition methods.
  Use for Configuration or laminar-column work.
---

# jaxfne Configuration procedure

Read `catalog-glossary-jaxfne` first. This skill describes API procedure, not
the mathematical specification.

## Tier choice

`Configuration` is the flat builder tier. `NeuronalTensor` is the structured
Areas × Layers × NeuronTypes tier. They are separate inputs to `construct()` and
both produce a `Model`. There is no Configuration-to-NeuronalTensor converter.

Use `Configuration` when the caller has a fluent, single-area, or flexible
layer specification. Use `NeuronalTensor` for structured multi-area or typed
circuits.

## Typical path

```python
import jaxfne as jtfne

cfg = (
    jtfne.laminar_cortex_config(
        seed=0,
        duration_ms=100.0,
        dt_ms=0.5,
        areas=["V1"],
        layers=["L1", "L2/3", "L4", "L5", "L6"],
        n=100,
    )
    .area_layer_cell_types("V1", layer_fractions)
    .runtime(seed=0, duration_ms=100.0, dt_ms=0.5)
)
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.5, seed=0)
```

Verify the builder signature and required field/probe declarations against
the live package before copying a recipe.

## Builder roles

- `.runtime(...)`: seed, duration, timestep, backend, dtype, JIT/VMAP policy.
- `.column(...)`/`.add_column(...)`: structural layer and population layout.
- `.area_layer_cell_types(...)`: per-area/per-layer population fractions.
- `.cell_types(...)`: global population weights; do not use it for a desired
  layer-specific gradient.
- `.connectivity(...)`: declared within/inter-area connection rules.
- `.set_emitter(...)`: emitter family and preset.
- `.field(...)`: field/readout declaration; it does not imply a PDE solve.
- `.probe(...)`/`.set_probes(...)`: requested output channels.
- `.hdp(...)`: runtime metadata; verify explicit runtime forwarding at simulate.
- `.homeostasis(...)` and `.plasticity(...)`: distinct mechanisms; verify their
  wiring before treating metadata as behavior.

Fluent methods follow immutable-style chaining. Verify this behavior with a
small identity/metadata check when changing a builder.

## Layer and population checks

Do not copy a canonical fraction table into a skill. Query the live constant
or inspect `model.neuron_table()`:

```python
fractions = jtfne.CANONICAL_LAYER_CELL_TYPE_FRACTIONS
table = model.neuron_table()
```

State layer names and population scope explicitly in user-facing examples.

## Validation

- Construct through the public `construct()` path.
- Check the model table for counts and labels.
- Check requested probes exist in `Signals`.
- Check finite output and status metadata.
- Run targeted configuration tests.

For scientific meaning and status vocabulary, use the current project source
documents and `docs/scope_and_status.md`.
