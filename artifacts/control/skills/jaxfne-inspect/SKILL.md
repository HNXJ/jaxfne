---
name: jaxfne-inspect
description: Read what a model or run actually contains (counts, storage, recorded outputs, tables) before using it.
metadata:
  audience: agents
---
# jaxfne inspect

## WHEN
Before reasoning about a model or run, and whenever a number will be quoted.

## AUTHORITIES
1. Live code: `jaxfne.agent.inspect`, `Model.neuron_table`, `Model.edge_table`,
   `jaxfne.util.canonical_compact_summary` (configured vs realized vs executed counts).

## RULES
- Quote counts, shapes and recorded outputs from inspection, never from recall or docs.
- Configured, realized and executed are different objects; name which one a number describes.
- A reporting surface is not the kernel: identity claims go through `jaxfne-verify`.

## STEPS
1. `jaxfne.agent.inspect(run)`: neurons, edges, storage, dt, steps, recorded outputs.
2. Tables: `model.neuron_table()`, `model.edge_table()` for per-neuron and per-edge detail.
3. Full taxonomy when needed: `canonical_compact_summary(model, signals)`.

## STOP
- A value needed for a claim is absent from every inspection surface.

## VERIFY
- Quoted values reproduce from the inspection call on the same run.

## DONE
- Every quoted quantity has the inspection call that produced it.
