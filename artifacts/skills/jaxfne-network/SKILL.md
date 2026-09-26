---
name: jaxfne-network
description: Specify a network in TFNE and realize it with identity preserved (weights, mechanisms, delays, geometry).
metadata:
  audience: agents
---
# jaxfne network

## WHEN
Building or changing connectivity: populations, projections, weights, probabilities,
mechanisms, delays, relative geometry.

## AUTHORITIES
1. TFNE grammar: `artifacts/project_sources/7_tfne_algebra.md`.
2. Capability rows: `artifacts/programme/capability_inventory.md` (Specify, Complete / realize).
3. Live code: `jaxfne/tfne.py` (`parse`, `resolve`, `realize`, `to_configuration`).

## RULES
- Specify in TFNE; `Configuration().connections(...)` is the hand-built path, not the default.
- Delay is declared in ms on the rule and realized as `round(delay_ms / dt_ms)` steps;
  a positive delay that rounds to 0 steps is refused, never dropped.
- `GABA` is ambiguous and refused; name `GABA_A` or `GABA_B`.
- Geometry is relative (fractions in [0, 1]); never mm or um.

## STEPS
1. Write the TFNE text; `r = jaxfne.agent.realize(text, seed=...)`.
2. Inspect `r.s` (`edge_pre`, `edge_post`, `edge_weight`, `edge_mechanism`, `edge_delay_ms`).
3. Execute with `jaxfne.agent.simulate(r, duration_ms=..., dt_ms=...)`.
4. Route identity checks to `jaxfne-verify`.

## STOP
- `TFNEError` from parse/resolve/realize; a scope that matches no realized neurons.

## VERIFY
- `jaxfne.agent.verify(run, p)` PASS for `weight_identity`, `mechanism_identity`, `delay_identity`.

## DONE
- Realized edges equal executed edges per class, with the TFNE text recorded.
