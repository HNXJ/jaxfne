# W16 scientific and developmental programme definition

Status: **PLAN** — v0.4.22 planning authority only. No implementation claims.

Source: `artifacts/roadmap/ROADMAP_0422_0424.md` W16 / §8.

## Distinctions (must not collapse)

| Concept | Is | Is not |
| --- | --- | --- |
| $H$ (RBS/RBD) | biophysical state of the current system | structural topology $G$ |
| HDP | parameter dynamics $\dot W$ | neuron birth or edge creation |
| $G$ / structural events | which neurons, edges, geometry exist | another name for $H$ |
| `develop()` (JDNA) | **implemented** build-time specification → tensor | runtime neurogenesis |
| `evolve()` (future) | developmental update of specification/realization | shipped today |

## Programme axes (0.4.22 scope: define only)

**W16.1 Flexible biophysical state.** Typed $H$ coordinates with explicit coupling
maps; not every model needs every coordinate.

**W16.2 Source → field → probe.** Single pipeline:
dynamics → source → field → probe → observation. Proxy ≠ calibrated measurement.

**W16.3 Developmental models (JDNA extension).** Future `evolve()` semantics for
specification change; syntax provisional; 0.4.23 substrate only if complete.

**W16.4 Structural biophysics.** Structural development changes the represented
system ($\dot G$ / events), distinct from $H$ dynamics.

**W16.5 Pseudo-neurogeneration.** Compact generative specifications → realized
systems; no biological fidelity claims without validation.

**W16.6 Neurobiophysical geometry.** $\mathrm{Aug}(F,c,\epsilon)$ research
programme — **0.4.24**, not 0.4.22/0.4.23.

## v0.4.22 deliverable

Written definitions and boundaries only. Public docs must not imply shipped
developmental runtime beyond documented `develop()` build-time scope.
