# BMTK / SONATA architecture study (W15)

Status: **PLAN** — planning receipt for v0.4.22. No SONATA import/export and no
BMTK interoperability in this release.

Source mapping: `artifacts/roadmap/ROADMAP_0422_0424.md` W15; measured internal
evidence W10/W11.

## Transferable lesson (not engine delegation)

BMTK separates build, persistent network representation, simulation, and analysis
across external engines. JaxFNE's differentiator is a JAX-native composable
execution stack. The transferable design is **representation boundaries and
backend specialization**, not delegating numerics to NEURON/NEST.

## Six principles mapped to JaxFNE evidence

1. **Do not retain representations the selected backend will not read.** Target:
   configuration → realized canonical topology → backend-specific executable layout.
2. **Backend selection depends on realized topology**, not N alone
   ($E = NK_{\max}$ vs $N^2$).
3. **Separate logical scientific model from physical execution layout** (class-shared
   edge parameters stored once; structural indices separate).
4. **Recording is observation-driven** ($O(TN)$ not $O(TE)$; explicit policy).
5. **Parallelism: independent batches first** (`vmap`/sharding across seeds and
   candidates before domain decomposition).
6. **Multi-resolution is product direction**, not a single-engine speed contest.

## Priority table (external → internal)

| External idea | JaxFNE analogue | Priority |
| --- | --- | ---: |
| Separate build from simulation | `construct` → `Model` | Very high |
| Persistent network representation | compact realized topology | Very high |
| Engine dispatch by regime | backend by realized structure | Very high |
| SONATA interoperability | optional import/export boundary (later) | High |
| Selective recording | requested observations only | High |

## v0.4.22 boundary

Study and record only. **v0.4.23 carry-forward:** realized topology authoritative;
dense `W` and edge lists are execution layouts, not co-equal permanent truths.

## Headline

Scientific model specification should be independent of the representation used to
execute it efficiently. JaxFNE achieves this through measured backend-specific
JAX layouts (W10/W11), not through adopting SONATA internally.
