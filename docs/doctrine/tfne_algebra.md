# TFNE Algebra

**Status:** canonical specification language (implementation: `jaxfne.tfne`;
verified by `tests/test_tfne_algebra.py`)
**Canonical source:** `artifacts/project_sources/7_tfne_algebra.md`

## Core form

A TFNE model is an explicitly modeled neural system bounded by an upstream
representation and a downstream representation:

$$
\boxed{x:\mathcal{A}:y}
$$

After realization, $\mathcal{A} \rightarrow (s,h_0)$ with execution normal
form:

$$
\boxed{(h_t,x_t;s)\mapsto(h_{t+1},y_t)}.
$$

See the [computation basis](../computation_basis.md) for tensor shapes and
the [containment architecture](tfne_containment_architecture.md) for the
operator factorization this language compiles into.

## Pipeline

| Stage | Representation | Purpose |
|---|---|---|
| Define | Named recursive objects and reusable rules | Local structure, models, parameters, mechanisms |
| Factor | Shared object and connection definitions | Reuse without duplicated numerical definitions |
| Compose | $O$, $X$, groups, named rules | Compact hierarchies and cross-connections |
| Realize | Expand $N$, $P$, $G$, models, rules | One explicit realized neural system |
| Flatten | $(s,h_0,\mathcal{I})$ | Flat execution arrays with identity preserved |
| Execute | $(h_t,x_t;s)\mapsto(h_{t+1},y_t)$ | JaxFNE numerical kernels |
| Inspect | $\mathcal{I}$ | Flat state back to TFNE names and paths |

Design principle: factor at specification time; flatten at execution time.
Flattening changes representation, not TFNE semantics.

## Mapping to JaxFNE components

| Algebra concept | JaxFNE realization |
|---|---|
| Named object / composite $\{E\}$ | `NeuronalTensor` areas, layers, groups; TFNE paths in $\mathcal{I}$ |
| Cell-type domain $C$, subdivision $L$, $L[C]$ | `NeuronType`, `Layer`, layer/type selection |
| Cardinality $N$, proportions $P$ | Neuron counts via deterministic allocation; `develop` constraint bands upstream |
| Geometry $G$ | `Geometry3D` / `Pose3D`; fixed geometry lands in $s$ |
| `model` tag | Emitter choice (Izhikevich, LIF, HH/Jaxley bridge); identity and model stay distinct |
| $O[k]$ / $X[k]$ rules | Named entries of the connection-rule compiler (`compile_connection_rules`) |
| Projection $>$, $<$, $<>$ / exclusion $\not>$, $\not<$ | Directed edge groups with rule provenance; exclusions are realization vetoes |
| $A^n$ replication | Indexed instances, no implied connectivity |
| $H$ H-state tensor | RBS coordinates; see [RBS/RBD/HDP](rbs_rbd_hdp.md) |
| $h$ mutable state | Kernel carry (voltage, recovery, spikes, synapses, $H$, $w$) |
| $s$ static state | Topology, fixed parameters, geometry, constants after realization |
| Typed $x$ / $y$ | Declared boundary interfaces (stimulus/readout side) |
| Index map $\mathcal{I}$ | Path/slice/rule/edge correspondence in both directions |

## Deterministic rules

The compiler (`jaxfne.tfne`) fixes the minimum deterministic semantics: bare
$O$/$X$ without a rule or projection generates structure with zero edges;
braces affect paths, not edge sets; replication indexes without connecting;
proportions allocate by largest remainder; rules declare their direction;
exclusions veto rather than delete; normalization replays idempotently and
seeds realization. Details live in the module docstring.

## Naming scope

Bare $O$/$X$ are composition operators and $H$ is the H-state tensor.
Pipeline stages are referenced by full name (Emitter, Source, Field, Probe,
Objective, Optimizer, Manifest); runtime state keeps its meanings ($X$
activity, $W$ weights, $B$ delay history, $Q$ source quantity). There is no
structural $H$ and no architectural $Q$ or $C[k]$ beyond selection. Only $O$,
$X$ and $H$ are rejected as structural object names.

## See Also

- [TFNE Containment and Composition](tfne_containment_architecture.md) — operator factorization and RBS containers
- [RBS, RBD and HDP](rbs_rbd_hdp.md) — state, dynamics, and plasticity grammar
- [Computation Basis](../computation_basis.md) — canonical tensor shapes
- [Relative Quantities](relative_quantity_grammar.md) — configured/realized/executed distinction
