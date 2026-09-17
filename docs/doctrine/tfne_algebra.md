# TFNE Algebra

**Language version:** `tfne/2` — `SEALED LANGUAGE`
**Canonical source:** `artifacts/project_sources/7_tfne_algebra.md`
**Compiler:** `jaxfne.tfne` (specification-time; exercised by
`tests/test_tfne_algebra.py`)

The language seal implies no parser or compiler. `jaxfne.tfne` targets this
language and does not define it: where the two disagree, the project source is
the authority and the compiler carries the defect. The current gap is recorded
under [compiler conformance](#compiler-conformance) rather than resolved by
restating the language to match the code.

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
| Projection $>$, $<$, $<>$ / exclusion $\not>$, $\not<$ | Directed edge groups with rule provenance; exclusions subtract from the rule expansion (see [compiler conformance](#compiler-conformance)) |
| $A^n$ replication | Indexed instances, no implied connectivity |
| $H$ H-state tensor | RBS coordinates; see [RBS/RBD/HDP](rbs_rbd_hdp.md) |
| $h$ mutable state | Kernel carry (voltage, recovery, spikes, synapses, $H$, $w$) |
| $s$ static state | Topology, fixed parameters, geometry, constants after realization |
| Typed $x$ / $y$ | Declared boundary interfaces (stimulus/readout side) |
| Index map $\mathcal{I}$ | Path/slice/rule/edge correspondence in both directions |

## Compiler conformance

`jaxfne.tfne` was built against `tfne/1` and has not yet been brought to
`tfne/2`. The clauses below are the measured difference between the sealed
language and the compiler, so that neither file silently stands in for the
other.

### Satisfied

Core form `x:A:y` and the execution normal form; recursive scale invariance
down to one cell; biological identity separate from `model`; exact realized
cardinality with `sum_c N[A.c] == N[A]`; replication that implies no
connectivity; geometry compiled into `s`; `H` reserved for the H-state tensor;
typed `x`/`y` boundaries; a realization index map supporting path->slice,
slice->path, rule->edges and edge->rule; idempotent normalization whose digest
seeds realization; hashes used as receipts rather than traversal keys.

### Not yet conformant

| Clause | `tfne/2` requires | `jaxfne.tfne` today |
|---|---|---|
| S6, S7 replication addressing | `A^n = {A.1 ... A.n}` | realizes `A.0 ... A.(n-1)` |
| S6 prefix rule application | `O[k](SEG^8)` expands to seven ordered adjacencies | `(` is a parse error |
| S8 group sensitivity | `{A O B} O C` composes through the composite frontier and is not generally equivalent to `A O B O C` | both realize identical edge sets |
| S9 frontiers | `in`/`out` are reserved interface components with derived defaults; `E_FRONTIER_UNRESOLVED` | no frontier concept; `A.out` resolves to no object |
| S11 cross associativity | ungrouped `A X[k] B X[k] C` requires grouping unless `k` declares an associative policy | accepted ungrouped |
| S12 rule binding | `$L` / `$R` metavariables bind the syntactic operands | `$` is a lexer error; rules carry flat parameter maps |
| S13 projection identity | redundant explicit projection is invalid | no redundancy check over `(src, dst, mechanism)` |
| S14 exclusions | `E_- subset G_0`, then `G = G_0 \ E_-`; an exclusion matching nothing is invalid | inverted: a matched exclusion raises, an unmatched exclusion is a silent no-op |
| S20 canonical ordering | typed natural order (`L1 < L2 < L10`) independent of declaration order | realization indexes in declaration order |
| S14, S25 statement atomicity | `;`-separated statements inside a composite, each atomic | `;` inside `{...}` is a parse error |
| S25 failure vocabulary | semantic classes (`E_ADDRESS_UNKNOWN`, `E_FRONTIER_UNRESOLVED`, `E_MECHANISM_UNRESOLVED`, `E_MECHANISM_NOT_PERMITTED`, `E_PROJECTION_REDUNDANT`, `E_EXCLUSION_UNKNOWN`) | untyped `TFNEError` with prose messages |

Two of these change realized meaning rather than surface syntax. Exclusion
direction is inverted, so a specification that `tfne/2` reads as "build the
rule expansion, then remove this projection" is currently read as "fail if this
projection exists". Instance addressing is off by one, so every replicated path
in `I` names a different instance than the language does. Neither may be
treated as a cosmetic difference.

### Rules the compiler fixes where the language leaves room

Bare `O`/`X` with no rule and no projection contribute structure and generate
zero edges; edges come only from explicit projections and rule applications.
Proportion-to-count allocation is largest-remainder with declaration-order
tiebreak, supplying the deterministic `R` of S5. Rule applications and bare
projections require an explicit direction; the compiler raises rather than
guessing one.

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
