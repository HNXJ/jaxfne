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

TFNE constrains, possibly underdetermined; completion under `D + K_D`
(JDNA), realization (Model), and execution are separate stages — see the
[TFNE–JDNA boundary](tfne_jdna_boundary.md). The pipeline above is the
language-internal view; the compiler path runs through JDNA completion
before construction.

One resolved representation, two consumers. `realize()` resolves the source
once and records the result in $\mathcal{I}$ as `connection_specs`;
`to_configuration()` builds the executable `Configuration` from those same
resolved specs, so inspection and execution read one resolution rather than
compiling the source twice. Topology, identity and declared connection
parameters therefore agree, verified edge for edge as multisets of
`(pre, post, weight, mechanism)` in `tests/test_tfne_parameter_transfer.py`.

Executable parameter ownership is explicit. A declared `weight`, `probability`,
`mechanism` and `direction` reach the kernel: the synaptic weight the
integrator uses is `_resolved_edge_weight(edge_list, dtype, emitter)`, and it
equals the realized `s["edge_weight"]`. A declared `plasticity` is a rule
identity for the separate registrable HDP surface rather than an edge
parameter, so it is preserved as inspectable provenance in `rule_params` and
`relation_origin` and does not alter the realized edges. A declared `delay`
(in ms) reaches the kernel as integer `delay_steps = round(delay_ms / dt_ms)`
at the construction timestep (0.5.2 decision 0b): configured ms and realized
steps are both recorded, and a positive delay rounding to 0 steps is refused
rather than dropped. Negative or non-numeric delays are refused at
realization.

Mechanism identity transfers, and mechanism kinetics now resolves with
it. A declared mechanism reaches the executed edges as a name, a receptor
index and an excitatory/inhibitory split, and its synaptic time constant
resolves through the mechanism vocabulary (`resolve_mechanism`): canonical
receptors execute at their canonical taus (AMPA 2.0, GABA_A 5.0, NMDA
100.0, GABA_B 150.0 ms), custom mechanisms at their declared `tau_ms`,
and anything unresolvable is refused at execution rather than inheriting
a placeholder. A rule without a mechanism means direct coupling
(`tfne_direct`, placeholder 0.1 ms, not a receptor claim). The realized
mechanism table still records `tau_ms: None` with status
`declared_not_simulated`: identity is realized, kinetics resolve
downstream. Synaptic kinetics is independent of `dt`, so refining the
timestep integrates the same synapse model rather than changing it.

Declared geometry is realized and executed as relative coordinates. `G`
is recorded in `s["geometry"]`, and each declared range is a pair of
fractions of the sampled (area, layer) block's extent within [0,1]: a
declared sub-range changes the executed positions (sampled inside it),
while an absent declaration — or a full [0,1] range — takes the
historical path bit-identically. A range outside [0,1] is refused
(`E_GEOMETRY_OUT_OF_RANGE`), never rescaled: relative coordinates carry
no physical (mm/um/conductivity/distance) semantics. Geometry is what
field observables are computed against, so field claims rest on the
manifest-recorded fractional domains (`tfne_geometry`).

`to_neuronal_tensor()` remains the structural bridge and is still used for
areas, layers and cell types, but it cannot carry connection parameters:
`InterConnection` and `AreaConnection` have no weight or probability
field (declared delays ride the bridge as `delay_ms` for inspection only).
That is why execution goes through the resolved specs rather than
through the tensor alone, and why the Execute row means the kernels consume the
realized connectivity, not merely a tensor rebuilt from the same text.

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
connectivity, indexed `A.1 ... A.n` (S6, S7); exclusions that subtract from the
rule expansion, with an unmatched exclusion rejected rather than ignored (S14);
ordered rules bound to their own adjacency, with composites composing through
derived `in`/`out` frontiers rather than through every member (S10, and the
S9 derived defaults S10 needs); geometry compiled from `s` into the executed
positions as fractional domains of each sampled block, with outside-[0,1]
ranges refused (0.5.2 decision 0a); `H` reserved for the H-state
tensor; typed `x`/`y` boundaries; a realization index map supporting path->slice, slice->path,
rule->edges and edge->rule; idempotent normalization whose digest seeds
realization; hashes used as receipts rather than traversal keys; declared
connection parameters carried from specification to execution without
substitution, with an unsupported parameter refused rather than dropped (see
[Pipeline](#pipeline)); typed natural ordering of canonical paths, so source
declaration order does not determine realization indexing, with
`order[A] := [...]` as the only override and a declaration that cannot be
honoured exactly refused rather than partially applied (S20, S20.1);
declared `in[A] := [...]` / `out[A] := [...]` composition frontiers naming
immediate-member subsets, each side overriding its derived default
independently, with anything unhonourable refused as
`E_FRONTIER_UNRESOLVED` (S9); rule bodies over `$L`/`$R` with per-side
scope addressing and per-statement mechanism, with flat rules keeping
exact legacy behavior (S12).

### Not yet conformant

| Clause | `tfne/2` requires | `jaxfne.tfne` today |
|---|---|---|
| S6 prefix rule application | `O[k](SEG^8)` expands to seven ordered adjacencies | `O[k](SEG^n)` chains per-instance adjacencies with head/tail frontiers (`(`/`)` lexed); `X[k](...)` and non-replication targets refused |
| S8 group sensitivity | `{A O B} O C` composes through the composite frontier and is not generally equivalent to `A O B O C` | composes through the frontier correctly; derived-only chains still coincide in edge set, and a declared frontier makes them differ |
| S9 frontiers | `in`/`out` reserved as interface path components, declarable, with an X-composite union default and `E_FRONTIER_UNRESOLVED` | declared `in[A]`/`out[A]` supported with per-side override and fail-closed validation; X-composite union default kept; `X[k]`-rule frontier override still needs rule bodies (TFNE2-05); `A.out` as a written path still resolves to no object |
| S11 cross associativity | ungrouped `A X[k] B X[k] C` requires grouping unless `k` declares an associative policy | refused as `E_AMBIGUOUS_EXPANSION` without grouping or `associative = true`; braces, definitions, and differing rules associate as before |
| S12 rule binding | `$L` / `$R` metavariables bind the syntactic operands | bodies supported (endpoints, collections, bidirectionality, per-statement mechanism); full `$L`/`$R` path tails beyond `.out`/`.in` still refused |
| S13 projection identity | redundant explicit projection is invalid | explicit projections overlapping rule output at leaf identity refused as `E_PROJECTION_REDUNDANT` |
| S14, S25 statement atomicity | `;`-separated statements inside a composite, each atomic | `{s1; s2}` expands each statement independently; invalid resolved projections contribute nothing while top-level ones still abort |
| S25 failure vocabulary | semantic classes (`E_ADDRESS_UNKNOWN`, `E_FRONTIER_UNRESOLVED`, `E_MECHANISM_UNRESOLVED`, `E_MECHANISM_NOT_PERMITTED`, `E_PROJECTION_REDUNDANT`, `E_EXCLUSION_UNKNOWN`) | implemented as `TFNEError` subclasses (plus `E_ORDER_*`, proportion/policy, and `E_AMBIGUOUS_EXPANSION` families); parser errors stay plain `TFNEError` |

The entries left are missing features rather than wrong answers: the compiler
rejects what it cannot express instead of realizing a different nervous system.
The three clauses that did change realized biology — inverted exclusions,
off-by-one instance addressing, and ordered chains that fabricated projections
the source never declared — are closed.

Declared rule parameters are no longer among the gaps. They were: a declared
`weight` of 0.5, 0.25 or 0.125 all executed at 0.353553, and a declared
`probability` of 0.5 realized 8 edges while executing 16, because `realize()`
and `to_neuronal_tensor()` were two independent compilations of one source.
Execution now consumes the single resolved representation, and the equivalence
is a dev-gate module rather than a claim in this page.

S20 is split. Typed natural ordering is implemented at one chokepoint, so
realization and `to_neuronal_tensor` cannot drift apart on it: `L1 < L2 < L10`
and `SEG.2 < SEG.10`, and a parent sorts before its children because its key is
a proper prefix. Cell types keep the order their `C = {...}` enumeration gives
them; only object paths are reordered.

The override is `order[A] := [...]`, defined by S20.1. It was previously a gap
in the language rather than the compiler: the sealed text said a definition may
"explicitly declare meaningful order" but gave no syntax, while its own opening
sentence ruled out reading an enumeration as that declaration. S20.1 now
settles it — ordering is metadata, so it takes a named property rather than a
new operator, and no other construct carries ordering.

A declaration is refused unless it names every immediate member of its scope
exactly once: `E_ORDER_INCOMPLETE`, `E_ORDER_DUPLICATE_MEMBER`,
`E_ORDER_MEMBER_UNKNOWN`, `E_ORDER_NOT_IMMEDIATE`, `E_ORDER_SCOPE_UNKNOWN`,
`E_ORDER_SCOPE_AMBIGUOUS`, `E_ORDER_DUPLICATE`. Ordering a partial or
contradictory declaration would index some members by declaration and the rest
by another rule. A bare scope name may address a nested object only when it
does so unambiguously.

Cell types still keep their enumeration order. `tfne/2` S20.1 states that
implicit source order never carries scientific semantics and that this is
retained only as temporary compatibility behaviour, so it remains a known
exception rather than a settled rule.

S9 declared frontiers are implemented: `in[A] := [...]` / `out[A] := [...]`
name immediate-member subsets (replica-aware), each side overriding its
derived default independently, with unhonourable declarations refused as
`E_FRONTIER_UNRESOLVED`. Ordered composites still expose `in(A)` and
`out(B)` by default, which is what S10 adjacency requires; the X-composite
union default is kept. Still open: an `X[k]`-rule frontier override (needs
TFNE2-05 rule bodies), and `A.out` as a written path still resolves to no
object.

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
