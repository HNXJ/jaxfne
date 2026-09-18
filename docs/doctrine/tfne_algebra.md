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
`relation_origin` and does not alter the realized edges. A declared `delay` is
refused with `E_PARAM_UNSUPPORTED` because no execution path consumes one —
there is no delay field on the connection-rule surface — so it fails closed
instead of being silently dropped.

Mechanism identity transfers; mechanism kinetics does not. A declared
mechanism reaches the executed edges as a name, a receptor index and an
excitatory/inhibitory split, but its synaptic time constant is the structural
bridge's per-connection `dT_ms` rather than the named receptor's own value, so
a declared `AMPA` edge executes at 0.1 ms and not the 2.0 ms of
`standard_receptor_specs()`. `tfne/2` rule bodies cannot declare a tau, so no
declared value is being substituted; the realized mechanism table records this
as `tau_ms: None` with status `declared_not_simulated`. Synaptic kinetics is
independent of `dt`, so refining the timestep integrates the same synapse
model rather than changing it.

Declared geometry is realized but not executed. `G` is recorded in
`s["geometry"]`, and at equal seed the executed positions are identical
whatever range is declared, so the declaration is inert rather than rescaled.
Geometry is what field observables are computed against, so a field or LFP
claim currently rests on coordinates the specification did not choose. Whether
absolute or relative coordinates are intended is undecided.

`to_neuronal_tensor()` remains the structural bridge and is still used for
areas, layers and cell types, but it cannot carry connection parameters:
`InterConnection` and `AreaConnection` have no weight, probability or delay
field. That is why execution goes through the resolved specs rather than
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
S9 derived defaults S10 needs); geometry compiled into `s` (realized only —
it does not reach the executed positions); `H` reserved for the H-state
tensor; typed `x`/`y` boundaries; a realization index map supporting path->slice, slice->path,
rule->edges and edge->rule; idempotent normalization whose digest seeds
realization; hashes used as receipts rather than traversal keys; declared
connection parameters carried from specification to execution without
substitution, with an unsupported parameter refused rather than dropped (see
[Pipeline](#pipeline)); typed natural ordering of canonical paths, so source
declaration order does not determine realization indexing (S20).

### Not yet conformant

| Clause | `tfne/2` requires | `jaxfne.tfne` today |
|---|---|---|
| S6 prefix rule application | `O[k](SEG^8)` expands to seven ordered adjacencies | `(` is a parse error |
| S8 group sensitivity | `{A O B} O C` composes through the composite frontier and is not generally equivalent to `A O B O C` | composes through the frontier correctly; the two still coincide in edge set for ordered chains, and no declared frontier can yet make them differ |
| S9 frontiers | `in`/`out` reserved as interface path components, declarable, with an X-composite union default and `E_FRONTIER_UNRESOLVED` | derived ordered defaults only (used by S10); `A.out` as a written path resolves to no object |
| S11 cross associativity | ungrouped `A X[k] B X[k] C` requires grouping unless `k` declares an associative policy | accepted ungrouped |
| S12 rule binding | `$L` / `$R` metavariables bind the syntactic operands | `$` is a lexer error; rules carry flat parameter maps |
| S13 projection identity | redundant explicit projection is invalid | no redundancy check over `(src, dst, mechanism)` |
| S20 declared order override | a biological definition that declares meaningful order overrides generic natural ordering | typed natural ordering is implemented; the override is not, because the sealed language specifies no syntax for declaring one |
| S14, S25 statement atomicity | `;`-separated statements inside a composite, each atomic | `;` inside `{...}` is a parse error |
| S25 failure vocabulary | semantic classes (`E_ADDRESS_UNKNOWN`, `E_FRONTIER_UNRESOLVED`, `E_MECHANISM_UNRESOLVED`, `E_MECHANISM_NOT_PERMITTED`, `E_PROJECTION_REDUNDANT`, `E_EXCLUSION_UNKNOWN`) | untyped `TFNEError` with prose messages |

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

What remains is the override. S20 says a biological definition that "explicitly
declares meaningful order" overrides natural ordering, but the sealed language
mentions this once and gives no syntax for declaring such an order — and its
own opening sentence, that source declaration order does not determine
indexing, rules out reading an enumeration as the declaration. Until that is
settled the override is unexpressible, so nothing can currently override
natural ordering. This is a gap in the language rather than the compiler.

S9 is listed as outstanding because only its derived defaults exist. Ordered
composites expose `in(A)` and `out(B)`, which is what S10 adjacency requires;
declared `in := [...]` / `out := [...]` bodies, the X-composite union default
and `E_FRONTIER_UNRESOLVED` are not implemented.

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
