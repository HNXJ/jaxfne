# TFNE Algebra — canonical project source (7 of 7)

**Status:** `tfne/2` — `SEALED LANGUAGE`. Canonical authority for the TFNE
specification language, superseding the `tfne/1` text previously carried here.
Verbatim source below; implementation-scope notes and the current compiler
conformance boundary live in `docs/doctrine/tfne_algebra.md`, not here.

The language seal implies no parser or compiler. `jaxfne.tfne` is a compiler
against this language and does not define it: where the two disagree, this
file is the authority and the compiler carries the defect.

---

# TFNE Algebra

**Version:** `tfne/2`
**Status:** `SEALED LANGUAGE`

TFNE is a scale-invariant specification algebra for neural systems.

Its fundamental rule is:

$$
\boxed{\text{new nervous-system biology introduces definitions, not new grammar}.}
$$

The primitive vocabulary is frozen. It may be extended only if a concrete neural-system definition demonstrates a definition-level proof of insufficiency.

TFNE specifies neural systems; it is not a simulator.

---

# 1. Core form

A bounded TFNE system is

$$
\boxed{x:\mathcal A:y}
$$

where:

* \(x\) is a typed upstream representation;
* \(\mathcal A\) is the explicitly modeled neural system;
* \(y\) is a typed downstream representation.

Examples range from

$$
x:\mathrm{Cell}:y
$$

through

$$
x:V1:y
$$

to arbitrary multi-system nervous architectures.

The construction and execution boundary is

$$
\boxed{
\mathcal A
\xrightarrow{\mathrm{normalize}}
NF(\mathcal A)
\xrightarrow[\rho]{R,\mathrm{realize}}
(s,h_0,\mathcal I)
}
$$

followed by

$$
\boxed{
(h_t,x_t;s)\mapsto(h_{t+1},y_t).
}
$$

Here:

* \(NF(\mathcal A)\) is the deterministic normalized TFNE specification;
* \(R\) is the declared deterministic realization/allocation policy;
* \(\rho\) is any required stochastic realization identity or seed;
* \(s\) is complete execution-static state;
* \(h_0\) is complete initial mutable state;
* \(\mathcal I\) is the realization index/address map.

Configured specification, normalized architecture, realized model, and executed state are distinct.

---

# 2. Fundamental vocabulary

| Form           | Permanent meaning                                                 |
| -------------- | ----------------------------------------------------------------- |
| \(A,B\)        | Arbitrary named neural/structural objects                         |
| `Capital...`   | Object/type identity, e.g. `V1`, `LGN`, `CTX`, `L4`               |
| `lowercase...` | Value, parameter, process, rule, interface, or representation     |
| \(A:=E\)       | Define object \(A\) by expression \(E\)                           |
| \(\{E\}\)      | Composite structural object/boundary                              |
| `[E]`          | Typed selection, specialization, definition body, or rule binding |
| `;`            | Independent-entry separator                                       |
| `.`            | Hierarchical address/path only                                    |
| \(A^n\)        | \(n\) indexed instances of \(A\); no connectivity implied         |
| \(x\)          | Typed upstream representation                                     |
| \(y\)          | Typed downstream representation                                   |
| `:`            | Explicit-model boundary                                           |
| \(A\,O\,B\)    | Ordered architectural composition                                 |
| \(A\,O[k]\,B\) | Ordered composition under rule \(k\)                              |
| \(A\,X\,B\)    | Nonordered cross/lateral architectural composition                |
| \(A\,X[k]\,B\) | Cross/lateral composition under rule \(k\)                        |
| \(A>B\)        | Explicit left-to-right projection                                 |
| \(A<B\)        | Explicit right-to-left projection                                 |
| \(A<>B\)       | Explicit bidirectional projection                                 |
| \(A\not>B\)    | Explicit left-to-right exclusion                                  |
| \(A\not<B\)    | Explicit right-to-left exclusion                                  |
| \(H\)          | H-state tensor only                                               |
| \(h\)          | Complete mutable execution state                                  |
| \(s\)          | Complete execution-static representation                          |
| \(N[A]\)       | Integer cardinality of \(A\)                                      |
| \(P[A]\)       | Requested composition/proportion specification                    |
| \(G[A]\)       | Geometry specification                                            |
| \(C\)          | Declared cell-type domain                                         |
| \(L\)          | Conventional structural subdivision, not mandatory hierarchy      |
| `model`        | Dynamical realization of a biological object                      |

No fundamental \(Q\) operator exists in TFNE/2.

Lateral relations are specializations of \(X\):

$$
A\,X[lateral]\,B.
$$

---

# 3. Recursive neural objects

TFNE imposes no mandatory anatomical hierarchy.

An object may represent:

* compartment;
* cell;
* population;
* layer;
* nucleus;
* ganglion;
* cortical area;
* spinal segment;
* peripheral structure;
* multi-area system;
* whole nervous system.

Thus:

$$
\boxed{
\text{complexity is cardinality, definition, and composition rather than new grammar}.
}
$$

A nucleus need not contain layers.

A cortical object may.

A one-cell model remains valid without a special language.

---

# 4. Biological identity and dynamical realization

Biological identity is distinct from numerical dynamics.

For example:

$$
PV[model=IZH]
$$

and

$$
PV[model=HH]
$$

may be different dynamical realizations of the same biological class.

A cell definition may contain:

$$
\{
\text{biological type};
\text{compartments};
\text{static parameters};
\text{dynamic variables};
H;
\text{plastic variables};
\text{mechanisms}
\}.
$$

Consequently:

$$
\boxed{\text{biological definition}\neq\text{dynamical model}.}
$$

Izhikevich, HH, LIF, multicompartment models, or future emitter families require definitions, not new TFNE primitives.

---

# 5. Cardinality and composition

For neural object \(A\),

$$
N[A]\in\mathbb N_0.
$$

For member/type set \(C_A\),

$$
0\le P_A[c]\le1,
\qquad
\sum_{c\in C_A}P_A[c]=1.
$$

Requested proportions need not produce integer products.

Realized counts are determined by a declared deterministic allocation map \(R\):

$$
\boxed{
N[A.c]=R_A\!\left(P_A[c]N[A]\right)
}
$$

subject to

$$
\boxed{
\sum_{c\in C_A}N[A.c]=N[A].
}
$$

Therefore \(P\) expresses requested composition while \(N[A.c]\) records exact realized cardinality.

Example:

$$
N[A]=101,\qquad P=(0.8,0.2)
$$

may realize under an appropriate \(R\) as

$$
(81,20).
$$

Noninteger requested counts are not errors.

Missing required realization policy is an error.

---

# 6. Replication

$$
\boxed{
A^n=\{A.1,A.2,\ldots,A.n\}
}
$$

creates \(n\) indexed instances and no connectivity.

Thus

$$
x:SEG^8:y
$$

is a valid system containing eight disconnected segment instances.

Connectivity must be introduced separately.

For an ordered replicated sequence,

$$
\boxed{
O[k](SEG^8)
}
$$

means

$$
SEG.1\,O[k]\,SEG.2\,O[k]\cdots O[k]\,SEG.8.
$$

Replication therefore never silently implies adjacency.

---

# 7. Addressing

`.` means hierarchical address/path only.

Examples:

```text
V1.L4.E
V1.L4.E.soma
SEG.3
SEG.3.inter
```

Numeric path components address replicated instances.

An unresolved path is invalid:

```text
V1.L1.E
```

fails if `E` is absent from `V1.L1`.

The corresponding semantic failure is:

`E_ADDRESS_UNKNOWN`.

`.` never means multiplication or architectural composition.

---

# 8. Structural composites

$$
\{E\}
$$

creates a structural object boundary.

Braces affect:

* object identity;
* scope;
* addressing;
* exposed interfaces;
* future composition semantics.

They need not change the internal projections generated by \(E\).

Thus:

$$
A\,O\,B
$$

and

$$
\{A\,O\,B\}
$$

may contain identical internal \(A\)-to-\(B\) projections while remaining different TFNE objects.

Consequently:

$$
A\,O\,B\,O\,C
$$

and

$$
\{A\,O\,B\}\,O\,C
$$

are not generally equivalent because the latter composes through the composite object's exposed frontier.

---

# 9. Interfaces and frontiers

Every composable object has typed interfaces:

$$
A.in,\qquad A.out.
$$

`in` and `out` are reserved interface path components, not fundamental operators.

Definitions may explicitly declare:

```text
in := [...]
out := [...]
```

or allow them to be derived.

For an ordered composite,

$$
\boxed{
in(\{A\,O\,B\})=in(A)
}
$$

and

$$
\boxed{
out(\{A\,O\,B\})=out(B).
}
$$

More generally,

$$
in(\{A_1O\cdots OA_n\})=in(A_1),
$$

$$
out(\{A_1O\cdots OA_n\})=out(A_n).
$$

For an \(X\)-composite, the default interface is the union of compatible constituent interfaces unless \(X[k]\) explicitly defines another frontier.

If a required interface cannot be derived uniquely:

`E_FRONTIER_UNRESOLVED`.

---

# 10. Ordered composition \(O\)

$$
A\,O\,B
$$

means ordered architectural composition.

It does not itself specify projection direction, mechanism, weight, density, delay, plasticity, or geometry.

Those belong to the bound rule.

Example:

$$
A\,O[ff]\,B.
$$

A mixed-rule ordered chain is valid:

$$
A\,O[k]\,B\,O[j]\,C.
$$

It normalizes as an adjacency-labelled ordered sequence:

$$
(A,k,B,j,C).
$$

Therefore unbraced ordered chains are associative at the structural-sequence level.

Each rule applies only to its own adjacency.

Braces create a new structural object and therefore interrupt this flattening.

---

# 11. Cross/lateral composition \(X\)

$$
A\,X\,B
$$

means a nonordered cross/lateral architectural relation.

Its detailed semantics come from its rule:

$$
A\,X[k]\,B.
$$

Unlike \(O\), \(X\) is not globally associative.

Therefore:

$$
A\,X[k]\,B\,X[k]\,C
$$

requires grouping unless rule \(k\) explicitly declares an associative composition policy.

Lateral, callosal, cross-level, recurrent cross-structure, and similar architectural relations are definitions of \(X[k]\), not new primitives.

---

# 12. Connection rules

Rules are reusable architectural-to-projection maps.

Example:

```text
O[ff] := [
    $L.out >[mech=AMPA] $R.in
]
```

Within an \(O[k]\) or \(X[k]\) rule:

* `$L` is the syntactic left operand;
* `$R` is the syntactic right operand.

They are rule-bound metavariables, not neural objects.

A rule may independently specify:

$$
\boxed{
\text{topology},
\text{mechanism},
\text{parameters},
\text{geometry},
\text{delay}.
}
$$

Direction alone implies none of these.

---

# 13. Explicit projections

Projection direction is separate from architecture:

$$
A>B,
\qquad
A<B,
\qquad
A<>B.
$$

An executable projection must resolve its required mechanism from either:

* its generating rule; or
* its explicit projection specification.

Projection identity includes at least:

$$
\boxed{
(src,dst,mechanism).
}
$$

Weight, density, delay, plasticity, and related quantities are projection parameters and need not define projection identity.

An explicit projection identical to one already generated canonically is invalid:

`E_PROJECTION_REDUNDANT`.

A mechanism not permitted on a route is invalid:

`E_MECHANISM_NOT_PERMITTED`.

A required but unresolved mechanism is invalid:

`E_MECHANISM_UNRESOLVED`.

---

# 14. Projection exclusions

Let canonical rule expansion produce projection set

$$
G_0.
$$

Resolve explicit exclusions into projection identities:

$$
E_-.
$$

Require:

$$
\boxed{
E_-\subseteq G_0.
}
$$

Then:

$$
G=G_0\setminus E_-.
$$

An exclusion matching no generated projection is invalid:

`E_EXCLUSION_UNKNOWN`.

If explicit additions form \(E_+\), require:

$$
\boxed{
E_+\cap G_0=\varnothing.
}
$$

Thus stale exclusions and redundant additions cannot silently survive normalization.

Statements are atomic: a statement with an invalid resolved projection contributes nothing.

---

# 15. Geometry

$$
G[A]
$$

is first-class structural specification.

It may describe, for example:

* point geometry;
* 1D paths;
* 2D sheets;
* 3D volumes;
* cortical depth;
* axonal trajectories;
* multicompartment morphology;
* electrode/contact geometry.

Geometry is neither connectivity nor dynamics.

When fixed for an execution, realized geometry ultimately contributes to \(s\).

This permits TFNE to represent both point-neuron systems and systems requiring morphology or field calculations without new architectural operators.

---

# 16. State algebra

\(H\) is reserved exclusively for the H-state tensor.

It is never a structural cortical unit.

The complete mutable execution state is

$$
\boxed{
h=
\{
h_{\mathrm{dyn}},
h_H,
h_{\mathrm{plastic}},
h_{\mathrm{history}},
h_{\mathrm{rng}}
\}.
}
$$

Where:

* \(h_{\mathrm{dyn}}\): ordinary dynamic state;
* \(h_H\): H-state values;
* \(h_{\mathrm{plastic}}\): mutable parameters;
* \(h_{\mathrm{history}}\): continuation/delay/history state;
* \(h_{\mathrm{rng}}\): stochastic continuation state.

Optional additional typed mutable state may be included when required by a model.

The defining invariant is continuation sufficiency:

$$
\boxed{
(s,h_t,x_{t:})
}
$$

with deterministic RNG semantics contains all information required to reproduce future execution.

---

# 17. Plasticity

Plasticity is not restricted to synaptic weight.

Any declared mutable parameter

$$
\theta\in h_{\mathrm{plastic}}
$$

may evolve through an explicit state-dependent rule.

Schematically,

$$
\Delta\theta
=
\mathcal P(H,X,\theta,\ldots).
$$

Thus STDP, homeostasis, intrinsic plasticity, synaptic scaling, efficacy modulation, and other declared mechanisms are definitions within TFNE rather than new algebra.

---

# 18. Static representation

\(s\) contains quantities fixed during an execution after realization.

This may include:

* realized fixed topology;
* fixed parameters;
* geometry;
* constants;
* fixed model configuration;
* immutable lookup/index structures.

However,

$$
\boxed{
\text{TFNE specification}\neq s.
}
$$

Instead:

$$
\boxed{
NF(\mathcal A)
\xrightarrow[\rho]{R,\mathrm{realize}}
(s,h_0,\mathcal I).
}
$$

Construction and execution remain distinct.

---

# 19. Typed boundaries

\(x\) and \(y\) are interfaces rather than specific physical quantities.

Examples include:

```text
x[type=current]
x[type=spikes]
x[type=sensory]
x[type=field]
y[type=spikes]
y[type=LFP]
y[type=muscle]
y[type=behavior]
```

Thus:

$$
x:\mathcal A:y
$$

may bound a single-cell experiment, a neural circuit, or a whole sensorimotor system.

Omitted upstream dependencies may be reduced into \(x\).

Omitted downstream consequences may be reduced into \(y\).

---

# 20. Canonical ordering

Source declaration order does not determine realization indexing.

TFNE uses deterministic typed natural ordering.

For example:

$$
L1<L2<L10
$$

and

$$
SEG.2<SEG.10.
$$

Canonical path components are compared by typed natural keys rather than raw lexical order.

If a biological definition explicitly declares meaningful order, that order becomes part of \(NF(\mathcal A)\) and overrides generic natural ordering.

---

# 21. Anonymous composites

An anonymous composite

$$
C=\{E\}
$$

has a canonical structural serialization derived from

$$
NF(E).
$$

That serialization determines deterministic structural ordering.

Its compact structural identity receipt may be:

$$
hash(serialization(NF(E))).
$$

The hash is not itself the traversal key.

Two structurally identical anonymous composites may share structural identity while remaining distinct instances through deterministic occurrence indices.

Therefore:

$$
\boxed{
\text{structural identity}
\neq
\text{instance identity}
\neq
\text{realization identity}.
}
$$

---

# 22. Realization index map

$$
\mathcal I
$$

preserves correspondence between TFNE identities and flattened numerical realization.

It must support, where applicable:

* TFNE path \(\rightarrow\) realized tensor/slice/index;
* realized index \(\rightarrow\) TFNE path;
* TFNE projection/rule \(\rightarrow\) realized edges;
* realized edge \(\rightarrow\) originating TFNE relation/rule;
* parameter/state targeting by TFNE identity;
* inspection of realized \(N,P,G\), parameters, states, and connectivity.

Example:

$$
\mathcal I[V1.L4.E]=[i_0,i_1).
$$

For fixed

$$
NF(\mathcal A),R,\rho,
$$

\(\mathcal I\) must be deterministic.

Flattening may change representation but never TFNE semantics:

$$
\boxed{
\text{flattening changes representation, not meaning}.
}
$$

---

# 23. Normalization

A TFNE source passes through:

$$
\boxed{
\begin{aligned}
\text{source}
&\rightarrow \text{parse}\\
&\rightarrow \text{type/address resolution}\\
&\rightarrow \text{replication expansion}\\
&\rightarrow \text{structural normalization}\\
&\rightarrow \text{frontier resolution}\\
&\rightarrow O/X\text{-rule expansion}\\
&\rightarrow G_0\\
&\rightarrow \text{exception resolution}\\
&\rightarrow NF(\mathcal A)\\
&\xrightarrow[\rho]{R,\mathrm{realize}}
(s,h_0,\mathcal I).
\end{aligned}
}
$$

Each stage fails locally and explicitly.

A valid fully specified TFNE expression has exactly one normalized structural meaning under fixed definition/motif versions.

---

# 24. Identity and hashing

Canonical serialization is authoritative.

Hashes are receipts.

At minimum distinguish:

$$
h_T=\operatorname{hash}(NF_{\mathrm{topology}})
$$

from parameter/model/realization identities where needed.

Architecture identity must not silently depend on runtime realization details.

Conversely:

$$
NF(\mathcal A_1)=NF(\mathcal A_2)
$$

does not imply identical numerical realizations unless realization policy and stochastic realization identity also agree.

---

# 25. Minimum semantic failures

TFNE/2 requires at least the failure classes demonstrated by its conformance corpus, including:

* unknown address;
* unresolved frontier;
* unresolved mechanism;
* mechanism conflict/not permitted;
* redundant projection;
* unknown exclusion;
* missing realization/allocation policy;
* ambiguous/nonunique expansion;
* invalid proportion/type specification.

Error vocabulary is semantic rather than parser-specific.

Invalid expressions must fail rather than silently normalize to a different nervous system.

---

# 26. Universality examples

## Single HH cell

```text
Cell := [...]
Cell[model=HH]
N[Cell] = 1

x[type=current] : Cell : y[type=voltage]
```

No special single-cell grammar is required.

## Nonlaminar nucleus

```text
C := {relay, inter}
LGN := [...]
N[LGN] = 1000
P_LGN[relay] = 0.8
P_LGN[inter] = 0.2
```

No artificial cortical layer is required.

## Cortical unit

```text
CTX[v1] := [...]
V1 := CTX[v1]
```

`CTX` is a biological definition family, not a fundamental TFNE symbol.

It may define cortical layers, populations, local connectivity, models, parameters, geometry, and interfaces.

## Replicated spinal chain

```text
SEG := [...]
N[SEG] = 500

CORD := O[spinal](SEG^8)
```

This creates eight indexed segments and seven ordered adjacencies.

## Heterogeneous nervous system

```text
VIS := {V1 X[callosal] V1_R}

SYS := {
    Retina O[retino] LGN O[thalamo] VIS
}

x[type=sensory] : SYS : y[type=neural]
```

Different anatomical organizations compose using the same algebra.

---

# 27. CTX definition layer

TFNE/2 does not define cortex as grammar.

A future canonical cortical family may be expressed as:

$$
CTX[k]:=[\ldots].
$$

Its definition should separately establish:

* **D4a:** populations and \(P_{\ell,c}\), including absent populations;
* **D4b:** \(N\)-scaling and allocation;
* **D4c:** `in`/`out` interfaces;
* **D4d:** local connectivity;
* **D4e:** biological identities and dynamical realizations;
* **D4f:** geometry/morphology required for observables.

Thus:

$$
CTX[IZH]
$$

and

$$
CTX[HH]
$$

may eventually be alternative realizations of common cortical biology rather than different TFNE grammars.

---

# 28. Spectrolaminar qualification is outside the algebra

A candidate `CTX` may later be scientifically qualified through separate gates:

$$
CTX\rightarrow SL0
$$

for native network generators,

$$
F_{\mathrm{field}}\rightarrow SL1
$$

for a physical extracellular-field forward model,

and

$$
SL0+SL1\rightarrow SL2
$$

for the empirical spectrolaminar phenotype.

SL0 concerns native circuit observables, not LFP.

SL1 concerns physical forward modeling.

SL2 compares the composed result with empirical laminar spectral observations.

Phenomenological visualization proxies must not be used as the optimization objective defining `CTX`.

These are scientific qualification requirements, not TFNE language semantics.

---

# 29. JaxFNE boundary

TFNE is a specification language targeting JaxFNE realization/execution, not a second simulator.

The intended boundary is:

$$
\boxed{
\text{TFNE}
\rightarrow
NF(\mathcal A)
\rightarrow
\text{explicit typed neural model}
\rightarrow
(s,h_0,\mathcal I)
\rightarrow
\text{JaxFNE execution}.
}
$$

JaxFNE may use flat, vectorized, indexed, JIT-compatible numerical structures.

No recursive Python traversal of the TFNE hierarchy is required inside the simulation loop.

Compiler limitations must not redefine TFNE biology.

Likewise, TFNE definitions must not imply that a JaxFNE capability has been qualified merely because the language can express it.

Configured, realized, executed, and causally effective mechanisms remain distinct.

---

# 30. Permanent invariants

$$
\boxed{\text{new biology requires new definitions, not new grammar}.}
$$

$$
\boxed{
x:\mathcal A:y
}
$$

$$
\boxed{
\mathcal A
\xrightarrow{\mathrm{normalize}}
NF(\mathcal A)
\xrightarrow[\rho]{R,\mathrm{realize}}
(s,h_0,\mathcal I)
}
$$

$$
\boxed{
(h_t,x_t;s)\mapsto(h_{t+1},y_t)
}
$$

$$
\boxed{
\text{structural identity}
\neq
\text{instance identity}
\neq
\text{realization identity}
}
$$

$$
\boxed{
A^n\text{ creates instances, not connectivity}
}
$$

$$
\boxed{
O,X\text{ specify architectural relations; }>,<,<>\text{ specify projection direction}
}
$$

$$
\boxed{
\text{flattening changes representation, not TFNE semantics}
}
$$

$$
\boxed{
h_t\text{ is continuation-sufficient under fixed }s,x_{t:}\text{ and RNG semantics}
}
$$

$$
\boxed{
\text{TFNE/2 primitives are frozen unless a concrete definition proves insufficiency}
}
$$

---

# 31. Status

`TFNE/2 LANGUAGE = SEALED`

The language layer is complete under the current conformance corpus.

The next TFNE work belongs to separate layers:

$$
\boxed{
\text{TFNE algebra}
\rightarrow
\text{biological definition library}
\rightarrow
\text{realization/compiler}
\rightarrow
\text{scientific qualification}.
}
$$

Current next-definition candidate:

$$
CTX.
$$

No parser or compiler is implied by the language seal.
