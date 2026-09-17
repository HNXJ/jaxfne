# TFNE Algebra — canonical project source (7 of 7)

**Status:** canonical algebra for TFNE specification language (adopted dev; see `docs/doctrine/tfne_algebra.md` for the implementation mapping). Verbatim source below; implementation-scope notes live in the doctrine page, not here.

---

## Core form

$$
\boxed{x:\mathcal{A}:y}
$$

A TFNE model is an explicitly modeled neural system $\mathcal{A}$ bounded by an upstream representation $x$ and a downstream representation $y$.

After realization,

$$
\mathcal{A} \rightarrow (s,h_0)
$$

and execution has the normal form

$$
\boxed{(h_t,x_t;s)\mapsto(h_{t+1},y_t)}.
$$

## Definitions

| Form | Definition |
|---|---|
| $A,B$ | Arbitrary neural tensor objects. |
| `Capital...` | Named tensor or structural object, e.g. `V1`, `LGN`, `L1`. |
| `lowercase...` | Value, process, parameter, rule name, or representation. |
| $A:=E$ | Define $A$ by expression $E$. |
| $\{E\}$ | Composite tensor. Creates and preserves a structural boundary. |
| $[E]$ | Typed definition, selection, or specialization body; meaning is determined by the object to which it is bound. |
| `;` | Separates independent entries within a definition. |
| `.` | Hierarchical address/path only, e.g. `A.L4.E.soma`. |
| $[k]$ | Named specialization or connection rule. |
| $A^n$ | $n$ indexed instances of $A$, with no connectivity implied. |
| $x$ | Typed upstream representation of dependencies not explicitly modeled. |
| $y$ | Typed downstream representation, including neural output, probe, field readout, motor output, behavior, or another declared output. |
| $x:A$ | Upstream boundary: explicit modeling begins at $A$. |
| $A:y$ | Downstream boundary: explicit modeling ends at $A$ relative to $y$. |
| $x:A:y$ | Explicit neural system $A$ between reduced upstream $x$ and downstream $y$. |
| $A\,O\,B$ | Ordered architectural composition. Does not itself imply projection direction. |
| $A\,O[k]\,B$ | Ordered composition using connection rule $k$. |
| $A\,X\,B$ | Nonserial, cross, or lateral architectural composition. Does not itself imply projection direction. |
| $A\,X[k]\,B$ | Cross/lateral composition using connection rule $k$. |
| $O[k]:=[E]$ | Define ordered connection rule $k$. |
| $X[k]:=[E]$ | Define cross/lateral connection rule $k$. |
| $A>B$ | Left-to-right projection. |
| $A<B$ | Right-to-left projection. |
| $A<>B$ | Bidirectional projection: $A>B$ and $A<B$. |
| $A\not>B$ | Explicit exclusion of a left-to-right projection. |
| $A\not<B$ | Explicit exclusion of a right-to-left projection. |
| $H$ | H-state tensor only. |
| $h$ | Complete mutable execution-state representation. |
| $s$ | Complete execution-static representation after realization. |
| $C$ | Declared cell-type domain, e.g. $C:=\{E,PV,SST,VIP\}$. |
| $L$ | Conventional structural subdivision; cortical layers are one specialization, not a mandatory universal hierarchy. |
| $L[C]$ | Cell-type subtensor $C$ within structural subdivision $L$. |
| $P[A]$ | Declared composition/proportion vector for object $A$. |
| $P_A[c]$ | Declared proportion of member/type $c$ within $A$. |
| $N[A]$ | Integer cardinality of neural object $A$. |
| $G[A]$ | Geometry specification of $A$. |
| `model` | Dynamical realization assigned to a biological object; biological identity and dynamical model are distinct. |

## Properties of definitions

| Property | Definition |
|---|---|
| Recursive scale | The same algebra applies recursively from cell substructure to cell, population, structural subdivision, area/nucleus, multi-area system, and whole nervous system. |
| Structural closure | A valid TFNE expression may be named and reused as a tensor object. |
| Composite preservation | Braces are semantic boundaries: $\{E\}$ is not silently flattened into $E$. |
| Group sensitivity | $A\,O\,B\,O\,C$, $\{A\,O\,B\}\,O\,C$, and $A\,O\,\{B\,O\,C\}$ are not generally equivalent. |
| Direction independence | $O$ and $X$ specify architectural relations; $>,<,<>$ specify projection direction. |
| Rule independence | A connection rule may independently define topology, mechanism, parameters, geometry, and delay. Direction does not imply excitation, inhibition, mechanism, density, weight, delay, or plasticity. |
| Replication independence | $A^n$ creates indexed instances only; relations among them require explicit composition or connection rules. |
| Typed brackets | `[]` is disambiguated by its bound object: selection in $L[C]$, rule binding in $O[k]$/$X[k]$, or declared specialization in $A[k]$. |
| Path semantics | `.` is hierarchical addressing only and never multiplication or architectural composition. |
| Optional hierarchy | Neural structures need not be cortical or laminar. A nucleus may directly contain populations; a minimal model may contain one object, one population, and one cell. |
| Degenerate validity | The algebra remains valid at cardinality one. No separate single-cell language is required. |
| Biological/model separation | Cell or population identity is distinct from its dynamical realization; the same biological type may use HH, Izhikevich, LIF, or another declared model. |
| Proportion normalization | For a declared member/type set $C_A$, $0\leq P_A[c]\leq1$ and $\sum_{c\in C_A}P_A[c]=1$. |
| Exact cardinality | Realized member counts are integers and satisfy $\sum_c N[A.c]=N[A]$. A declared deterministic allocation map converts proportions into integer counts. |
| Geometry separation | $G[A]$ is first-class structural specification but compiles into execution-static state when fixed. Geometry is not connectivity or dynamics. |
| Static state | $s$ contains all quantities fixed during an execution after realization, including applicable realized topology, fixed parameters, geometry, and constants. |
| Mutable state | $h$ contains every quantity required from the current execution state to determine future evolution. |
| Mutable decomposition | $h=\{h_{\mathrm{dyn}},h_H,h_{\mathrm{plastic}},h_{\mathrm{history}},h_{\mathrm{rng}}\}$, with components retained as distinct types. |
| Plasticity generality | Any declared mutable parameter $\theta\in h_{\mathrm{plastic}}$ may evolve by an explicit state-dependent rule; plasticity is not restricted to synaptic weight. |
| Boundary typing | $x$ and $y$ are interfaces, not specific physical quantities. Their type must be declared when ambiguity matters. |
| Boundary reduction | $x$ may represent omitted upstream neural or environmental dependencies without explicitly modeling their causes. $y$ may similarly represent omitted downstream consequences. |
| Realization separation | Structural TFNE is realized into executable static and initial mutable state: $\mathcal{A}\rightarrow(s,h_0)$. Configured structure is not identical to realized or executed state. |
| Continuation sufficiency | Given $s$, future inputs $x_{t:}$, and deterministic RNG semantics, $h_t$ contains all mutable information required to reproduce future execution. |
| Computational normal form | Every executable TFNE system reduces to $(h_t,x_t;s)\mapsto(h_{t+1},y_t)$. |
| Compact dependency form | $y=s.h.x$ may be used as compact dependency/composition notation; `.` is not scalar multiplication. |
| Deterministic normalization | A valid fully defined TFNE expression must have one deterministic expansion into its explicit tensor/graph hierarchy before execution. |

## Reserved fundamental names

| Name | Unique meaning |
|---|---|
| $O$ | Ordered architectural composition |
| $X$ | Cross/lateral architectural composition |
| $x$ | Upstream representation |
| $y$ | Downstream representation |
| $H$ | H-state tensor |
| $h$ | Mutable execution-state representation |
| $s$ | Static execution representation |
| $N$ | Cardinality map |
| $P$ | Proportion/composition map |
| $G$ | Geometry map |
| $C$ | Cell-type domain |
| `:` | System boundary |
| `:=` | Definition |
| `{}` | Composite structural boundary |
| `[]` | Typed selection/specialization/rule binding |
| `.` | Hierarchical addressing |
| $>,<,<>$ | Projection direction |
| $\not>,\not<$ | Projection exclusion |
## Purpose and implementation boundary

TFNE separates the representation used to specify and inspect a neural model from the representation used to execute it.

| Stage | Representation | Purpose |
|---|---|---|
| Define | Named recursive objects and reusable rules | Express local neural structure, models, parameters, and mechanisms once. |
| Factor | Shared object and connection definitions | Reuse common biology without duplicating specification. |
| Compose | $O$, $X$, groups, and named rules | Express arbitrarily large hierarchies and cross-connections compactly. |
| Realize | Expand definitions, $N$, $P$, $G$, models, parameters, and connection rules | Produce one explicit realized neural system. |
| Flatten | $(s,h_0,\mathcal{I})$ | Convert hierarchical semantics to execution-efficient arrays while preserving identity. |
| Execute | $(h_t,x_t;s)\mapsto(h_{t+1},y_t)$ | Run the numerical model. |
| Inspect | $\mathcal{I}$ | Map flattened state, parameters, objects, and connections back to TFNE names and paths. |

The design principle is

$$
\boxed{\text{factor at specification time; flatten at execution time}.}
$$

A complex neural system therefore does not require a more complex execution grammar. Complexity is carried by recursive definitions, cardinality, composition, geometry, and connection rules, then compiled away from the runtime hierarchy.

### Factoring

Objects and rules may be defined independently and reused:

```text
V1 := CTX[v1]
V2 := CTX[v2]
MT := CTX[mt]

O[ff] := [...]
X[lat] := [...]
```

The architecture can then remain compact:

```text
x : V1 O[ff] V2 O[ff] {V3 X[lat] V3A} O[ff] MT : y
```

Local definitions determine internal biology; composition determines relations among objects. Shared definitions do not imply shared numerical parameter values unless explicitly declared.

### Hierarchical-to-flat realization

A hierarchical TFNE specification is construction-time structure:

$$
\boxed{
\mathcal{A}
\xrightarrow{\mathrm{resolve}}
\mathcal{A}_{\mathrm{explicit}}
\xrightarrow{\mathrm{realize}}
(s,h_0,\mathcal{I})
}
$$

where $\mathcal{I}$ is the realization index map.

Runtime need not recursively traverse the TFNE hierarchy. Named paths such as

```text
V1.L4.E.v
```

may resolve to slices or indexed subsets of flattened state, while projections such as

```text
V1.L4.E > V2.L3.E
```

may resolve to flattened connectivity/index representations.

The required invariant is

$$
\boxed{\text{flattening changes representation, not TFNE semantics}.}
$$

### Realization index map

$\mathcal{I}$ preserves correspondence between TFNE specification and realized numerical representation.

It must support, where applicable:

- TFNE object/path to realized tensor or slice;
- realized tensor or slice to TFNE object/path;
- TFNE projection/rule to realized edges;
- realized edges to their originating TFNE relation/rule;
- parameter and mutable-state targeting by TFNE identity;
- inspection of realized cardinality, geometry, parameters, state, and connectivity.

Flattening must therefore preserve inspectability even when the runtime representation is fully numerical and flat.

### JaxFNE compilation target

TFNE is a specification language for JaxFNE, not a second simulator.

The intended boundary is

$$
\boxed{
\text{TFNE}
\rightarrow
\text{explicit typed neural model}
\rightarrow
(s,h_0,\mathcal{I})
\rightarrow
\text{JaxFNE execution}
}
$$

with runtime

$$
\boxed{
(h_t,x_t;s)\mapsto(h_{t+1},y_t).
}
$$

The recursive TFNE hierarchy should therefore impose no required recursive Python execution inside the simulation loop. JaxFNE may use flat, indexed, vectorized, JIT-compatible numerical representations provided the TFNE meaning and realization map are preserved.

### Scope

The algebra is intended to remain invariant across neural scale and anatomical organization. It may describe a single cell, nonlaminar nucleus, layered cortical area, multi-area hierarchy, peripheral circuit, or larger nervous system without introducing a new fundamental grammar.

Increasing model complexity should primarily increase definitions and realized data, not the number of fundamental operators.
