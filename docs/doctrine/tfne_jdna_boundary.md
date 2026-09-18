# TFNE–JDNA Boundary: Constraints, Completion, Realization

TFNE specifies scientific constraints. JDNA completes the generative choices
required for realization. The realized JaxFNE model executes. Each stage owns
distinct decisions, and every realized value records which stage chose it.

```text
TFNE -> JDNA -> JaxFNE Model -> Simulation -> Observation -> Evidence
```

## Stage ownership

| Stage | Owns | Example |
|---|---|---|
| TFNE | Structural/scientific constraints | areas, multiplicity, `O`/`X`, cell types, connection rules |
| JDNA | Developmental completion | positions, distributions, allocation, stochastic realization, defaults |
| JaxFNE Model | Realized executable arrays | coordinates, neuron indices, edges, parameters |
| Simulation | Dynamics | trajectories and observations |

TFNE can be intentionally underdetermined: a specification may leave
quantities open that realization requires. That is not a defect in the
specification. JDNA is the single owner of completing them, under explicit
developmental rules `D` and an explicit development RNG domain `K_D`:

$$
\boxed{
\underbrace{\mathcal A}_{\text{TFNE constraints}}
+
\underbrace{D}_{\text{JDNA developmental rules/defaults}}
+
\underbrace{K_D}_{\text{development RNG}}
\longrightarrow
\underbrace{M}_{\text{realized JaxFNE model}}.
}
$$

## Replication vs the relation among replicas

TFNE distinguishes instantiating from relating. `A^{n}` creates `n` indexed
instances and no connectivity. Where the relation among instances is part of
the constraint, it is written explicitly:

| Form | Meaning |
|---|---|
| `A^{n}` | `n` instances, no inter-instance relation implied |
| `A^{nX}` | `n` instances developed under `X` |
| `A^{nO}` | `n` instances in sequential `O`-composition |

So `V1^{10X}` develops structurally into
`V1_1 X V1_2 X ... X V1_{10}`, and `V1^{10X} O V2^{10X}` first develops two
repeated systems, then realizes the declared `O` relation between them
according to its rule. Compressed TFNE expands before realization; expansion
is structural and deterministic, while concrete contents (counts, positions,
sampled edges) are JDNA's under `K_D`.

## Defaults: what JDNA does with what TFNE omits

| Missing from TFNE | JDNA behavior |
|---|---|
| Deliberately optional and has canonical default | Apply default and record provenance |
| Determined by another TFNE constraint | Derive deterministically |
| Stochastic developmental quantity | Sample using explicit `K_D` domain |
| Required consequential quantity without default | Reject |
| Ambiguous biological choice | Reject / require specification |

Refusing is load-bearing: a defaulted value and a rejected specification are
different scientific facts, and the realized model must distinguish them. A
completer that invents consequential values silently would make two different
specifications realize identically with no record of the difference.

## Value origin provenance

Every realized value carries its origin:

```text
value origin in {TFNE-declared, JDNA-derived, JDNA-default, JDNA-sampled}
```

- `TFNE-declared` — stated in the specification (`G[V1] := Ω`, `N[V1]`, a rule).
- `JDNA-derived` — fixed by another declared constraint (integer allocation
  from `N`/`P`, replica indexing).
- `JDNA-default` — a canonical default applied where the specification is
  deliberately optional (`G[V1] = ∅` with a permitted default domain).
- `JDNA-sampled` — drawn under `K_D` (positions, thinned edges).

This is what makes the boundary inspectable: given a realized model, every
value answers which stage chose it. It extends the standing invariant that
realization-affecting metadata belongs in normalization and digest: JDNA
outputs carry their origins plus `K_D` in their own provenance, so the
TFNE→JDNA→Model chain is auditable end to end.

## Geometry belongs to JDNA

TFNE declares geometry (`G[A]` is first-class structural specification) or
omits it. Realization of coordinates is JDNA's:

- `G[V1] := Ω` — JDNA obeys the declared domain.
- `G[V1] = ∅` with a permitted default — JDNA supplies its defined default,
  e.g. `rᵢ ~ U(Ω_default)`, recorded as `JDNA-default`/`JDNA-sampled`.
- `G[V1] = ∅` with no permitted default — JDNA rejects rather than invents.

Geometry is therefore never patched independently inside the TFNE-to-model
bridge: declared geometry, defaults, distributions, RNG, and resulting
coordinates have one explicit owner. The TFNE bridge carries declarations
through unchanged; JDNA turns them into coordinates.

## RNG domains

Development, construction/simulation, and optimization use independently
controlled PRNG domains: `K_D ≠ K_S ≠ K_A`. `K_D` is threaded explicitly
(JAX keys are split, never reused); changing one domain never silently alters
another. Same `K_D` reproduces the same completion; different `K_D` realizes
different phenotypes within the same constraint bands.
