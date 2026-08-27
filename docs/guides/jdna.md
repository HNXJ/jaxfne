# PseudoGenome

## Definition

A **PseudoGenome** is a finite, model-defined generative specification whose coordinates and rules determine the development of a neuronal phenotype. In JaxFNE, this model is implemented by **JDNA** (JAX Developmental Neural Architecture). JDNA develops a `PseudoGenome` into a `NeuronalTensor`. The genome analogy is computational: its components need not correspond to literal genes, DNA sequences, chromosomes, or molecular genomic mechanisms.

```text
PseudoGenome --[JDNA/develop; K_D]--> NeuronalTensor  (K_D: developmental random key/domain)
```

Canonical grammar:

```text
PseudoGenome --develop--> NeuronalTensor --construct--> Model --simulate--> Signals
```

`PseudoGenome` is the scientific object; `JDNA` is the implementation; `development` is the `PseudoGenome` → `NeuronalTensor` map; `phenotype` is the realization. The terminal structural phenotype is a [NeuronalTensor](../api/neuronal_tensor.md). No literal genetics implication is made.

## Illustrative examples

**Rules → realization.** A PseudoGenome may declare per-layer population fractions (e.g., excitatory/inhibitory base fractions), bounded developmental ranges (fraction tolerance bands), laminar geometry (depth bands, x/y ranges, distribution), and distance- or class-dependent connection rules. Development realizes these rules as explicit arrays and edges in the NeuronalTensor: integer counts per cell type within declared bands, concrete positions, per-neuron parameters, edges, weights, and delays.

**Cortical column (1000 neurons).** The canonical 6-layer cortical-column PseudoGenome declares layer-wise counts and composition rules, geometry, and typed inter-layer connection schemes. Developing it with a given `K_D` realizes a concrete 1000-neuron phenotype — specific counts, positions, parameters, edges, weights, and delays — within the declared constraints. The PseudoGenome stores the generative rules, not 1000 per-neuron records.

**Same genome, different K_D; same K_D, same phenotype.** Two developments of the same PseudoGenome with different `K_D` values (e.g., `seed=0` vs `seed=1`) share the same rules but realize different phenotypes within the same constraint bands. Conversely, development is deterministic in `K_D`: the same PseudoGenome and the same `K_D` reproduce the same phenotype.

## Mathematics

### The development operator

Let

\[
G = \texttt{PseudoGenome}.
\]

Let the developmental state be

\[
Z_D(t) = \bigl(\mathcal N(t),\ H_D(t),\ \ldots\bigr),
\]

where \(\mathcal N(t)\) is the developing network structure and \(H_D(t)\) is
the developmental state vector (of dimension \(d_D\), possibly zero). A
PseudoGenome *parameterizes* developmental rules; the developmental operator
`D` executes them:

\[
\boxed{
Z_D(t + \Delta t) = D\bigl(Z_D(t),\ U_D(t);\ G,\ K_D\bigr).
}
\]

`U_D` are development-stage inputs, `K_D` the development PRNG domain. The
terminal phenotype is the network structure at development end:

\[
\boxed{
\operatorname{phenotype}(Z_D(T)) = \mathcal N_T = \texttt{NeuronalTensor}.
}
\]

Structural invariants:

\[
G \neq H_D, \qquad G \neq D.
\]

A PseudoGenome parameterizes developmental rules (`G` provides coordinates and
constants); the developmental operator executes them (`D` is the rule engine);
developmental state evolves under them (`H_D`, `Z_D`); the terminal structural
phenotype is a NeuronalTensor.

### Generative rules, not stored phenotype

The defining invariant:

\[
\boxed{
\text{PseudoGenome describes how the phenotype is generated}
\neq
\text{PseudoGenome stores the terminal phenotype}.
}
\]

A PseudoGenome declares *rules*: laminar bands and depths, per-layer cell-type
base fractions with tolerance bands, geometry, and typed connection schemes.
Development realizes a phenotype within the declared constraints. The genome
never stores positions, edges, or population arrays.

 ### Developmental versus runtime state

Do not assume \(H_D = H_R\). Developmental state lives in
\(\mathbb R^{d_D}\); runtime RBS lives in \(\mathbb R^{d_R}\); in general
\(d_D \ne d_R\). No developmental-to-runtime state projection exists in 0.4.17
unless an implementation requires and scientifically defines one.

### Build-time scope (0.4.17)

v0.4.17 pseudogenomes are **build-time generative specifications**:

\[
\boxed{G,\,K_D \xrightarrow{\mathrm{develop}} \mathcal N}
\]

`develop` runs at build time and returns a terminal `NeuronalTensor`
structure, which is then passed to ordinary `construct`/`simulate`. It does
**not** mutate network topology during the simulation clock. In particular,
**activity-dependent structural growth is not implemented in 0.4.17** — no
runtime H-triggered layer insertion (the proposed
\(\mathcal N(t) \to \mathcal N(t^+)\) developmental-growth experiment) is
present. That remains future work after the frozen-use period.

### PRNG separation

Development, simulation, and optimization use independently controlled PRNG
domains:

\[
\boxed{K_D \ne K_S \ne K_A.}
\]

`K_D` = `develop(genome, seed=...)`; `K_S` = `RuntimeConfiguration.seed`
(construction/edge realization and simulation noise); `K_A` = optimizer seed
(`Model.tune`). Changing one never silently alters the others.

## Assumptions

1. The generative analogy is computational; no molecular-genomic claims are made.
2. Development is deterministic in `K_D`: `D(G, K) = D(G, K)`.
3. Development realizes only phenotypes satisfying the genome-declared
   constraint bands (exact neuron counts, per-layer fractions within declared
   tolerance).
4. No structural mutation occurs inside `simulate()` in 0.4.17.
5. No developmental state crosses into the runtime model: the runtime RBS
   (`h_state`) arises from the ordinary construct path, not from development.
6. Development may be ordinary Python/JAX orchestration; it is not forced into
   JIT-compiled simulation kernels.

## Relationships

| Concept | Relationship |
|---|---|
| PseudoGenome | Generative specification (rules + constraints). |
| JDNA | Theory/model governing pseudo-genomic generation. |
| develop | `(G, K_D) -> NeuronalTensor`. |
| NeuronalTensor | Terminal structural phenotype; ordinary pipeline input. |
| construct | Ordinary `(NeuronalTensor, RuntimeConfiguration) -> Model`. |
| Model / simulate | Ordinary execution; no JDNA branches. |
| RBS (runtime) | `H_R` arises at construct (PlasticParams.H aggregation); development declares no `H_D` for the canonical genome. |
| AGSDR | Optimizes runtime/model/objective parameters \(\Theta_R'\); does not evolve PseudoGenomes in 0.4.17. |

## Implementation mapping

| Symbol | Implementation |
|---|---|
| `PseudoGenome`, `AreaGenome`, `LayerGenome`, `ConnectionRuleGenome` | `jaxfne.jdna.genome` (frozen dataclasses). |
| `develop` | `jaxfne.jdna.genome.develop` — PRNG split per area/layer, jittered fractions projected onto the box-constrained simplex (bands + sum-to-one), largest-remainder integer counts, tensor assembly, provenance attach, constraint verification. |
| Canonical genomes | `jaxfne/jdna/genomes/*.json` (shipped package data, `pseudogenome_v1` schema). |
| Loaders | `load_pseudogenome`, `load_canonical_pseudogenome`, `list_canonical_pseudogenomes`. |
| Provenance | `NeuronalTensor.provenance`: genome identity hash, schema version, development seed, development parameters, phenotype hash. |
| Constraint declarations | `jaxfne.jdna.declared_constraints` (machine-readable, used by tests/audits). |

Root exports: `jtfne.PseudoGenome`, `jtfne.develop`, `jtfne.load_pseudogenome`,
`jtfne.load_canonical_pseudogenome`, `jtfne.list_canonical_pseudogenomes`.

## Example

```python
import jaxfne as jtfne
jtfne.enable_x64()
genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor = jtfne.develop(genome, seed=0)
model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
```

This path uses the ordinary tensor → model → signals pipeline. JDNA is
optional; the existing `Configuration`/`NeuronalTensor` direct paths remain
first-class.

## Limitations

- Development is structural: population composition, geometry declaration,
  connection schemes. Positions and edges are realized by the ordinary
  construct path under `K_S`.
- No birth, death, migration, or developmental-time structural mutation in
  0.4.17.
- No genome-level optimization: AGSDR optimizes runtime parameters
  (\(\Theta_R'\)), never \(\Theta_D\) or \(\Theta_G\).
- No developmental RBS (`H_D`) is declared by the canonical genome; the
  `H_D \subset Z_D` possibility is documented but not exercised by an
  implemented genome.
- The stochastic variation produced is *computational pseudo-genomic /
  developmental variation*, not biological genetic variability.

## References

- [Public surface contract](../public_surface_contract.md) — additive JDNA
  surface (5 root symbols).
- [NeuronalTensor API](../api/neuronal_tensor.md) — phenotype schema.
- [RBS / RBD / HDP doctrine](../doctrine/rbs_rbd_hdp.md) — runtime state semantics.
- [References](../reference/references.md) — scholarly ancestry for
  developmental/generative neural modeling.
