# JDNA API — PseudoGenome and development

## Overview

JDNA is the theory/model governing pseudo-genomic generation; the concrete
specification object is a `PseudoGenome`. The root API is deliberately small:
five symbols.

```python
import jaxfne as jtfne

jtfne.PseudoGenome                # generative specification (dataclass)
jtfne.develop(genome, seed=0)     # (G, K_D) -> NeuronalTensor
jtfne.load_pseudogenome(path)     # load from JSON file or dict
jtfne.load_canonical_pseudogenome(name)  # load a shipped canonical genome
jtfne.list_canonical_pseudogenomes()     # shipped genome names
```

Deeper inspection helpers live in `jaxfne.jdna` (`genome_rules_hash`,
`phenotype_sha256`, `declared_constraints`, `validate_genome`,
`save_pseudogenome`, `genomes_dir`).

## PseudoGenome

**What it is.** A frozen dataclass describing *generative rules* for a neuronal
phenotype: areas, laminar depth bands, per-layer cell-type base fractions with
declared tolerance bands, geometry, and typed connection schemes.

**Why it exists.** It separates *what a phenotype is* (NeuronalTensor) from
*how it is generated* (PseudoGenome). It never stores the terminal phenotype.

**Input semantics.**
- `name` — genome identifier (also the developed tensor's name).
- `schema_version` — `"pseudogenome_v1"` (JSON schema version).
- `description` — free prose; excluded from the rules hash.
- `areas` — tuple of `AreaGenome` (layers + within-area connection scheme + pose).
- `area_connections` — declared between-area rules.
- `development_parameters` — global development knobs, e.g.
  `{"fraction_jitter_sigma": 0.01}` (std of the Gaussian jitter applied to
  cell-type base fractions before integer allocation).

**Output semantics.** `PseudoGenome` is not executable itself; pass it to
`develop`.

**Ownership.** `jaxfne.jdna.genome`. Serialized as JSON data
(`pseudogenome_v1`), never as code.

**Invariants.** Every layer declares positive `n_neurons`, a valid depth band
`0 <= lo < hi <= 1`, and cell-type fractions summing to 1. Tolerances are
absolute fraction bands; realized counts always fall inside
`[floor(n*tol_lo), ceil(n*tol_hi)]`.

**Adjacent objects.** `AreaGenome`, `LayerGenome`, `ConnectionRuleGenome` —
the rule containers; `NeuronalTensor` — the developed phenotype.

**When to use.** To express phenotype families by rules; to generate
constraint-satisfying phenotype ensembles; to record development provenance.

**When not to use.** For a single fixed circuit, load a canonical
NeuronalTensor directly — JDNA is optional.

**Determinism.** Purely deterministic; equality is structural (frozen
dataclass).

**Status.** CANONICAL (0.4.17).

## develop

`develop(genome, seed=0, *, development_parameters=None) -> NeuronalTensor`

**What it is.** The development operator `D(G, K_D)`.

**Input semantics.** `seed` is the development PRNG key `K_D` (integer). The
same `(genome, seed)` reproduces the identical phenotype. Optional
`development_parameters` overrides merge over the genome's declared parameters
(e.g. `{"fraction_jitter_sigma": 0.0}` disables composition jitter).

**Output semantics.** A `NeuronalTensor` whose per-layer neuron counts are
exact (`n_neurons`), whose cell-type counts respect the declared tolerance
bands, and whose `provenance` records: genome identity hash (`genome_sha256`),
schema version, development seed, development parameters, and phenotype hash
(`phenotype_sha256`). Realization of `N` via `develop` does not establish
effectiveness (`ΔX` under intervention).

**Ownership.** `jaxfne.jdna.genome`.

**Invariants.** Type closure: always a `NeuronalTensor`. Determinism:
`D(G, K) = D(G, K)`. Constraints: realized counts within declared bands.
Boundary: positions and edges are realized later by the ordinary construct
path under the runtime seed `K_S`; `develop` itself never touches runtime
kernels.

**Deterministic/stochastic behavior.** Deterministic in `seed`; the jitter is
pseudo-random (JAX PRNG), i.e. reproducible. Variation across seeds is
*computational pseudo-genomic variation*, not biological genetic variability.

**Status.** CANONICAL (0.4.17).

## Loaders

- `load_pseudogenome(path | dict)` — load a genome from a JSON file or dict;
  rejects unknown `schema_version` values explicitly (no silent
  interpretation as v1; a registered migration path is required for future
  schemas).
- `load_canonical_pseudogenome(name)` — load a shipped genome from
  `jaxfne/jdna/genomes/` (package data); supports with/without `.json` suffix.
- `list_canonical_pseudogenomes()` — shipped genome names.

**Minimal executable example**

```python
import jaxfne as jtfne

genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor = jtfne.develop(genome, seed=0)
model = jtfne.construct(
    tensor,
    jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5),
)
signals = jtfne.simulate(model)
```

## Provenance

`NeuronalTensor.provenance` is an additive optional field (default `None`),
populated by `develop` and ignored by the ordinary pipeline
(`construct`/`simulate` require no JDNA knowledge). JSON tensor saves exclude
`provenance`; the canonical workflow for provenance-preserving archives is the
manifest/evidence path.

## Related pages

- [JDNA theory](../guides/jdna.md) — definitions, mathematics, limitations.
- [NeuronalTensor](neuronal_tensor.md) — the phenotype schema.
- [RBS / RBD / HDP](../guides/hdp.md) — runtime state semantics.
- [Public surface contract](../public_surface_contract.md).
