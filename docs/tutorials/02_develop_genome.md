# 02 — Develop: genome → phenotype

> Continued from [01 — Define](01_define_genome.md). Same `genome` variable; no rebuild. Next: [03 — Inspect](03_inspect_tensor.md).

Development is deterministic in `K_D`: `develop(G, K_D)` maps rules to a concrete `NeuronalTensor` (counts, `fraction`s, provenance). Different `K_D` → different phenotypes within the same bands; same `K_D` → identical phenotype.

```python
import jaxfne as jtfne
# continued — genome from 01_define_genome.md
genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")

tensor0 = jtfne.develop(genome, seed=0)
tensor1 = jtfne.develop(genome, seed=1)
assert jtfne.develop(genome, seed=0).provenance["phenotype_sha256"] == tensor0.provenance["phenotype_sha256"]
print(tensor0.provenance)  # genome_sha256, schema_version, development_seed, development_parameters, phenotype_sha256
```

Jitter within declared bands (σ=0.01, deterministic JAX PRNG per layer):

| layer | base E fraction | seed 0 realized | seed 1 realized | band |
|-------|----------------|-----------------|-----------------|------|
| L1 | 0.50 | 0.49 (49 E) | 0.50 (50 E) | [0.45,0.55] |
| L2 | 0.648 | 0.648 (162 E) | 0.648 (162 E) | [0.60,0.70] |

Provenance is additive: `tensor.provenance["genome_sha256"]` == `genome_rules_hash(genome)`; JSON saves of `NeuronalTensor` exclude provenance (manifest/evidence path preserves it).

Next: inspect the realized tensor before any simulation.
