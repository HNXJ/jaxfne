# 01 — Define: the canonical PseudoGenome

> Cumulative step 01/08 — endpoint `canonical-v1-column-1000n`. This page defines the generative rules; the next page develops them. See [Tutorials overview](index.md) and [PseudoGenome guide](../guides/jdna.md).

A **PseudoGenome** stores rules, never the phenotype (`G ≠ D`, `G ≠ H_D`). The canonical 1000-neuron V1 column declares 6 laminar layers, per-layer cell-type base fractions with tolerance bands, geometry, and 48 typed connection rules.

```python
import jaxfne as jtfne

genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
jtfne.jdna.genome.validate_genome(genome)
print(genome.name)  # canonical-v1-column-1000n
print(f"areas={len(genome.areas)} layers={len(genome.areas[0].layers)} rules={len(genome.areas[0].inter_connections)}")
# 1 area (V1), 6 layers (L1 100, L2 250, L3 200, L4 100, L5 200, L6 150), 48 rules
```

Declared constraints (machine-readable, used by tests):

```python
from jaxfne.jdna.genome import declared_constraints, genome_rules_hash

print(genome_rules_hash(genome)[:16])  # content hash (description excluded)
print(declared_constraints(genome)["areas"]["V1"]["layers"]["L2"])
# {'n_neurons': 250, 'cell_type_count_bands': {'E': [150, 175], 'PV': [37, 62], ...}}
```

Tolerance example (L1): `E (0.45,0.55) SST (0.1,0.2) VIP (0.3,0.4)` — realized counts must fall in those bands. Development parameter `fraction_jitter_sigma=0.01` controls how far the jitter pushes the base fractions before projection onto the box-constrained simplex.

Next: [02 — Develop](02_develop_genome.md) — `develop(genome, K_D)` realizes one phenotype.
