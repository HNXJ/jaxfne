# 03 — Inspect: realized vs configured vs effective

> Continued from [02 — Develop](02_develop_genome.md). Same `tensor` variable (no new genome).

Configured (`p_EI` prior) → realized (concrete `NeuronalTensor` + `EdgeList` after `construct`) → effective (`ΔX` under intervention). Realization does not imply effectiveness.

```python
import jaxfne as jtfne, collections
# continued — tensor from 02_develop_genome.md
genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor = jtfne.develop(genome, seed=0)

# layer counts are exact; cell-type fractions are within declared bands
counts = collections.Counter((r["layer"], r["cell_type"]) for r in jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1)).neuron_table())
print(len(tensor.areas[0].layers))  # 6
print(sum(l.n_neurons for l in tensor.areas[0].layers))  # 1000
print(sum(1 for a in genome.areas for c in a.inter_connections))  # 48 declared rules

# round-trip: configs are data, never code
path = jtfne.save_neuronal_tensor(tensor, "/tmp/cumulative_1000n.json")
assert jtfne.load_neuronal_tensor(path).name == tensor.name
```

**Box: n=1 vs 1000 contrast.** A single-cell tutorial (`configuration().network(n=1)`) and this 1000n tensor share no variables — the point is that isolated `n=1 → 2 → 100 → 600` progression resets the model. The cumulative path does not.

Next: [04 — Simulate](04_simulate_tensor.md) — construction realizes positions and edges.
