# 04 — Simulate: construct → Signals

> Continued from [03 — Inspect](03_inspect_tensor.md). Same `tensor`; now compile and run.

Construction realizes geometry positions and the edge list under runtime key `K_S` (distinct from development `K_D`). Simulation is cheap; construction is the expensive step (~2 s at 1k, ~40 s at 10k).

```python
import jaxfne as jtfne, numpy as np
# continued — tensor from 02_develop_genome.md
genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor = jtfne.develop(genome, seed=0)

model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
print(len(model.neuron_table()))  # 1000
print(model.params["edge_list"].n_edges)  # ~215k (48 rules × p=1.0 bipartite; 215785 for direct canonical tensor, 215190 for develop seed 0)
print(signals.get("spikes").shape)  # (2000, 1000)  — (T, N)
print(float(signals.get("spikes").mean() * 1000 / 0.5))  # ~8-12 Hz population rate
```

`model.params["positions"]` now holds pose-correct 3-D placement (apply `Pose3D` per area). `model.params["edge_list"].weight`/`.delay_steps` are inspectable. Vary only `K_S` or `drive_per_neuron` for sweeps via `model.with_emitter_parameters(drive_per_neuron=...)` without rebuilding.

**Box: two on-ramps, one compiler.** The fluent `Configuration` path (`Configuration().column().cell_types()...`) and this `develop → NeuronalTensor` path converge on `construct → simulate`. See [Configuration Grammar](../guides/configuration_grammar.md).

Next: [05 — Observe](05_observe_fields.md) — post-hoc observation operators on the frozen trajectory.
