# 04 — Simulate: construct → Signals

> Continued from [03 — Inspect](03_inspect_tensor.md). Same `tensor`; now compile and run.

Construction realizes geometry positions and the edge list under runtime key `K_S` (distinct from development `K_D`). Simulation is cheap; construction is the expensive step (seconds at 1k, tens of seconds at 10k; machine- and builder-dependent).

```python
import jaxfne as jtfne, numpy as np
# continued — tensor from 02_develop_genome.md
genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor = jtfne.develop(genome, seed=0)

model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))
signals = jtfne.simulate(model)
print(len(model.neuron_table()))  # 1000
print(model.params["edge_list"].n_edges)  # ~215k observed 2026-10, re-run to confirm (215785 direct canonical tensor, 215190 develop seed 0; 48 rules x p=1.0 bipartite)
print(signals.get("spikes").shape)  # (2000, 1000)  — (T, N)
print(float(signals.get("spikes").mean() * 1000 / 0.5))  # ~8-12 Hz population rate
```

`model.params["positions"]` now holds pose-correct 3-D placement (apply `Pose3D` per area). `model.params["edge_list"].weight`/`.delay_steps` are inspectable. Vary only `K_S` or `drive_per_neuron` for sweeps via `model.with_emitter_parameters(drive_per_neuron=...)` without rebuilding.

**Box: two on-ramps, one compiler.** The fluent `Configuration` path (`Configuration().column().cell_types()...`) and this `develop → NeuronalTensor` path converge on `construct → simulate`. See [Configuration Grammar](../../guides/grammar/configuration_grammar.md).

<!-- jx-figure -->
## Figure

![Spike raster of the 1000 ms reference run](../../assets/readme/raster.png)

[Interactive version](https://hnxj.github.io/jaxfne/figures/readme/raster.html)

Spike raster of the pinned 1000 ms reference run (canonical tensor, runtime seed 0, dt 0.5 ms); the code above uses runtime seed 1, so its raster differs in detail. [Open full page](https://hnxj.github.io/jaxfne/figures/readme/raster.html).

Next: [05 — Observe](../extend/05_observe_fields.md) — post-hoc observation operators on the frozen trajectory.

## Interactive atlas (dark)

Dark-theme Plotly panels from the pinned 1000 ms reference run of this same canonical column (the page runs 1000 ms — the same duration as the pinned run): [Circuit schematic](https://hnxj.github.io/jaxfne/figures/readme/schema.html) · [Network 3D](https://hnxj.github.io/jaxfne/figures/readme/network_3d.html) · [Spike raster](https://hnxj.github.io/jaxfne/figures/readme/raster.html) · [LFP proxy](https://hnxj.github.io/jaxfne/figures/readme/lfp.html) · [Oscillatory response](https://hnxj.github.io/jaxfne/figures/readme/oscillatory.html)

Regenerate: `python scripts/generate_readme_atlas.py --html-only`.
