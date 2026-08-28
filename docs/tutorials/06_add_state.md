# 06 — Add state: H / RBS container

> Continued from [05 — Observe](05_observe_fields.md). Same `tensor, model`; add state without dynamics.

Per-neuron `H` seeds the homeostatic controller's initial state only when HDP is later enabled. It is stored but inert when HDP is off (the default).

```python
import jaxfne as jtfne, numpy as np
# continued — tensor, model from 04_simulate_tensor.md
genome = jtfne.load_canonical_pseudogenome("canonical-v1-column-1000n")
tensor = jtfne.develop(genome, seed=0)
model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=1, duration_ms=1000.0, dt_ms=0.5))

# PlasticParams.H aggregation (mean across incoming edges touching the neuron; untouched → 1.0)
# In the canonical genome this is done by construct_neuronal_tensor; here shown explicitly:
model_h = model.with_hdp_initial_state(H0=np.ones(len(model.neuron_table())))
print(model_h.last_hdp_diagnostics())  # None — HDP not enabled

# pause/resume identity of the container
state = jtfne.checkpoint_state(model)
model2 = jtfne.restore_state(model, state)
```

`H_D` (developmental, `ℝ^{d_D}`) ≠ `H_R` (runtime RBS, `ℝ^{d_R}`); the canonical genome declares no `H_D`.

Next: [07 — Add dynamics](07_add_dynamics.md) — enable `enable_hdp`.
