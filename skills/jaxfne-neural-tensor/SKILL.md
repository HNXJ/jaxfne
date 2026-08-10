---
name: jaxfne-neural-tensor
description: >-
  Build or audit a jaxfne NeuronalTensor circuit, its Areas/Layers/NeuronTypes
  and connections, or its tensor runtime and HDP dispatch. Use for structured
  tensor inputs, canonical JSON tensors, or full-state pipeline work.
---

# jaxfne NeuronalTensor procedure

Read `catalog-glossary-jaxfne` and the mathematical project source before
changing tensor or HDP semantics. This skill records API procedure only.

## Tensor path

```python
import jaxfne as jtfne

tensor = jtfne.load("jaxfne/configs/canonical-v1-column-1000n.json")
runtime_configuration = jtfne.RuntimeConfiguration(
    seed=0, duration_ms=100.0, dt_ms=0.5
)
model = jtfne.construct(tensor, runtime_configuration)
signals = jtfne.simulate(model, duration_ms=100.0, dt_ms=0.5, seed=0)
```

The structured types are `NeuronalTensor`, `Area`, `Layer`, `NeuronType`,
`InterConnection`, and `AreaConnection`. Use `NeuronType.make(...)` for typed
population declarations and the package loaders for JSON.

`RuntimeConfiguration` is distinct from `RuntimeConfig`:

- `RuntimeConfiguration`: tensor construction/execution settings.
- `RuntimeConfig`: simulation runtime policy, including `enable_hdp` and
  `hdp_params`.

## HDP dispatch procedure

When HDP is requested, pass an explicit runtime:

```python
runtime = jtfne.RuntimeConfig(enable_hdp=True, hdp_params=params)
signals = jtfne.simulate(
    model, duration_ms=100.0, dt_ms=0.5, seed=0, runtime=runtime
)
```

Do not assume a direct `Model.simulate()` call inherits configuration metadata.
Verify the returned runtime metadata for the actual call path.

Do not change HDP equations, parameter defaults, weight dynamics, or topology
behavior under a context/governance-only task. Those are specification/code
reconciliation items requiring their own tests.

## Continuation

`Model.with_hdp_initial_state()` is an initialization convenience. It is not a
general full-state continuation contract.

For full turn-to-turn state, use the internal pipeline only when the task
explicitly requires it:

```text
_pipeline.DynamicState
    v, u, prev_spikes, syn_state, H, w
_pipeline.compile_step_fn / scan_network
```

Verify internal signatures before use; do not promote them to a stable public
API without a separate contract decision.

## Connection checks

Tensor-declared connections are compiled into configuration rules and edges.
Inspect `model.edge_list` or the owning model/connectivity contract after
construction. If the tensor graph is intended to be exclusive, verify that
default recurrent generation is disabled or has zero contribution. Do not
silently accept an unexpected edge count.

## JSON and validation

- Use `jtfne.load` and package tensor serializers.
- Preserve schema version and JSON-safe values.
- Verify layer/population counts from `model.neuron_table()`.
- Check finite signals and status metadata.
- Run targeted tensor tests.

Current implementation truth is in `jaxfne/neuronal_tensor.py`,
`jaxfne/_pipeline.py`, `jaxfne/_model*.py`, and tests. Mathematical meaning
belongs to the project source documents.
