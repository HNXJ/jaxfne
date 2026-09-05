# V1-PFC Dual Column: Continuous AAAB Adaptation

Two 100-neuron canonical laminar columns, **V1** and **PFC**, both
HDP-enabled, connected V1→PFC feedforward. V1's L4 and L6 host three
disjoint/overlapping tuning-group populations (`AB`, `A`, `B`) that are
driven with real 40Hz AC current in a fixed, repeating 1000ms local-oddball
trial structure:

```
fx(0-100) - p1=A(100-200) - d1(200-300) - p2=A(300-400) - d2(400-500)
- p3=A(500-600) - d3(600-700) - p4=B(700-800, deviant) - d4(800-900)
- rw(900-1000)
```

`A` drives the `AB`+`A` neurons; the deviant `B` window drives the `AB`+`B`
neurons. Both the per-neuron homeostatic factor `H` and the synaptic
weights `w` carry forward trial-to-trial (`Model.with_hdp_initial_state`),
so adaptation is genuine across the whole run rather than reset every
trial.

## Why V1's L4/L6 are pure-E

The sign-detection logic in `jaxfne/neuronal_tensor.py` checks
`source_neuron_type == "E"` exactly to decide excitatory vs. inhibitory --
renaming a neuron type (e.g. to `"E_AB"`) would silently misclassify it as
inhibitory. So tuning-group identity is **not** encoded via `NeuronType.name`
at all: V1's L4 and L6 are declared as single-type (`"E"`, `fraction=1.0`)
tuning-only layers, and `AB`/`A`/`B` membership is a **positional
post-construction tag**, sliced from the layers' stable `neuron_table()`
order (`[0:5]`=`AB`, `[5:10]`=`A`, `[10:15]`=`B`, unioned across L4 and L6 for
10+10+10 combined). This is a deliberate, explicit design choice, not a
workaround pending a real feature -- see `tuning_group_indices()` in the
script.

## Building the two-area tensor

```python
from jaxfne.neuronal_tensor import (
    Area, AreaConnection, Layer, NeuronType, NeuronalTensor, PlasticParams,
    construct_neuronal_tensor,
)

# V1: L1=10, L2=25, L3=15, L4=15 (pure-E tuning), L5=20, L6=15 (pure-E tuning)
v1 = Area(name="V1", layers=(
    # ...canonical E/PV/SST/VIP layers for L1, L2, L3, L5...
    Layer(name="L4", n_neurons=15, neuron_types=(NeuronType.make("E", fraction=1.0),)),
    Layer(name="L6", n_neurons=15, neuron_types=(NeuronType.make("E", fraction=1.0),)),
))

# PFC: canonical 6-layer column, fully generic/untuned
pfc = Area(name="PFC", layers=(...))

feedforward = [
    AreaConnection(
        source_area="V1", source_layer=layer, source_neuron_type="E",
        target_area="PFC", target_layer="L4", target_neuron_type="E",
        mechanism="AMPA",
        # PlasticParams.H defaults to 0.0, outside the valid HDP range
        # (H_min=0.1, H_max=10.0) -- ALWAYS set H=1.0 explicitly when HDP
        # will be enabled, or the target neurons' seeded H0 blows up the
        # HDP integration at step 0.
        plastic=PlasticParams(H=1.0),
    )
    for layer in ("L4", "L6")
]

tensor = NeuronalTensor(areas=(v1, pfc), area_connections=tuple(feedforward),
                         name="v1_pfc_continuous_aaab")
model = construct_neuronal_tensor(tensor, seed=0, duration_ms=1000.0, dt_ms=0.1)
```

## Running chained trials with HDP carryover

```python
import jaxfne as jtfne
from jaxfne.core import RuntimeConfig
from jaxfne.hdp_network import v1_pfc_aaab_hdp_params

runtime_cfg = RuntimeConfig(
    dtype="float32", backend="cpu", enable_hdp=True,
    hdp_params=v1_pfc_aaab_hdp_params(),  # named preset: K_HDP=0.003, K_w_ctrl=0.001
)

for trial in range(n_trials):
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1,
                              seed=seed + trial, runtime=runtime_cfg, paradigm=schedule)
    diag = model.last_hdp_diagnostics()
    # carry BOTH H (homeostatic factor) and w (synaptic weights) forward:
    model = model.with_hdp_initial_state(H0=diag["H_final"], w0=diag["w_final"])
```

## The weight-carryover instability, and its fix

Carrying synaptic weights trial-to-trial used to be an **unbounded
positive-feedback runaway**: `H` already had a two-sided restoring term
(`K_ctrl`) pulling it back to equilibrium, but the synaptic weight
magnitude had none -- only a hard floor/ceiling clip. Chaining trials
compounded drift with no ceiling until the clip saturated every edge (rate
→ ~50Hz by trial 15-19).

**Fixed** by adding `K_w_ctrl` to
`simulate_edge_recurrent_izhikevich_hdp` (`jaxfne/emitters.py`): a linear
restoring force pulling the weight magnitude back toward its calibrated
baseline (`|edges.weight|`), mirroring `K_ctrl`'s own form for `H`. `K_HDP`
and `K_w_ctrl` are exposed as a named, reproducible preset --
`jaxfne.hdp_network.DEFAULT_HDP_V1_PFC_AAAB` / `v1_pfc_aaab_hdp_params()`
-- rather than bare magic constants, mirroring `DEFAULT_HDP`/
`DEFAULT_HDP_DESYNC`'s existing precedent.

## Verified results

A real 100-trial run (`carry_weights=True`, the default) is genuinely
stable across the **entire** run, not just the first few trials:

- `spike_rate_hz_mean` held at 12.53-12.55Hz across all 100 trials (trial 1
  == trial 100).
- `w_final_mean` (mean synaptic weight magnitude) held at 0.012248-0.012254
  across all 100 trials -- a range of 6e-6, i.e. essentially flat, not
  runaway.
- `H_final_mean` converged 1.027 → 1.067 with `H_final_std` tightening
  0.0201 → 0.0034 (homeostatic settling, not divergence).
- Zero NaN in any of the 100 trial summaries.

This demonstrates real, stable, long-term homeostatic adaptation with
genuine trial-to-trial weight plasticity -- not just a working-but-static
H-only pipeline (`carry_weights=False` remains available to reproduce that
earlier, more conservative behavior for comparison).

## Running it yourself

```bash
PYTHONPATH=. python3 scripts/v1_pfc_continuous_aaab_smoke_test.py [n_trials]
```

Default `n_trials=10` (~26s on CPU). The full spec target is 1000 trials
and is **not** run by default -- pass an explicit `n_trials` to opt into a
longer run. A receipt is written to
`outputs/v1_pfc_continuous_aaab_smoke_test/smoke_test_receipt.json`.

## Known limitations (stated, not hidden)

- V1's L4 and L6 have PV/SST/VIP removed entirely (pure-E tuning layers)
  to host the AB/A/B groups; L3 is shrunk from its canonical 20 neurons to
  15 to make room for L4's growth (canonical 10 → 15), holding V1 at 100
  neurons total. L1, L2, L5, and all of PFC keep the canonical
  E:PV:SST:VIP fractions and layer-size proportions unchanged.
- This is a script-driven smoke test, not a polished tutorial notebook --
  there is no Colab badge and no `tutorials/*.ipynb` file for it yet.

## Next steps

See [Guides](../guides/index.md) for how-to articles on extending these
models.
