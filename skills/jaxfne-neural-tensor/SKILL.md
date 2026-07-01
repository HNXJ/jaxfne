---
name: jaxfne-neural-tensor
description: >-
  Verified NeuronalTensor build path (Areas × Layers × NeuronTypes ×
  AreaConnections) and HDP homeostatic-plasticity module. USE when building or
  editing a NeuronalTensor circuit, enabling HDP, or naming a NeuronalTensor
  symbol. Do NOT construct NeuronalTensor(some_configuration) — it silently
  type-confuses (verified 2026-06-30, no runtime validation on the dataclass);
  there is no Configuration -> NeuronalTensor converter yet (open TBI, see
  jaxfne repo artifacts/developer/progress.json).
---

# jaxfne Neural Tensor

USE FIRST: `catalog-glossary-jaxfne` §1b/§1c.

## What it is

`NeuronalTensor = [Areas, AreaConnections]`, `Area = [Layers × NeuronTypes, InterConnections]`.
This is a **separate** build path from `Configuration` (`jaxfne-config`) — both
converge on the same `Model` via `construct()`, dispatching on input type:

```python
model = jtfne.construct(cfg)                                    # Configuration path
model = jtfne.construct(tensor, RuntimeConfiguration(...))      # NeuronalTensor path
```

## Building a tensor

```python
from jaxfne.neuronal_tensor import Area, Layer, NeuronalTensor, NeuronType

tensor = NeuronalTensor(areas=[
    Area(name="V1", layers=[
        Layer(name="L4", n_neurons=100,
              neuron_types=[NeuronType.make("E", fraction=0.8),
                            NeuronType.make("PV", fraction=0.2)]),
    ]),
])
```

`NeuronalTensor`/`Area`/`Layer`/`NeuronType`/`InterConnection`/`AreaConnection`
are all **frozen dataclasses with no runtime type validation** — passing the
wrong type into a field (e.g. a `Configuration` object where `areas:
Sequence[Area]` is expected) does NOT raise; it silently stores the wrong
object. **Verified landmine (2026-06-30):** `NeuronalTensor(some_cfg)` succeeds
and produces `tensor.areas == some_cfg` (a `Configuration`, not a list of
`Area`) with zero error — this is always a bug if it happens, never a valid
shortcut. There is no `Configuration -> NeuronalTensor` converter anywhere in
the codebase (only the reverse, `neuronal_tensor_to_configuration`) — don't
invent one by passing a `Configuration` into `NeuronalTensor()`.

`NeuronType.make(name, *, relative_size=None, fraction=None, value_tag=...)` —
`fraction` declares an explicit population fraction; if **every** type in a
`Layer` declares one, those normalized fractions populate
`Configuration.metadata["area_layer_cell_types"][area][layer]` at construct
time; if any type omits it, the whole layer falls back to an even split
(backward-compatible all-or-nothing behavior).

## Loading / saving JSON

```python
tensor = jtfne.load(path)                       # canonical loader (raises ValueError if no top-level "areas" key)
jtfne.load_neuronal_tensor(path)                 # compat alias, prefer jtfne.load
jtfne.load_canonical_neuronal_tensor(name)        # named presets, e.g. "canonical-v1-column-1000n"
jtfne.list_canonical_neuronal_tensors()
jtfne.merge_neuronal_tensors(t1, t2, ...)
```

`schema_version` drift (present but not matching `NEURONAL_TENSOR_SCHEMA_VERSION`)
**warns, does not raise** — forward-readability, not strict lockstep. Missing
`"areas"` top-level key raises `ValueError` immediately.

Existing canonical config JSONs (all E/PV/SST/VIP, 6-layer): `jaxfne/configs/
canonical-v1-column-1000n.json`, `canonical-v1-v4-pfc-multiarea.json`,
`default-column.json`, `homeostatic-h-override-demo.json`,
`laminar-column-4layer.json`, `two-area-feedforward.json`,
`default_macaque_V1.json` (built 2026-06-30, LAP-manuscript-derived, AMPA+NMDA+GABA_A wired).
Legacy `Configuration`-schema archives live at `jaxfne/configs/legacy/`
(not NeuronalTensor-schema — do not load them via `jtfne.load()`).

`RuntimeConfiguration` (`neuronal_tensor.py`, frozen, execution-only:
seed/duration_ms/dt_ms/dtype/emitter/device/jit/vmap) is **distinct from**
`RuntimeConfig` (`core.py`, has `enable_hdp`/`hdp_params`). `RuntimeConfiguration`
has **no** HDP field.

## Enabling HDP (no new public API needed)

Build via `construct(tensor, RuntimeConfiguration(...))`, then pass an explicit
`runtime=RuntimeConfig(enable_hdp=True, hdp_params={...})` to `simulate()` — the
explicit `runtime=` kwarg overrides any `Configuration`-derived runtime:

```python
tensor = jtfne.load("jaxfne/configs/canonical-v1-column-1000n.json")
model = jtfne.construct(tensor, jtfne.RuntimeConfiguration(seed=0, duration_ms=1000.0, dt_ms=0.5))
from jaxfne.hdp_network import DEFAULT_HDP
sig = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0,
                      runtime=jtfne.RuntimeConfig(enable_hdp=True, hdp_params=dict(DEFAULT_HDP)))
```

## HDP module (`jaxfne/hdp_network.py`) — generic, config-driven, no per-N functions

```python
DEFAULT_HDP = dict(K_HDP=0.01, tau_0_ms=200.0, K_ctrl=5.0, rho_passive=0.0,
                    barrier_c=0.01, barrier_d=0.01)
```

`rho_passive=0.0` is canonical (F-017 sweep found no working non-zero window —
wild H oscillation <0.24, neuron silencing ≥0.36; `K_ctrl=5.0` is the tuned
restoring term instead).

`BASE_HDP_KWARGS_DEFAULT` (H_min=0.1, H_max=10.0, alpha=0.01, beta=0.0,
gamma=0.0, delta=0.0, C_spike=0.0, …), `BASE_DRIVE_BY_CELL_TYPE_DEFAULT =
{"E":4.0,"PV":4.0,"SST":4.0,"VIP":4.0}`, `DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT`.

Kernel: `simulate_edge_recurrent_izhikevich_hdp` (`emitters.py`) —
**`tau_i = tau_0_ms * size_i**3`** (cube law, verified; NOT `size_i**2`; a
`relative_size=2.0` neuron integrates H 8× slower). `hdp_params` is a free-form
dict forwarded through `core.py`'s `_hdp_packed` — any new key (e.g.
`size_scale_by_cell_type`, `size_scale_override`) must be explicitly added
there or it is silently dropped (verify with `grep -n size_scale_by_cell_type
jaxfne/core.py` before trusting a new key reaches the kernel).

`jaxfne.neuronal_tensor.DEFAULT_RELATIVE_SIZE` (re-exports
`emitters.DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE`, verified: `E=5.0, PV=1.0, Inl=1.0,
SST=1.5, Ing=1.5, VIP=1.5`) is the single source of truth for `NeuronType` sizes
and the HDP tau-law scaling — not re-exported at the top-level `jtfne` namespace,
import it from `jaxfne.neuronal_tensor` or `jaxfne.emitters` directly.

`model.last_hdp_diagnostics()` → dict with `H_trace`, weight trace, per-edge
`receptor_index`.

## Internal pure-function layer (`jaxfne/_pipeline.py`, Phase 0–3a)

Thin, tested wrappers for the tensor/HDP path — `load_tensor`, `save_tensor`,
`tensor_to_configuration`, `build_network`, `run_network`, `select_signal`
(added 2026-06-30, thin wrapper over `Signals.get`), `checkpoint_state`,
`restore_state`, `dynamic_state_from_model`, `compile_step_fn` + `scan_network`
(the real `Y, H(t+dt) = step(X(t), H(t))` `jax.lax.scan` pattern — HDP
edge-list backend only, not general-purpose across backends).
**Open TBIs (unresolved as of 2026-06-30):** `configuration_to_tensor`
(reverse converter, does not exist), `tensor_to_graph` (internal flattening
pass, not extracted from `construct()`). Not a public API yet.

## Related skills

- `jaxfne-config` — the `Configuration` fluent-builder path
- `jaxfne-neural-network` — `construct → simulate → Signals` runtime chain
- `jaxfne-spectrolaminar-suite` — scale path, crossover caveats
