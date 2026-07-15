---
name: jaxfne-neural-tensor
description: >-
  Verified NeuronalTensor build path (Areas × Layers × NeuronTypes ×
  AreaConnections) and HDP homeostatic-plasticity module. USE when building or
  editing a NeuronalTensor circuit, enabling HDP, or naming a NeuronalTensor
  symbol. As of 2026-07-03, NeuronalTensor/Area raise TypeError at
  construction time on wrong-typed elements (closes the old silent
  type-confusion landmine — NeuronalTensor(some_configuration) now raises
  immediately instead of corrupting state). There is still no
  Configuration -> NeuronalTensor converter (deliberately out of scope —
  Configuration is the simpler tier, promoting it up adds no information;
  see AGENTS.md § "Config complexity tiers").
---

# jaxfne Neural Tensor

USE FIRST: `catalog-glossary-jaxfne` §1b/§1c.

## What it is

`NeuronalTensor = [Areas, AreaConnections]`, `Area = [Layers × NeuronTypes, InterConnections]`.
This is the structured "IC/PCB schematic" tier — `Area`/`Layer`/`NeuronType` are
elements, `InterConnection`/`AreaConnection` are wires (see `AGENTS.md` §
"Config complexity tiers" for how this relates to `Configuration` and
`HDPColumnConfig`). It is a **separate** build path from `Configuration`
(`jaxfne-config`) — both converge on the same `Model` via `construct()`,
dispatching on input type:

```python
model = jtfne.construct(cfg)                                    # Configuration path
model = jtfne.construct(tensor, RuntimeConfiguration(...))      # NeuronalTensor path
```

`NeuronalTensor`/`Area` validate their element types at construction (`__post_init__`,
added 2026-07-03) — passing the wrong type (e.g. a `Configuration` where an `Area`
belongs) raises `TypeError` immediately with a clear message, rather than silently
producing a malformed tensor that fails confusingly later.

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

`NeuronalTensor`/`Area` validate their element types in `__post_init__` (added
2026-07-03, see above) — `NeuronalTensor(some_cfg)` now raises `TypeError`
immediately instead of the pre-2026-07-03 landmine (silently storing
`tensor.areas == some_cfg`, a `Configuration`, with zero error). `Layer`/
`NeuronType`/`InterConnection`/`AreaConnection` remain plain frozen dataclasses
without their own `__post_init__` checks — the two-level validation (top-level
`NeuronalTensor`/`Area`) catches the documented landmine case; nested
mis-construction at the `Layer`/`NeuronType` level is not separately guarded.
There is no `Configuration -> NeuronalTensor` converter anywhere in the
codebase (only the reverse, `neuronal_tensor_to_configuration`) — this is
deliberate, not a gap (see `AGENTS.md` § "Config complexity tiers").

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
`RuntimeConfig` (moved 2026-07-03 from `core.py` to `jaxfne/_runtime_config.py`,
re-exported from `jaxfne.core` unchanged — `from jaxfne.core import RuntimeConfig`
still works; has `enable_hdp`/`hdp_params`). `RuntimeConfiguration` has **no** HDP field.

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

**Two more landmines found 2026-07-14, building a small (N=20) chained-turn
HDP habituation test:**

1. **`Model.simulate(sim)` (the OO method) does NOT get the same
   `cfg`-derived runtime inheritance as the top-level `jtfne.simulate(model,
   ...)` shown above.** Verified via source: the top-level free function
   (`_construct.py`'s module-level `simulate()`) calls
   `_runtime_config_from_metadata(cfg.metadata)` when no explicit `runtime=`
   is given, so a `Configuration.hdp(...)`/`.runtime(...)` declaration takes
   effect automatically — but `Simulation.resolved_runtime` (what
   `Model.simulate()` actually uses) never touches `cfg.metadata` at all; a
   bare `Simulation(...)` there silently gets `enable_hdp=False` regardless
   of what the `Configuration` declared. If you need the OO `.simulate()`
   call (e.g. because you're also calling `Model.with_hdp_initial_state()`,
   which only exists on `Model`), pass `runtime=RuntimeConfig(enable_hdp=True,
   hdp_params={...})` explicitly every call — don't rely on `cfg.hdp(...)`
   alone once you're off the top-level-function path.

2. **`Model.with_hdp_initial_state(H0=..., w0=...)` does NOT give true
   turn-to-turn simulation continuity.** It only carries `H` and edge weight
   `w` forward — membrane voltage `v`, recovery `u`, `prev_spikes`, and
   synaptic state `syn_state` are hard-reset to the model's native
   `v0`/`u0`/zero on every call (see `_model.py`'s `init_state` construction
   in the HDP dispatch branch). Combined with an identical PRNG seed across
   calls, this can produce bit-identical firing-rate/oscillation output
   across "chained" turns despite `H`/`w` visibly drifting — a real trap for
   anyone building a genuinely continuous multi-turn simulation. The verified
   full-state path is `jaxfne._pipeline.compile_step_fn`/`scan_network`'s
   `DynamicState` (`v, u, prev_spikes, syn_state, H, w` — all six carried),
   documented above under "Internal pure-function layer" but easy to miss
   since `with_hdp_initial_state` looks like it should be sufficient.

## HDP module (`jaxfne/hdp_network.py`) — generic, config-driven, no per-N functions

```python
DEFAULT_HDP = dict(K_HDP=0.01, tau_0_ms=200.0, K_ctrl=5.0, rho_passive=0.0,
                    barrier_c=0.01, barrier_d=0.01)
```

`rho_passive=0.0` is canonical (F-017 sweep found no working non-zero window —
wild H oscillation <0.24, neuron silencing ≥0.36; `K_ctrl=5.0` is the tuned
restoring term instead).

**`barrier_c`/`barrier_d` ratio gap (found 2026-07-14, F-029):** `barrier_c==
barrier_d==0.01` is a 1:1 ratio, but `emitters.py`'s own docstring says the
barrier potential's minimum only coincides with `H*=1` at a 100:1 ratio
(`barrier_d/barrier_c=((H_max-1)/(1-H_min))**2=100` at the canonical
`H_min=0.1`/`H_max=10.0`). Left as-is deliberately, not a bug to "fix" by
rebalancing — `DEFAULT_HDP`'s own verified dynamics keep H tightly pinned
near `H*=1` via `K_ctrl`, so H rarely nears either boundary and this
asymmetry is dormant in practice. Re-tune `barrier_d` before relying on
barrier repulsion near a boundary in a new use case.

**`K_w_ctrl=0.0` unbounded-drift gap (found 2026-07-14):** `DEFAULT_HDP`
ships `K_w_ctrl=0.0` (no weight-restoring force). Verified this permits
unbounded `|w|` growth on long/custom runs outside the presets already
checked (80s continuous run on a custom 20-neuron all-to-all network: `|w|`
grew monotonically the entire window, no asymptote). `K_w_ctrl=0.001` (the
repo's own verified value, from `DEFAULT_HDP_V1_PFC_AAAB`) over-corrects on
a different topology — collapses weight differentiation ~150x vs a
no-restoring control. If building a new long-horizon HDP network directly
on `DEFAULT_HDP`, sweep `K_w_ctrl` for that specific topology rather than
assuming either `0.0` or `0.001` is safe by default.

`BASE_HDP_KWARGS_DEFAULT` (H_min=0.1, H_max=10.0, alpha=0.01, beta=0.0,
gamma=0.0, delta=0.0, C_spike=0.0, …), `BASE_DRIVE_BY_CELL_TYPE_DEFAULT =
{"E":4.0,"PV":4.0,"SST":4.0,"VIP":4.0}`, `DRIVE_CORRECTION_BY_CELL_TYPE_DEFAULT`.

Kernel: `simulate_edge_recurrent_izhikevich_hdp` (`emitters.py`) —
**`tau_i = tau_0_ms * size_i**3`** (cube law, verified; NOT `size_i**2`; a
`relative_size=2.0` neuron integrates H 8× slower). `hdp_params` is a free-form
dict forwarded through `_model.py`'s `_hdp_packed` dispatch (moved here from
`core.py` during the 2026-07-04/05 monolith split; core.py itself is now a
233-line pure re-export aggregator — see `jaxfne-worker-context-router`'s
ownership map) — any new key (e.g. `size_scale_by_cell_type`,
`size_scale_override`, `K_w_ctrl`) must be explicitly added there or it is
silently dropped (verify with `grep -n size_scale_by_cell_type
jaxfne/_model.py` before trusting a new key reaches the kernel — this exact
bug bit `K_w_ctrl`'s first implementation pass, see below).

**`K_w_ctrl` (added 2026-07-04) — the weight-magnitude analogue of `K_ctrl`.**
Real bug found and fixed 2026-07-04: `H` already had a two-sided restoring
term (`K_ctrl`) pulling it toward equilibrium, but synaptic weight magnitude
had none — only a hard floor/ceiling clip — so carrying weights trial-to-trial
(`Model.with_hdp_initial_state(H0=..., w0=...)`) was an unbounded
positive-feedback runaway (rate → ~50Hz by trial 15-19). `K_w_ctrl` adds
`dwmag/dt += K_w_ctrl*(wmag_baseline - wmag)` (`wmag_baseline = |edges.weight|`,
the network's declared wiring, not the carried value), mirroring `K_ctrl`'s own
form one level down. Default `K_w_ctrl=0.0` (fully backward compatible).
Verified stable at 100 chained trials with `K_w_ctrl=0.001`: weight magnitude
held flat (range 6e-6 across all 100 trials), rate held at ~12.53-12.55Hz, zero
NaN. Named, reproducible presets exist for this — don't hand-roll a magic
`K_w_ctrl` constant:

```python
DEFAULT_HDP_DESYNC = dict(K_HDP=0.01, tau_0_ms=5.0, K_ctrl=0.15, rho_passive=0.0,
                           barrier_c=0.01, barrier_d=0.01, alpha=0.05, gamma=0.5, C_spike=0.0)
DEFAULT_HDP_V1_PFC_AAAB = dict(DEFAULT_HDP_DESYNC, K_HDP=0.003, K_w_ctrl=0.001)
from jaxfne.hdp_network import v1_pfc_aaab_hdp_params  # full BASE + preset + size-scale assembly
```

`DEFAULT_HDP_DESYNC` ("responsive H" — faster `tau_0_ms=5` + rate-drain
`gamma=0.5`, vs. `DEFAULT_HDP`'s near-static `tau_0_ms=200`) is the family
`DEFAULT_HDP_V1_PFC_AAAB` builds on; both are real, named presets in
`jaxfne/hdp_network.py`, not one-off script constants.

`jaxfne.neuronal_tensor.DEFAULT_RELATIVE_SIZE` (re-exports
`emitters.DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE`, verified: `E=5.0, PV=1.0, Inl=1.0,
SST=1.5, Ing=1.5, VIP=1.5`) is the single source of truth for `NeuronType` sizes
and the HDP tau-law scaling — not re-exported at the top-level `jtfne` namespace,
import it from `jaxfne.neuronal_tensor` or `jaxfne.emitters` directly.

`model.last_hdp_diagnostics()` → dict with `H_trace`, weight trace, per-edge
`receptor_index`.

### Footgun: `PlasticParams.H` defaults to 0.0, outside the valid HDP range

`PlasticParams.H` (on `InterConnection`/`AreaConnection`, `neuronal_tensor.py`)
defaults to **`0.0`** — outside `BASE_HDP_KWARGS_DEFAULT`'s valid homeostatic
range (`H_min=0.1, H_max=10.0`). `construct_neuronal_tensor` averages every
connection's `plastic.H` touching a target neuron (mean across
`InterConnection`/`AreaConnection`, untouched neurons default to the HDP
equilibrium `1.0`) into that neuron's initial HDP state via
`Model.with_hdp_initial_state`. **Any `InterConnection`/`AreaConnection`
declared without an explicit `plastic=PlasticParams(H=1.0)` silently seeds
that neuron's H0 outside its valid range** — this is inert until HDP is
enabled (`RuntimeConfig.enable_hdp=True`, default `False`), but once enabled
it can blow up HDP integration to NaN from step 0 (verified: single-area HDP
with no `H` set = fine; adding a 2-area `AreaConnection` with the `H=0.0`
default = NaN immediately). Always set `plastic=PlasticParams(H=1.0, ...)`
explicitly on every `InterConnection`/`AreaConnection` if HDP will be enabled.

## Per-layer neuron count IS preserved by the tensor→Configuration bridge (fixed 2026-07-04)

`neuronal_tensor.py::neuronal_tensor_to_configuration` used to call the plain
`.column(area.name, layers=layer_names, n=area_n)` (total area count only),
which meant a `NeuronalTensor`'s declared per-layer `Layer.n_neurons` was
**silently ignored** — the constructed model split the area's total count
across layers using jaxfne's hardcoded default (`core._SUITE2_LAYER_FRACTIONS`:
L1=10%, L2=15%, L3=20%, L4=10%, L5=30%, L6=15%) regardless of what you
declared. Fixed by routing through `Configuration.population(area_n,
neurons={layer.name: layer.n_neurons, ...}, name=area.name, layers=...)`
instead — this records `metadata["area_layer_count_frac"][area]`, which
`core._area_layer_count_frac` resolves *ahead of* the thickness fallback.
Verified: a declared L1=10/L2=25/L3=15/L4=15/L5=20/L6=15 area now constructs
with exactly those per-layer counts (previously silently became
L1=10/L2=15/L3=20/L4=10/L5=30/L6=15 for any 100-neuron area, regardless of
what was declared). Per-layer cell-type *composition* was always preserved
correctly (via `.area_layer_cell_types(...)`); it was only layer *size* that
was affected.

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

## Construct-once / checkpoint / reload (verified 2026-07-01 — 2 real landmines)

`construct()` is the expensive step at scale (231s at N=100k vs. ~5s to
simulate 100 steps) — a checkpoint/reload pattern avoids paying it twice, but
two non-obvious things WILL silently corrupt the reload if skipped:

1. **A treedef from a differently-sized dummy `construct()` is NOT a safe
   substitute for the real model's treedef.** `IzhikevichParams.labels`/
   `layer_labels` are pytree **aux data** (length N, baked into the treedef
   itself, not leaves) — `jax.tree_util.tree_unflatten(dummy_treedef,
   real_leaves)` does not error, it silently produces a model whose aux
   metadata (labels) mismatches its array lengths, and `simulate()` then
   produces different-but-still-finite output with no error. Caught by a
   real bit-identical-output check that failed, not by inspection.
2. **`construct()` mutates `Configuration.metadata` in place** (adds a
   `recurrent_backend` key, flips `circuit.connections[*].status`) —
   reusing a pre-construct `cfg`'s metadata after a checkpoint reload
   reproduces a *different*, still-finite `V_m` trace, not an error.

**The safe pattern** (verified bit-identical across separate processes, at
N=2000 and N=20000): persist `model.params["emitter"]`/`edge_list`/
`positions`'s raw arrays + their non-array aux fields (`labels`,
`layer_labels`, calibration strings) + `model.cfg.metadata` explicitly, then
reconstruct the dataclasses **directly** (`IzhikevichParams(...)`,
`EdgeList(...)`) rather than via any `tree_unflatten` shortcut — see
`scripts/cortical_column_localized_workflow.py::save_column`/`load_column`
for the reference implementation. Full detail: `skills/FRICTIONS_STACK.md`
F-020/F-021.

`jaxfne/_pipeline.py::checkpoint_state`/`restore_state` (above) predate this
fix and only save `model.params`/`model.static` — if reusing them directly,
also persist `model.cfg.metadata` yourself and reconstruct via a fresh
`Model` with **matching N** (per their own docstring), not a dummy.

## Related skills

- `jaxfne-config` — the `Configuration` fluent-builder path
- `jaxfne-neural-network` — `construct → simulate → Signals` runtime chain
- `jaxfne-spectrolaminar-suite` — scale path, crossover caveats
