# jaxfne HPC Refactoring Roadmap — Patch 202606

## Canonical object vocabulary

Use these names going forward:

```text
Config      declarative bio-circuit PCB sketch
Net         compiled executable biophysical circuit
Paradigm    task/trial/stimulus software
Objective   programmer's measure
Trainer     programmer/tuner loop
Signals     tensor outputs and query API
FlatNet     JAX-native array representation for JIT/vmap/pmap
```

Compatibility aliases for one release line:

```text
Configuration -> Config
Model         -> Net
FlatModel     -> FlatNet
```

## Module boundaries

```text
jaxfne.core           facade only
jaxfne.config         Config and typed sub-specs
jaxfne.identity       NodeIdentity, SelectorSpec
jaxfne.connectivity   MechanismSpec, ConnectionRuleSpec, compile_connection_rules
jaxfne.net            Net, construct, clone
jaxfne.weld           config/net welding and rename maps
jaxfne.flatten        FlatNet, TrackingMaps, flatten_net, unflatten_net
jaxfne.signals        Signals, SignalTensor, get_signal, layout conversion
jaxfne.paradigm       Paradigm, TrialSpec, EventSpec, stimulation mapping
jaxfne.objective      ObjectiveOutputSpec, ObjectiveResult, metrics/gates
jaxfne.optim.trainer  AGSDRTrainer, TrainingResult
jaxfne.fields         source/field/probe tensor operators
jaxfne.vis            raster/LFP/CSD/EEG/MEG/PSD/spectrogram/spectrolaminar/connectivity only
```

## Deprecation map

| Current surface | Future home | Deprecation policy |
|---|---|---|
| `Configuration` | `Config` | alias for one release line |
| `Model` | `Net` | alias for one release line |
| `FlatModel` if introduced | `FlatNet` | avoid adding `FlatModel` public name |
| `core.Paradigm*` | `jaxfne.paradigm` | re-export from core until 0.3.34 |
| `core.Objective*` | `jaxfne.objective` | re-export from core until 0.3.34 |
| `core.suite2_*` | `jaxfne.presets` / tutorials | keep wrappers, move implementations |
| `fields.proxy.spectrolaminar_psd` CPU path | `fields.spectral_numpy` | add `fields.spectral_jax` |
| `runtime.safe_jit` silent fallback | `runtime.safe_jit(..., fallback=...)` | default `fallback='raise'` in tests |

## Implementation ladder

### 0.3.28-hpc-a — contracts only

- Add contract module with TBI skeletons.
- No behavior changes.
- Tests assert TBI methods raise `>TBI-not-ready`.

### 0.3.28-hpc-b — Config sub-specs

- Add `Config` dataclass with typed sub-specs.
- Add `schema_version` and migration stubs.
- Keep `Configuration` wrapper.

### 0.3.29-hpc — identity and selectors

- Standardize `area_id:local_id:layer:cell_type` with six-digit local id.
- Add `SelectorSpec.resolve(neuron_table, allow_empty=False)`.

### 0.3.30-hpc — sparse connectivity compiler

- Add `pattern` field: `all_to_all`, `bernoulli`, `fixed_indegree`, `matrix`, `artifact_ref`.
- Add direct edge-array compiler.
- Dense weights become opt-in.

### 0.3.31-hpc — weld

- `weld_config` is canonical.
- `weld_net` follows after identity maps are stable.
- No cross-connections are created by weld unless explicit rules are added after weld.

### 0.3.32-hpc — construct/reconstruct/constant Paradigm

- Minimal `Paradigm.constant_dc(...)` lands before roundtrip simulation gates.
- `construct(cfg) -> Net -> to_config() -> construct()` preserves counts.

### 0.3.33-hpc — FlatNet/JIT/pmap

- Add `FlatNet` pytree.
- Add `simulate_flat_izhikevich(flat, runtime_static, drive, key)`.
- Use leading candidate axis for `vmap` and `pmap`.

### 0.3.34-hpc — gate

- Run full 35-condition gate plus transform tests.

## Runtime split

Use this shape for compiled functions:

```python
def simulate_flat_izhikevich(
    flat: FlatNet,
    runtime_static: RuntimeStatic,
    drive: jax.Array,
    key: jax.Array,
) -> SignalTensor:
    ...
```

`runtime_static` must be hashable/static. Dynamic arrays must be separate JAX arrays.

## Weight artifact policy

`artifact_ref.path` resolves relative to `Config.artifact_root` or the parent directory of the config file.

Load timing:

```text
compile-time for construct/flatten
never inside JIT
```

Missing or hash-mismatched files fail before simulation.

## Backend policy

```text
CPU: mandatory
CUDA GPU: optional but expected for Colab/A100 smoke
TPU: optional
Apple Silicon: run through JAX backend when available; MLX bridge is future optional adapter, not core dependency
```
