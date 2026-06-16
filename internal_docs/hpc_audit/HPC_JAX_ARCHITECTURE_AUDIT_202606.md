# jaxfne HPC/JAX Architecture Audit — 2026-06

## Scope

Input snapshot: uploaded `jaxfne-main (1).zip`, package version `0.3.27` in `pyproject.toml`.

Review target: maximize JAX purity, tensor-field mathematical validity, backend independence, transform compatibility (`jit`, `vmap`, `pmap`/`shmap`), and future `Config -> Net -> Paradigm -> Objective -> Trainer -> Signals -> Vis/Export` architecture.

Status boundary preserved:

```yaml
claim_level: computational_scaffold
field_solver_status: linear_solver
physical_amplitude_calibrated: false
```

## Executive assessment

`jaxfne` already has a strong JAX numerical core in `emitters.py`: the main Izhikevich paths use `jax.lax.scan`, `IzhikevichParams` is registered as a pytree, and `EdgeList` is a pytree dataclass. This is the best current foundation for GPU/XLA performance.

The bottleneck is not the core emitter kernels. The bottleneck is architectural mixing: `core.py` is a public facade, config container, model object, runtime adapter, paradigm layer, objective evaluator, optimizer loop, tutorial preset store, manifest schema adapter, and validation bridge in one file. That makes JIT boundaries less clear and makes future `pmap`/`shmap`/FlatNet work harder.

The next safe direction is not a rewrite. It is a staged split:

```text
jaxfne.core          -> thin facade + backward-compatible aliases
jaxfne.config        -> typed immutable specs and JSON schema
jaxfne.net           -> compiled executable model object
jaxfne.paradigm      -> task/trial/stimulus schedules
jaxfne.signals       -> named tensor outputs and layout conversion
jaxfne.objective     -> output metrics and loss/gate aggregation
jaxfne.optim.trainer -> trainable path apply/evaluate/fit/save/load
jaxfne.flatten       -> FlatNet pytrees and tracking maps
jaxfne.connectivity  -> selector rules, mechanisms, edge compiler
jaxfne.fields        -> source/leadfield/readout tensor operators only
jaxfne.vis           -> visualization only
```

## Critical bottlenecks and anti-patterns

### P0 — `core.py` is too broad for JAX transform hygiene

**Evidence**

- `jaxfne/core.py:461-558` defines `Configuration` as frozen but contains mutable `list`/`dict` fields and mutates with `object.__setattr__` in `__post_init__`.
- `jaxfne/core.py:2713-3005` defines `Model`, simulation cache, simulation dispatch, field projection, metadata assembly, and signal construction in one object.
- `jaxfne/core.py:3800-3865` performs optimizer candidate evaluation from inside the model class.
- `jaxfne/core.py:4863-4931` includes finite-difference/Adam tutorial optimization logic in the same module.

**Risk**

- JIT cache is object-owned and mutation-backed (`_compiled_cache`), which is useful now but weak for pure functional APIs.
- Model objects are not pytrees and are not a stable JAX transform boundary.
- Config metadata is a dict bag, so static vs dynamic values are not explicit.

**Fix**

Introduce typed `Config`, `Net`, and `FlatNet` modules. Keep `Configuration` and `Model` aliases for one release line.

---

### P0 — object-local JIT cache causes hidden side effects

**Evidence**

- `jaxfne/core.py:2802-2827` mutates `self._compiled_cache` inside `_simulate_arrays`.
- `jaxfne/core.py:2833-2858` repeats the same cache mutation for the dense path.
- `jaxfne/core.py:3068-3091` mutates `_compiled_cache` in `simulate_batch`.

**Risk**

- The object becomes stateful after a first simulation call.
- Cache key construction omits some static values that can affect compiled behavior, such as edge count and schedule shape details.
- Object mutation conflicts with clean functional composition and makes reproducibility reports less direct.

**Fix**

Move compilation into `jaxfne.runtime.compile_simulator(flat, runtime_static)` and use explicit `RuntimeStatic` keys. Keep object cache as a compatibility wrapper only.

---

### P0 — NumPy/SciPy paths block JIT and backend independence in field/spectrolaminar utilities

**Evidence**

- `jaxfne/fields/proxy.py:14-15` imports NumPy and SciPy at module import time.
- `jaxfne/fields/proxy.py:787-818` converts signal data to NumPy and uses SciPy Welch or NumPy FFT in `spectrolaminar_psd`.
- `jaxfne/objectives.py:115-149` uses `np.random` for null generation.
- `jaxfne/objectives.py:195-208` computes binned correlations with Python loops and NumPy correlation.

**Risk**

- CPU fallback becomes hidden in functions that look like JAX functions.
- GPU/TPU execution cannot include spectrolaminar PSD or null generation.
- Reproducibility depends on global NumPy RNG state.

**Fix**

Provide separate paths:

```text
spectrolaminar_psd_jax(...)     # JAX/XLA path, shape-static
spectrolaminar_psd_numpy(...)   # CPU/reporting path
null_distribution_jax(key, ...) # deterministic JAX RNG
```

---

### P1 — `safe_jit` / `safe_vmap` hide transform failures

**Evidence**

- `jaxfne/runtime.py:103-123` returns the original function if `jax.jit` raises.
- `jaxfne/runtime.py:126-147` returns the original function if `jax.vmap` raises.

**Risk**

- Performance regressions are silent.
- A function can appear JIT-compatible while running eagerly.

**Fix**

Add a strict mode default for internal tests:

```python
safe_jit(fn, fallback="raise")
safe_vmap(fn, fallback="raise")
```

Keep fallback mode only for user notebooks with an explicit report field.

---

### P1 — dense matrices remain the default connection compiler target

**Evidence**

- `jaxfne/core.py:384-399` builds dense `W` using Python list comprehensions over labels.
- `jaxfne/core.py:5016` converts dense `W` to an `EdgeList` after construction.
- `jaxfne/fields/proxy.py:570-605` builds multiple dense connectivity matrices for tutorial laminar connectivity.

**Risk**

- Memory scales as `O(N^2)` before converting to sparse edges.
- Large models will OOM on GPU before the edge path receives sparse arrays.

**Fix**

Make connection rules compile directly to edge arrays:

```text
ConnectionRuleSpec -> edge_pre, edge_post, edge_weight, edge_mechanism
```

Do not build dense `W` unless explicitly requested by `weights(format="dense")`.

---

### P1 — precision policy is globally mutable

**Evidence**

- `jaxfne/runtime.py:62-97` mutates `jax.config.update("jax_enable_x64", True)`.
- `jaxfne/core.py:5504` exposes `enable_x64()`.

**Risk**

- JAX's X64 flag is process-global and should be set at program start.
- Late mutation can confuse tests and Colab notebooks.

**Fix**

Keep this as a startup helper only. Add warnings if called after arrays/devices are already initialized.

---

### P2 — custom containers are partially pytree-compatible

**Evidence**

- `jaxfne/emitters.py:78-156`: `IzhikevichParams` is registered manually as pytree.
- `jaxfne/emitters.py:437-466`: `EdgeList` is a pytree class.
- `jaxfne/core.py:1495-1504`: `Signals` is not a pytree.
- `jaxfne/core.py:2713-2718`: `Model` is not a pytree.
- `jaxfne/fields/proxy.py:19-32`: `FieldOutput` is not a pytree.

**Risk**

- Some data structures are transform-safe; others become opaque leaves.
- `Signals` and `FieldOutput` cannot be naturally mapped or donated.

**Fix**

Use `FlatNet`, `SignalTensor`, and `FieldTensor` pytrees for numerical paths; leave high-level `Net` and rich reports as Python objects.

---

### P2 — module import cost and optional dependencies

**Evidence**

- `jaxfne/fields/proxy.py:15` imports `scipy.signal` at module import.
- `jaxfne/vis/*` pulls plotting dependencies on demand in most places, which is better.

**Risk**

- Core import may fail or slow if SciPy is absent or heavy.

**Fix**

Make SciPy import lazy inside CPU-only helper functions.

## Refactoring scope catalogue

### Stage 1: introduce contracts without behavior mutation

Add experimental modules with TBI functions that fail loudly:

```text
jaxfne/experimental_hpc/contracts.py
```

This patch includes that file. It does not change existing behavior.

### Stage 2: typed config and schema version

Add canonical `Config` with typed sub-specs:

```text
RuntimeSpec, GeometrySpec, CircuitSpec, ProbeSpec, ParadigmSpec, ObjectiveSpec, OptimizerSpec
```

Keep `Configuration` as an alias/wrapper.

### Stage 3: Net and FlatNet split

`Net` owns Python-side metadata and user-facing methods.
`FlatNet` owns only JAX arrays and static maps.

### Stage 4: connection compiler

Add direct sparse edge compilation. Avoid dense all-to-all matrices unless explicitly requested.

### Stage 5: JAX-native analysis kernels

Add JAX PSD, JAX nulls, vectorized synchrony metrics, and shape-static bandpower.

### Stage 6: pmap/shmap/pjit readiness

Make candidate dimension the leading axis for optimizer population sweeps. Shard candidates, replicate model arrays unless memory requires partitioned edges.

## Testing scope

### Transform tests

```text
- jit(emitters.simulate_*): no tracer leaks
- vmap over seeds/candidates
- grad through differentiable surrogate paths only
- pmap/shmap smoke when >=2 devices are visible
```

### PyTree tests

```text
- tree_flatten/tree_unflatten FlatNet
- tree_map over SignalTensor
- donation smoke for large arrays
```

### Backend tests

```text
- CPU mandatory
- CUDA optional GitHub/self-hosted or Colab
- Apple Silicon optional local runner
- dtype policy: float32 default, float64 only when x64 enabled at startup
```

### Numerical tolerance

```text
spikes: exact equality for deterministic seed and same backend
voltage/source: rtol=1e-5, atol=1e-5 for float32 smoke
PSD/bandpower: rtol=1e-4, atol=1e-6 due FFT/window differences
optimizer scores: compare bounded metrics, not full histories
```

## Pass criteria for the HPC refactor line

```text
1. core.py shrinks to facade/aliases.
2. FlatNet contains only JAX arrays + static maps.
3. Net.simulate delegates to pure functional kernels.
4. All numerical kernels accept explicit PRNGKey, dtype, and static runtime.
5. No NumPy/SciPy is used inside JIT paths.
6. No plotting/JSON/file I/O is called inside JIT paths.
7. Signals/Field tensors have JAX-native layout conversion.
8. Connection rules compile directly to sparse edge arrays.
9. Dense weights are opt-in.
10. Pytree tests pass for FlatNet and SignalTensor.
11. vmap over candidate population passes.
12. pmap/shmap smoke passes when multi-device is available.
13. PyPI release artifacts match tested SHA256 hashes.
14. Truth/status gates remain unchanged.
```
