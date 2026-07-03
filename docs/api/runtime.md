# Runtime API

Runtime configuration and execution helpers for neural simulations.

## Overview

The runtime module provides configuration objects for controlling JAX execution backend, device selection, numerical precision, and simulation parameters.

---

## RuntimeConfig

```python
jaxfne.RuntimeConfig(
    backend="auto", dtype="float32", jit=False, vmap=False,
    precision="default", seed=0, n_steps=0, recurrent_backend="dense",
    synaptic_kernel="exponential", recompilation_guard="warning",
    enable_homeostasis=False, homeostasis_params={...},
    enable_hdp=False, hdp_params={...},
)
```

A plain frozen-by-convention dataclass (no fluent `with_*` methods) holding
JAX execution/backend/dtype policy and emitter control. `dtype='float64'` is
honored only when JAX x64 is enabled globally; the manifest reports both the
requested and actual dtype policy.

### Fields

- `backend` (str): `"auto"` | `"cpu"` | `"gpu"` | `"tpu"`
- `dtype` (str): `"float32"` | `"float64"`
- `jit` (bool | str): `True`/`False`/`"auto"`
- `vmap` (bool | str): `True`/`False`/`"auto"`
- `precision` (str): `"default"` | `"high"`
- `seed` (int): default `0`
- `n_steps` (int): default `0`
- `recurrent_backend` (str): `"dense"` | `"edge_list"`
- `synaptic_kernel` (str): `"exponential"` | `"receptor_exponential"`
- `recompilation_guard` (str): `"warning"` | `"exception"` | `"off"`
- `enable_homeostasis` (bool): per-neuron activity-trace feedback, default `False`
- `homeostasis_params` (dict): `{r_star, tau_r_ms, alpha, k_gain, g_min, g_max, r_max}`; `k_gain=0` disables
- `enable_hdp` (bool): per-neuron master-state (H) plasticity controller, default `False`; mutually exclusive with `enable_homeostasis`
- `hdp_params` (dict): `{K_HDP, tau_0_ms, alpha, beta, gamma, delta, C_spike, K_ctrl, barrier_c, barrier_d}`; all gains default `0.0` (the null). See [the HDP guide](../guides/hdp.md)

To change a setting, construct a new `RuntimeConfig(...)` with the desired
fields — there is no `with_seed`/`with_dtype`/`with_device` mutator API.

**Example:**
```python
runtime_cfg = jtfne.RuntimeConfig(seed=42, dtype="float32")
```

---

## Configuring via Configuration

The preferred way to set runtime parameters is via the chainable Configuration API:

```python
import jaxfne as jtfne

cfg = jtfne.Configuration()
cfg = cfg.runtime(
    seed=7,
    dtype='float32',
    duration_ms=1000.0,
    dt_ms=0.1
)
```

This stores runtime metadata in the configuration's metadata dictionary, which is available when constructing and simulating.

---

## Numerical Precision

### float32 (Default)

```python
cfg = cfg.runtime(dtype='float32')
```

**Pros:**
- Faster computation
- Lower memory usage
- JAX default for most operations

**Cons:**
- Limited precision for long simulations
- May accumulate numerical error

**Use when:**
- Running tutorials or quick prototypes
- Training with large networks
- Comparing relative dynamics

### float64

```python
cfg = cfg.runtime(dtype='float64')
# or
jtfne.enable_x64()  # Global flag
```

**Pros:**
- High numerical precision
- Suitable for long simulations (>10s)
- Better for conservation checks

**Cons:**
- Slower computation (~2-4× slower)
- Higher memory usage

**Use when:**
- Validating conservation properties
- Long-duration simulations
- Requiring maximum numerical accuracy

---

## Random Seed & Reproducibility

### Setting the Seed

```python
cfg = cfg.runtime(seed=7)
# or
model = jtfne.construct(cfg)
signals = jtfne.simulate(model, seed=7)
```

**Property:** With same seed and configuration, simulations produce identical outputs (bitwise reproducible on same hardware/JAX version).

### Deterministic Simulation

```python
import jaxfne as jtfne

# Build config
cfg = jtfne.Configuration()
cfg = cfg.runtime(seed=42, dtype='float32', duration_ms=1000.0, dt_ms=0.1)
# ... configure network ...

# Construct model
model = jtfne.construct(cfg)

# Run simulation (multiple runs with same seed give identical results)
signals_1 = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=42)
signals_2 = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=42)

assert (signals_1.V_m == signals_2.V_m).all()  # Bitwise identical
```

### Varying Trials with Different Seeds

```python
for trial_idx in range(10):
    signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=trial_idx)
    # ... process signals ...
```

---

## Device Selection

The `RuntimeConfig`/`cfg.runtime(...)` field is **`backend`**, not `device` —
`cfg.runtime(device=...)` is silently ignored (`_runtime_config_from_metadata`
only reads the `backend`/`device_type` metadata keys, never `device`).

### CPU (Default)

```python
cfg = cfg.runtime(backend='cpu')
```

**Use when:**
- Prototyping small networks
- No GPU available
- Running on personal machines

### GPU

```python
cfg = cfg.runtime(backend='gpu')
```

**Requirements:**
- CUDA-capable GPU
- CUDA Toolkit and CuDNN installed
- JAX compiled with GPU support

**Performance:** Typically 10-100× faster for large networks

### TPU (Google Cloud)

```python
cfg = cfg.runtime(backend='tpu')
```

**Available on:** Google Colab, Google Cloud TPU pods

---

## Runtime Report

```python
jaxfne.runtime_report(runtime_config=None)
```

Get runtime environment information.

**Returns:** Dictionary with runtime details

**Contents (actual keys returned by `RuntimeConfig.runtime_report()`):**
- `jax_version` (str), `jaxlib_version` (str)
- `default_backend` (str): JAX's own default backend
- `available_devices` (list[str]): Available compute devices
- `selected_backend` / `backend` / `requested_backend` / `actual_backend` (str), `backend_enforced` (bool), `backend_warning` (str | None) — if a non-`"auto"` backend is requested but unavailable on this machine, `runtime_report()` performs an "honest downgrade": `actual_backend` falls back to JAX's real default, `backend_enforced=False`, and `backend_warning` is set to `requested_backend_unavailable:requested=...`  rather than falsely claiming the requested device is in use
- `requested_dtype` / `actual_dtype` / `dtype` (str), `x64_enabled` (bool)
- `jit` / `vmap` (bool), `precision` (str), `seed` (int), `n_steps` (int)
- `recurrent_backend` / `synaptic_kernel` (str)
- `enable_homeostasis` / `homeostasis_params`, `enable_hdp` / `hdp_params`

For a smaller, standalone device/dtype probe that doesn't require a `RuntimeConfig`, use
`get_jax_backend_report()` instead — keys `available_devices`, `default_backend`,
`x64_enabled`, `dtype_default` (it does NOT return jaxfne/numpy/python version or platform
info; no jaxfne helper currently reports those). See "jaxfne.runtime Module Helpers" below
for the access path — note it is `importlib.import_module("jaxfne.runtime")`, not
`jaxfne.runtime.get_jax_backend_report()` (that expression resolves to a different function
due to a name collision, see below).

**Example:**
```python
report = jtfne.runtime_report()
print(f"JAX version: {report['jax_version']}")
print(f"Available devices: {report['available_devices']}")
```

---

## jaxfne.runtime Module Helpers

`jaxfne/runtime.py` re-exports `RuntimeConfig`/`Configuration.runtime`/`runtime_report`
from `jaxfne/core.py` (see above) and additionally defines three standalone helpers
(`__all__` in `jaxfne/runtime.py`), independent of `RuntimeConfig`.

**Name-collision landmine:** `jaxfne.runtime` (attribute access on the `jaxfne` package)
resolves to the `runtime()` *function* (re-exported from `core.py`), not the `runtime.py`
*module* — this is deliberate, see the `_RuntimeModuleWrapper` in `jaxfne/__init__.py`.
Neither `jtfne.runtime.get_jax_backend_report()` nor `import jaxfne.runtime as jtfne_runtime`
reaches the module (both resolve `runtime` to the function via the same `__getattr__`
interception, so `jtfne_runtime.get_jax_backend_report()` raises `AttributeError`). The only
way to reach these helpers is an explicit `importlib.import_module`:

```python
import importlib
jtfne_runtime = importlib.import_module("jaxfne.runtime")
```

### `get_jax_backend_report()`

```python
report = jtfne_runtime.get_jax_backend_report()
```

Lightweight environment/device probe. Returns a dict with `available_devices` (list[str]),
`default_backend` (str), `x64_enabled` (bool), `dtype_default` (str). Does not require a
`RuntimeConfig`/`Configuration` — call it standalone to check the environment before
building anything.

### `set_precision_policy(dtype="float32", enable_x64=False)`

```python
result = jtfne_runtime.set_precision_policy(dtype="float64", enable_x64=True)
```

Sets JAX's *global* `jax_enable_x64` flag (process-wide, not per-`RuntimeConfig`). Returns
`{requested_dtype, actual_dtype, x64_enabled, status}`. If `dtype="float64"` is requested
without `enable_x64=True`, it warns (or raises under `JAXFNE_STRICT=1`) and falls back to
float32 rather than silently claiming float64. Call this once at startup, matching the
project-wide "x64 is a startup policy, never mid-code" rule — do not call it after arrays
have already been built.

### `safe_jit(fn, strict=False, **jit_kwargs)` / `safe_vmap(fn, in_axes=0, strict=False, **vmap_kwargs)`

```python
fast_fn = jtfne_runtime.safe_jit(my_fn)
batched_fn = jtfne_runtime.safe_vmap(my_fn, in_axes=0)
```

Wrap a function with `jax.jit`/`jax.vmap`; on compilation/vectorization failure, they warn
and return the original (eager/unbatched) function instead of raising — an honest fallback,
not a silent one, since a `RuntimeWarning` is always emitted. Pass `strict=True` (or set
`JAXFNE_STRICT=1`) to raise instead of falling back.

---

## Best Practices

1. **Set seed at configuration time:** Ensures reproducibility
2. **Use float32 by default:** Unless precision is critical
3. **Match precision across workflow:** Use consistent dtype throughout
4. **Check runtime report:** Verify JAX/device setup before long runs
5. **Document runtime choices:** Include seed and dtype in published results

**Example: Full Runtime Setup**

```python
import jaxfne as jtfne

# Configuration
cfg = jtfne.Configuration()
cfg = cfg.runtime(
    seed=42,
    dtype='float32',
    duration_ms=1000.0,
    dt_ms=0.1,
    backend='gpu'
)
cfg = cfg.column("V1", layers=["L2/3"], n=100)
cfg = cfg.cell_types({"E": 0.8, "I": 0.2})
cfg = cfg.connectivity()
cfg = cfg.set_emitter("izhikevich", "cortical_eig")
cfg = cfg.probes(["SPK", "Vm", "LFP-proxy"])

# Build model
model = jtfne.construct(cfg)

# Run simulation
signals = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.1, seed=42)

# Verify runtime environment
report = jtfne.runtime_report()
print(f"JAX {report['jax_version']}, backend={report['actual_backend']}")

# For a standalone device/dtype probe (see "jaxfne.runtime Module Helpers" above):
import importlib
jtfne_runtime = importlib.import_module("jaxfne.runtime")
backend_report = jtfne_runtime.get_jax_backend_report()
print(f"Devices: {backend_report['available_devices']}")
```

---

## See also

- [Core API](core.md) — Configuration and Model
- [Quickstart](../quickstart.md) — Getting started
- [API reference](index.md)
