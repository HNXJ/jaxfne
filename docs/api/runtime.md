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

### CPU (Default)

```python
cfg = cfg.runtime(device='cpu')
```

**Use when:**
- Prototyping small networks
- No GPU available
- Running on personal machines

### GPU

```python
cfg = cfg.runtime(device='gpu')
```

**Requirements:**
- CUDA-capable GPU
- CUDA Toolkit and CuDNN installed
- JAX compiled with GPU support

**Performance:** Typically 10-100× faster for large networks

### TPU (Google Cloud)

```python
cfg = cfg.runtime(device='tpu')
```

**Available on:** Google Colab, Google Cloud TPU pods

---

## Runtime Report

```python
jaxfne.runtime_report(runtime_config=None)
```

Get runtime environment information.

**Returns:** Dictionary with runtime details

**Contents:**
- `jaxfne_version` (str): jaxfne package version
- `jax_version` (str): JAX version
- `numpy_version` (str): NumPy version
- `python_version` (str): Python version
- `platform` (str): OS and hardware info
- `available_devices` (list[str]): Available compute devices
- `default_device` (str): Default device for operations

**Example:**
```python
report = jtfne.runtime_report()
print(f"JAX version: {report['jax_version']}")
print(f"Available devices: {report['available_devices']}")
```

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
    device='gpu'
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
print(f"jaxfne {report['jaxfne_version']} on {report['platform']}")
```

---

## See also

- [Core API](core.md) — Configuration and Model
- [Quickstart](../quickstart.md) — Getting started
- [API reference](index.md)
