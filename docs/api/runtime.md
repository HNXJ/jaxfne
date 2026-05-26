# Runtime API

Runtime configuration and execution helpers for neural simulations.

## Overview

The runtime module provides configuration objects for controlling JAX execution backend, device selection, numerical precision, and simulation parameters.

---

## RuntimeConfig

```python
jaxfne.RuntimeConfig(seed=None, dtype='float32', **kwargs)
```

Execution backend and device settings for simulations.

### Attributes

- `seed` (int, optional): Random seed for JAX PRNG
- `dtype` (str): JAX dtype ('float32', 'float64')
- `device` (str, optional): Compute device ('cpu', 'gpu', 'tpu')
- `enable_x64` (bool): Enable 64-bit precision (default: False)
- `xla_flags` (dict, optional): XLA compiler flags

### Methods

#### `with_seed(seed: int) -> RuntimeConfig`

Create a new RuntimeConfig with specified seed.

**Parameters:**
- `seed` (int): New random seed

**Returns:** Updated `RuntimeConfig`

**Example:**
```python
runtime_cfg = jtfne.RuntimeConfig(seed=42)
runtime_cfg_new = runtime_cfg.with_seed(100)
```

#### `with_dtype(dtype: str) -> RuntimeConfig`

Create a new RuntimeConfig with specified dtype.

**Parameters:**
- `dtype` (str): JAX dtype ('float32' or 'float64')

**Returns:** Updated `RuntimeConfig`

**Example:**
```python
runtime_cfg = jtfne.RuntimeConfig(dtype='float32')
runtime_cfg_f64 = runtime_cfg.with_dtype('float64')
```

#### `with_device(device: str) -> RuntimeConfig`

Create a new RuntimeConfig with specified device.

**Parameters:**
- `device` (str): Device type ('cpu', 'gpu', 'tpu')

**Returns:** Updated `RuntimeConfig`

**Example:**
```python
runtime_cfg = runtime_cfg.with_device('gpu')
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

## XLA Compiler Flags

Advanced: Control JAX/XLA compiler behavior:

```python
runtime_cfg = jtfne.RuntimeConfig(
    xla_flags={
        'xla_force_host_platform_device_count': 4,  # Simulate multi-device
        'xla_gpu_autotune_level': 2,  # Autotuning aggressiveness
    }
)
```

**Common flags:**
- `xla_gpu_autotune_level`: Autotuning (0-4, higher = more thorough)
- `xla_dump_to`: Debug output directory
- `xla_force_host_platform_device_count`: Virtual device count

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
