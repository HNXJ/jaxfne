# Sharding Utilities API

Multi-device (multi-GPU/TPU) population-sweep sharding for jaxfne.

```python
from jaxfne.sharding_utils import (
    make_population_mesh,
    make_candidate_sharding,
    make_replicated_sharding,
    get_sharding_context,
)
```

These utilities enable distributing parameter sweeps across JAX devices.
On single-device environments (CPU, single GPU), all functions return `None`
so downstream code can gate sharding with a simple `if ctx is not None` check.

---

## `make_population_mesh`

```python
def make_population_mesh() -> jax.sharding.Mesh | None
```

Return a 1-D named `Mesh` across all visible JAX devices, with axis name
`"population_sweep"`.

Returns `None` on single-device environments.

```python
mesh = make_population_mesh()
if mesh is not None:
    with mesh:
        pass  # distribute work here
```

---

## `make_candidate_sharding`

```python
def make_candidate_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding
```

Return a `NamedSharding` that slices the first (batch/population) axis across
`mesh`. Shape convention: `(B, n_params)` — partition on axis 0, replicate
axis 1.

**Returns:** `PartitionSpec("population_sweep", None)`

Use for candidate parameter arrays where each device handles a slice of the
population (e.g. AGSDR/SDR sweep candidates).

---

## `make_replicated_sharding`

```python
def make_replicated_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding
```

Return a `NamedSharding` that fully replicates an array across all devices in
`mesh`.

**Returns:** `PartitionSpec(None)`

Use for model-parameter tensors (weight matrices, emitter params) that must
not be partitioned, to avoid cross-device gradient communication overhead.

---

## `get_sharding_context`

```python
def get_sharding_context() -> dict | None
```

Convenience wrapper. Returns a dict with all three sharding objects, or
`None` on single-device environments.

**Returns:**

```python
{
    "mesh":       Mesh,           # the logical device mesh
    "candidate":  NamedSharding,  # slice batch dim across devices
    "replicated": NamedSharding,  # replicate on every device
}
```

**Pattern:**

```python
ctx = get_sharding_context()
if ctx is not None:
    with ctx["mesh"]:
        candidate_arr = jax.device_put(candidate_arr, ctx["candidate"])
        model_params  = jax.device_put(model_params,  ctx["replicated"])
```
