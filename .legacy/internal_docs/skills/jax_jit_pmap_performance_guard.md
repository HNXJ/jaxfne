---
summary: Use for JAX/JIT/vmap/pmap/pjit performance, FlatNet arrays, scan loops, GPU/A100 work, sharding, and compile-time guards.
trigger: Use whenever the task mentions JAX, jit, jax.jit, lax.scan, vmap, pmap, pjit, shard, GPU, A100, Colab GPU, performance, benchmark, FlatNet, flatten, trace, static_argnums, compile time, OOM, batching, vectorization.
---

# JAX JIT PMAP Performance Guard

## JAX-native rules

- Numerical kernels use `jax.numpy` arrays.
- Use explicit PRNG keys.
- Time loops use `jax.lax.scan`.
- Candidate/trial/neuron batches use `vmap` when shape-stable.
- `pmap`/`pjit` requires FlatNet or equivalent array-only structures.
- Plotting, JSON, pandas, file I/O, markdown, and Python object mutation stay outside JIT.

## FlatNet contract

Dynamic traced arrays:

```text
neuron_params [N, P]
positions [N, 3]
area_id/layer_id/cell_type_id [N]
edge_pre/edge_post/edge_weight/edge_mechanism [E]
probe/readout arrays [C, ...]
```

Static tracking metadata:

```text
row_to_quartet
global_id_to_row
area_vocab
layer_vocab
cell_type_vocab
mechanism_vocab
edge_rule_origin
```

## Runtime split

Do not pass arbitrary `Mapping[str, Any]` through JIT. Split:

```text
runtime_static: Python metadata marked static or closed over
runtime_dynamic: JAX arrays/scalars
```

## Equivalence testing

Flat path vs non-flat smoke path must declare tolerances:

```text
spikes: exact or event-count tolerance if stochastic path differs by key split
voltage/source: rtol <= 1e-4, atol <= 1e-5 for deterministic smoke
readout proxies: rtol <= 1e-4, atol <= 1e-5
```

Pin seed, dtype, dt, n_steps, and key splitting.

## Memory/scale checks

Report:

```text
N nodes
E edges
edge density
array dtypes
estimated bytes
compile time
run time
peak memory if available
```

Prefer sparse edge arrays over dense `[N, N]` weights for large networks.

## Stop conditions

```text
Python object mutation inside JIT
shape changes across scan steps
non-JAX arrays in numerical hot path
unbounded dense connectivity for large N
JIT function returns Python objects
```
