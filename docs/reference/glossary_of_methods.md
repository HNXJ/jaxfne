# Glossary of methods

This glossary defines the package-level terms used by `jaxfne` docs, examples, tests, and manifests.

## Objective grammar

```text
Emitter -> Source -> FieldProxy -> Probe -> Objective -> Optimizer -> Manifest
```

| Term | Meaning in `jaxfne` |
|---|---|
| `Emitter` | Local state generator for voltage, spikes, synaptic traces, or typed neural state. |
| `Source` | Declared source tensor derived from emitter state under one source mode per run. |
| `FieldProxy` | Fixed computational projection from source tensors to named proxy readouts. |
| `Probe` | Selector and sampling operator for a readout family and channel layout. |
| `Objective` | Metrics, gates, nulls, and scores computed from readouts. |
| `Optimizer` | Bounded parameter search or gradient path over declared objectives. |
| `Manifest` | Strict JSON receipt for runtime, artifacts, hashes, truth gates, and validation status. |

## Numerical integration

### Forward Euler and scan integration

```math
y_{n+1} = y_n + \Delta t\, f(y_n, t_n)
```

`jaxfne` uses explicit state updates for emitter dynamics and wraps hot time loops with `jax.lax.scan` where practical.

## Proxy readout equations

### Source projection

```math
\Phi_{k,c} = \sum_n S_{k,n} W_{c,n}
```

`S` is the source tensor, `W` is a declared projection kernel, `k` indexes time or sample, `c` indexes channels, and `n` indexes source support.

### LFP proxy

```math
\mathrm{lfp\_proxy}_{k,c} = \sum_n S_{k,n} K_{c,n}
```

The kernel `K` encodes the laminar or spatial projection used by the run.

### CSD proxy

```math
\mathrm{csd\_proxy}_{k,c} = D_{zz}\Phi_{k,c}
```

`D_{zz}` is the declared depth-axis finite-difference operator.

### EEG and MEG proxies

```math
\mathrm{eeg\_proxy}_{k,c} = \sum_n S_{k,n} L^{\mathrm{eeg}}_{c,n}
```

```math
\mathrm{meg\_proxy}_{k,c} = \sum_n S_{k,n} L^{\mathrm{meg}}_{c,n}
```

`L` is the fixed readout kernel declared for the run.

## Runtime methods

| Method | Package behavior |
|---|---|
| `jax.numpy` kernels | numerical arrays and vectorized tensor operations |
| explicit PRNG keys | deterministic seeded simulations and search paths |
| `jax.lax.scan` | hot time loops for emitter and trace updates |
| `jax.vmap` | candidate, seed, readout, and batch axes where supported |
| `jax.jit` | pure numerical kernels with I/O and plotting kept outside compiled functions |
| dtype policy | `float32` default with `float64` opt-in through JAX x64 configuration |

## Optimization methods

| Optimizer | Role |
|---|---|
| `SDR` | spectral descent response transform |
| `GSDR` | generalized spectral descent response transform |
| `AGSDR` | adaptive generalized spectral descent response path |
| random search | bounded stochastic search over declared parameter spaces |

## Configuration objects

| Object | Role |
|---|---|
| `Config` / `Configuration` | typed circuit, runtime, source, probe, and task specification |
| `Model` / `Net` | constructed computational graph and identity map |
| `Signals` | tensor output container with semantic selectors |
| `ProbeReport` | readout metadata and interpretation status |
| `RuntimeReport` | backend, dtype, device, and timing receipt |

## See also

- [API reference](../api/index.md)
- [Tutorials](../tutorials/index.md)
- [Limitations and future plans](../limitations_and_future_plans.md)
