# Plasticity API

STDP (Spike-Timing Dependent Plasticity) support for jaxfne models.

```python
from jaxfne.plasticity import (
    STDPPlasticityConfig,
    STDPState,
    update_stdp_weights_jax,
    summarize_stdp_adaptation,
)
```

---

## `STDPPlasticityConfig`

```python
@dataclass
class STDPPlasticityConfig:
    ...
```

Configuration class for STDP activity-dependent plasticity. Holds all
hyperparameters controlling a single STDP learning rule.

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `A_plus` | `float` | LTP (long-term potentiation) rate — scales weight increase on near-coincident pre→post firing. |
| `A_minus` | `float` | LTD (long-term depression) rate — scales weight decrease on post→pre ordering. |
| `tau_plus_ms` | `float` | Presynaptic trace decay time constant (ms). |
| `tau_minus_ms` | `float` | Postsynaptic trace decay time constant (ms). |
| `w_min` | `float` | Hard lower bound on synaptic weights. |
| `w_max` | `float` | Hard upper bound on synaptic weights. |
| `plasticity_scale` | `float` | Global scaling factor applied to every weight update. |

---

## `STDPState`

```python
@dataclass
class STDPState:
    ...
```

Container for the state variables of the STDP synapse model. Carries the
running pre- and postsynaptic eligibility traces alongside the current weight
matrix, so the full STDP state can be passed through `jax.lax.scan` cleanly.

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `W` | `jax.Array` | Synaptic weight matrix, shape `(n_neurons, n_neurons)`. |
| `trace_pre` | `jax.Array` | Presynaptic eligibility traces, shape `(n_neurons,)`. |
| `trace_post` | `jax.Array` | Postsynaptic eligibility traces, shape `(n_neurons,)`. |

---

## `update_stdp_weights_jax`

```python
def update_stdp_weights_jax(
    W: jax.Array,
    trace_pre: jax.Array,
    trace_post: jax.Array,
    spiked: jax.Array,
    exc_mask: jax.Array,
    A_plus: float,
    A_minus: float,
    plasticity_scale: float,
    w_min: float,
    w_max: float,
) -> jax.Array
```

JAX-native STDP weight-update kernel. Designed to be called inside
`jax.lax.scan` or `jax.vmap` without Python-side loops.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `W` | `jax.Array` `(n, n)` | Current synaptic weight matrix. |
| `trace_pre` | `jax.Array` `(n,)` | Presynaptic eligibility traces. |
| `trace_post` | `jax.Array` `(n,)` | Postsynaptic eligibility traces. |
| `spiked` | `jax.Array` `(n,)` | Boolean spike indicator for this timestep. |
| `exc_mask` | `jax.Array` `(n,)` | Boolean mask selecting excitatory neurons. |
| `A_plus` | `float` | LTP rate parameter. |
| `A_minus` | `float` | LTD rate parameter. |
| `plasticity_scale` | `float` | Global scaling factor. |
| `w_min` | `float` | Minimum weight (hard clamp). |
| `w_max` | `float` | Maximum weight (hard clamp). |

**Returns:** Updated weight matrix, shape `(n_neurons, n_neurons)`.

```python
W_new = update_stdp_weights_jax(
    W, trace_pre, trace_post, spiked, exc_mask,
    A_plus=0.01, A_minus=0.012,
    plasticity_scale=1.0,
    w_min=0.0, w_max=5.0,
)
```

---

## `summarize_stdp_adaptation`

```python
def summarize_stdp_adaptation(
    W_before: jax.Array,
    W_after: jax.Array,
) -> dict
```

Compute synapse-by-synapse adaptation statistics after an STDP run.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `W_before` | `jax.Array` | Weight matrix before adaptation. |
| `W_after` | `jax.Array` | Weight matrix after adaptation. |

**Returns:** `dict` with summary metrics (mean/std of weight changes, fraction
of potentiated vs. depressed synapses, etc.).

```python
summary = summarize_stdp_adaptation(W_before, W_after)
print(summary["mean_delta_W"], summary["frac_ltp"])
```
