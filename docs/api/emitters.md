# Emitters API

Neuron models and emitter implementations for neural dynamics simulation.

## Izhikevich Model

```python
cfg = jtfne.Configuration().set_emitter("izhikevich", "cortical_eig")
```

The Izhikevich neuron model is a phenomenological spiking neuron model with two state variables (v, u). It provides a good balance between computational efficiency and biological realism for tutorial-scale simulations.

`Configuration.set_emitter(family="izhikevich", preset="cortical_eig")` is a thin chainable
wrapper over `Configuration.emitter(**kwargs)`; it records `family`/`preset` as metadata on
the config (`jaxfne/_config.py:860`). At build time, `"izhikevich"` and `"homeostatic_ei"`
are the two supported families (`_SUPPORTED_EMITTER_FAMILIES`, `jaxfne/_construct_core.py:389`
— see the `homeostatic_ei` section further down this page for the second family); other family
strings raise at build time. `preset` is a
free-form string tag; `emitters.py` has no dedicated per-preset dynamics table. The emitter's
per-neuron behavior differentiation comes from **cell type** (E/PV/SST/VIP/Inl/Ing), rather than
from `preset`. Treat `preset="cortical_eig"` as the conventional default tag, rather than a switch between
named dynamical regimes.

### State Variables

- `v` (internal units, uncalibrated relative to mV): Membrane-potential-like state variable. Reset value
  `c` defaults to `-65.0`; spike threshold is a fixed `30.0` (see Dynamics below).
- `u` (internal units): Recovery variable (adaptation/refractory effects).

### Dynamics

```
dv/dt = 0.04*v^2 + 5*v + 140 - u + I
du/dt = a*(b*v - u)
```

If `v >= 30.0`, a spike occurs and the states reset:
```
v <- c
u <- u + d
```

This is implemented inside the `jax.lax.scan` step functions of
`simulate_eig_izhikevich`, `simulate_edge_recurrent_izhikevich`,
`simulate_edge_recurrent_izhikevich_homeostatic`,
`simulate_edge_recurrent_izhikevich_hdp`, and `simulate_receptor_exponential_izhikevich`
(`jaxfne/emitters.py`); each simulate function inlines the same dynamics with its own
synaptic/plasticity wrapper, rather than sharing one standalone "Izhikevich integrator" function.

`I` (`current_native` in source) is an **uncalibrated internal drive**, distinct from a physical current in
amperes — every kernel in this module sets `source_calibration_status` to a value such as
`"uncalibrated_izhikevich_native_current"`, always short of a physical-calibration claim.

### Canonical source representation

Canonical `Q^(r)` map (see [Source/Field Equations](../source_field_equations.md) and [Source Schema](source_schema.md)):

```text
Q^(r) = source_scale * (current_native + DEFAULT_SPIKE_IMPULSE_GAIN * spikes)
current_native = drive + recurrent_synaptic + noise
```

| Field | Source |
|---|---|
| `source_scale` | per-neuron scale |
| `DEFAULT_SPIKE_IMPULSE_GAIN` | `jaxfne.presets` (shared across kernels) |

### Parameters

Canonical per-cell-type parameter defaults live in
`IZHIKEVICH_CELL_TYPE_DEFAULTS` (`jaxfne/emitters.py:25`), keyed by cell-type label
(`E`, `PV`, `Inl`, `SST`, `Ing`, `VIP`):

| Label | `a` | `b` | `c` (mV-like) | `d` | `drive` | `sign` |
|-------|-----|-----|-----|-----|---------|--------|
| `E`   | 0.02 | 0.20  | -65.0 | 8.0 | 5.0 | +1.0 |
| `PV`  | 0.10 | 0.20  | -65.0 | 2.0 | 3.0 | -1.0 |
| `Inl` | 0.10 | 0.20  | -65.0 | 2.0 | 3.0 | -1.0 |
| `SST` | 0.05 | 0.25  | -65.0 | 2.0 | 3.5 | -1.0 |
| `Ing` | 0.05 | 0.25  | -65.0 | 2.0 | 3.5 | -1.0 |
| `VIP` | 0.02 | -0.10 | -55.0 | 6.0 | 3.0 | -1.0 |

An unrecognized cell-type label falls back to the `VIP` row (`_get_cell_type_params`,
`jaxfne/emitters.py:35`). These values feed the `a`/`b`/`c`/`d`/`drive`/`sign` fields of
`IzhikevichParams` — see the dataclass table below for the full field list. There is
no `I_injected` field; the closest analogue is the per-cell `drive` value plus, at simulate time, an
optional `drive_schedule` argument.

### Building Parameters

```python
# From explicit cell-type fractions (random assignment order):
params = jtfne.emitters.izhikevich_eig_params(
    n=128,
    cell_type_fractions={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03},
    dtype="float32",
)

# From an explicit, ordered list of per-neuron labels (deterministic):
params = jtfne.emitters.izhikevich_params_from_labels(
    labels=("E", "E", "PV", "SST"),
    layer_labels=("L4", "L4", "L4", "L2/3"),   # optional, must match len(labels)
    dtype="float32",
    drive_overrides={"E": 6.0},                # optional per-type drive override
    source_scale=1.0,
)
```

`izhikevich_eig_params(n, cell_type_fractions, *, dtype="float32")` (`jaxfne/emitters.py:212`)
assigns cell-type labels by walking `cell_type_fractions` in dict order and rounding counts;
any unknown-label edge cases are absorbed into the last fraction bucket. `izhikevich_params_from_labels`
(`jaxfne/emitters.py:262`) instead takes an explicit, ordered `labels` sequence and raises
`ValueError` when the `labels` tuple is empty, when `layer_labels` length differs, or when
a label falls outside `IZHIKEVICH_CELL_TYPE_DEFAULTS`.

---

## IzhikevichParams

```python
jaxfne.IzhikevichParams
```

Frozen dataclass parameter container for a reduced Izhikevich population
(`jaxfne/emitters.py:84`), registered as a JAX pytree.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `a` | `jax.Array` | Per-neuron recovery timescale |
| `b` | `jax.Array` | Per-neuron coupling strength |
| `c` | `jax.Array` | Per-neuron reset voltage (internal units) |
| `d` | `jax.Array` | Per-neuron recovery reset increment |
| `drive` | `jax.Array` | Per-neuron baseline internal drive current |
| `sign` | `jax.Array` | Per-neuron +1.0 (excitatory) / -1.0 (inhibitory) sign |
| `W` | `jax.Array` | Dense recurrent weight matrix, shape `(n, n)` |
| `v0` | `jax.Array` | Initial membrane state, shape `(n,)` |
| `u0` | `jax.Array` | Initial recovery state, shape `(n,)` |
| `source_scale` | `jax.Array` | Scalar (or per-neuron) scale applied to the source proxy |
| `labels` | `tuple[str, ...]` | Per-neuron cell-type labels |
| `layer_labels` | `tuple[str, ...] \| None` | Optional per-neuron layer labels (default `None`) |
| `source_calibration_status` | `str` | Default `"uncalibrated_izhikevich_native_current"` |

### Properties

#### `n_neurons -> int`

Number of neurons, i.e. `v0.shape[0]`.

**Example:**
```python
params = jtfne.emitters.izhikevich_eig_params(128, {"E": 0.8, "PV": 0.2})
print(params.n_neurons)  # 128
```

`IzhikevichParams` has no `class_method(...)` factory; construction goes through the
module-level functions `izhikevich_eig_params(...)` and `izhikevich_params_from_labels(...)`
documented above.

---

## ReceptorSpec

```python
jaxfne.ReceptorSpec
```

Frozen dataclass metadata declaration for a synaptic receptor (`jaxfne/emitters.py:44`).
**Metadata only** — distinct from a biological kernel; `emitters.py` leaves the conductance
and reversal-potential equations undefined for these fields.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Receptor name, e.g. `"AMPA"` |
| `receptor_index` | `int` | Integer channel index used by `edges.receptor_index` |
| `sign` | `int` | `+1` excitatory / `-1` inhibitory |
| `tau_ms` | `float` | Time constant (ms) |
| `reversal_mV` | `float \| None` | Reversal potential, metadata only (not used in dynamics) |
| `source_calibration_status` | `str` | Default `"metadata_only_uncalibrated"` |
| `claim_level` | `str` | Default `"computational_scaffold"` |

### Standard Receptors

Access via `jaxfne.standard_receptor_specs()` (`jaxfne/emitters.py:66`), which returns a
`dict[str, ReceptorSpec]`:

| Name | `receptor_index` | `sign` | `tau_ms` | `reversal_mV` |
|------|-------------------|--------|----------|---------------|
| `AMPA` | 0 | +1 | 2.0 | 0.0 |
| `GABA_A` | 1 | -1 | 5.0 | -80.0 |
| `NMDA` | 2 | +1 | 100.0 | 0.0 |
| `GABA_B` | 3 | -1 | 150.0 | -95.0 |

**Example:**
```python
receptors = jtfne.standard_receptor_specs()
print(receptors["AMPA"].tau_ms)  # 2.0
```

Note: the recurrent-simulation kernels' binary excitatory/inhibitory split (see `EdgeList` below)
consumes `receptor_index` values `0` and `1` only; `NMDA`/`GABA_B` are declared
here, while the kernels instantiate them as combined excitatory/inhibitory channels rather than as separate synaptic populations.

---

## SynapseSpec

```python
jaxfne.SynapseSpec
```

Frozen dataclass metadata declaration for a synapse (`jaxfne/emitters.py:57`). **Distinct from** a
per-connection edge record — it is a small wrapper bundling a tuple of `ReceptorSpec` objects
plus calibration-status metadata, with `source_idx`/`target_idx`/`weight`/`delay` fields left out.
Per-connection edges are represented by `EdgeList` (below), a distinct type from `SynapseSpec`.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `receptors` | `tuple[ReceptorSpec, ...]` | Receptor specs composing this synapse |
| `source_calibration_status` | `str` | Default `"metadata_only_uncalibrated"` |
| `physical_amplitude_calibrated` | `bool` | Default `False` |

---

## EIGNetwork

```python
jaxfne.EIGNetwork
```

Frozen dataclass — lightweight description of an E/PV/SST/VIP-like reduced network
(`jaxfne/emitters.py:172`).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `params` | `IzhikevichParams` | The population's Izhikevich parameters |
| `positions` | `jax.Array` | Per-neuron `(x, y, depth)` positions, shape `(n, 3)` |
| `metadata` | `dict` | Free-form metadata (e.g. `emitter_family`, `source_calibration_status`, `position_units`) |

### Properties

#### `n_neurons -> int`

Delegates to `self.params.n_neurons`.

`EIGNetwork` has no `EIGNetwork(n_exc=..., n_inh=...)` constructor and no `to_dense()` method — build an
`EIGNetwork` via `make_eig_network(...)` (below), the alternative to constructing the dataclass directly with
excitatory/inhibitory counts.

---

## EdgeList

```python
jaxfne.EdgeList
```

Frozen dataclass registered as a JAX pytree class (`@jax.tree_util.register_pytree_node_class`) —
sparse recurrent connectivity (`jaxfne/emitters.py:459`). Weights stay internal/unphysical
pending a future calibration bridge.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `pre` | `jax.Array` | Presynaptic neuron index per edge, shape `(n_edges,)` |
| `post` | `jax.Array` | Postsynaptic neuron index per edge, shape `(n_edges,)` |
| `weight` | `jax.Array` | Signed internal weight per edge |
| `receptor_index` | `jax.Array` | Integer receptor channel per edge (`0`=excitatory, `1`=inhibitory in the recurrent kernels) |
| `tau_ms` | `jax.Array` | Per-edge synaptic decay time constant (ms) |
| `source_calibration_status` | `str` | Default `"uncalibrated_izhikevich_native_current"` |

### Properties and Methods

#### `n_edges -> int`

Number of edges, i.e. `pre.shape[0]`.

#### `tree_flatten()` / `tree_unflatten(aux, children)`

JAX pytree protocol methods (children = the five arrays; aux = `source_calibration_status`).

#### `to_dict() -> dict`

Returns a JSON-safe summary: `backend`, `n_edges`, a `receptors` label map
(`{"0": "excitatory_native", "1": "inhibitory_native"}`), `source_calibration_status`, and
`physical_amplitude_calibrated=False`.

`EdgeList` has no `from_dense(...)` classmethod. The conversion path is the module-level
function `make_edge_list_from_dense(weights, *, threshold=1e-12, dtype="float32")`
(`jaxfne/emitters.py:503`):

```python
W = jnp.ones((100, 100))
edges = jtfne.make_edge_list_from_dense(W, threshold=1e-12, dtype="float32")
```

`weights` uses rows as postsynaptic targets and columns as presynaptic sources (matching
`weights @ spikes` in the baseline dense backend). Entries with `abs(weight) <= threshold` are
dropped. `tau_ms` per edge is set to `2.0` (excitatory, `weight >= 0`) or `5.0` (inhibitory,
`weight < 0`), matching the excitatory/inhibitory tau split used elsewhere in this module.

---

## Emitter Functions

### `make_eig_network(n=128, cell_type_fractions=None, *, dtype="float32") -> EIGNetwork`

Build a minimal EIG network with laminar depth positions (`jaxfne/emitters.py:323`).

**Parameters:**
- `n` (int, default 128): Number of neurons; depth positions are spread evenly over `[0, 1]`
  (x/y are fixed at 0).
- `cell_type_fractions` (`Mapping[str, float]`, optional): E/PV/SST/VIP fractions. Default:
  `{"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03}`.
- `dtype` (str, keyword-only, default `"float32"`): Array dtype policy.

**Returns:** `EIGNetwork`, with `metadata` containing `emitter_family="izhikevich"`,
`source_calibration_status`, and `position_units="relative_laminar_depth_proxy"`.

**Example:**
```python
network = jtfne.make_eig_network(n=128, cell_type_fractions={"E": 0.8, "PV": 0.1, "SST": 0.07, "VIP": 0.03})
```

---

### `simulate_eig_izhikevich(params, n_steps, dt_ms, key, *, dtype="float32", drive_schedule=None, silence_mask=None, noise_scale=None) -> (voltages, spikes, sources)`

Simulate a reduced EIG Izhikevich scaffold using `jax.lax.scan`, with dense (`params.W`)
recurrent coupling (`jaxfne/emitters.py:358`).

**Parameters:**
- `params` (`IzhikevichParams`): Population parameters, including `W`.
- `n_steps` (int): Number of simulation timesteps.
- `dt_ms` (float): Timestep in milliseconds.
- `key` (`jax.Array`): PRNG key.
- `dtype` (str, keyword-only, default `"float32"`).
- `drive_schedule` (`jax.Array | None`, keyword-only): Optional `(n_steps, n_neurons)` internal
  current added to `params.drive` at every step.
- `silence_mask` (`jax.Array | None`, keyword-only): Optional `(n_neurons,)` mask; neurons with
  mask `<= 0.5` are held at `c` and cannot spike.
- `noise_scale` (`jax.Array | float | None`, keyword-only): Stochastic-current coefficient;
  `None` keeps the historical `0.5` scalar.

**Returns:** `(voltages, spikes, sources)`, each shape `(n_steps, n_neurons)`.

**Example:**
```python
key = jax.random.PRNGKey(7)
voltages, spikes, sources = jtfne.simulate_eig_izhikevich(params, n_steps=1000, dt_ms=0.1, key=key)
```

A backwards-compatible alias `simulate_izhikevich_eig` (from v0.0.3) points at the same function.

---

### `make_edge_list_from_dense(weights, *, threshold=1e-12, dtype="float32") -> EdgeList`

Convert a dense recurrent weight matrix into a sparse `EdgeList`. See the `EdgeList` section
above for the exact semantics.

---

### `simulate_edge_recurrent_izhikevich(params, edges, n_steps, dt_ms, key, *, dtype="float32", drive_schedule=None, silence_mask=None, noise_scale=None) -> (voltages, spikes, sources, final_state)`

Simulate reduced Izhikevich emitters with **sparse** recurrent synapses, using `jax.lax.scan`
over time and `jax.ops.segment_sum` over edges (`jaxfne/emitters.py:532`). JIT/vmap compatible.

**Parameters:** same as `simulate_eig_izhikevich`, plus `edges` (`EdgeList`) in place of dense
`params.W` coupling.

**Returns:** `(voltages, spikes, sources, final_state)` where `final_state` is a dict with keys
`v`, `u`, `prev_spikes`, `syn_state` — usable to resume a simulation.

---

### `simulate_edge_recurrent_izhikevich_homeostatic(...) -> (voltages, spikes, sources, diagnostics_dict)`

As `simulate_edge_recurrent_izhikevich`, plus a per-neuron homeostatic excitability bias
`g_i = clip(k_gain * (r_star - r_i), g_min, g_max)` driven by a slow activity trace `r_i`, and
optional homeostatic synaptic plasticity (`eta != 0`) (`jaxfne/emitters.py:650`). Key extra
keyword-only parameters: `r_star=0.05`, `tau_r_ms=300.0`, `alpha=1.0`, `k_gain=1.0`,
`g_min=-12.0`, `g_max=8.0`, `r_max=1.0`, `eta=0.0` (disables plasticity when `0.0`),
`tau_x_ms=100.0`, `w_min=-10.0`, `w_max=10.0`, plus hard numerical-stability bounds
(`v_floor`, `v_ceiling`, `u_abs_max`, `syn_abs_max`) and `init_state` for pause/resume.
`k_gain` is a one-sided damper (can suppress firing below baseline, not reliably drive it above).
`diagnostics_dict` includes `g_bias` and `r_trace`, each shape `(n_steps, n_neurons)`, plus
(when `eta != 0`) `w_trace` shape `(n_steps, n_edges)`.

---

### `simulate_edge_recurrent_izhikevich_rbd(...) -> (voltages, spikes, sources, final_state)`

Protocol H1 — **Relative Biophysical Dynamics** with **fixed weights** (\(\dot W=0\),
\(d_H=1\)). Composes the edge-recurrent Izhikevich kernel (including Protocol D
``delay_steps`` when nonzero) with scalar per-neuron RBS coordinates ``H_i``.

**H1c-C (postsynaptic recurrent gain):**

\[
I_i^{\mathrm{drive}} = I_i^{\mathrm{ext}} + G_H(H_i;\beta_H)\,I_i^{\mathrm{rec}} + \text{noise},
\quad G_H = 1 + \beta_H(H-1).
\]

External drive is untouched. ``F_H`` uses **pre-gain** ``I_i^rec`` for
``kappa_H * I_i^rel``. ``beta_h=0`` and ``H=1`` recover H1a / legacy activity.
``G_H`` must stay positive; values at or below zero end the step.

**H2 continuation:** pass ``init_state`` with ``v``, ``u``, ``prev_spikes``,
``syn_state``, ``H_final``/``H``, ``delay_state`` (when delays active), and
``continuation_step_offset``. Returns the same fields plus ``H_trace``.

``rbd_family`` selects:

| ID | Dynamics |
|----|----------|
| ``"f0"`` | RBS disabled: ``H_i ≡ 1`` |
| ``"f1"`` | ``tau_H * dH_i/dt = (1 - H_i) + kappa_H * I_i^rel`` |
| ``"f2"`` | ``tau_H * dH_i/dt = (H_i^{-1} - 1) + kappa_H * I_i^rel`` (requires ``H>0``) |

``I_i^rel`` is total recurrent synaptic input at neuron ``i`` divided by ``i_ref``.
F2 trajectories with ``H<=0`` propagate non-finite ``H``; values are unclipped.
``final_state`` includes ``H_trace``, ``H_final``, and ``w_fixed`` (the constant edge
weights). Zero-delay continuation only; nonzero-delay continuation awaits Protocol H2.

Authority: ``docs/doctrine/protocol_h_rbd_memory.md``.

---

### `simulate_edge_recurrent_izhikevich_hdp(...) -> (voltages, spikes, sources, diagnostics_dict)`

As `simulate_edge_recurrent_izhikevich`, plus **Hidden-state Dependent Plasticity (HDP)**:
per-neuron Relative Biophysical State (RBS) coordinates `H_i` (default scalar
1.0, clamped to `[H_min, H_max]`) that both synaptic
drive and the neuron's own spiking feed back into, and that drives a weight-update rule selected
by `hdp_rule` (`"signed_linear"` default, or `"signed_quadratic"`, `"hebbian_product"`)
(`jaxfne/emitters.py:1039`). For an edge `i -> j`, define
`Delta_H = H_post - H_pre` and `w = q*m`; the difference-family magnitude term is
`q*K_HDP*phi(Delta_H)*m`, with `phi(x)=x` for `"signed_linear"` and
`phi(x)=x*abs(x)` for `"signed_quadratic"`. `"hebbian_product"` is separate product
modulation using `H_pre*H_post`, not another difference rule. The independent
`K_w_ctrl*(m0-m)` term restores magnitude toward the declared baseline. All
H income/spending gains (`alpha`, `beta`, `gamma`, `delta`, `C_spike`) default to
`0.0`, and `K_HDP` defaults to `1.0` but multiplies a zero weight-update
term when those gains are `0.0` — so the defaults are an H-state/weight-term null,
not a general full-system equivalence claim. Per-neuron `tau_i =
tau_0_ms * size_i**3`, with per-cell-type `size_i` from `DEFAULT_HDP_SIZE_SCALE_BY_CELL_TYPE`
(`jaxfne/emitters.py:978`: `E=5.0`, `PV=1.0`, `Inl=1.0`, `SST=1.5`, `Ing=1.5`, `VIP=1.5`), unless
overridden via `size_scale_by_cell_type` or `size_scale_override`. `diagnostics_dict` includes
The current scalar HDP realization is the `d_H=1` case. `H_trace` has shape
`(n_steps, n_neurons)` for `h_state_dim=1` and
`(n_steps, n_neurons, h_state_dim)` for vector H. The diagnostics also include
`w_trace`, `H_final`, and `w_final`, plus optional
per-term `dH_*_trace` diagnostics when `record_dH_components=True` and `edge_current_trace` when
`record_edge_current=True`. `record_weight_trace` (default `True`) controls whether `w_trace`
(shape `(n_steps, n_edges)`, the dominant memory cost at scale — e.g. 10,000 steps x 2,000,000
edges x 4 bytes = 80GB) is stacked at all; set `False` to get `w_trace=None` while `w_final` and
HDP's actual dynamics stay unaffected. For `h_state_dim>1`, componentwise H
dynamics are the default; `h_state_readout` supplies the current scalar
adaptation projection and `h_state_coupling` supplies an optional square
component-coupling matrix. See the docstring in `jaxfne/emitters.py:1042` for the full parameter
reference — it is extensive and kept in the source docstring.

---

### `simulate_receptor_exponential_izhikevich(params, edges, n_steps, dt_ms, key, *, dtype="float32", drive_schedule=None, silence_mask=None) -> (voltages, spikes, sources, final_state)`

v0.0.11 receptor-indexed exponential recurrent kernel (`jaxfne/emitters.py:1538`). Keeps one
scalar synaptic state per edge and selects the per-edge decay from `edges.receptor_index` via
`standard_receptor_tau_table()`. Two receptor channels on the same anatomical connection are
represented as two separate edges with identical `pre`/`post` but different `receptor_index` —
the kernel does not expand state to `(n_edges, n_receptors)`. Reversal potentials remain
metadata-only; no `g * (V - E_rev)` conductance equation is computed. `final_state` additionally
includes `tau_per_edge`.

---

### `simulate_dynamic_ei_coupling(params, n_steps, dt_ms, key, *, g_ei=5.0, g_ie=3.0, tau_syn_e_ms=5.0, tau_syn_i_ms=10.0, dtype="float32") -> (voltages, spikes, syn_currents, sources)`

Simulates a fixed **two-neuron** E/I pair (`params` must have `n_neurons=2`, neuron 0 = E,
neuron 1 = I) with dynamic exponential synaptic coupling (`jaxfne/emitters.py:1665`). Not a
general-`n` recurrent kernel — intended for small illustrative demos.

---

### `simulate_multi_area_izhikevich(neurons_df, positions_m, W, source_tensor=None, control_params=None, cfg=None, n_steps=None, dt_ms=0.1, seed=0, dtype="float32") -> (spikes, voltages)`

Multi-area Izhikevich simulation driven from a neuron metadata mapping (`neurons_df` with
`area`/`layer`/`cell_type` keys), a dense connectivity matrix `W` (rescaled by `0.1` for gain
compatibility), and an optional `source_tensor` used as `drive_schedule`
(`jaxfne/emitters.py:1780`). Note the return order is `(spikes, voltages)`, the reverse of the
other `simulate_*` functions' `(voltages, spikes, ...)` order.

---

### `standard_receptor_specs() -> dict[str, ReceptorSpec]`

Get standard synaptic receptor specifications (`jaxfne/emitters.py:66`).

**Returns:** Dictionary of receptor type -> `ReceptorSpec`.

**Available types:** `AMPA`, `GABA_A`, `NMDA`, `GABA_B`.

**Example:**
```python
receptors = jtfne.standard_receptor_specs()
print(receptors["AMPA"])
```

---

### `standard_receptor_tau_table(dtype="float32") -> jax.Array`

Returns the `receptor_index -> tau_ms` lookup table, built from `standard_receptor_specs()` so
the kernels and the declarative metadata cannot drift apart (`jaxfne/emitters.py:1422`).

---

### `synaptic_tau_from_mechanism(mechanism, *, dtype="float32") -> tau_ms`

Map declared receptor-mechanism names to per-edge tau (Synaptic Tensor, tau stage).

**Parameters:**
- `mechanism` (`Sequence[str]`): per-edge mechanism names, e.g. `["AMPA", "NMDA", ...]`.
  Valid names match `standard_receptor_specs()`: `AMPA`, `GABA_A`, `NMDA`, `GABA_B`.

**Returns:** `tau_ms` (`jax.Array`, milliseconds, shape `[E]`).

**Description:**
Vectorized lookup over `standard_receptor_specs()` / `standard_receptor_tau_table()`
keyed by mechanism name. **Additive only** — this does not change how
`core._compile_connection_rules` infers tau today (it still derives receptor type
from weight sign only, hardcoding `tau=2.0` for excitatory / `5.0` for inhibitory
edges regardless of any declared `mechanism` string — a known, separately-tracked
inertness). Raises `ValueError` on an unrecognized mechanism name rather than
silently substituting the wrong kinetics.

**Example:**
```python
tau_ms = jtfne.synaptic_tau_from_mechanism(["AMPA", "GABA_A", "NMDA", "GABA_B"])
# -> [2.0, 5.0, 100.0, 150.0]
```

---

### `synaptic_current_tensor(spikes_pre, tau_ms, dt_ms) -> filtered`

Standalone single-pole synaptic current tensor (Synaptic Tensor, filter stage).

**Parameters:**
- `spikes_pre` (`jax.Array`): per-channel spike/input trace, shape `[T, E]`.
- `tau_ms` (`jax.Array`): per-channel time constant in milliseconds, shape `[E]`.
  Build with `synaptic_tau_from_mechanism`.
- `dt_ms` (float): simulation timestep in milliseconds.

**Returns:** synaptic state trace, shape `[T, E]`.

**Description:**
Factors out the exact per-edge synaptic state update used inline by
`simulate_edge_recurrent_izhikevich` / `simulate_receptor_exponential_izhikevich`
(`syn_next = syn_state * exp(-dt/tau) + spike`) as an explicit, named, reusable
operator — usable outside the full `simulate()` orchestration for diagnostics or
parameter sweeps. Single-exponential decay only (no separate rise time constant),
matching the kernels exactly.

Validated falsification (500 ms, periodic 20 Hz input, `order` n/a — single pole):
NMDA (tau=100 ms) sustains a mean current ~37.6x AMPA's (tau=2 ms) under identical
input, confirming the tensor is genuinely mechanism-selective.

**Example:**
```python
tau_ms = jtfne.synaptic_tau_from_mechanism(["AMPA", "NMDA"])
trace = jtfne.synaptic_current_tensor(spikes_pre, tau_ms, dt_ms=0.5)
```

---

### `synaptic_tensor_report(tau_ms, mechanism=None) -> dict`

JSON-safe truth-gate report for a `synaptic_current_tensor` call.

**Parameters:**
- `tau_ms` (`jax.Array`): the same per-channel tau array passed to `synaptic_current_tensor`.
- `mechanism` (`Sequence[str] | None`): the mechanism names used, if any.

**Returns:** a `dict` with `tau_ms_mean/min/max`, `mechanism`, `finite_tau`,
`source_calibration_status="metadata_only_uncalibrated"`,
`physical_amplitude_calibrated=False`, `claim_level="computational_scaffold"`.

**Example:**
```python
report = jtfne.synaptic_tensor_report(tau_ms, mechanism=["AMPA", "NMDA"])
assert report["finite_tau"]
```

---

## Emitter Facade Classes

`emitters.py` also exposes a small object-oriented facade layer used by tutorials/smoke tests
(`jaxfne/emitters.py:1896` onward), separate from the functional `simulate_*` kernels above:

- **`EmitterState`** (`NamedTuple`): `v`, `u`, `spikes`, `key`, `step_count`.
- **`EmitterOutput`** (`NamedTuple`): `voltage`, `spikes`, `source`, `finite`; has a `dtype`
  property returning `str(self.voltage.dtype)`.
- **`Emitter`**: base class with `initial_state(seed=0) -> EmitterState` and
  `step(state, input_t, *, dt_ms=0.1) -> (EmitterState, EmitterOutput)`, both `NotImplementedError`
  stubs on the base class.
- **`IzhikevichEmitter(Emitter)`**: concrete single-step facade over the same dense-`W` Izhikevich
  dynamics as `simulate_eig_izhikevich`. Constructor: `IzhikevichEmitter(n=None, *, n_neurons=None,
  dtype="float32", cell_type_fractions=None)` — accepts either `n` or `n_neurons` (at least one
  required, `n` takes precedence), default `cell_type_fractions={"E": 0.75, "PV": 0.10, "SST": 0.08,
  "VIP": 0.07}`.
- **`GLIFEmitter(Emitter)`**, **`LIFEmitter(Emitter)`**: unimplemented stubs — both raise
  `NotImplementedError` on construction (`__init__` immediately raises). Not usable yet.
- **`SynapseState`** (`NamedTuple`): `trace` — a JAX-pytree-compatible carry for `SynapseLayer`.
- **`SynapseLayer`**: `SynapseLayer(n, W, tau_ms=5.0, dtype="float32")`, dense exponential synapse
  layer with `initial_state() -> SynapseState` and `step(state, pre_spikes, *, dt_ms=0.1) ->
  (SynapseState, current)`.

```python
emitter = jtfne.IzhikevichEmitter(n=64, cell_type_fractions={"E": 0.8, "PV": 0.2})
state = emitter.initial_state(seed=0)
state, output = emitter.step(state, input_t=jnp.zeros(64), dt_ms=0.1)
```

---

## homeostatic_ei (second canonical HDP sanity emitter)

`jaxfne/emitters_homeostatic_ei.py`. A **separate emitter family from Izhikevich** —
not a variant of it. State `x` is continuous, bounded, differentiable rate-like state,
**not** a hard Izhikevich threshold-and-reset (no `v`/`u`, no spike-triggered reset). It is
the smallest dynamical system built to exercise HDP: three explicit, independently staged
timescales (never fused into one rule):

```
dx/dt = f(x, G, u)        fast neuronal dynamics      (tau_x_ms)
dG/dt = f_G(x, H, is_e)    intermediate conductance     (tau_G_ms)
dH/dt = f_H(x, H, is_e)    slow HDP homeostasis         (tau_H_ms)
```

```python
cfg = (
    jtfne.Configuration()
    .runtime(seed=0, duration_ms=1000.0, dt_ms=0.5)
    .network(name="ei8", n=8)
    .set_emitter("homeostatic_ei", activation_rule="cubic", conductance_rule="hebbian",
                 homeostasis_rule="linear", bound_mode="minimal")
    .field(domain="none")
    .probe(modes=["vm"])
)
model = jtfne.construct(cfg)
signals = model.simulate(jtfne.simulation(duration_ms=1000.0, dt_ms=0.5, seed=0))
```

`n` (from `.network(n=...)`) can be any value `>=2` — split E/I via
`_homeostatic_ei_cell_type_split` (`~75%`/`~25%`, at least 1 of each).
`Model.summary()`/`.neuron_table()`/`.checkpoint()`/`.with_emitter_parameters()`/
`.simulate_batch()` all raise `NotImplementedError` for this family (not yet generalized).

### Rule registries (`ACTIVATION_RULES`/`CONDUCTANCE_RULES`/`HOMEOSTASIS_RULES`)

Registry **names** pass through `Configuration` (which must stay JSON-safe); a
custom Python callable bypasses `Configuration` entirely, via
`jaxfne.emitters_homeostatic_ei.simulate_homeostatic_ei(...)`.

- **activation_rule**: `"linear"` (`dx = -x + G@x + u`), `"cubic"` (`dx = -x^3 + G@x + u`,
  the default — bounds runaway growth; `"linear"` diverges once G-adaptation is on),
  `"logistic"` (`dx = -x + G@sigmoid(x) + u`).
- **conductance_rule**: `"hebbian"` (`dG_ij = H_i*x_i*x_j - G_ij`), `"bcm"`, `"linear"`,
  **`"hebbian_pairwise"`** — independent gains per population pair (E-E/E-I/I-E/I-I)
  instead of one flat Hebbian rate; default gains all `1.0` (numerically identical to
  plain `"hebbian"`). Custom gains: `conductance_rule=jaxfne.emitters_homeostatic_ei.make_hebbian_pairwise_rule(k_ee=1.0, k_ei=5.0, k_ie=0.2, k_ii=1.0)` (a callable, reachable via
  `simulate_homeostatic_ei`, bypassing `Configuration`).
- **homeostasis_rule**: `"linear"` (`dH = -(x-1)*H` — one-sided rate-drain; collapses to
  `H_min` at higher N, no term restoring H from below), `"logistic"`, `"cubic_penalty"`
  (adds a two-sided cubic restoring force toward `H=1`, avoiding the floor-collapse),
  **`"cubic_penalty_coupled"`** — adds an E<->I cross-population coupling term on top of
  `cubic_penalty` (I's H rises when E's population-mean activity exceeds target, and
  vice versa) — every other rule's `dH` depends solely on that neuron's own `x`.

### `bound_mode` (`"minimal"` | `"stable"`, default `"minimal"`)

- `"minimal"`: `jnp.clip` on `G`/`H`; `x` is entirely **unbounded**. A large enough
  step (explicit Euler on the cubic activation term overshoots once `|x|` exceeds a
  real numerical-stability radius, `~2.58` at the canonical `dt_x`) can diverge to
  `NaN` — reproduced at N=16 with the shipped flat `G_max=5.0` default.
- `"stable"`: a smooth `tanh` soft-bound (`_soft_bound`) applied to `x`, `G`, and `H`
  every step instead of `jnp.clip` — a bounded *codomain*, not a *force*: cannot be
  numerically outrun by any step size, N, or gain, and is gradient-friendly everywhere
  (unlike `jnp.clip`'s zero-gradient boundary). Requires `HomeostaticEIParams.x_min`/
  `.x_max` (new fields; default a very wide `+-1e6`, safe/inert for `"minimal"` mode).

### `make_minimal_ei_params(n=8, e_fraction=0.75, **kwargs) -> HomeostaticEIParams`

Builds a minimal all-pairwise E/I `HomeostaticEIParams` for any `n>=2` — the
`scripts/`-level analog of `Configuration.set_emitter("homeostatic_ei")`, for ad hoc
scripts/experiments that want a `HomeostaticEIParams` without going through
`Configuration`/`construct()`. `G_max` defaults to `10.0/n` (not a flat constant) —
holds the aggregate per-row Hebbian feedback ceiling (`n * G_max`) constant across `n`.
</content>
