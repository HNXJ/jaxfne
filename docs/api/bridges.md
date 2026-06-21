# Bridges API

Optional bridges that let external biophysical simulators participate in the
jaxfne object grammar. The Jaxley bridge turns a Jaxley module into a first-class
jaxfne **emitter**: it produces `Signals` that flow through the same probes,
projections, objectives, and `jaxfne.vis.*` readouts as the built-in emitter.

All bridge outputs are conservative proxies — `field` arrays are proxy readouts,
`physical_amplitude_calibrated=False`, and `claim_level="computational_scaffold"`.
Biophysical fidelity follows the Jaxley model you provide (channels, morphology):
jaxfne is the mathematical backend; the configuration sets the detail.

Jaxley is an optional dependency:

```bash
pip install "jaxfne[jaxley]"
```

## `require_jaxley()` and the JAX clip shim

```python
jx = jtfne.bridges.require_jaxley()
```

Imports Jaxley lazily and installs an idempotent `jax.numpy.clip(a_min=/a_max=)`
compatibility shim. Jaxley (≤0.13) channel gating calls `jnp.clip(x, a_min=...,
a_max=...)`; recent JAX removed those keyword names, so without the shim *no*
Jaxley channel can integrate. Every Jaxley entry point routes through
`require_jaxley()`, so the shim is always in force before a model runs. It
self-disables when the running JAX natively supports `a_max`.

## `jaxley_to_signals(module, recordings, ...)`

```python
rec = jx.integrate(module, delta_t=0.025, t_max=100.0, solver="bwd_euler")
signals = jtfne.jaxley_to_signals(module, rec, dt_ms=0.025, state="v")
```

Converts a Jaxley module plus its `jaxley.integrate` output (shape
`(n_recordings, n_time)`) into jaxfne `Signals`. Maps recordings to canonical
`[T, N]` `V_m`, derives a threshold spike proxy, and pulls the recorded
compartments' xyz from `module.nodes` into `metadata["recorded_positions_xyz"]`
for downstream projection. `field=None` (no field computed here).

| Parameter | Meaning |
|-----------|---------|
| `module` | Integrated Jaxley `Module`/`Cell`/`Network` (read for layout/geometry) |
| `recordings` | `jaxley.integrate(...)` output, `(n_recordings, n_time)` or `(n_time,)` |
| `dt_ms` | Integration timestep in ms (must match the `delta_t` used) |
| `state` | Recorded state name to attribute (default `"v"`) |
| `spec` | Optional `JaxleyTraceSpec` (claim gates, spike threshold) |
| `source` | Optional explicit source array; defaults to the voltage proxy |

## `JaxleyBridge`

```python
bridge = jtfne.JaxleyBridge(model=module)
```

Wraps a Jaxley module and exposes three run modes.

### `simulate(...)` — voltage-proxy Signals

```python
signals = bridge.simulate(duration_ms=100.0, dt_ms=0.025, solver="bwd_euler")
```

Runs the module end-to-end and returns voltage-proxy `Signals` (`field=None`).
Ensures a recording exists, integrates with a stable implicit solver, and
converts via `jaxley_to_signals`. Pass `checkpoint_lengths` for long
BPTT-friendly runs; `return_recordings=True` to also get the raw recordings.

### `simulate_laminar_field(...)` — Emitter → Source → Field

```python
signals = bridge.simulate_laminar_field(
    duration_ms=60.0, dt_ms=0.025, n_contacts=16,
    projection_mode="density_preserving",
)
lfp = signals.get("lfp_proxy")   # [T, n_contacts]
csd = signals.get("csd_proxy")
fig = jtfne.vis.lfp(signals)     # vis.* read straight off signals.field
```

Closes `Emitter → Source → Field → Probe` for the Jaxley bridge: a Jaxley **HH**
network gets the same LFP/CSD/spectrolaminar readouts as the built-in emitter,
but from a physically meaningful generator. The extracellular field is driven by
the **reconstructed HH transmembrane ionic current**

```
I_ionic = gNa·m³·h·(v − eNa) + gK·n⁴·(v − eK) + gLeak·(v − eLeak)
```

rebuilt from the recorded gating states (`HH_m/HH_h/HH_n`) and the channel
parameters in `module.nodes`. This requires HH: the Izhikevich and Fire channels
are non-capacitive (`compute_current` returns zero) and cannot generate a field.

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_contacts` | `16` | Laminar contacts the source is projected onto |
| `width` | `0.1` | Gaussian projection kernel width (relative-depth units) |
| `projection_mode` | `"density_preserving"` | `"row_normalize"` erases source density and flattens depth structure, so it is *not* the default |
| `spike_threshold_mV` | `0.0` | Upward-crossing threshold for the spike proxy |
| `positions` | recorded xyz | Explicit `(n_comp, 3)` positions; defaults to the recorded compartments' xyz |
| `normalize_depth` | `True` | Min-max rescale the z axis to `[0, 1]`; `project_laminar_sources` fixes contacts to `[0, 1]`, so µm-scale geometry must be normalized or sources collapse onto one contact |

!!! note "Depth is relative"
    `project_laminar_sources` reads `positions[:, 2]` as *relative* laminar depth
    in `[0, 1]` and spans its contacts over `[0, 1]`. With `normalize_depth=True`
    (default) the bridge rescales the z axis for you; the raw range is preserved in
    `metadata["depth_raw_range"]`.

### `simulate_homeostatic(...)` — windowed excitability controller

```python
signals, diag = bridge.simulate_homeostatic(
    duration_ms=400.0, base_current_nA=0.01,
    target_rate_hz=40.0, k_gain=0.2, return_diagnostics=True,
)
```

Wraps the Jaxley emitter in an **outer-loop windowed homeostatic controller** so
you keep Jaxley's channel/morphology toolset and gain jaxfne's intrinsic-plasticity
homeostasis. Per recorded compartment, a slow rate trace `r` drives a restoring
excitability bias injected as a per-cell current:

```
rate_w = spikes / window_seconds                       # per-window rate (Hz)
r      = decay_r · r + (1 − decay_r) · rate_w
g      = clip(k_gain · (target_rate_hz − r), g_min, g_max)
I_cell = base_current_nA + bias_current_scale · g
```

Windows are stitched with continuous state resume (`all_states`); per-cell current
is injected via grad-safe `data_stimulate`. `k_gain=0` disables the controller
(a clean null). Because Jaxley's `integrate` is a closed scan, the feedback runs
at *window* cadence — a documented approximation of the per-step built-in kernel.
The injected current is hard-bounded (`current_clip_nA`) and finiteness is
verified (`strict_finite=True` raises rather than masking).

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `window_ms` | `20.0` | Controller update cadence (and integration window) |
| `base_current_nA` | `0.01` | Baseline injected current (scalar or per-cell array) |
| `target_rate_hz` | `40.0` | Homeostatic set-point firing rate |
| `tau_r_ms` | `300.0` | Rate-trace time constant |
| `k_gain` | `0.2` | Controller gain (`0` = null) |
| `g_min`, `g_max` | `-12.0`, `8.0` | Bias clamp range |
| `current_clip_nA` | `1.0` | Hard bound on injected current (stability lever) |
| `strict_finite` | `True` | Raise on non-finite output instead of masking |

!!! warning "Stay in the monotonic f-I band"
    Jaxley single-compartment HH has a non-monotonic f-I curve (high current →
    depolarization block). Keep `base_current_nA` and the bias range inside the
    monotonic band (roughly 0.002–0.02 nA for the default compartment) so the
    controller operates where rate increases with current.

## `hh_jaxley_reference_trace(...)`

```python
t, V, I_inj = jtfne.bridges.hh_jaxley_reference_trace(duration_ms=500.0, dt_ms=0.1)
```

Single-compartment Hodgkin-Huxley reference trace via Jaxley (real channel
integration). Returns time (ms), membrane potential (mV, proxy), and injected
step current (nA). A NumPy-only equivalent, `hh_numpy_reference_trace`, is
available without the Jaxley dependency for tutorials.

## See also

- [Jaxley Interoperability guide](../guides/jaxley_interop.md) — end-to-end usage.
- [Fields API](fields.md) — `project_laminar_sources` and the proxy field engine.
