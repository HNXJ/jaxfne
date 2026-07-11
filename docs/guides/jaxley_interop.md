# Jaxley interoperability

**Combine biophysical detail with population-scale readouts.** Use [Jaxley](https://jaxley.readthedocs.io)
for multi-compartment, conductance-based neurons; use jaxfne for laminar LFP/CSD
proxies, spectrolaminar summaries, objectives, and manifests on the same signals.

| Tool | Best for |
|------|----------|
| **Jaxley** | Differentiable HH/compartment models, morphology, channel dynamics |
| **jaxfne** | Cortical columns, field-proxy chains, multi-trial laminar workflows |

## Using Jaxley with jaxfne

### End-to-end: a Jaxley model → tfne Signals (one call)

Wire channels + stimulus + recordings on a Jaxley model, then hand it to
`JaxleyBridge.simulate(...)`. The bridge integrates with a stable implicit solver
and returns a proxy `Signals` ready for `jaxfne.vis` and the readout/field layer:

```python
import jaxley as jx
from jaxley.channels import HH
import jaxfne as jtfne

# Build a Jaxley emitter: single HH compartment, recorded and stimulated
cell = jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1])
cell.insert(HH())
cell.record("v")
cell.stimulate(jx.step_current(i_delay=10, i_dur=50, i_amp=0.1, delta_t=0.025, t_max=100))

# One call: integrate the Jaxley model and convert to jaxfne Signals
sig = jtfne.JaxleyBridge(model=cell).simulate(duration_ms=100.0, dt_ms=0.025)

print(sig.V_m.shape)                                   # [T, N] proxy voltage
print(sig.metadata["physical_amplitude_calibrated"])   # False (proxy gate)
fig = jtfne.vis.vm(sig)                                 # plot like any tfne run
```

Already ran `jaxley.integrate(...)` yourself? Convert the recordings directly —
recorded-compartment xyz are pulled from `module.nodes` into the metadata for
downstream projection:

```python
rec = jx.integrate(cell, delta_t=0.025, t_max=100.0, solver="bwd_euler")  # (n_rec, n_time)
sig = jtfne.jaxley_to_signals(cell, rec, dt_ms=0.025)
sig.metadata["recorded_positions_xyz"]                 # [[x, y, z], ...] per recording
```

### tfne homeostasis on a Jaxley emitter

Keep Jaxley's channels/morphologies *and* tfne's homeostatic controller. Record the
compartments to control, then call `simulate_homeostatic(...)`. An outer-loop windowed
controller maintains a slow firing-rate trace per cell and injects an excitability
current bias with tfne's restoring law `g = clip(k_gain * (target_rate_hz - r), g_min, g_max)`,
stitching windows with continuous state resume:

```python
import jaxley as jx
from jaxley.channels import HH
import jaxfne as jtfne

# 4 cells with graded drive -> divergent rates without control
net = jx.Network([jx.Cell(jx.Branch(jx.Compartment(), ncomp=1), parents=[-1]) for _ in range(4)])
net.insert(HH())
for c in range(4):
    net.cell(c).branch(0).comp(0).record("v")

sig, diag = jtfne.JaxleyBridge(model=net).simulate_homeostatic(
    duration_ms=400.0,
    base_current_nA=[0.004, 0.008, 0.013, 0.018],   # stay in the monotonic f-I band
    target_rate_hz=70.0, k_gain=0.3, bias_current_scale=0.0008,
    return_diagnostics=True,
)
diag["rate_hz"][-1]   # per-cell rate converges toward the set-point
sig.metadata["homeostasis"]   # controller params + framing gates
```

`k_gain=0` is a clean null (no bias). The controller runs at *window* cadence (Jaxley's
`integrate` is a closed scan), a documented approximation of the per-step Izhikevich
homeostatic kernel. **Framing:** this is a computational control proxy, not a biological
mechanism — `biological_learning_claim=False`, `mechanism_claim_status="not_claimed"`.
Note: Jaxley single-compartment HH has a non-monotonic f-I curve (high current →
depolarization block), so keep currents in the ~0.002–0.02 nA band for the default compartment.

> **JAX compatibility shim.** Jaxley (≤0.13) channels call the
> `jnp.clip(a_min=, a_max=)` keywords that recent JAX removed. Every Jaxley entry
> point routes through `require_jaxley()`, which lazily installs a backward-compatible
> shim so channel emitters actually integrate on current JAX — you do not need to do
> anything. The metadata records `jax_clip_compat_installed` for transparency.

### Array-first trace bridge

jaxfne provides a minimal array-first bridge for converting Jaxley-style voltage traces to jaxfne Signals without running a full simulation:

```python
import jaxfne as jtfne
import numpy as np

# Jaxley-style voltage trace: [time, neurons] in mV
trace = np.random.randn(1000, 16) * 10 - 70  # 1000 timesteps, 16 neurons

# Convert via bridge
spec = jtfne.JaxleyTraceSpec(dt_ms=0.1)  # 0.1 ms timestep
signals = jtfne.jaxley_trace_to_signals(trace, spec=spec)

# Result: jaxfne.core.Signals with time_ms, V_m, spikes, metadata
print(f"V_m shape: {signals.V_m.shape}")  # (1000, 16)
print(f"Claim level: {signals.metadata['claim_level']}")  # computational_scaffold
```

**Key features:**
- No Jaxley installation required (optional dependency)
- Accepts NumPy or JAX arrays
- Flexible layout support: `time_by_unit` [T,N], `unit_by_time` [N,T], `recording_by_time` [R,T]
- Spike proxy derivation via voltage threshold (default 0.0 mV, configurable)
- Conservative voltage-proxy source (ionic current mapping deferred)
- Scope declaration: `computational_scaffold`, `amplitude_status=False`

**Scope specification:**
All outputs are marked as:
- `claim_level: "computational_scaffold"` — Designed for computational workflows and tutorial scenarios
- `physical_amplitude_calibrated: False` — Voltage is treated as a proxy readout, not a physical status
- `source_calibration_status: "uncalibrated_jaxley_voltage_proxy"` — Calibration to physical units pending
- `field_solver_status: "not_computed"` — Field/LFP computation is optional downstream work

**Important:** The voltage trace is treated as a proxy external readout. Voltage is treated as a computational proxy without biological mechanism statements; field computation is optional; ionic current modeling is deferred. Field computation is deferred to the jaxfne probe/field layer if needed.

**See also:**
- `examples/07_jaxley_trace_bridge.py` — Full tutorial with layout conversion and threshold variation
- `JaxleyTraceSpec` — Configuration for trace metadata, dt, spike threshold
- `jaxley_trace_to_signals()` — Main conversion function

### Bridging: array layout conventions

Jaxley outputs spike times or voltage traces. jaxfne expects:

- **Shape:** `[time, neurons]` or `[time, neurons, features]`
- **Units:** Declared (Jaxley units → jaxfne declared mapping)
- **Indexing:** Neuron/compartment ID → jaxfne neuron index

Check [Output bundles](output_bundles.md) for metadata conventions.

## Design philosophy

jaxfne provides optional Jaxley interoperability. Key features:

- Works with or without Jaxley installation (optional dependency)
- Accepts any JAX arrays shaped consistently
- Preserves Jaxley model autonomy and differentiability

Use jaxfne when you want to:

- Organize Jaxley outputs into field/probe workflows
- Add LFP-proxy, CSD-proxy, EEG-proxy readouts downstream
- Build local/global interaction summaries
- Serialize workflows with validation metadata

## Next steps

- **[Tensor-field workflows](tensor_field_workflows.md)** for pipeline overview
- **[Tutorials](../tutorials/index.md)** for end-to-end examples
- **[Jaxley documentation](https://jaxley.readthedocs.io)** for Jaxley-specific topics
