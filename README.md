# jaxfne

**JAX Field Neural Equations** (`jaxfne`) is a JAX-native source-to-field neurophysiology engine for Tensor-Field Neural Equations (TFNE).

```python
import jaxfne as jtfne

cfg = (
    jtfne.configuration()
    .network(name="V1", kind="cortical_column", n=64)
    .emitter(family="izhikevich", preset="cortical_eig")
    .field(domain="laminar_column", conductivity="proxy", boundary="mean_zero_neumann", gauge="mean_zero")
    .probe(name="laminar_probe", modes=["spikes", "V_m", "CSD", "LFP"])
)

model = jtfne.construct(cfg)
rt = jtfne.runtime(dtype="float32", backend="auto")
sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0, runtime=rt)
signals = model.simulate(sim)
readout = model.probe(signals, modes=["spikes", "V_m", "CSD", "LFP"])
manifest = model.manifest(signals, readout)
```

## Identity

`jaxfne` is not primarily a neuron model, optimizer, plotting library, or data format. It is the composition layer:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

JAX handles arrays, compilation, batching, and device execution. Jaxley can later provide detailed emitters. Optax can later provide differentiable optimizers. `jaxfne` handles TFNE source-to-field/readout contracts, diagnostics, invariant checks, and manifests.

## Current status

This local package is a **v0.0.4 candidate** that hardens source/probe invariants on top of public v0.0.3.

It still does **not** solve the full resistive extracellular TFNE PDE:

```text
J_e = -sigma_e grad(phi_e)
div(J_e) = q
div(-sigma_e grad(phi_e)) = q
CSD = div(J_e)
```

Current field outputs are **laminar proxy readouts**:

```text
source_projection_mode = proxy_no_field_solve
field_solver_status = laminar_proxy_no_pde
field_claim_level = proxy_readout_only
physical_amplitude_claim_allowed = false
```

## Version roadmap

```text
v0.0.1  skeleton
v0.0.2  API/object hardening
v0.0.3  runtime + source-field status metadata
v0.0.4  source projection + probe invariant tests
v0.0.5  objective/evaluation path
v0.0.6  optax-free tuning scaffold
v0.0.7  Jaxley bridge skeleton hardening
```

## Package structure

```text
jaxfne/
  __init__.py        public API surface
  core.py            Configuration, Model, Simulation, RuntimeConfig, Signals, Probe, Objective, Paradigm
  emitters.py        Izhikevich EIG scaffold
  fields.py          laminar proxy source/probe layer and invariant diagnostics
  optim.py           AGSDR placeholder + require_optax guard
  bridges.py         Jaxley bridge scaffold + require_jaxley guard
  io.py              strict JSON manifest, hashing, save/load
```

## Runtime doctrine

- JAX is required.
- Jaxley and Optax are optional and deferred.
- CPU must work by default.
- GPU/TPU should not be blocked by object design.
- `float32` is the default dtype.
- `float64` is used only when JAX x64 is enabled; otherwise manifests report the actual dtype.
- Simulation kernels use JAX arrays and explicit PRNG keys.
- Manifests record runtime, backend, devices, dtype, source-field status, and truth gates.

## Development smoke

```bash
python -m compileall -q jaxfne tests examples
python -m pytest -q
python examples/minimal_eig_column.py
python examples/global_local_oddball_sketch.py
```

## Truth status

- `truth_mode`: `truth_safe_unverified`
- `claim_level`: `computational_scaffold`
- `source_calibration_status`: `uncalibrated_izhikevich_native_current`
- No calibrated physical CSD/LFP/EEG/MEG amplitude claim is made.
