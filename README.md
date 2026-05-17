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
sim = jtfne.simulation(duration_ms=100.0, dt_ms=0.1, seed=0)
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

## Current status (v0.0.5 release candidate)

v0.0.5 adds the task-flow, objective, and optimizer metadata layers on top of the v0.0.4 source/probe invariant foundation.

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

## v0.0.5 API surface

### Task paradigm

```python
# Standard visual omission task: 12 conditions, condition-number mapping
paradigm = jtfne.standard_visual_omission()
print(paradigm.condition_names())        # ['AAAB', 'AXAB', ..., 'RRRX']
print(paradigm.omission_conditions())    # 9 conditions with omission_position set
print(paradigm.event_codes)              # {'fx': 10, 'p1': 101, ..., 'rw': 96}
print(paradigm.analysis_windows)        # {'baseline': (-500, 0), ...}

# Paradigm is JSON-safe
import json
json.dumps(paradigm.to_dict(), allow_nan=False)
```

### Objective / evaluation

```python
obj = (
    jtfne.objective()
    .loss("rate_loss", target=20.0, weight=1.0, metric="spike_rate_hz_mean")
    .regularizer("vm_reg", target=-65.0, weight=0.1, metric="mean_V_m")
    .gate("rate_gate", threshold=200.0, criterion="below", metric="spike_rate_hz_mean")
)

eval_report = model.evaluate(signals, obj)
# eval_report["evaluation_status"] == "objective_evaluate_v0.0.5"
# eval_report["all_gates_pass"] in {True, False}  <- computational diagnostic only
# eval_report["truth_mode"] == "truth_safe_unverified"
# eval_report["physical_amplitude_claim_allowed"] == False
```

Known metrics: `spike_rate_hz_mean`, `spike_count_total`, `mean_V_m`,
`source_proxy_abs_mean`, `csd_proxy_abs_mean`, `lfp_proxy_abs_mean`.

Gate criteria: `below`, `above`, `equal`, `in_range`.

**Gate pass/fail is a computational diagnostic only.** It does not imply
empirical validation, biological calibration, or mechanism proof.

### Optimizer metadata / tune scaffold

```python
# Blackbox path (always allowed, no gradient required)
same_model, tune_report = model.tune(obj, optimizer="GSDR", steps=1)
assert same_model is model                          # model never mutated
assert tune_report["tuning_status"] == "metadata_only_v0.0.5"
assert tune_report["same_model_unchanged"] is True

# OptimizerSpec with explicit differentiability declaration
spec = jtfne.gsdr(alpha=0.7, exploration=0.05)
spec = jtfne.agsdr()
spec = jtfne.random_search()
# Optax path (requires declared_surrogate or differentiable, Optax installed)
spec = jtfne.optax_adam(learning_rate=1e-3, differentiability_status="declared_surrogate")
```

`Model.tune()` in v0.0.5 is a **metadata-only scaffold**. No optimization loop
runs and no parameters are changed. The report documents what would happen in
a future real tuning pass.

The Optax path requires explicit `differentiability_status="declared_surrogate"` or
`"differentiable"`. The default `"not_checked"` is blocked because spiking networks
are not differentiable through spike resets without a surrogate gradient declaration.

### Manifest with v0.0.5 metadata

```python
manifest = model.manifest(
    signals=signals,
    readout=readout,
    paradigm=paradigm.to_dict(),
    objective={"name": obj.name, "losses": obj.losses, ...},
    evaluation=eval_report,
    tuning=tune_report,
)
# manifest["v005_claim_labels"]["empirical_validation_status"] == "not_empirically_validated"
# manifest["v005_claim_labels"]["mechanism_claim_status"] == "not_claimed"
# All v0.0.4 truth gates still present
```

## Version roadmap

```text
v0.0.1  skeleton
v0.0.2  API/object hardening
v0.0.3  runtime + source-field status metadata
v0.0.4  source projection + probe invariant tests
v0.0.5  task paradigm + objective/evaluate + optimizer metadata scaffold
v0.0.6  (future) real optimization loop — GSDR/AGSDR/Optax
v0.0.7  (future) Jaxley bridge skeleton hardening
```

## Package structure

```text
jaxfne/
  __init__.py        public API surface
  core.py            Configuration, Model, Simulation, RuntimeConfig, Signals,
                     Probe, Objective, Paradigm, ParadigmEvent, ParadigmCondition
  emitters.py        Izhikevich EIG scaffold
  fields.py          laminar proxy source/probe layer and invariant diagnostics
  optim.py           OptimizerSpec, GSDR/AGSDR/random_search/Optax specs,
                     require_optax guard, legacy AGSDR class
  bridges.py         Jaxley bridge scaffold + require_jaxley guard
  io.py              strict JSON manifest, hashing, save/load
```

## Development smoke

```bash
python -m compileall -q jaxfne tests examples
python -m pytest -q
python examples/minimal_eig_column.py
python examples/global_local_oddball_sketch.py
python examples/02_omission_scaffold.py
python examples/03_objective_and_tune_smoke.py
```

Expected: 55 passed, 0 failed.

## Truth status

```text
truth_mode:                    truth_safe_unverified
claim_level:                   computational_scaffold
source_calibration_status:     uncalibrated_izhikevich_native_current
source_projection_mode:        proxy_no_field_solve
field_solver_status:           laminar_proxy_no_pde
field_claim_level:             proxy_readout_only
physical_amplitude_claim_allowed: false
empirical_validation_status:   not_empirically_validated
mechanism_claim_status:        not_claimed
```

No calibrated physical CSD/LFP/EEG/MEG amplitude claim is made.
No biological mechanism is implied by gate pass/fail or tuning metadata.
Optimizer success does not constitute empirical validation.
