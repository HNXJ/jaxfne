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
# API scaffold for Paper 2.0 (Omission deferred)
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

## Version roadmap (Publication Event 1.0)

The roadmap centers around **Paper 1.0: in-silico spectrolaminar motif**, deferring omission (Paper 2.0) and global/local oddball (Paper 3.0).

**Pre-v0.1 bridge lane:**
* `v0.0.11`  uncalibrated multi-receptor synaptic kernel
* `v0.0.12`  paradigm/stimulus injection into recurrent model
* `v0.0.13`  laminar population builder / source geometry metadata
* `v0.0.14`  trial runner and condition-aligned outputs
* `v0.0.15`  v0.1 readiness audit

**Publication Event 1.0 sequence:**
* `v0.1.0`   practical OOP core freeze
* `v0.2.x`   spectrolaminar objective/readout base
* `v0.3.x`   generative spectrolaminar workflow
* `v0.4.0`   Paper 1.0 minimum sufficient release (in-silico spectrolaminar motif)

**Deferred (Post-Paper 1.0):**
* Paper 2.0: Omission mismatch workflows
* Paper 3.0: Global/local oddball workflows
* Dense Jaxley/NEURON compartment bridges
* EEG/MEG physical solvers

## Claims Discipline

**Allowed:**
- v0.1.x provides the compact JAX-native OOP core required to build reproducible TFNE workflows.

**Forbidden:**
- v0.1.x validates spectrolaminar mechanisms.
- v0.1.x produces calibrated LFP/CSD amplitudes.
- v0.1.x is a full simulator.

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

## v0.0.9 edge-list backend

`v0.0.9` adds a sparse `EdgeList` recurrent backend using JAX pytrees, `jax.lax.scan`, and `jax.ops.segment_sum`. It is selected with `runtime(recurrent_backend="edge_list")`. This is a computational backend upgrade only; field output remains `laminar_proxy_no_pde`, source calibration remains uncalibrated, and optimizer-selected candidates do not establish empirical or mechanistic claims.

## v0.0.10 synapse metadata

`v0.0.10` hardens source and synapse declarations. It introduces metadata-only `ReceptorSpec` and `SynapseSpec` definitions without adding new conductance-based physical solvers or biological kernels. The backend manifest now flows `EdgeList` details transparently. Dense-vs-edge computations maintain statistical parity. No calibrated synapse claim and no new PDE/field/empirical/mechanism claim is made.

## v0.0.11 receptor-indexed exponential synaptic kernel

`v0.0.11` adds an opt-in second synaptic kernel selected via `runtime(recurrent_backend="edge_list", synaptic_kernel="receptor_exponential")`. The default remains `synaptic_kernel="exponential"`, preserving the v0.0.9/v0.0.10 edge-list path. The new path keeps `syn_state.shape == (n_edges,)` and looks up the per-edge decay time constant from `edge.receptor_index` against the standard `ReceptorSpec` table (AMPA/GABA_A/NMDA/GABA_B). Aggregation uses `jax.ops.segment_sum(weight * syn_state, post, n_neurons)`, so each edge contributes exactly once to its postsynaptic native recurrent input; multiple edges may legitimately converge on the same neuron. Receptor reversal potentials remain metadata-only and are not used in the current computation. Weights remain native/unphysical and the source readout remains a laminar proxy. No conductance equation, no physical-amplitude claim, no PDE upgrade, and no empirical-validation or biological-mechanism claim is introduced.

## v0.0.12 native stimulus injection

`v0.0.12` adds explicit event-aligned native-drive stimulus injection. It introduces the `StimulusSchedule` value object and `stimulus_schedule()` factory to build timed drive arrays from `ParadigmCondition` events. Injected drive is added as native (uncalibrated) current to the recurrent kernels (`dense`, `edge_list`, and `receptor_exponential`). This enables basic condition-aligned trial workflows for Paper 1.0. It does not implement a full `Paradigm.batch()` trial runner, and no cognitive omission or global-local logic is encoded. No calibrated current, physical amplitude, PDE upgrade, or empirical/mechanism claim is introduced.

## v0.0.13 laminar source geometry

`v0.0.13` adds explicit laminar population and source geometry metadata. It introduces the `LaminarPopulation` and `LaminarSourceGeometry` frozen dataclasses and the `laminar_source_geometry()` factory. An optional `geometry` kwarg is added to `construct(cfg, *, geometry=None)`; when provided, the geometry's `n_units_total` must match the network `n` or a `ValueError: geometry_n_units_total_mismatch` is raised. Population depths use deterministic NumPy linspace — no random placement. Co-located populations (depth overlap) are allowed and not treated as errors, as this is anatomically valid for different cell types. The geometry manifest propagates into `Model.manifest()` under `source_geometry`. No new field solver, no physical-amplitude claim, no PDE upgrade, no empirical or mechanism claim is introduced.

## v0.0.14 sequential trial runner

`v0.0.14` adds a sequential trial runner for executing batches of condition-aligned simulations. It introduces `TrialSpec`, `TrialBatch`, `TrialResult`, and `TrialBatchResult` dataclasses, along with a `trial_batch()` factory for deterministic batch creation. The `Model.run_trials(batch, sim)` method performs a sequential loop over `batch.trials`, replacing `sim.seed` with the trial's seed and capturing any exceptions into the `TrialResult`. Serialization via `to_dict()` is strictly JSON-safe and automatically excludes large JAX arrays (`V_m`, `spikes`, `sources`, `field`) from the results, providing instead a compact `Signals.summary()`. This enables robust condition-aligned workflow orchestration for Paper 1.0. No parallel execution, no new field solver, no physical-amplitude claim, no PDE upgrade, and no empirical or mechanism claim is introduced.

## v0.0.15 config/object standard foundation

`v0.0.15` adds the first formal `.jcfg.json` declarative configuration standard.
It introduces `JaxFNEConfig`, `ConfigValidationResult`, `load_config`, `validate_config`,
`config_to_simulation`, `config_to_geometry`, `config_to_configuration`,
`config_to_trial_batch`, and `config_truth_boundary`.

The config schema maps directly onto existing proven objects (`Simulation`,
`Configuration`, `LaminarSourceGeometry`, `TrialBatch`) without introducing new kernels,
field solvers, or physical-amplitude claims.  Truth boundary fields are required in
every config file and any escalation is a blocking validation error.
Geometry depths remain normalized proxy coordinates in [0, 1] with default
`position_units = "relative_laminar_depth_proxy"` — no physical spatial units (mm, µm)
are introduced.
No new PDE solver, no calibrated current, no empirical validation, and no mechanism
claim is introduced.



## v0.0.17 readout spec

`v0.0.17` adds the declarative feature-extraction standard for Paper 1.0 workflows.
It introduces `ReadoutSpec`, `ReadoutResult`, the `readout_spec()` factory, and
`Model.compute_readout(signals, specs)`.

A `ReadoutSpec` declares a named scalar metric to extract from `Signals`.
Supported metrics:

| Metric | Description |
|--------|-------------|
| `spike_rate_hz` | Mean firing rate across all units (Hz) |
| `spike_count` | Total spike count |
| `mean_V_m` | Mean membrane voltage |
| `csd_abs_mean` | Mean absolute CSD proxy |
| `lfp_abs_mean` | Mean absolute LFP proxy |
| `source_abs_mean` | Mean absolute source proxy |

Optional `time_window_ms=(start, end)` and `n_contacts_slice=(start, end)` allow
temporal and depth windowing.  `ReadoutResult.status` is `"computed"`,
`"no_field"`, or `"unknown_metric"`.

`_KNOWN_READOUT_METRICS` is exported from both `jaxfne` and `jaxfne.core`.

No new PDE solver, no calibrated amplitude, no empirical validation, and no
mechanism claim is introduced.

## v0.0.16 run receipt

`v0.0.16` adds `RunReceipt` — an immutable, JSON-safe record of a single simulation run.
It introduces `RunReceipt`, `Model.run_receipt()`, the module-level `run_receipt()` factory,
and `save_receipt()` for writing receipts to disk with an overwrite guard.

A `RunReceipt` captures: config fingerprint (`config_hash`), simulation parameters,
`Signals.summary()`, frozen truth gates, claim labels, and backend metadata.
The `receipt_id` is deterministic: same configuration + seed + version always produces
the same 16-char SHA256 prefix.  `save_receipt(receipt, path)` raises
`ValueError("receipt_file_exists: ...")` if the path already exists, unless
`overwrite=True` is passed.

The `manifest_schema_version` default in `io.manifest()` is bumped from `"0.0.4"` to
`"0.0.16"` to reflect the current package version.

No new PDE solver, no calibrated current, no empirical validation, and no mechanism
claim is introduced.

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
