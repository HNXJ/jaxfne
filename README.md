## Installation

For the future PyPI release, the intended Colab entry point is %pip install jaxfne. Until PyPI publication, use the repository or built wheel.

See [docs/COLAB.md](docs/COLAB.md) and [docs/PACKAGING.md](docs/PACKAGING.md).

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

# canonical v0.0.16+ workflow
signals = model.simulate(sim)
receipt = model.run_receipt(signals)
readout = model.compute_readout(signals, [
    jtfne.readout_spec("rate", "spike_rate_hz"),
    jtfne.readout_spec("csd",  "csd_abs_mean"),
])
```

## Release Scope (v0.1.1)

**JaxFNE is a compact JAX-native framework for Tensor-Field Neural Equation (TFNE) workflows.** It provides emitter, source, field, probe, readout, objective, receipt, and manifest interfaces for reproducible computational neurophysics experiments.

**v0.1.1 supports:**
- CPU-first spectrolaminar proxy workflows
- Deterministic reproducible receipts and JSON-safe manifests
- Configurable laminar population and probe geometry
- Conservative source/field metadata with explicit claim boundaries

**Status Metadata:**
- `truth_mode`: `truth_safe_unverified` — no empirical or biological claims
- `claim_level`: `computational_scaffold` — reference architecture, not calibration
- `source_calibration_status`: `uncalibrated_izhikevich_native_current` — native, unscaled units
- `field_solver_status`: `laminar_proxy_no_pde` — forward-field proxy, not PDE-solved
- `field_claim_level`: `proxy_readout_only` — readout and metadata, no physical unit scaling
- `physical_amplitude_claim_allowed`: `false` — no physical unit claims without validation

Receipts and manifests are computational validation artifacts that document model state and metadata. Empirical validation and physical calibration require external evidence and workflow-supplied validation gates.

## Identity

`jaxfne` is not primarily a neuron model, optimizer, plotting library, or data format. It is the composition layer:

```text
Emitter -> Source -> Field -> Probe -> Objective -> Optimizer
```

JAX handles arrays, compilation, batching, and device execution. Jaxley can later provide detailed emitters. Optax can later provide differentiable optimizers. `jaxfne` handles TFNE source-to-field/readout contracts, diagnostics, invariant checks, and manifests.

## Current status (v0.1.1)

v0.1.1 refines spectrolaminar computational correctness for Paper 1.0 in-silico workflows. It adds VIP Izhikevich cell type presets, population-to-neuron slicing, and a systematic preset registry for neurons and receptors. All truth gates remain frozen at v0.0.16+ baselines. Install and Colab smokes validated from PyPI. v0.1.1 is a stable maintenance release supporting reproducible, deterministic TFNE laminar workflows.

**Field solver approach:** `jaxfne` models source-to-field coupling through a **laminar proxy readout** based on layered source contributions rather than solving the full resistive extracellular PDE:

```text
Resistive PDE formulation (out of scope):
  J_e = -sigma_e grad(phi_e)
  div(J_e) = q
  → CSD = div(J_e)

Proxy approach (in scope):
  sources → laminar depth integrations
  → CSD proxy, LFP proxy (unsolved, metadata-only)
```

This proxy design keeps the model compact, differentiable, and deterministic while providing structured metadata for future full-PDE workflows:

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

### Manifest with v0.0.5 metadata (compatibility alias)

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

`model.manifest()` is a compatibility alias retained from v0.0.4–v0.0.14.
For the canonical v0.1 workflow, prefer `model.run_receipt()` and
`model.evaluate_report()`.

## Version roadmap (Publication Event 1.0)

The roadmap centers around **Paper 1.0: in-silico spectrolaminar motif**, deferring omission (Paper 2.0) and global/local oddball (Paper 3.0).

**Pre-v0.1 bridge lane:**
* `v0.0.11`  uncalibrated multi-receptor synaptic kernel
* `v0.0.12`  paradigm/stimulus injection into recurrent model
* `v0.0.13`  laminar population builder / source geometry metadata
* `v0.0.14`  trial runner and condition-aligned outputs
* `v0.0.15`  config/JaxFNEConfig standard (.jcfg.json)
* `v0.0.16`  RunReceipt deterministic audit receipt
* `v0.0.17`  ReadoutSpec declarative feature extraction
* `v0.0.18`  ObjectiveReport structured evaluation result

**Publication Event 1.0 sequence:**
* `v0.0.20`   semantic correctness hardening  ← **current release**
* `v0.1.0`   practical OOP core freeze (pending v0.0.20 gate)
* `v0.2.x`   spectrolaminar objective/readout base
* `v0.3.x`   generative spectrolaminar workflow
* `v0.4.0`   Paper 1.0 minimum sufficient release (in-silico spectrolaminar motif)

**Deferred (Post-Paper 1.0):**
* Paper 2.0: Omission mismatch workflows
* Paper 3.0: Global/local oddball workflows
* Dense Jaxley/NEURON compartment bridges
* EEG/MEG physical solvers

## Scope Boundaries

**What v0.1.1 is:**
- A compact JAX-native OOP framework for reproducible TFNE workflows
- A deterministic laminar proxy model for spectrolaminar population dynamics
- A computational scaffold with metadata-driven architecture
- A testbed for condition-aligned trial batching and feature extraction

**What v0.1.1 does not claim:**
- Mechanistic validation of spectrolaminar circuit function
- Calibrated extracellular field amplitudes (LFP/CSD at physical units)
- Full biophysical realism (simplified Izhikevich models, proxy fields)

External validation, empirical benchmarking, and physical calibration are responsibilities of downstream analysis workflows, not this package.

## Package structure

```text
jaxfne/
  __init__.py        public API surface
  core.py            Configuration, Model, Simulation, RuntimeConfig, Signals,
                     Probe, Objective, Paradigm, ParadigmEvent, ParadigmCondition,
                     RunReceipt, ReadoutSpec, ReadoutResult, ObjectiveReport,
                     JaxFNEConfig, ConfigValidationResult (v0.0.15–v0.0.18)
  emitters.py        Izhikevich EIG scaffold
  fields.py          laminar proxy source/probe layer and invariant diagnostics
  optim.py           OptimizerSpec, GSDR/AGSDR/random_search/Optax specs,
                     require_optax guard, legacy AGSDR class
  bridges.py         Jaxley bridge scaffold + require_jaxley guard
  io.py              strict JSON manifest, hashing, save/load, save_receipt
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

Expected: 178 passed, 0 failed.

## v0.0.9 edge-list backend

`v0.0.9` introduces a sparse `EdgeList` recurrent backend using JAX pytrees, `jax.lax.scan`, and `jax.ops.segment_sum`. Enabled with `runtime(recurrent_backend="edge_list")`. This provides a memory-efficient computational path for networks with structured sparsity while maintaining deterministic reproducibility and JSON-safe serialization. The dense backend remains the default.

## v0.0.10 synapse metadata

`v0.0.10` adds `ReceptorSpec` and `SynapseSpec` metadata structures for declaring synaptic properties. These provide structured documentation of receptor types, kinetics, and synaptic parameters in the model manifest. The edge-list backend propagates receptor and receptor-index information transparently. Dense and sparse backends maintain numerical parity under matched initial conditions.

## v0.0.11 receptor-indexed exponential synaptic kernel

`v0.0.11` adds an optional receptor-indexed synaptic kernel (`synaptic_kernel="receptor_exponential"`) that looks up per-receptor decay constants from the `ReceptorSpec` registry. Synaptic state remains compact (`n_edges` shape) and aggregation uses `jax.ops.segment_sum()` for efficient postsynaptic integration. Both single-exponential and receptor-indexed pathways maintain the laminar proxy architecture. Receptor reversal potentials are documented in metadata but not used in the current neuronal current model.

## v0.0.12 native stimulus injection

`v0.0.12` adds `StimulusSchedule` and event-aligned stimulus injection for condition-based trial workflows. The `stimulus_schedule()` factory builds timed current arrays from `ParadigmCondition` event timing. Injected current is native and unsacaled, allowing trial-by-trial customization. This supports repetition-based experimental designs for spectrolaminar context effects.

## v0.0.13 laminar source geometry

`v0.0.13` introduces explicit `LaminarPopulation` and `LaminarSourceGeometry` structures that map neural populations to laminar depth coordinates. The `geometry` parameter in `construct(cfg, geometry=...)` enables deterministic population-to-neuron indexing. Co-located populations (depth overlap) are permitted and documented as anatomically valid. Population slicing is provided via the `population_slices()` method on `LaminarSourceGeometry`.

## v0.0.14 sequential trial runner

`v0.0.14` adds deterministic batch trial execution via `TrialSpec`, `TrialBatch`, and `Model.run_trials()`. Each trial is seeded independently and exceptions are captured for graceful error handling. Results are compact and JSON-safe; large JAX arrays are replaced with summary statistics for efficient storage and transfer.

## v0.0.15 config/object standard foundation

`v0.0.15` introduces the `.jcfg.json` declarative configuration format via `JaxFNEConfig` and associated validation/loading utilities. The schema maps directly to proven objects (`Simulation`, `Configuration`, `LaminarSourceGeometry`, `TrialBatch`). Truth boundary fields are required and validated in all configs. Geometry depths use normalized laminar-proxy coordinates (relative depth in [0, 1]); physical units are not inferred.




## v0.0.16 run receipt

`v0.0.16` introduces `RunReceipt` — a deterministic, immutable, JSON-safe record of a single simulation run. The receipt captures config fingerprint, simulation parameters, signal summary, truth gates, and backend metadata. Receipt IDs are deterministic SHA256 prefixes: identical configuration + seed + version always produce the same receipt ID, enabling deduplication and content-addressed archive workflows. `save_receipt(receipt, path)` provides atomic file creation with overwrite protection.

## v0.0.17 readout spec

`v0.0.17` adds declarative feature extraction via `ReadoutSpec` and `Model.compute_readout(signals, specs)`. Supported metrics include spike rates, counts, membrane voltage, and field proxies (CSD, LFP). Optional temporal and depth windowing enable focused analysis on specific layers and time windows. Results are structured, composable, and JSON-safe.

| Metric | Description |
|--------|-------------|
| `spike_rate_hz` | Mean firing rate across all units (Hz) |
| `spike_count` | Total spike count |
| `mean_V_m` | Mean membrane voltage |
| `csd_abs_mean` | Mean absolute CSD proxy |
| `lfp_abs_mean` | Mean absolute LFP proxy |
| `source_abs_mean` | Mean absolute source proxy |

## v0.0.18 objective report

`v0.0.18` adds structured objective evaluation via `ObjectiveReport` and `Model.evaluate_report()`. Reports embed losses, regularizers, and gate diagnostics alongside optional readout results. Gate status is a computational diagnostic; pass/fail indicates whether objective targets are met, not empirical validation. Truth metadata is frozen in every report for audit transparency.

## Release rehearsal (no credentials required)

```bash
./scripts/release_rehearsal.sh
```

Runs the full pre-publish gate locally: clean build → twine check → fresh venv wheel+sdist install smokes from /tmp → pytest → all examples. No upload, no credentials, no tagging.

## Colab smoke

See [`docs/COLAB_SMOKE_V010.md`](docs/COLAB_SMOKE_V010.md) for copy-pasteable Colab cells to validate `pip install jaxfne==0.1.0` from TestPyPI or real PyPI.

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
