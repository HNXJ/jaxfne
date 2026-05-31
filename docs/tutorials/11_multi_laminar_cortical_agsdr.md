# Multi-Laminar Cortical Modeling and Stable Fine-Tuning with AGSDR

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb)

This tutorial demonstrates a package-native workflow for configuring a multi-laminar cortical scaffold, simulating baseline activity, stimulating a selected subpopulation, tuning with AGSDR, and rendering spectrolaminar proxy readouts.

## Scope

This tutorial uses simulated/proxy readouts and preserves the release gates:

```yaml
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

It covers configuration, simulation, native-drive proxy stimulation, black-box AGSDR tuning, and visualization. It does not provide calibrated LFP/CSD amplitudes, real EEG/MEG forward modeling, biological metabolism, mechanism proof, or a solved PDE field model.

---

## v0.3.21 Etude No. 1 Compatibility Addendum

> **Add-only.** The original plan below remains the workflow spine. This addendum defines the release-compatible notebook behavior and supersedes earlier values where they differ.

### Artifact class

```yaml
artifact_class: etude
artifact_id: etude_no_1
suite_id: null
scope: advanced_full_detail_usage
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_claim_level: proxy_readout_only
physical_amplitude_claim_allowed: false
```

Etude No. 1 is separate from Suite numbering. It is allowed to be denser than a Suite, but it must remain package-native, executable, JSON-safe, and proxy-safe.

### Part 1 — Single-unit Izhikevich warmup

Before the multi-laminar scaffold, the notebook includes a warmup for the four cell classes used later (E, PV, SST, VIP) using **package-native functions only** — no local Izhikevich solver:

```python
WARMUP_LABELS      = ("E", "PV", "SST", "VIP")
WARMUP_LAYERS      = ("warmup", "warmup", "warmup", "warmup")
WARMUP_DRIVE       = {"E": 5.0, "PV": 3.0, "SST": 3.5, "VIP": 3.0}
WARMUP_DURATION_MS = 300.0
WARMUP_SOURCE_SCALE = 1.0
WARMUP_DTYPE       = "float32"

warmup_params = jtfne.izhikevich_params_from_labels(
    labels=WARMUP_LABELS, layer_labels=WARMUP_LAYERS,
    dtype=WARMUP_DTYPE, drive_overrides=WARMUP_DRIVE,
    source_scale=WARMUP_SOURCE_SCALE,
)
warmup_key   = jax.random.PRNGKey(SEED)
warmup_steps = int(round(WARMUP_DURATION_MS / DT_MS))

warmup_V, warmup_spikes, warmup_sources = jtfne.simulate_eig_izhikevich(
    params=warmup_params, n_steps=warmup_steps, dt_ms=DT_MS,
    key=warmup_key, dtype=WARMUP_DTYPE,
    drive_schedule=None, silence_mask=None,
)
```

Output: a rate table (one row per cell class) and a native V\_m trace figure. These are reduced-model diagnostics, not calibrated physical voltages.

### Explicit-editability rule

Every important argument is shown explicitly, including defaults. Avoid compact one-line scientific calls in Etude cells:

```python
# Etude style — verbose, editable
sim = jtfne.Simulation(
    duration_ms=DURATION_MS, dt_ms=DT_MS, plasticity=0.0,
    seed=SEED, record_sources=True, record_fields=True,
    poisson_drive=None, runtime=None, ablation=None,
)
```

### Target selection (supported arguments only)

```python
TARGET_AREA      = "V1"
TARGET_LAYER     = None   # set to e.g. "L4" to restrict
TARGET_CELL_TYPE = "E"

target_indices = jtfne.select_neurons(
    model, area=TARGET_AREA, layer=TARGET_LAYER, cell_type=TARGET_CELL_TYPE,
)
```

### Stimulus schedule (named variables)

```python
STIM_LABEL       = "V1_E_native_drive"
STIM_ONSET_MS    = 100.0
STIM_DURATION_MS = 100.0
STIM_AMPLITUDE   = 1.5

STIM_EVENTS = [{"label": STIM_LABEL, "onset_ms": STIM_ONSET_MS,
                "duration_ms": STIM_DURATION_MS, "amplitude": STIM_AMPLITUDE,
                "target_indices": target_indices.tolist()}]

stim = jtfne.stimulus_schedule(STIM_EVENTS, n_neurons=n_units)
```

### Objective (metric-backed, v0.3.21+)

```python
TARGET_RATE_HZ          = 3.5   # supersedes earlier 5.0 Hz value
TARGET_KAPPA_SYNCHRONY  = 0.0
RATE_WEIGHT             = 1.0
SYNCHRONY_WEIGHT        = 0.25

objective = jtfne.rate_synchrony_targets(
    target_rate_hz=TARGET_RATE_HZ,
    target_kappa_synchrony=TARGET_KAPPA_SYNCHRONY,
    rate_weight=RATE_WEIGHT,
    synchrony_weight=SYNCHRONY_WEIGHT,
)
```

The objective uses `spike_rate_hz_mean` and `kappa_synchrony` — both computed engine metrics in v0.3.21.

### Optimizer (explicit all kwargs, supported parameter only)

```python
AGSDR_PARAMETERS      = {"drive_gain": (0.1, 1.5)}  # not noise_amplitude
AGSDR_GENERATIONS     = 3
AGSDR_POPULATION_SIZE = 2
AGSDR_SEED            = SEED
AGSDR_ALPHA           = 0.7
AGSDR_EXPLORATION     = 0.05
AGSDR_DESELECT_FACTOR = 2.0

optimizer = jtfne.agsdr(
    alpha=AGSDR_ALPHA, exploration=AGSDR_EXPLORATION,
    deselect_factor=AGSDR_DESELECT_FACTOR, metadata=None,
    parameters=AGSDR_PARAMETERS, generations=AGSDR_GENERATIONS,
    population_size=AGSDR_POPULATION_SIZE, seed=AGSDR_SEED,
    inner_optimizer=None, inner_steps=0, inner_objective=None,
)
```

### Visualization (explicit all kwargs)

```python
FIG_FREQ_MIN_HZ = 1.0
FIG_FREQ_MAX_HZ = 150.0
FIG_FREQ_COUNT  = 128
FIG_PSD_NPERSEG = 128
FIG_SIZE        = (12, 8)
FIG_DPI         = 150
FIG_TITLE       = "Etude No. 1 - simulated proxy readouts"

fig = jtfne.vis.spectrolaminar_suite(
    signals_tuned,
    freq_min_hz=FIG_FREQ_MIN_HZ, freq_max_hz=FIG_FREQ_MAX_HZ,
    freq_count=FIG_FREQ_COUNT,   psd_nperseg=FIG_PSD_NPERSEG,
    figsize=FIG_SIZE, dpi=FIG_DPI, title=FIG_TITLE,
)
```

### Output path (v0.3.21)

Outputs write to `outputs/etude_no_1/` (supersedes `outputs/multi_laminar_cortical_agsdr/`):

```text
outputs/etude_no_1/
  manifest.json
  validation_report.json
  metrics.json
  asset_hashes.json
  figures/spectrolaminar.png
```

`manifest.json` includes an `editable_inputs` object covering all 11 groups: `runtime`, `columns`, `cell_types`, `drive`, `inter_column_connectivity`, `field`, `probes`, `objective`, `optimizer`, `stimulus`, `visualization`.

`metrics.json` includes real optimizer evidence:

```yaml
best_score:          finite_number
best_parameters:     non_empty_object   # e.g. {"drive_gain": 0.709}
tuning_status:       ACCEPT_CANDIDATE
same_model_unchanged: false
rate_improvement_hz: positive_number
kappa_improvement:   number
```

### Acceptance criteria (v0.3.21)

```yaml
nbformat_valid:                    true
install_cells_use_jaxfne_viz:      true
git_install_targets_main:          true
colab_badge_targets_main:          true
canonical_import:                  "import jaxfne as jtfne"
single_unit_warmup_present:        true
package_native_paths_only:         true
all_major_inputs_explicit:         true
editable_inputs_exported:          true
unsupported_tunable_parameter:     absent
best_score_finite:                 true
best_parameters_non_empty:         true
same_model_unchanged:              false
rate_improvement_hz_positive:      true
strict_json_exports:               true
png_figure_present:                true
TFNE_SMOKE_1_nbclient_pass:        true
TFNE_SMOKE_0_nbclient_pass:        true
mkdocs_nav_entry_present:          true
mkdocs_build_strict_pass:          true
```

---

## Learning objectives

1. Create a multi-area spectrolaminar cortex configuration.
2. Edit every major configuration domain explicitly.
3. Construct and simulate a model from the edited configuration.
4. Select a custom neuron subset using package-native metadata.
5. Apply targeted native-drive proxy stimulation.
6. Tune with AGSDR toward overall 3.5 Hz firing rate and kappa synchrony 0.0 (see addendum).
7. Render a spectrolaminar suite with custom frequency limits and resolution.
8. Export JSON-safe manifest and validation reports.

## Colab installation cells

The notebook includes both supported install paths:

```python
!pip install -q "jaxfne[viz]"
```

```python
!pip install -q "jaxfne[viz] @ git+https://github.com/HNXJ/jaxfne.git@main"
```

The first cell is the stable PyPI install (with the `viz` extra for matplotlib-backed figures). The second cell installs the current `main` release candidate from GitHub before PyPI catches up.

## Configuration domains exercised

| Domain | Example API | Purpose |
|---|---|---|
| Runtime | `cfg.runtime(...)` | Seed, duration, time step, dtype, JIT metadata |
| Columns | `default_spectrolaminar_config(...)` plus `.column(...)` metadata | V1/V4 multi-column scaffold |
| Cell types | `.cell_types(...)` | E/PV/SST/VIP fractions |
| Drive | `.drive(...)` | Baseline native-drive proxy and event metadata |
| Inter-column connectivity | `.inter_column_connectivity(...)` | Sparse feedforward/feedback metadata |
| Field/proxy | `.field(...)` | Laminar proxy status and boundary/gauge metadata |
| Probes | `.probes(...)` | SPK, Vm, source, LFP-like, CSD-like, EEG-like, MEG-like, EMM-proxy labels |
| Objective | `.objective(...)` and `rate_synchrony_targets(...)` | 5 Hz rate and kappa synchrony target |
| Optimizer | `.optimizer(...)` and `agsdr(...)` | AGSDR search metadata and execution |

## Main workflow

```python
import jaxfne as jtfne

cfg = jtfne.default_spectrolaminar_config(
    areas=["V1", "V4"],
    n_per_area=80,
    seed=20260530,
    duration_ms=1000.0,
    dt_ms=0.1,
)
```

The notebook then edits all major domains, constructs a model, simulates baseline activity, selects a V1/L4/E target subset, applies native-drive proxy stimulation, tunes with AGSDR, and exports manifest artifacts.

## Custom subset stimulation

The notebook uses package-native selection and targeted schedules:

```python
target_indices = jtfne.select_neurons(model, area="V1", layer="L4", cell_type="E")

stim = jtfne.stimulus_schedule(
    [{
        "label": "custom_V1_L4_E_drive",
        "onset_ms": 250.0,
        "duration_ms": 150.0,
        "amplitude": 1.25,
        "target_indices": target_indices,
    }],
    n_neurons=model.summary()["n_units"],
)
```

The schedule report records selected-index targeting and remains JSON-safe.

## AGSDR fine tuning

The objective is explicit and proxy-scaffold bounded:

```python
objective = jtfne.rate_synchrony_targets(
    target_rate_hz=3.5,          # v0.3.21: 3.5 Hz (see addendum)
    target_kappa_synchrony=0.0,
    rate_weight=1.0,
    synchrony_weight=0.25,
)
```

The AGSDR tuning report includes best parameters, best score, achieved firing rate, achieved kappa synchrony, and `differentiability_status=nondifferentiable_spiking`.

## Spectrolaminar suite visualization

The notebook demonstrates custom frequency limits and resolution:

```python
fig = jtfne.vis.spectrolaminar_suite(
    signals_tuned,
    freq_min_hz=1.0,
    freq_max_hz=150.0,
    freq_count=128,
    psd_nperseg=512,
    figsize=(14, 10),
    dpi=160,
    title="Multi-laminar cortical AGSDR tuning - simulated proxy readouts",
)
```

The title remains proxy-safe and the function checks finite LFP-like/CSD-like arrays before plotting.

## Exported artifacts

The notebook writes outputs under:

```text
outputs/etude_no_1/          ← v0.3.21 path (supersedes outputs/multi_laminar_cortical_agsdr/)
```

Expected artifacts:

```text
manifest.json
validation_report.json
metrics.json
asset_hashes.json
figures/spectrolaminar.png
```

## Failure modes

| Failure mode | Likely cause | Action |
|---|---|---|
| Empty target subset | Compact scaffold metadata has sparse layer coverage | Select by area/cell type or increase `n_per_area` |
| High kappa after tuning | Synchrony penalty too light | Increase `SYNCHRONY_WEIGHT` |
| High rate after tuning | Drive bounds too high | Widen `AGSDR_PARAMETERS` bounds downward |
| Silent run | Drive bounds too low or inhibition too strong | Raise drive or lower inhibitory gain |
| Slow Colab run | Large `n_per_area`, long duration, or AGSDR budget | Reduce population or budget for smoke mode |
| Tuning returns unchanged model | Unsupported parameter or unscorable objective | Use `drive_gain`; verify finite `best_score` and non-empty `best_parameters` |
| All AGSDR scores are null | Objective metric names do not match computed metrics | Use the fixed `rate_synchrony_targets` path backed by `spike_rate_hz_mean` and `kappa_synchrony` (v0.3.21+) |
| Figure cell fails in a bare kernel | Visualization extras missing | Install with `jaxfne[viz]` |
| Colab opens stale branch content | Badge points to `dev` or old path | Point badge to `blob/main/tutorials/etudes/...` |
| Warmup rates are silent or saturated | Native drive values too low or high | Edit `WARMUP_DRIVE` per cell class |
| Exported metrics show no improvement | AGSDR budget too small or bounds too narrow | Increase `AGSDR_GENERATIONS`, `AGSDR_POPULATION_SIZE`, or widen `drive_gain` bounds |

## Exercises

**Original exercises (workflow spine):**

1. Change the target subset from V1/E to V4/PV.
2. Tune to 6.0 Hz instead of 3.5 Hz.
3. Increase `SYNCHRONY_WEIGHT` and inspect `kappa_improvement`.
4. Set `FIG_FREQ_MAX_HZ` to 80.0 Hz in the visualization.
5. Add a second targeted event with another cell type.
6. Compare feedforward-only and feedback-only metadata settings.
7. Save two validation reports and compare their achieved rate and kappa values.

**Addendum exercises (Etude depth):**

8. Change `WARMUP_DRIVE` for PV (e.g. to 6.0) and compare the single-unit rate table.
9. Increase `AGSDR_POPULATION_SIZE` to 8 and compare `best_score`.
10. Export two `metrics.json` files (default vs. wider `drive_gain` bounds) and compare `rate_improvement_hz`.
