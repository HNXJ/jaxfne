# Multi-Laminar Cortical Modeling and Stable Fine-Tuning with AGSDR

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/dev/notebooks/jaxfne_multi_laminar_cortical_agsdr.ipynb)

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

## Learning objectives

1. Create a multi-area spectrolaminar cortex configuration.
2. Edit every major configuration domain explicitly.
3. Construct and simulate a model from the edited configuration.
4. Select a custom neuron subset using package-native metadata.
5. Apply targeted native-drive proxy stimulation.
6. Tune with AGSDR toward overall 5 Hz firing rate and kappa synchrony 0.0.
7. Render a spectrolaminar suite with custom frequency limits and resolution.
8. Export JSON-safe manifest and validation reports.

## Colab installation cells

The notebook includes both supported install paths:

```python
!pip install -q jaxfne
```

```python
!pip install -q "jaxfne @ git+https://github.com/HNXJ/jaxfne.git@dev"
```

The second cell intentionally overwrites the PyPI install when users want the development branch.

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
    target_rate_hz=5.0,
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
outputs/multi_laminar_cortical_agsdr/
```

Expected artifacts:

```text
manifest.json
validation_report.json
figures/multi_laminar_spectrolaminar_suite.png
```

## Failure modes

| Failure mode | Likely cause | Action |
|---|---|---|
| Empty target subset | Compact scaffold metadata has sparse layer coverage | Select by area/cell type or increase `n_per_area` |
| High kappa after tuning | Synchrony penalty too light | Increase `synchrony_weight` |
| High rate after tuning | Drive bounds too high | Lower `drive_scale` / `noise_amplitude` bounds |
| Silent run | Drive bounds too low or inhibition too strong | Raise drive or lower inhibitory gain |
| Slow Colab run | Large `n_per_area`, long duration, or AGSDR budget | Reduce population or budget for smoke mode |

## Exercises

1. Change the target subset from V1/L4/E to V4/L2/3/PV.
2. Tune to 8 Hz instead of 5 Hz.
3. Increase `synchrony_weight` to 1.0.
4. Set `freq_max_hz=80.0` in the visualization.
5. Add a second targeted event with another cell type.
6. Compare feedforward-only and feedback-only metadata settings.
7. Save two validation reports and compare their achieved rate and kappa values.
