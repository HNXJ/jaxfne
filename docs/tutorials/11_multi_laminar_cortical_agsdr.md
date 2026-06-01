# Etude No. 1: Multi-Laminar Cortical AGSDR

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb)

Artifact class: Etude. This page tracks the release notebook used for the multi-laminar cortical AGSDR workflow.

## What this tutorial does

This tutorial builds and runs a JAXFNE scaffold with four sections:

1. Interactive single-emitter waveform preview.
2. Uniform-density cortical column scaffold.
3. Baseline, stimulus, tuned readouts, and spectrolaminar suites.
4. Run artifacts, validation metrics, hashes, exercises, and scope.

The workflow follows the package-native path:

```text
configure -> construct -> simulate -> visualize -> optimize -> export
```

## Run status

```yaml
artifact_class: etude
artifact_id: etude_no_1
run_status: tutorial_scaffold
model_status: computational_scaffold
field_solver_status: laminar_proxy_no_pde
field_readout_status: proxy_readout_only
amplitude_status: native_unscaled
```

The workflow exports simulated proxy readouts and validation artifacts for the configured run. LFP/CSD-like outputs are laminar proxy readouts. No PDE solver is executed in this workflow.

## Colab setup

The notebook contains install cells for both public package use and current `main` branch testing:

```python
!pip install -q "jaxfne[viz]>=0.3.24"
```

```python
!pip install -q --force-reinstall --no-cache-dir "jaxfne[viz] @ git+https://github.com/HNXJ/jaxfne.git@main"
```

The notebook verifies the installed package and hard-fails when required visualization APIs are unavailable.

## Learning objectives

1. Use a centralized full-detail config as the edit anchor.
2. Explore single-emitter waveform intuition in an interactive HTML panel.
3. Run package-native E/PV/SST/VIP warmup traces.
4. Construct a uniform-density multi-area laminar scaffold.
5. Simulate baseline, stimulus, and tuned conditions.
6. Visualize declared geometry, activity, and spectrolaminar proxy readouts.
7. Tune with AGSDR toward a 5.0 Hz average firing-rate target and low synchrony.
8. Export JSON-safe manifest, validation report, metrics, hashes, PNG figures, and Plotly HTML.

## Notebook structure

```text
setup
centralized config
Section 1: interactive single-emitter waveform preview
Section 2: uniform-density cortical column scaffold
Section 3: baseline / stimulus / tuned simulations and readouts
Section 4: run artifacts, validation metrics, hashes, exercises, scope
```

## Configuration domains

Expose these as named values or dictionaries and export them under `manifest["editable_inputs"]`:

```text
runtime, geometry, areas, layers, cell_types, cell_colors, cell_signs,
layer fractions, native drive, connectivity metadata, field/proxy metadata,
probes, objective, optimizer, stimulus, visualization, artifact paths, status fields
```

Core tutorial settings:

```yaml
dt_ms: 0.1
dtype: float32
seed: deterministic
column_height_mm: 1.5
column_radius_mm: 0.125
target_rate_hz: 5.0
target_kappa: 0.0
primary_tunable: drive_gain
```

## Section 1: interactive single-emitter waveform preview

Section 1 embeds `tutorials/etudes/assets/izhikevich_waveform_control_panel.html` and copies it to:

```text
outputs/etude_no_1/html/izhikevich_waveform_control_panel.html
```

The panel is a browser-side Euler preview for parameter intuition. The package-native JAXFNE warmup cells remain the executable evidence path for the tutorial.

## Section 2: uniform-density cortical column scaffold

Section 2 builds the cortical column scaffold with uniform cell density across layers. This section is the reference condition for later readout comparisons.

Network visualization requirements:

```text
Plotly 3D HTML: required
Static PNG: required
E/PV/SST/VIP labels: required
Per-area cylinder geometry validation: required
```

Geometry target:

```text
height approximately 1.5 mm
radius metadata 0.125 mm
observed node-cloud extent validated with tolerance
```

## Section 3: simulations and spectrolaminar readouts

The notebook runs three conditions:

```text
baseline
stimulus
tuned + stimulus
```

Each condition exports:

```text
activity suite PNG
activity suite Plotly HTML
spectrolaminar suite PNG
spectrolaminar suite Plotly HTML
```

The spectrolaminar suite uses cortical position from L4 as the shared depth axis.

## Section 4: AGSDR metrics and artifacts

The optimizer target is:

```yaml
target_rate_hz: 5.0
target_kappa: 0.0
```

Required JSON outputs:

```text
manifest.json
validation_report.json
metrics.json
asset_hashes.json
```

Required PNG outputs:

```text
figures/cortical_circuit_network.png
figures/activity_suite_baseline.png
figures/activity_suite_stimulus.png
figures/activity_suite_tuned.png
figures/spectrolaminar_baseline.png
figures/spectrolaminar_stimulus.png
figures/spectrolaminar_tuned.png
figures/optimization_summary.png
```

Required HTML outputs:

```text
html/izhikevich_waveform_control_panel.html
plotly/cortical_circuit_network.html
plotly/activity_suite_baseline.html
plotly/activity_suite_stimulus.html
plotly/activity_suite_tuned.html
plotly/spectrolaminar_baseline.html
plotly/spectrolaminar_stimulus.html
plotly/spectrolaminar_tuned.html
```

## Package-native calls

```python
import jaxfne as jtfne
```

Typical flow:

```python
cfg = jtfne.default_spectrolaminar_config(...)
model = jtfne.construct(cfg)
sim = jtfne.Simulation(...)
signals = model.simulate(sim)
```

Objective and optimizer:

```python
objective = jtfne.rate_synchrony_targets(
    target_rate_hz=5.0,
    target_kappa_synchrony=0.0,
    rate_weight=1.0,
    synchrony_weight=0.25,
)

opt = jtfne.agsdr(
    parameters={"drive_gain": (0.1, 1.5)},
    generations=3,
    population_size=2,
    seed=SEED,
)
```

## ReadTheDocs build

ReadTheDocs uses:

```text
.readthedocs.yaml
mkdocs.yml
docs/requirements.txt
```

The docs build command is:

```bash
mkdocs build --strict
```

The notebook link in this page opens the current GitHub notebook directly in Colab.
