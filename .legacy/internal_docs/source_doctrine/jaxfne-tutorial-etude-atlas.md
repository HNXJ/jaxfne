# JAXFNE Tutorial and Etude Atlas

## Tutorial roles

- Suite: structured release/tutorial unit.
- Tutorial: public-facing runnable learning path.
- Etude: full-detail workflow with explicit configuration, diagnostics, visualizations, artifacts, and execution receipts.

## Hard gates

```yaml
duration_ms: ">=1000 in full mode"
dt_ms: 0.1
dtype: float32
seed: deterministic
canonical_import: import jaxfne as jtfne
package_native_path: true
local_scientific_engines: false
finite_outputs: true
strict_json: true
png_figures: required
plotly_html: optional
proxy_safe_titles: true
```

## Current multi-area laminar tutorial standard

Required spine:

```text
setup -> centralized Config -> single-unit warmup -> construct Net -> visualize 3D scaffold -> simulate baseline/stimulus/tuned -> activity suites -> spectrolaminar suites -> AGSDR trainer evidence -> export artifacts
```

## Required figures

```text
figures/cortical_circuit_network.png
figures/activity_suite_baseline.png
figures/activity_suite_stimulus.png
figures/activity_suite_tuned.png
figures/spectrolaminar_baseline.png
figures/spectrolaminar_stimulus.png
figures/spectrolaminar_tuned.png
plotly/cortical_circuit_network.html optional
```

Every visualization cell must display the figure and save an artifact.

## Editable-input export

Tutorials export major knobs under `manifest["editable_inputs"]` or the unified Config export:

```text
runtime, geometry, areas, layers, cell_types, cell_params, mechanisms,
connections, lesions, probes, paradigm, objective_outputs, trainables,
optimizer, visualization, artifact_paths, status_metadata
```

## AGSDR/trainer evidence

`metrics.json` includes:

```yaml
best_score: finite_number
best_parameters: non_empty_object
training_status: string
same_model_unchanged: false
rate_mean_hz: finite_number
rate_max_hz: finite_number
rate_stable: bool
objective_outputs: object
```

Use declared trainable paths such as:

```text
cell.E.drive
cell.PV.noise
conn.local_exc_gain
conn.feedforward_gain
mechanism.AMPA.g
```

Unsupported knobs must fail validation before training.

## Visualization doctrine

`jaxfne.vis` owns:

```text
raster
voltage traces
source traces
LFP/CSD/EEG/MEG proxy traces
PSD
spectrogram
spectrolaminar suite
activity suite
connectivity weight maps
network 3D
2-photon image proxy
```

Notebook-local visualization code may format one-off figures, but reusable plotting options belong in `jaxfne.vis`:

```text
dark theme
axis flip/range/aspect/camera
activity height/scale
relative_power_mode="per_frequency_depth_max"
trained-vs-initial comparison
```

## Docs navigation

Use public user-facing labels such as `Tutorial: Multi-area Laminar Model`. File paths may keep legacy names if tests and Colab links depend on them.
