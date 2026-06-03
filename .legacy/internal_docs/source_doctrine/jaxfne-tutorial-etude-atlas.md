# JAXFNE Tutorial and Etude Atlas

## Tutorial roles

- Suite: structured release/tutorial unit.
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

## Etude No. 1 current standard

Etude No. 1 is a legacy-inspired, jaxfne-native multi-laminar cortical AGSDR workflow.

Required spine:

```text
setup -> centralized config -> single-unit E/PV/SST/VIP warmup -> construct scaffold -> visualize 3D network -> simulate baseline/stimulus/tuned -> activity suites -> spectrolaminar suites -> AGSDR evidence -> export artifacts
```

Install cells:

```python
!pip install -q "jaxfne[viz]"
!pip install -q "jaxfne[viz] @ git+https://github.com/HNXJ/jaxfne.git@main"
```

Colab badge targets:

```text
blob/main/tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb
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

Etudes export all major knobs under `manifest["editable_inputs"]`:

```text
runtime, geometry, areas, layers, cell_types, cell_colors, cell_signs,
layer_fractions, drive, connectivity, field, probes, objective, optimizer,
stimulus, visualization, artifact_paths, truth_gates
```

Config cells may be long. Scientific API calls should show important defaults explicitly when useful.

## AGSDR evidence

`metrics.json` must include:

```yaml
best_score: finite_number
best_parameters: non_empty_object
tuning_status: string
same_model_unchanged: false
rate_improvement_hz: positive_number
kappa_improvement: number
```

Use supported tunable parameters such as `drive_gain`. Avoid unsupported knobs such as `noise_amplitude` unless the package supports them.

## Docs navigation

MkDocs nav includes:

```yaml
Etude No. 1 (Multi-Laminar Cortical AGSDR): tutorials/11_multi_laminar_cortical_agsdr.md
```

## Near-term package gaps

Move reusable notebook glue into package APIs:

```text
jtfne.vis.visualize_network_3d(...)
shared JSON-safe artifact helpers
shared metric registry for objectives/tutorial helpers
static network PNG helper with compatibility wrappers
```
Binary files repo_orig/jaxfne/__pycache__/__init__.cpython-313.pyc and repo_final/jaxfne/__pycache__/__init__.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/bridges.cpython-313.pyc and repo_final/jaxfne/__pycache__/bridges.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/builders.cpython-313.pyc and repo_final/jaxfne/__pycache__/builders.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/core.cpython-313.pyc and repo_final/jaxfne/__pycache__/core.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/emitters.cpython-313.pyc and repo_final/jaxfne/__pycache__/emitters.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/io.cpython-313.pyc and repo_final/jaxfne/__pycache__/io.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/objectives.cpython-313.pyc and repo_final/jaxfne/__pycache__/objectives.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/paradigm.cpython-313.pyc and repo_final/jaxfne/__pycache__/paradigm.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/presets.cpython-313.pyc and repo_final/jaxfne/__pycache__/presets.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/runtime.cpython-313.pyc and repo_final/jaxfne/__pycache__/runtime.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/sharding_utils.cpython-313.pyc and repo_final/jaxfne/__pycache__/sharding_utils.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/tutorial_utils.cpython-313.pyc and repo_final/jaxfne/__pycache__/tutorial_utils.cpython-313.pyc differ
Binary files repo_orig/jaxfne/__pycache__/validation.cpython-313.pyc and repo_final/jaxfne/__pycache__/validation.cpython-313.pyc differ
Binary files repo_orig/jaxfne/fields/__pycache__/__init__.cpython-313.pyc and repo_final/jaxfne/fields/__pycache__/__init__.cpython-313.pyc differ
Binary files repo_orig/jaxfne/fields/__pycache__/probes.cpython-313.pyc and repo_final/jaxfne/fields/__pycache__/probes.cpython-313.pyc differ
Binary files repo_orig/jaxfne/fields/__pycache__/proxy.cpython-313.pyc and repo_final/jaxfne/fields/__pycache__/proxy.cpython-313.pyc differ
Binary files repo_orig/jaxfne/optim/__pycache__/__init__.cpython-313.pyc and repo_final/jaxfne/optim/__pycache__/__init__.cpython-313.pyc differ
Binary files repo_orig/jaxfne/optim/__pycache__/agsdr.cpython-313.pyc and repo_final/jaxfne/optim/__pycache__/agsdr.cpython-313.pyc differ
Binary files repo_orig/jaxfne/optim/__pycache__/bounds.cpython-313.pyc and repo_final/jaxfne/optim/__pycache__/bounds.cpython-313.pyc differ
Binary files repo_orig/jaxfne/optim/__pycache__/core.cpython-313.pyc and repo_final/jaxfne/optim/__pycache__/core.cpython-313.pyc differ
Binary files repo_orig/jaxfne/optim/__pycache__/gsdr.cpython-313.pyc and repo_final/jaxfne/optim/__pycache__/gsdr.cpython-313.pyc differ
Binary files repo_orig/jaxfne/optim/__pycache__/gsgd.cpython-313.pyc and repo_final/jaxfne/optim/__pycache__/gsgd.cpython-313.pyc differ
Binary files repo_orig/jaxfne/optim/__pycache__/manifests.cpython-313.pyc and repo_final/jaxfne/optim/__pycache__/manifests.cpython-313.pyc differ
Binary files repo_orig/jaxfne/optim/__pycache__/sdr.cpython-313.pyc and repo_final/jaxfne/optim/__pycache__/sdr.cpython-313.pyc differ