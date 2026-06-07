<!--
Updated jaxfne project-source bundle.
Generated from attached repo zip: jaxfne-pub-ed08-tutorial-atlas-coverage.zip
Zip SHA256: ebcc98621b0542a9fca1de1ad5790d6508258fe66c63365a95cefe3bbbde6761
Repo checklist SHA: 9a8c7db58f588bde9f5e8c31b664d56c4982958e
Repo checklist branch: pub/ed08-tutorial-atlas-coverage
jaxfne version: 0.3.29
Generated UTC: 2026-06-07T22:34:39Z
-->
# JAXFNE Tutorial and Etude Atlas

## Roles

- Suite: structured release/tutorial unit.
- Etude: full-detail workflow with explicit configuration, diagnostics, visualizations, artifacts, and execution receipts.

## Hard gates

```yaml
duration_ms: >=1000 in full mode
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

## Latest tutorial atlas status from inspected zip

```text
tutorial files: 24
notebooks: 18
docs/tutorials pages: 16 markdown pages plus index/standards/output contract
ED8: tutorial atlas coverage implemented
remaining tutorial publication work: rerun receipts and output manifests on live cur
```

## Tutorial stack observed

| Path family | Role |
|---|---|
| `tutorials/jaxfne_v031_single_neuron.ipynb` to `tutorials/jaxfne_v038_lfp_csd_like_readout.ipynb` | v0.3 scenario spine tutorials |
| `tutorials/jaxfne_v0310_eeg_meg_emm_proxy_bundle.ipynb` | multimodal proxy bundle |
| `tutorials/jaxfne_v0313_omission_oddball.ipynb` | omission/oddball scaffold |
| `tutorials/jaxfne_suite_no_*.ipynb` | suite-level curriculum notebooks |
| `tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb` | full multi-laminar cortical AGSDR Etude |
| `tutorials/etudes/jaxfne_etude_no_2_spectrolaminar_power.ipynb` | spectrolaminar power Etude |

## Etude No. 1 current standard

Required spine:

```text
setup -> centralized config -> single-unit E/PV/SST/VIP warmup -> construct scaffold -> visualize 3D network -> simulate baseline/stimulus/tuned -> activity suites -> spectrolaminar suites -> AGSDR evidence -> export artifacts
```

Install cells:

```python
!pip install -q "jaxfne[viz]"
!pip install -q "jaxfne[viz] @ git+https://github.com/HNXJ/jaxfne.git@main"
```

Required Etude artifacts:

```text
figures/cortical_circuit_network.png
figures/activity_suite_baseline.png
figures/activity_suite_stimulus.png
figures/activity_suite_tuned.png
figures/spectrolaminar_baseline.png
figures/spectrolaminar_stimulus.png
figures/spectrolaminar_tuned.png
plotly/cortical_circuit_network.html optional
manifest.json
validation_report.json
metrics.json
asset_hashes.json
```

PNG is required for docs/GitHub/PDF stability. Plotly HTML may augment; it never replaces PNG.

## Editable-input export

Etudes export all major knobs under `manifest["editable_inputs"]`:

```text
runtime, geometry, areas, layers, cell_types, cell_colors, cell_signs,
layer_fractions, drive, connectivity, field, probes, objective, optimizer,
stimulus, visualization, artifact_paths, truth_gates
```

## AGSDR evidence

`metrics.json` includes:

```yaml
best_score: finite_number
best_parameters: non_empty_object
tuning_status: string
same_model_unchanged: false
rate_improvement_hz: positive_number
kappa_improvement: number
```

Use supported tunable parameters such as `drive_gain`. Do not use unsupported knobs such as `noise_amplitude` unless the package exposes them.

## Tutorial-to-package rule

Tutorials may contain plotting, display, path, and formatting glue. Reusable scientific logic belongs in `jaxfne`: simulator kernels, source operators, probe/readout operators, objective engines, optimizer engines, field solvers, JSON-safe exporters, and metric registries.

## Near-term package gaps

- Keep `jtfne.vis.visualize_network_3d(...)` available and covered by compatibility tests.
- Move shared JSON-safe artifact helpers into package utilities.
- Share metric registries between objectives and tutorials.
- Keep static network PNG helpers package-native.
- Preserve API wrappers when moving notebook helpers into package.
