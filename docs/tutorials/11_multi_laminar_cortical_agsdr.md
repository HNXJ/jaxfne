# Etude No. 1: Multi-Laminar Cortical AGSDR

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb)

Artifact class: Etude. This is a full-detail, legacy-inspired, jaxfne-native workflow. It is not a Suite-numbered tutorial.

## Scope

```yaml
artifact_class: etude
artifact_id: etude_no_1
truth_mode: truth_safe_unverified
claim_level: computational_scaffold
field_solver_status: laminar_proxy_no_pde
physical_amplitude_claim_allowed: false
```

The Etude demonstrates a package-native scaffold: configure -> construct -> simulate -> visualize -> optimize. It uses simulated/proxy readouts and does not report calibrated LFP/CSD/EEG/MEG amplitudes or solved PDE fields.

## Install cells

```python
!pip install -q "jaxfne[viz]"
```

```python
!pip install -q "jaxfne[viz] @ git+https://github.com/HNXJ/jaxfne.git@main"
```

## Learning objectives

1. Use a centralized full-detail config as the edit anchor.
2. Run single-unit E/PV/SST/VIP warmup with package-native emitters.
3. Construct a multi-area laminar scaffold.
4. Simulate baseline, stimulus, and tuned conditions.
5. Visualize declared/simulated geometry, activity, and spectrolaminar proxy readouts.
6. Tune with AGSDR toward 3.5 Hz and kappa synchrony 0.0.
7. Export JSON-safe manifest, validation, metrics, hashes, PNG figures, and optional Plotly HTML.

## Required notebook structure

```text
setup
centralized config
single-unit warmup
construct model
3D network visualization
baseline simulation
stimulus simulation
AGSDR tuning
activity suites: baseline/stimulus/tuned
spectrolaminar suites: baseline/stimulus/tuned
artifact export
validation summary
```

## Configuration domains

Expose these as named values or dictionaries and export them under `manifest["editable_inputs"]`:

```text
runtime, geometry, areas, layers, cell_types, cell_colors, cell_signs,
layer fractions, native drive, connectivity metadata, field/proxy metadata,
probes, objective, optimizer, stimulus, visualization, artifact paths, truth gates
```

Config cells may exceed normal line limits. Scientific calls should show important defaults explicitly.

## Package-native calls

```python
import jaxfne as jtfne
```

Warmup:

```python
jtfne.izhikevich_params_from_labels(...)
jtfne.simulate_eig_izhikevich(...)
```

Model and simulation:

```python
cfg = jtfne.default_spectrolaminar_config(...)
model = jtfne.construct(cfg)
sim = jtfne.Simulation(...)
signals = model.simulate(sim)
```

Targeting and stimulus:

```python
target_indices = jtfne.select_neurons(model, area="V1", layer=None, cell_type="E")
stim = jtfne.stimulus_schedule(events, n_neurons=model.summary()["n_units"])
```

Objective and optimizer:

```python
objective = jtfne.rate_synchrony_targets(
    target_rate_hz=3.5,
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

## Visualization requirements

Visualization is part of the user-facing evidence. Each visualization cell must display the figure and save an artifact.

Required outputs:

```text
figures/cortical_circuit_network.png
plotly/cortical_circuit_network.html optional
figures/activity_suite_baseline.png
figures/activity_suite_stimulus.png
figures/activity_suite_tuned.png
figures/spectrolaminar_baseline.png
figures/spectrolaminar_stimulus.png
figures/spectrolaminar_tuned.png
```

Network visualization should use a package-level API such as `jtfne.vis.visualize_network_3d(...)` for interactive HTML and a stable matplotlib path for PNG. Titles must say simulated/proxy/declared geometry.

## Artifact outputs

Output directory:

```text
outputs/etude_no_1/
```

Required files:

```text
manifest.json
validation_report.json
metrics.json
asset_hashes.json
figures/*.png
plotly/*.html optional
```

`metrics.json` must include:

```yaml
baseline_rate_hz: number
stimulus_rate_hz: number
tuned_rate_hz: number
target_rate_hz: 3.5
baseline_kappa_synchrony: number
stimulus_kappa_synchrony: number
tuned_kappa_synchrony: number
target_kappa_synchrony: 0.0
best_score: finite_number
best_parameters: non_empty_object
tuning_status: string
same_model_unchanged: false
rate_improvement_hz: positive_number
kappa_improvement: number
```

## Failure modes

| Failure mode | Cause | Action |
|---|---|---|
| Tuning returns unchanged model | unsupported parameter or unscorable objective | use `drive_gain`; verify finite `best_score` and non-empty `best_parameters` |
| AGSDR scores are null | metric names not backed by evaluator | use fixed `rate_synchrony_targets` path |
| Figure cell fails in bare kernel | missing viz dependencies | install `jaxfne[viz]` |
| Colab opens stale content | badge points to `dev` or old path | use `blob/main/tutorials/etudes/...` |
| Warmup is silent/saturated | native drive too low/high | edit `WARMUP_DRIVE` |
| No improvement | budget too small or bounds too narrow | increase AGSDR generations/population or widen `drive_gain` bounds |

## Acceptance checks

```bash
python -m compileall -q jaxfne tests examples scripts
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest tests/ -q --tb=line
python scripts/audit_notebooks_and_assets.py --check
TFNE_SMOKE=1 nbclient tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb
TFNE_SMOKE=0 nbclient tutorials/etudes/jaxfne_etude_no_1_multi_laminar_cortical_agsdr.ipynb
mkdocs build --strict
```
Binary files repo_orig/examples/__pycache__/00_generalized_izhikevich_3d_smoke.cpython-313.pyc and repo_final/examples/__pycache__/00_generalized_izhikevich_3d_smoke.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/00_minimal_column.cpython-313.pyc and repo_final/examples/__pycache__/00_minimal_column.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/01_generalized_readout_smoke.cpython-313.pyc and repo_final/examples/__pycache__/01_generalized_readout_smoke.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/01_source_field_manifest.cpython-313.pyc and repo_final/examples/__pycache__/01_source_field_manifest.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/02_generalized_vis_smoke.cpython-313.pyc and repo_final/examples/__pycache__/02_generalized_vis_smoke.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/02_omission_scaffold.cpython-313.pyc and repo_final/examples/__pycache__/02_omission_scaffold.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/02_spectrolaminar_oddball_scaffold.cpython-313.pyc and repo_final/examples/__pycache__/02_spectrolaminar_oddball_scaffold.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/03_jaxley_bridge_smoke.cpython-313.pyc and repo_final/examples/__pycache__/03_jaxley_bridge_smoke.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/03_objective_and_tune_smoke.cpython-313.pyc and repo_final/examples/__pycache__/03_objective_and_tune_smoke.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/03_single_neuron_multimodal_probe.cpython-313.pyc and repo_final/examples/__pycache__/03_single_neuron_multimodal_probe.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/04_blackbox_tuning_loop.cpython-313.pyc and repo_final/examples/__pycache__/04_blackbox_tuning_loop.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/04_two_neuron_ei_multimodal.cpython-313.pyc and repo_final/examples/__pycache__/04_two_neuron_ei_multimodal.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/05_dataset_bridge_manifest.cpython-313.pyc and repo_final/examples/__pycache__/05_dataset_bridge_manifest.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/05_network_100_ei_multimodal.cpython-313.pyc and repo_final/examples/__pycache__/05_network_100_ei_multimodal.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/06_edge_list_recurrent_backend.cpython-313.pyc and repo_final/examples/__pycache__/06_edge_list_recurrent_backend.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/07_jaxley_trace_bridge.cpython-313.pyc and repo_final/examples/__pycache__/07_jaxley_trace_bridge.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/v031_single_izhikevich_neuron.cpython-313.pyc and repo_final/examples/__pycache__/v031_single_izhikevich_neuron.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/v032_single_neuron_parameter_sweep.cpython-313.pyc and repo_final/examples/__pycache__/v032_single_neuron_parameter_sweep.cpython-313.pyc differ
Binary files repo_orig/examples/__pycache__/v033_two_neuron_ei_multimodal.cpython-313.pyc and repo_final/examples/__pycache__/v033_two_neuron_ei_multimodal.cpython-313.pyc differ