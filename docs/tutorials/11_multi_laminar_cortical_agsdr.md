# Tutorial: Multi-area Laminar Model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HNXJ/jaxfne/blob/main/tutorials/etudes/jaxfne_etude_no_1_base.ipynb)

A thin, Colab-ready, maximally customizable tutorial. Heavy computation lives in `jaxfne`; the notebook exposes the controls and runs entirely through the package engine (`import jaxfne as jtfne`) — there is no local simulator.

## What this tutorial does

1. **Explore neuron types and waveforms** — E (excitatory) vs PV / SST / VIP (inhibitory) Izhikevich action potentials, with editable per-type `a, b, c, d, drive`.
2. **Build a two-area laminar network** — 6 layers × 4 cell types per area (extend freely), with within-area recurrence and between-area projections. Defaults: thalamic input → L4 of V1; V1 L2/3 (E) **feedforward** → V4 L4; V4 L2/3 (E) **feedback** → V1 L5/6.
3. **AGSDR fine-tuning** — adaptively tune control toward a target firing rate while minimizing spike synchrony (kappa).
4. **Custom everything** — stimulus, stimulation target, lesions, cell-type parameters, drive, connectivity.
5. **Spectrolaminar motif play** — manipulate cfg, cell types, ratios, densities, drives, and waveforms and watch the alpha-beta / gamma depth profile respond.
6. **Knock-out experiments** — lesion a layer / cell type / area and measure the downstream effect.

## Run status

```yaml
artifact_class: tutorial
artifact_id: multi_area_laminar_workshop
run_status: tutorial_scaffold
claim_level: computational_scaffold
field_solver_status: linear_solver
field_claim_level: proxy_readout
physical_amplitude_calibrated: false
```

LFP/CSD outputs are laminar **proxy** readouts produced by a depth-dependent Gaussian leadfield over real per-neuron Izhikevich source traces. No PDE/field solver is executed, and no calibrated-amplitude claim is made. The spectrolaminar motif is **emergent** from the dynamics and the leadfield — it is not imposed.

## Colab setup

The first cell installs the package. Set `INSTALL_MODE` to `pypi`, `github-main`, `local-editable`, or `skip`:

```python
%pip install -q "jaxfne[viz]"                                      # pypi
%pip install -q "jaxfne[viz] @ git+https://github.com/HNXJ/jaxfne.git@main"  # github-main
```

Under `TFNE_SMOKE=1` (CI / local validation) the install mode is forced to `skip` so the checked-out code is used.

## Package-native path

The tutorial uses the `jaxfne.tutorial_utils` helpers end to end:

```python
import jaxfne as jtfne

cfg = jtfne.tutorial_utils.make_laminar_column_config(
    areas=("V1", "V4"),
    layer_cell_type_frac=LAYER_CELL_TYPE_FRAC,
    cell_type_izh_params=CELL_TYPE_IZH,     # per-type waveform (wider-E default)
    connectivity_spec=CONNECTIVITY_SPEC,    # inter-area feedforward / feedback
    lesion_spec=LESION_SPEC,                # knock-out rules
)
model = jtfne.tutorial_utils.build_laminar_column(cfg)
stimulus = jtfne.tutorial_utils.make_stimulus(kind="sine", duration_ms=cfg.duration_ms, dt_ms=cfg.dt_ms)
target = jtfne.tutorial_utils.select_cells(model, area="V1", layers=("L4",), cell_types=("E",))
trials = jtfne.tutorial_utils.simulate_laminar_trials(model, cfg, cfg.base_control, stimulus, target)
```

## Notebook structure

```text
1.  Install + import
2.  Runtime / scale controls (smoke / interactive / release)
3.  Column geometry (x, y, z, radius, L4 reference)
4.  Layers and per-layer neuron counts
5.  Per-layer cell-type ratios + per-cell-type Izhikevich params (CELL_TYPE_IZH)
5b. Inter-area connectivity (CONNECTIVITY_SPEC) and lesions (LESION_SPEC)
6.  Build the jaxfne configuration
7.  Izhikevich control panel
7b. Single-cell E/PV/SST/VIP waveform explorer
8.  Build model + interactive 3D cortical scaffold
9.  Stimulus, target cells, simulation
10. Activity suite (raster / Vm / extracellular proxy / CSD-proxy / PSD)
11. Spectrolaminar relative-power motif (A density / B power spectrum / C alpha-beta vs gamma)
12. Export artifacts (manifest / metrics / validation / hashes)
13. AGSDR fine-tuning (target firing rate + minimize kappa)
14. Knock-out experiment
```

## Customization controls

| Control | Where | Effect |
| --- | --- | --- |
| `CELL_TYPE_IZH` | §5 | Per-cell-type `a, b, c, d, drive` (waveform / frequency content) |
| `LAYER_CELL_TYPE_FRAC`, `LAYER_COUNT_FRAC` | §4–5 | Per-layer composition and density |
| `CONNECTIVITY_SPEC` | §5b | Inter-area feedforward / feedback projections |
| `LESION_SPEC` | §5b / §14 | Silence a layer / cell type / area (knock-out) |
| `STIMULUS_*`, `TARGET_*` | §9 | Custom stimulus and stimulation target |
| `base_control` gains | §6 | `local_exc_gain`, `local_inh_gain`, `feedforward_gain`, `feedback_gain` |
| `tune_laminar_agsdr(...)` | §13 | AGSDR adaptive-greedy control tuning |

## AGSDR tuning

`tune_laminar_agsdr` applies the AGSDR algorithm (adaptive greedy selection / deselection with stochastic restart) to the control vector, scoring each candidate with the real `population_rate_hz` and `kappa_synchrony` from short simulations:

```python
best_control, history = jtfne.tutorial_utils.tune_laminar_agsdr(
    model, cfg, target_rate_hz=10.0, kappa_weight=1.0,
    parameters={"local_exc_gain": (0.3, 2.5),
                "local_inh_gain": (0.3, 2.5),
                "feedforward_gain": (0.0, 3.0)},
    generations=8, population_size=6,
)
```

This tunes the scaffold's proxy control knobs; it is distinct from `jtfne.agsdr` + `Model.tune`, which optimizes the core-engine `Model`.

## Knock-out experiment

```python
import dataclasses
cfg_lesion = dataclasses.replace(cfg, lesion_spec=({"area": "V1", "layers": ("L2", "L3"), "cell_types": ("E",)},))
trials_lesion = jtfne.tutorial_utils.simulate_laminar_trials(model, cfg_lesion, cfg.base_control, stimulus, target)
```

The lesioned population is silenced for the run, and per-area firing rates are compared to the intact network. Inter-area projections are modulatory, so knock-out of a feedforward/feedback source shifts the downstream area's rate by a measurable amount.

## Artifacts

```text
manifest.json, metrics.json, validation_report.json   (JSON-safe, finite)
scaffold_3d.html                                       (interactive Plotly)
activity_suite.png                                     (raster / Vm / proxy / CSD / PSD)
spectrolaminar_initial_<area>.png                      (3-panel motif per area)
```

## ReadTheDocs build

```bash
mkdocs build --strict
```

The Colab badge above opens the current `main` notebook directly.
