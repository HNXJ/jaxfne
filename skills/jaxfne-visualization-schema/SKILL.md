---
name: jaxfne-visualization-schema
summary: Design, fix, or audit jaxfne.vis for raster, Vm/LFP/CSD/EEG/MEG traces, PSD, spectrogram, spectrolaminar suites, activity suites, connectivity weightmaps, 3D scaffold, and 2-photon proxy visualizations.
trigger: Use whenever the task mentions plot, figure, visualization, vis, raster, spike raster, Vm, trace, LFP, CSD, EEG, MEG, EMM, PSD, spectrogram, frequency, relative power, spectrolaminar, activity suite, connectivity map, weightmap, Plotly, 3D, scaffold, camera, dark theme, axis flip, z axis, contact count, 2-photon, PNG, or HTML.
---

# jaxfne Visualization Schema

## Purpose

Make visualization package-level, stable, and proxy-safe. Do not leave reusable plotting behavior as notebook-only patches.

## Visualization taxonomy

`jaxfne.vis` should own:

```text
raster
vm_trace
rate_trace
source_trace
lfp_trace
csd_trace
eeg_trace
meg_trace
emm_proxy_trace
PSD
spectrogram
spectrolaminar_suite
activity_suite
connectivity_weightmap
connectivity_graph
network3d
2_photon_image_proxy
trained_vs_initial_comparison
```

## Module ownership

```text
rasters.py          -> spike rasters and event plots
traces.py           -> Vm/rate/source/LFP/CSD/EEG/MEG/EMM traces
spectra.py          -> PSD and spectrogram
spectrolaminar.py   -> laminar depth-frequency suites and profile crossing
connectivity.py     -> weight maps, edge tables, graph views
network3d.py        -> Plotly 3D scaffold and camera/axis controls
tutorial_panels.py  -> high-level tutorial suites only
```

## Required package options

Implement notebook hacks as parameters:

```python
jtfne.vis.spectrolaminar_suite_3panel(
    specs,
    model,
    cfg,
    relative_power_mode="per_frequency_depth_max",
    power_vmin=0.0,
    power_vmax=1.0,
    theme="light",
)
```

```python
jtfne.vis.visualize_network_3d(
    neurons,
    theme="dark",
    z_flip=True,
    axis_aspect=(1, 1, 3),
    camera_eye=(1.7, -1.8, 1.2),
)
```

```python
jtfne.vis.activity_trace_suite(
    trials,
    cfg,
    figure_size=(14, 16),
    dpi=180,
)
```

## Relative power rule

For reference-style spectrolaminar heatmaps, frequency remains in Hz and power is normalized per frequency over depth.

If power has shape `(n_freqs, n_contacts)`:

```python
relative_power = power / power.max(axis=1, keepdims=True)
```

Set heatmap color range to `0.0-1.0` for this mode.

## Proxy wording

Titles and colorbars should use proxy-safe labels:

```text
LFP proxy
CSD proxy
EEG-like proxy
MEG-like proxy
spectrolaminar proxy readout
relative power
```

Do not title proxy plots as real EEG/MEG/CSD or calibrated amplitude.

## Release artifact rules

- PNG required for core figures.
- Plotly HTML optional.
- Save after post-processing figure size/theme/camera changes.
- Do not put matplotlib/plotly imports in core numerical modules.

## Stop conditions

```text
plotting code added to core/net/fields numerical kernels
vis function changes data/operator semantics silently
frequency axis confused with relative power normalization
contact/frequency axes transposed without shape check
proxy readout titled as real EEG/MEG/CSD
PNG artifact missing for release/tutorial figure
```
