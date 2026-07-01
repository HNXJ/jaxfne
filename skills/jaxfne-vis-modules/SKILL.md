---
name: jaxfne-vis-modules
description: >-
  Design, fix, or audit jaxfne.vis — raster, Vm/LFP/CSD/EEG/MEG traces, PSD,
  spectrogram, spectrolaminar suites, activity suites, connectivity
  weightmaps, 3D scaffold, and 2-photon proxy visualizations. Renamed from
  jaxfne-visualization-schema (2026-06-30). USE whenever the task mentions
  plot, figure, visualization, vis, raster, trace, LFP, CSD, EEG, MEG, EMM,
  PSD, spectrogram, spectrolaminar, activity suite, connectivity map,
  weightmap, Plotly, 3D, scaffold, or PNG/HTML export.
---

# jaxfne Vis Modules

USE FIRST: `catalog-glossary-jaxfne` §8, `jaxfne-neural-network` (for the
`Signals` object every vis function consumes).

## Purpose

Make visualization package-level, stable, and proxy-safe. Do not leave reusable
plotting behavior as notebook-only patches.

## Signal-driven signature (pass a `Signals`, get a `Figure`)

```python
sig = jtfne.simulate(model, duration_ms=1000.0, dt_ms=0.5, seed=0)
jtfne.vis.raster(sig); jtfne.vis.vm(sig); jtfne.vis.rate(sig); jtfne.vis.source(sig)
jtfne.vis.lfp(sig); jtfne.vis.lfp_traces(sig); jtfne.vis.csd(sig); jtfne.vis.csd_traces(sig)
jtfne.vis.eeg(sig); jtfne.vis.meg(sig); jtfne.vis.emm(sig)
jtfne.vis.psd(sig); jtfne.vis.spectrogram(sig); jtfne.vis.bandpower(sig)
jtfne.vis.connectivity(model)  # / connectivity_matrix
jtfne.vis.laminar_profile(sig)  # / layer_celltype_counts
jtfne.vis.geometry3d(model)  # / circuit3d / column_geometry
jtfne.vis.multi_area_layout(...); jtfne.vis.summary(sig); jtfne.vis.objective_report(...)
jtfne.vis.spectrolaminar(sig)          # 3-panel spectrolaminar from a Signals object
jtfne.vis.spectrolaminar_suite(sig)    # Suite No. 2 readout panel — preferred single-run readout
```

Each returns a matplotlib `Figure`; a `*_with_meta` variant returns a JSON-safe
metadata container alongside it.

**All 4 previously-dead-stub functions were fixed 2026-06-30** — they now
render real data instead of an empty title-only figure: `plot_laminar_field_interpolation`
(fields.py, imshow), `plot_spike_rasters` (rasters.py, scatter from nonzero spike
coords), `plot_continuous_traces` (traces.py, overlaid line plots),
`plot_spectrogram_profiles` (spectra.py, real `scipy.signal.spectrogram` +
pcolormesh). All 4 now return the `Figure` (previously returned `None`).

## Module ownership (actual `jaxfne/vis/` layout — verified on disk)

```text
rasters.py            -> spike rasters and event plots
raster_arrays.py      -> raw-array raster helpers
traces.py             -> Vm/rate/source/LFP/CSD/EEG/MEG/EMM traces
spectra.py             -> PSD, spectrogram, windowed_band_power
fields.py              -> spectrolaminar(), spectrolaminar_suite(), laminar field plots
canonical.py           -> canonical/report plot wrappers (incl. plot_band_power)
tutorial_panels.py     -> spectrolaminar_suite_3panel, activity_trace_suite (trial/specs-driven)
tutorial_array_plots.py -> array-driven quick plots (also wrapped in tutorial_utils)
network3d.py           -> Plotly 3D scaffold and camera/axis controls
layout.py              -> shared figure layout/grid helpers
plotly/*               -> interactive Plotly dashboards, manuscript figures
report_plots.py        -> script/report figure helpers
script_reports.py      -> batch report generation
hdp_diagnostics.py     -> HDP-specific diagnostic plots
plasticity_viz.py      -> plasticity/homeostasis visualization
exporters.py           -> figure export helpers
core.py                -> require_matplotlib, close_all, prepare_static_plot_matrix
```

There is **no** top-level `spectrolaminar.py` or `connectivity.py` under `vis/`.

## `jtfne.vis.tutorial_panels.*` (trial/specs-driven suites)

- `spectrolaminar_suite_3panel(specs, model, cfg, areas=..., output_dir=..., theme="dark")` → `{area: Figure}` (re-exported at `jtfne.vis.spectrolaminar_suite_3panel`)
- `activity_trace_suite(trials, cfg, ...)` — raster + LFP + CSD + PSD
- `visualize_laminar_column_3d(model, cfg, ...)`
- `visualize_network_3d(data, *, output_html=..., show_edges=...)` — **interactive Plotly** 3D network, HTML export, pan/zoom

## `jtfne.tutorial_utils.plot_*` (array-driven quick plots)

`plot_raster`, `plot_spectrolaminar_power(t, signal, freq_min, freq_max, n_freqs)`,
`plot_laminar_readout`, `plot_population_rate`, `plot_voltage_samples`,
`plot_connectivity_matrix`, `save_png(fig, name, fig_dir)`.

## Required package options (implement notebook hacks as parameters, not one-offs)

```python
jtfne.vis.spectrolaminar_suite_3panel(specs, model, cfg,
    relative_power_mode="per_frequency_depth_max", power_vmin=0.0, power_vmax=1.0, theme="light")
jtfne.vis.visualize_network_3d(neurons, theme="dark", z_flip=True,
    axis_aspect=(1, 1, 3), camera_eye=(1.7, -1.8, 1.2))
jtfne.vis.activity_trace_suite(trials, cfg, figure_size=(14, 16), dpi=180)
```

## Relative power rule

For reference-style spectrolaminar heatmaps, frequency stays in Hz and power is
normalized per frequency over depth. If power has shape `(n_freqs, n_contacts)`:

```python
relative_power = power / power.max(axis=1, keepdims=True)
```

Set heatmap color range to `0.0-1.0` for this mode.

## Proxy wording (always)

Titles/colorbars: `LFP proxy`, `CSD proxy`, `EEG-like proxy`, `MEG-like proxy`,
`spectrolaminar proxy readout`, `relative power`. Never title a proxy plot as
real EEG/MEG/CSD or calibrated amplitude.

## Release artifact rules

- PNG required for core figures; Plotly HTML optional.
- Save after post-processing figure size/theme/camera changes.
- Do not put matplotlib/plotly imports in core numerical modules (`core.py`, `fields/*`, `optim/*`).

## Stop conditions

```text
plotting code added to core/net/fields numerical kernels
vis function changes data/operator semantics silently
frequency axis confused with relative power normalization
contact/frequency axes transposed without shape check
proxy readout titled as real EEG/MEG/CSD
PNG artifact missing for release/tutorial figure
new plotting function that builds a Figure, sets only title/labels, never plots data (the dead-stub pattern fixed 2026-06-30)
```

## Related skills

- `jaxfne-neural-network` — the `Signals` object every function here consumes
- `jaxfne-spectrolaminar-suite` — scale path (N≥1000), crossover-regime caveats
