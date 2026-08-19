# Visualization API (`jaxfne.vis`)

The `jaxfne.vis` package provides matplotlib/plotly figure generation for the
standard workflows. It is a **lazy** import — `import jaxfne` does not load
matplotlib or plotly; they are pulled in only when `jaxfne.vis` is first used
(install the `viz` extra for these backends).

All visualization is derived from the package's public data structures
(`Signals`, `Model`, `Objective`, `TuneResult`, `NeuronalTensor`,
`PseudoGenome` phenotype). No figure duplicates a scientific engine locally —
figures plot the package's own outputs.

## Entry points

- `jtfne.vis.plot_raster(...)`, `plot_population_rate(...)`,
  `plot_membrane_potentials(...)`, `plot_lfp(...)`, `plot_csd(...)`,
  `plot_psd(...)`, `plot_spectrogram(...)`, `plot_band_power(...)`,
  `plot_depth_profile(...)`, `plot_connectivity(...)`,
  `plot_objective_history(...)` — canonical single-figure plotters.
- `jtfne.vis.circuit3d(...)` / `visualize_network_3d(...)` — 3-D circuit
  layout.
- `jtfne.vis.traces.*` — `vm`, `rate`, `source`, `lfp`, `csd`, `eeg`, `meg`,
  `emm`, `summary` (+ `*_with_meta`) and multi-trace/panel helpers.
- `jtfne.vis.rasters.*` — spike rasters.
- `jtfne.vis.spectra.*` — PSD / spectrogram / windowed band power.
- `jtfne.vis.fields.*` — laminar field, spectrolaminar, connectivity, EI
  circuit, objective-report panels.
- `jtfne.vis.hdp_diagnostics.*` — HDP H-trace / dH-component /
  bifurcation / gain-sweep / LFP-by-depth panels.
- `jtfne.vis.report_plots.*` — `dual_raster_comparison`,
  `optimization_progress_line`, `spike_grid_heatmap`, `gain_matrix_heatmap`,
  `agsdr_rate_tuning_panel_grid`, PDF helpers.
- `jtfne.vis.export_figure(fig, path, formats=...)` /
  `jtfne.vis.export_figures(figures, output_dir, formats=...)` — export to
  html/png/svg.
- `jtfne.vis.plotly_*` and `jtfne.vis.tutorial_*` helpers for tutorial/script
  reports.

## Backend

`jaxfne.vis` selects matplotlib or plotly per callable. All figure objects are
standard library figure handles; export via `export_figure`/`export_figures`
or the host tool (e.g. `fig.write_image` for plotly, `fig.savefig` for
matplotlib).

## Notes

- Figures are proxy-aware: amplitude/unit labels follow the
  [relative-quantity grammar](../doctrine/relative_quantity_grammar.md) — a proxy
  readout is never labeled as a calibrated physical measurement.
- `kaleido` is not imported by the package; plotly PNG export needs it
  installed separately (optional).
