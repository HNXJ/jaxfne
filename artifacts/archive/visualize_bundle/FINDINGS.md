# Standard simulation visualization bundle — findings (artifact-backed)

Head: 350730a -> current (visualize bundle additive)

## Contract
`signals = jtfne.simulate(model)` is unchanged. `jtfne.visualize(model, signals)` is an **optional** post-hoc layer.
No numerics added, no field solver change, no calibration claim. Receipt: `receipt.json`.

## 8 panels (via existing vis layer)
- `01_network_structure`: positions/layers/E-I (geometry3d / plot_network_3d)
- `02_connectivity`: matrix/sparse (edge_list scatter + weight hist)
- `03_parameter_summary`: weights/delays/state ownership + calibration
- `04_raster`: spike raster
- `05_population_rates`: mean rate + per-type
- `06_state_traces`: V, H, W (H/W optional, from last_hdp_diagnostics)
- `07_source`: source proxy Q(t) + |Q|
- `08_field_probe`: LFP-like (lfp_proxy heatmap)

Reused vis entry points: geometry3d, connectivity_matrix, raster, rate, vm, source, lfp, plot_network_3d, plot_raster, plot_population_rate, plot_lfp.

## Backend
- `backend="static"` -> matplotlib (Agg-safe) — default
- `backend="plotly"` -> plotly.go.Figure (requires jaxfne[viz])
- `backend="both"`  -> 16-figure bundle (`*_static` + `*_plotly`)

## Large networks (deterministic sampling)
- LARGE_N_GEOMETRY=5000 -> sampled 3D scatter
- LARGE_N_CONNECTIVITY=600 -> downsampled matrix or sparse scatter capped at 8000 edges
- LARGE_N_RASTER=2000 / MAX_RASTER_POINTS=200000 -> spike-dot subsample

Evidence: `static/` (N=80, ~0.5s) + `large_network/` (N=2000, ~0.4s) exported without OOM; dense NxN never allocated for N>600.

## State traces (V, H, W)
- Always shows V sample traces.
- When enable_hdp=True (or model.last_hdp_diagnostics() present) adds H mean+/-std and W mean-evolution / hist. Verified with HDP example (`hdp_example/`).

## Source / Field
- Source panel = mean+/-std or sampled per-neuron Q(t) + |Q| summary.
- Field panel = LFP proxy heatmap (or "field not recorded" placeholder). Backed by signals.field.lfp_proxy.

## Zero overhead when unused
Fresh `import jaxfne.core` loads no matplotlib/plotly — test_simulation_engine_has_zero_graphics_overhead passes. `jtfne.visualize` is lazy via _RuntimeModuleWrapper and lazily imports matplotlib/plotly inside panels only.

## Artifacts
- `static/png/*.png` — 8 static figures for N=80
- `plotly/html/*.html` + `plotly/png/*.png` — interactive variants
- `both/` — 16 figures
- `large_network/png/*.png` — 8 figures for N=2000 (sampling active)
- `receipt.json` — machine-readable contract + timings + manifests

Verifiable: `python -m pytest tests/test_v0321_migration_boundaries.py::test_simulation_engine_has_zero_graphics_overhead -xvs`
