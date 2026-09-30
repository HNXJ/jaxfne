# jnwb transition inventory (todo item 0d, step 1)

Date 2026-09-29. Basis: jaxfne `dev` at `4223a0f`; jnwb 0.2.5 as installed from
PyPI (`C:/Python314/Lib/site-packages/jnwb`). `E:/repos/jnwb-dev` holds 0.1.8
and is not a git repository, so it is not the reference.

jnwb: data/signal → processing → analysis → results.
jaxfne: model/signal → implementation → simulation → results.

## Seam facts (observed)

| Quantity | jaxfne (`Signals`, `FieldOutput`) | jnwb | Adapter rule |
|---|---|---|---|
| time | `time_ms`, ms | onsets and spike times in s; windows and bins in ms | convert ms → s at the seam, once |
| sampling rate | implied by `dt_ms` | `fs` in Hz, required | `fs = 1000 / dt_ms`; refuse a non-uniform `time_ms` |
| spikes | dense counts, `(n_steps, n_units)` | per-unit 1D spike-time arrays (s) | count c at step k → c times `t0 + k*dt` (float64, from `dt_ms`, not float32 `time_ms`); non-integer values refused |
| V_m, sources | `(n_steps, n_units)` | time on axis 0 by default | pass as is, time axis 0 |
| LFP, CSD, phi_e | `*_proxy`, `(n_steps, n_contacts)`, `epistemic_level="RELATIVE_PROXY"` | `current_source_density_1d` needs volts, µm pitch and S/m; returns A/m³ | kept as `*_proxy` with `field_level`; never relabelled volts, so a volts-only jnwb call on a `RELATIVE_PROXY` view is the caller's error, stated in the module docstring |
| contact depths | `FieldOutput.contact_depths` | `pitch_um` scalar | passed raw; the depth unit is not declared, so no pitch is derived yet |
| neuron identity | `metadata["neuron_metadata"]` | unit metadata tables | pass the table; no renaming |
| trials | none on `Signals`; `TrialBatchResult` | `as_trials`, `epoch_continuous` | per-trial `Signals` → trial axis at the seam |

Python floor: jaxfne `>=3.11`, jnwb `>=3.12`. The extra needs a marker
(`jnwb; python_version >= "3.12"`) or jaxfne raises its floor.

## Operations

Class: **overlap** (both have it; migrate), **jaxfne-only** (keep in jaxfne),
**gap** (belongs in jnwb; add it there, not here).

| jaxfne | Class | jnwb counterpart | Note |
|---|---|---|---|
| `vis/core.welch_psd`, `vis/spectra.psd` | overlap, not equal | `compute_psd` | jaxfne `nperseg=256`; jnwb `nperseg=min(n, int(fs))` (20000 at 0.05 ms). Output changes; a gap in jnwb (no `nperseg` argument) or a declared output change |
| `vis/core.inband_power_mean`, `analysis/spectral.bandpower_jax` | overlap | `band_power`, `relative_power` | check band-edge inclusion |
| `vis/spectra.spectrogram`, `windowed_band_power` | overlap | `complex_tfr`, `morlet_wavelet` | Morlet ≠ STFT: different estimator |
| `vis/traces.csd`, `canonical.plot_csd` | overlap on proxy only | `current_source_density_1d` | physical units; only after calibration |
| `vis/core.binned_population_rate_hz` | overlap | `bin_spikes`, `gaussian_smooth_rate` | |
| `vis/rasters.*`, `raster_arrays.raster_from_arrays` | overlap | `raster_psth` | |
| `analysis/metrics.fano_factor`, `mean_pairwise_spike_correlation`, `burst_index` | gap | none found | move to jnwb spiking |
| `analysis/metrics.fleiss_kappa_binary` | gap | none found | jnwb statistics |
| `analysis/spectral.spectrolaminar_*_jax` | gap (jax) | `laminar.vflip`, `label_layers` | jnwb laminar is numpy; jax kernels stay for gradients |
| connectivity (Granger, PSI, TE, coherence, wPLI) | jnwb-only | `granger`, `phase_slope_index`, `transfer_entropy`, `wpli`, `imaginary_coherency` | new capability for jaxfne via the seam |
| decoding, RSA, trajectory, statistics | jnwb-only | `nested_cv_linear_svm`, `jrsa`, `compute_population_trajectory`, `cluster_permutation_test` | new capability via the seam |
| `vis/network3d`, `column_viewer`, `network_glow`, `network_inspect`, `fields.ei_circuit_diagram`, `multi_area_layout` | jaxfne-only | none | model structure, not data |
| `vis/hdp_diagnostics`, `plasticity_viz`, `pseudogenome_viewer` | jaxfne-only | none | model state and plasticity |
| `vis/evidence_*`, `export.py`, `report_plots` | jaxfne-only | `save_figure_suite` | receipts and manifests stay |
| `pynwb_compat.write_nwb`, `read_nwb` | wired 2026-09-29 | `read_nwb`, `unit_spike_times` | writes a `to_jnwb` view: Units + proxy `TimeSeries` (not `ElectricalSeries`, so `acquisition_channel` does not read proxies); reads via jnwb |

Scale: `jaxfne/vis` + `jaxfne/analysis` hold 11,799 lines.

## Consequence for the plan

- The step-3 equality bound does not hold for spectra: the estimator
  parameters differ. Either jnwb takes an `nperseg` argument (then equality
  within float tolerance), or the migration is a declared output change
  listed like a seal note.
- The adapter is pure numpy/metadata and imports jnwb lazily; it needs no
  jaxfne module other than `Signals` and `FieldOutput`.
