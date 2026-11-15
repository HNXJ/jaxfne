# P-056 diagnostic specification v1 (DRAFT, 2026-11-15; unsigned, no run authorized)

Separately versioned from the pilot's unrecorded metric. Any change after a candidate result is observed becomes v2; v1 numbers are never retuned. Todo T-90. Parent issue: `artifacts/assurance/issue_log/ISSUE_LOG.md` P-056.

## 1. Estimand (frozen before any run)

Column: Billeh 2020 wiring, Hagen 2016 cells, AdEx conductance route, Vogels 2011 iSTDP settle. Layers L2/3, L4, L5, L6 map to populations e2, e4, e5, e6 plus their interneurons by the digit in the name (the Billeh-to-Hagen map is a design input, recorded with the run).

- Contacts: one per layer, at the layer midpoint of the Hagen source bounds `0, 81.6, 587.1, 922.2, 1170.0, 1491.7` um, scaled to the column.
- `P_l` = band-integrated Welch power of the LFP at contact l: 8 to 25 Hz (package default), Welch Hann, 1 s segments, 50% overlap, `detrend=False`, `fs` = 1000/dt, window = after dropping the first 2.0 s, keep at least 10.0 s (`_ref42_window`). Alpha-beta peak: prominence of the 8 to 25 Hz maximum above a 1/f fit of 2 to 50 Hz excluding 8 to 25 Hz.
- `R_l = P_l / P_L2/3`. Deficit as logged: `R_L5, R_L6 < 0.005`.
- Reference population: L2/3 contact. Normalization: ratio of band powers, never per-channel max (`bandpower_jax` divides by the channel maximum; not used here).
- Numerical tolerance: a repeat run with a different seed must reproduce `R_l` within a spread measured from 5 seeds before any intervention; interventions are judged against that spread, not a fixed number.

## 2. Measurement validity (done and to do)

| control | status |
|---|---|
| positive: known 15 Hz deep drive, tone power = analytic | done, `tests/test_kernel_lfp_band_power.py` (5e-8) |
| negative: zero modulation; time-shuffled drive | done (1e-14; 0.7%) |
| column positive: inject a synthetic 15 Hz count into one deep class on the column geometry; R must equal the analytic ratio of kernel responses | to do before the baseline |
| null: spike-time shuffle per population in the real column run; alpha-beta peak prominence must vanish | to do with the baseline |
| absent vs defective: report per-layer rate, spike count and `|H(f)|` at each contact next to `P_l`; deep rate 0 means absent activity, deep rate > 0 with `R` near 0 means kernel or geometry | to do with the baseline |

## 3. Sequence

1. Column positive control (no network simulation).
2. Baseline: settle the AdEx column, record per-layer rates and N, compute `R_l` for 5 seeds. Report the deficit whether or not it reproduces.
3. One mechanism at a time, each its own intervention row, baseline preserved, failures recorded: deep rates (drive or iSTDP target rate), deep kernel amplitude (peak current per class), coherence radius, longer settle.
4. Independent review before interpretation; numerical artefact vs mechanism stated per row.

## 4. Sizing (estimate only; nothing run)

Not retained: pilot N, dt, duration. Planning values: dt 0.1 ms, 12 s (settle 2 s plus 10 s) = 120000 steps; N = 4000 first, 10000 only if the smoke passes.

| quantity | formula | N = 4000 | N = 10000 |
|---|---|---|---|
| dense recurrent weights, f32 | 4 N^2 B | 64 MB | 400 MB |
| per-class spike counts for the kernel (not per-neuron rasters) | 4 T C B | 120000 x 12 classes: 5.8 MB | same |
| per-neuron raster if kept as bool | T N B | 480 MB | 1.2 GB |
| kernel tensor | 4 C P K B | below 10 MB | same |

Keep counts per class, not rasters (the kernel input is class counts). Compile time and step cost are unmeasured: the smoke is 1 s of simulated time at N = 1000, reporting wall time, peak RSS and compile seconds, then scaled by N^2 for the recurrent term. A full run starts only after this spec is reviewed and the smoke is read.

## 5. Geometry-only prediction (derived 2026-11-15; no network run; scripts in a session scratchpad, not committed)

Hagen 2016 reduced cells of L23E, L4E, L5E, L6E (`merge=3`) on the source bounds, contacts at the five layer midpoints (40.8, 334.4, 754.7, 1046.1, 1330.9 um), unit synaptic weights, 15 Hz drive of equal amplitude on one class at a time, 8 to 25 Hz Welch power as in section 1.

| source class (placement) | own-layer contact power / L2/3 class at the L2/3 contact |
|---|---|
| L5E apical-targeted | 0.90 |
| L5E basal-targeted | 0.58 |
| L6E apical-targeted | 0.16 |
| L6E basal-targeted | 0.58 |

At equal drive and unit weights the deep kernels are within a factor 1 to 6 of L2/3, so geometry and kernel amplitude alone cannot give `R < 0.005`. Power scales with drive squared, so the logged deficit needs the deep 8 to 25 Hz modulation or the deep synaptic current about 10 to 100 times smaller than L2/3 (inferred; rates not retained). That ranks rates and synaptic current above coherence radius for the intervention order.

Design caution (observed): synapses spread over the whole cell give kernels of about 1e-17 mV for L23E, L4E and L5E (an order 1e-12 below the apical or basal placements; consistent with near-complete cancellation of source and return currents, inferred) but 1e-5 to 1e-4 mV for L6E. The L6E exception is explained (observed): 7.5% of the L6E membrane area has its midpoint below the white-matter bound (z up to 1651 um against 1491.7), so a whole-cell weight range drops those compartments and the balance fails; L23E, L4E and L5E have 0% outside. A column must therefore place synapses by explicit depth ranges per class, never by the layer bounds alone. Synapse placement per connection class is therefore a design input.

## 6. Owner rulings needed before any candidate result is read

- The threshold for "deep alpha-beta present" (the issue gives none; proposal: `R_L5` or `R_L6 >= 0.05` and peak prominence above the shuffle null's 99th percentile). Without it only the baseline and controls are run.
- Billeh-to-Hagen population map.
