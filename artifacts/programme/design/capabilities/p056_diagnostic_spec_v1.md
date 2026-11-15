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
| column positive: inject a synthetic 15 Hz count into one deep class on the column geometry; R must equal the analytic ratio of kernel responses | done 2026-11-15 for L5E and L6E, apical and basal placement, all five layer contacts (rtol 3%), plus the whole-cell cancellation case; `tests/test_kernel_lfp_band_power.py` |
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

At equal drive and unit weights the deep kernels are within a factor 1 to 6 of L2/3, so geometry and kernel amplitude alone cannot give `R < 0.005`. Closing the gap to 0.005 needs a further power attenuation of 0.16/0.005 to 0.90/0.005 = 32 to 180; with power quadratic in drive amplitude that is an amplitude attenuation of 5.7 to 13.4 in the deep 8 to 25 Hz rate modulation or the deep synaptic current relative to L2/3 (inferred under the equal-drive assumption; rates not retained; a hypothesis, not a measured mechanism). That ranks rates and synaptic current above coherence radius for the intervention order.

Design caution (observed): synapses spread over the whole cell give kernels of about 1e-17 mV for L23E, L4E and L5E (an order 1e-12 below the apical or basal placements; consistent with near-complete cancellation of source and return currents, inferred) but 1e-5 to 1e-4 mV for L6E. The L6E exception is explained (observed): 7.5% of the L6E membrane area has its midpoint below the white-matter bound (z up to 1651 um against 1491.7), so a whole-cell weight range drops those compartments and the balance fails; L23E, L4E and L5E have 0% outside. A column must therefore place synapses by explicit depth ranges per class, never by the layer bounds alone. Synapse placement per connection class is therefore a design input.

## 5b. Smoke protocol (declared 2026-11-15 before any smoke run; the receipt is kept whether or not it passes)

- Configuration: N = 1000 split over the eight Hagen populations by a fixed table recorded in the receipt; dt 0.1 ms; 1.0 s simulated (10000 steps); seed 0; float32 unless the route requires x64 (then x64, stated); CPU.
- Budget (declared now): compile at most 180 s; execute at most 120 s; peak RSS at most 4 GB; output at most 50 MB.
- Timing: compile (first call minus second call) and execute (second call) reported separately, each with `block_until_ready`.
- Finite and coverage checks: all outputs finite; spike counts integer-valued and non-negative; per-class count array shape `[T, C]` equals the kernel's class axis; every population present.
- Input-engagement control (before any intervention is read): per-population external input actually reaches each population. Required: for every population, mean membrane input current or rate is non-zero and changes by the intended factor when its drive is scaled by 2 (a 2x drive on one population changes that population's rate by more than the seed spread; run this check with recurrent weights zeroed so that the other populations are expected unchanged, and again with recurrence on, reporting the downstream changes without a pass criterion). A deep population that does not respond to its own drive makes an intervention on it vacuous and stops the sequence.
- A failed smoke is reported with its log; nothing is retuned and re-labelled as the smoke.
- Result (2026-11-15, darwin arm64, jax 0.10.1, float32). The tree is at its file budget (2796), so no script or receipt file is added: the script is `git show a9d09921:scripts/p056_resource_smoke.py` (sha256 prefix 64dc5dda8a70523d; removed from the tip in the next commit, history keeps it), run on a clean tree at that commit, and the receipt follows in the next block, verbatim apart from whitespace: within the budget on every line. First call 6.37 s, second call 6.05 s (compile estimate 0.31 s), 6.05 s per 1.0 s simulated at N = 1000 with 100151 edges, peak RSS 0.58 GB; outputs finite, counts `[10000, 3]` non-negative integers, repeat run bit-identical. Engagement with recurrence zeroed: doubling the E drive moves the E rate 8.0 to 22.0 Hz (x2.75) and leaves PV (47.0) and SOM (10.0) unchanged. With recurrence on and iSTDP active: E 2.85, PV 24.8, SOM 6.46 Hz. Sizing, inferred from one point and an edge-linear cost: a 12 s run at N = 1000 is about 73 s; N = 4000 (about 1.6 M edges, 16x) is about 20 min per run; memory above N = 1000 is not measured (the route takes an edge list, so cost follows edges, not N squared), so a smoke at N = 2000 precedes it. Drive, weights and the E/PV/SOM table are arbitrary here and carry no P-056 meaning.

```json
{"schema":"jaxfne.p056_resource_smoke.v1","label":"resource smoke only: generic E/PV/SOM network, arbitrary weights and drive; says nothing about P-056","config":{"n_e":800,"n_pv":100,"n_som":100,"dt_ms":0.1,"n_steps":10000,"seed":0,"p_connect":0.1,"w_e_nS":1.0,"w_i_nS":3.0,"delay_steps":20,"drive_pA":300.0,"receptor_reversal_mV":[0.0,-80.0],"receptor_tau_ms":[2.0,5.0],"precision":"float32"},"budget":{"compile_s":180.0,"execute_s":120.0,"peak_rss_gb":4.0,"output_mb":50.0},"n_edges":100151,"timing_s":{"first_call":6.366013667080551,"second_call":6.054645749740303,"compile_estimate":0.3113679173402488,"execute":6.054645749740303,"zero_recurrence_base":6.078275667037815,"zero_recurrence_E_x2":6.164872499648482},"peak_rss_gb":0.578094482421875,"rates_hz":{"recurrent":{"E":2.8475,"PV":24.81,"SOM":6.46},"zero_recurrence_base":{"E":8.0,"PV":47.0,"SOM":10.0},"zero_recurrence_E_drive_x2":{"E":22.0325,"PV":47.0,"SOM":10.0}},"checks":{"spikes_bool":true,"counts_shape":[10000,3],"counts_nonnegative_integer":true,"w_final_finite":true,"H_final_finite":true,"populations_present":true,"deterministic_repeat":true},"engagement":{"E_rate_ratio_x2_over_base_zero_recurrence":2.7540625,"PV_rate_ratio":1.0,"SOM_rate_ratio":1.0},"within_budget":{"compile":true,"execute":true,"rss":true},"env":{"python":"3.13.7","platform":"macOS-27.0.1-arm64-arm-64bit-Mach-O","jax":"0.10.1","jaxlib":"0.10.1","numpy":"2.4.6","x64":false,"git_sha":"a9d099217f104cb9ab0a2784d8a4b85eb99a5a92","git_dirty":false}}
```

## 5c. What the AdEx route supports today (inventory 2026-11-15, read from source at `fdb62a93`; nothing run)

- Labels are `E`, `PV`, `SOM` only (`make_adex_params` refuses others, `emitters_adex.py:97-101`), so the 17 Billeh populations need a declared map to three classes; the module has no per-population N, no synaptic weight table and no positions (`billeh2020.py`).
- The HDP kernel (needed for iSTDP) takes one Poisson drive set (one weight, receptor, delay); per-population Poisson weights exist only in the plain kernel. Per-population current drive (`I_ext_pA`, `[N]` or `[T, N]`) is available in both.
- The kernel returns the spike raster `[T, N]` bool and has no per-step per-population count output. At N = 1000 and 10000 steps the raster is 10 MB, so counts are formed after the run (`raster @ onehot`); a scan-body change is needed only at large N.
- Each call re-traces the scan (no `jax.jit` in `emitters_adex.py`), so compile and execute timing in section 5b needs two calls with identical shapes; largest tested AdEx N is 100.
- Consequence: the N = 1000 smoke below is a resource smoke on a generic E/PV/SOM network (current drive, fixed table of E 800, PV 100, SOM 100, receptor reversals 0 and -80 mV). It measures cost, stability and input engagement. It says nothing about P-056, and its receipt is labelled so. The P-056 baseline additionally needs the population map, per-class N, synaptic weights and drive amplitudes: scientific inputs for Hamm, not defaults for an agent to choose.

## 6. Owner rulings needed before any candidate result is read

- The threshold for "deep alpha-beta present" (the issue gives none; proposal: `R_L5` or `R_L6 >= 0.05` and peak prominence above the shuffle null's 99th percentile). Without it only the baseline and controls are run.
- Billeh-to-Hagen population map, and the Billeh-to-{E, PV, SOM} class map the AdEx route requires.
- Per-class N, synaptic weights (nS) and external drive amplitudes for the baseline column (none is in the repository).
